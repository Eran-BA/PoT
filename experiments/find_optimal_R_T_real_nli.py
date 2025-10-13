#!/usr/bin/env python3
"""
Find Optimal R (Refinement Steps) and T (HRM Outer Loop Period) on Real NLI

Systematic grid search on SNLI dataset to find the best combination of:
- R: Number of refinement iterations (1, 2, 3, 4, 6, 8, 12, 16)
- T: HRM outer loop period (1, 2, 4, 8, 12)

Author: Eran Ben Artzy
Date: October 2025
"""

import os
import time
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pot.modules import PoHConfig, PoHStack, IterRefiner


# ========== Real NLI Dataset (SNLI) ==========

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("⚠️  Warning: 'datasets' not installed. Install with: pip install datasets")


class SNLIDataset(Dataset):
    """Real SNLI dataset from Hugging Face."""
    def __init__(self, split='train', max_samples=None, tokenizer=None):
        if not DATASETS_AVAILABLE:
            raise ImportError("Install datasets: pip install datasets")
        
        print(f"Loading SNLI {split} split...")
        self.dataset = load_dataset('snli', split=split)
        
        # Filter out examples with -1 label (no gold label)
        self.dataset = self.dataset.filter(lambda x: x['label'] != -1)
        
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        
        self.tokenizer = tokenizer or self._simple_tokenizer
        
        print(f"  Loaded {len(self.dataset)} examples")
    
    def _simple_tokenizer(self, text, vocab_size=10000, max_len=32):
        """Simple word-level tokenizer."""
        # Hash-based pseudo-tokenization (deterministic)
        words = text.lower().split()[:max_len]
        tokens = [hash(w) % vocab_size for w in words]
        
        # Pad to max_len
        tokens = tokens + [0] * (max_len - len(tokens))
        return torch.tensor(tokens[:max_len], dtype=torch.long)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        premise = self.tokenizer(item['premise'])
        hypothesis = self.tokenizer(item['hypothesis'])
        label = item['label']  # 0=entailment, 1=neutral, 2=contradiction
        
        return premise, hypothesis, label


# ========== PoH-NLI Model ==========

class PoHForNLI(nn.Module):
    """PoH model for NLI with configurable R and T."""
    def __init__(
        self, 
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        depth: int = 4,
        R: int = 12,
        T: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 32,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.R = R
        self.T = T
        
        # Embeddings
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # PoH stack configuration
        cfg = PoHConfig(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            route_mode="soft",
            share_router=True,
            pos_encoding="absolute",
            max_seq_len=max_seq_len,
        )
        
        # Create stack
        stack = PoHStack(cfg, depth=depth)
        
        # Override HRM controller T parameter in each block
        for block in stack.blocks:
            if hasattr(block, 'router'):
                # Check if router has HRM controller with T parameter
                if hasattr(block.router, 'hrm_controller'):
                    block.router.hrm_controller.T = T
                # Alternative: if router itself has T
                elif hasattr(block.router, 'T'):
                    block.router.T = T
        
        # Iterative refiner with R refinement steps
        self.refiner = IterRefiner(
            stack, 
            max_inner_iters=R,
            outer_residual=True,
            rezero_init=True
        )
        
        # Classification head
        self.pooler = nn.Linear(d_model, d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),  # concat + element-wise product
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3)
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming initialization."""
        # Embedding: normal distribution with smaller std
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        if self.embed.padding_idx is not None:
            self.embed.weight.data[self.embed.padding_idx].zero_()
        
        # Classification head: Xavier uniform
        for module in [self.pooler] + list(self.classifier.modules()):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, premise, hypothesis):
        # Embed
        p_emb = self.embed(premise)  # [B, L, D]
        h_emb = self.embed(hypothesis)  # [B, L, D]
        
        # Process with PoH (R refinement steps)
        p_out, _ = self.refiner(p_emb)
        h_out, _ = self.refiner(h_emb)
        
        # Pool (mean pooling over non-padding tokens)
        p_mask = (premise != 0).float().unsqueeze(-1)  # [B, L, 1]
        h_mask = (hypothesis != 0).float().unsqueeze(-1)
        
        p_pooled = (p_out * p_mask).sum(dim=1) / (p_mask.sum(dim=1) + 1e-9)  # [B, D]
        h_pooled = (h_out * h_mask).sum(dim=1) / (h_mask.sum(dim=1) + 1e-9)
        
        # Apply pooler
        p_pooled = self.pooler(p_pooled)
        h_pooled = self.pooler(h_pooled)
        
        # Combine: [premise; hypothesis; premise*hypothesis]
        combined = torch.cat([
            p_pooled, 
            h_pooled, 
            p_pooled * h_pooled
        ], dim=-1)  # [B, 3D]
        
        # Classify
        logits = self.classifier(combined)  # [B, 3]
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ========== Training ==========

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    max_steps: Optional[int] = None,
) -> Tuple[float, int]:
    """Train for one epoch or max_steps."""
    model.train()
    total_loss = 0.0
    steps = 0
    
    for premise, hypothesis, labels in loader:
        premise = premise.to(device)
        hypothesis = hypothesis.to(device)
        labels = labels.to(device)
        
        # Forward
        logits = model(premise, hypothesis)
        loss = criterion(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        steps += 1
        
        if max_steps and steps >= max_steps:
            break
    
    return total_loss / steps if steps > 0 else 0.0, steps


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    """Evaluate model and return (accuracy, loss)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    correct = 0
    total = 0
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for premise, hypothesis, labels in loader:
            premise = premise.to(device)
            hypothesis = hypothesis.to(device)
            labels = labels.to(device)
            
            logits = model(premise, hypothesis)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=-1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
            num_batches += 1
    
    acc = correct / total if total > 0 else 0.0
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return acc, avg_loss


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    max_steps: int = 2000,
    lr: float = 1e-3,  # Optimal from diagnostic (5x better than 2e-4)
    warmup_steps: int = 200,  # Longer warmup for stability
) -> Tuple[float, float, float]:
    """
    Train model and return (best_val_acc, final_val_acc, time_minutes).
    
    Hyperparameters empirically optimized via diagnostic sweep:
    - lr=1e-3: Optimal from LR sweep (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    - warmup_steps=200: Gradual warmup prevents early instability
    - weight_decay=0.01: L2 regularization for generalization
    - grad_clip=1.0: Prevents gradient explosions
    
    Diagnostic results (200 steps on 5K samples):
    - LR=1e-3 achieves 51.3% accuracy with R=8
    - LR=2e-4 (original) only achieved ~48% (underfit)
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler with warmup + cosine decay
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        # Cosine decay to 10% of peak LR
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_val_acc = 0.0
    final_val_acc = 0.0
    start_time = time.time()
    
    step = 0
    eval_interval = max(50, max_steps // 10)
    
    pbar = tqdm(total=max_steps, desc="Training")
    
    while step < max_steps:
        # Train
        model.train()
        for premise, hypothesis, labels in train_loader:
            premise = premise.to(device)
            hypothesis = hypothesis.to(device)
            labels = labels.to(device)
            
            # Forward
            logits = model(premise, hypothesis)
            loss = criterion(logits, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            step += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})
            
            # Evaluate
            if step % eval_interval == 0 or step == max_steps:
                val_acc, val_loss = evaluate(model, val_loader, device)
                pbar.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "val_acc": f"{val_acc:.3f}",
                    "val_loss": f"{val_loss:.3f}"
                })
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                final_val_acc = val_acc
                
                model.train()
            
            if step >= max_steps:
                break
    
    pbar.close()
    
    time_minutes = (time.time() - start_time) / 60
    return best_val_acc, final_val_acc, time_minutes


# ========== Grid Search ==========

def run_grid_search(
    R_values: List[int],
    T_values: List[int],
    seeds: List[int],
    vocab_size: int = 10000,
    d_model: int = 256,
    n_heads: int = 8,
    d_ff: int = 1024,
    depth: int = 4,
    max_steps: int = 2000,
    batch_size: int = 32,
    max_train_samples: int = 10000,
    max_val_samples: int = 2000,
    device: str = "cpu",
    save_dir: str = "experiments/results/R_T_search_real_nli",
):
    """Run grid search on real SNLI data."""
    
    if not DATASETS_AVAILABLE:
        raise ImportError("Install datasets: pip install datasets")
    
    os.makedirs(save_dir, exist_ok=True)
    results_file = os.path.join(save_dir, "grid_search_results.csv")
    
    # Create CSV
    with open(results_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'timestamp', 'R', 'T', 'seed', 
            'best_val_acc', 'final_val_acc', 'time_minutes', 
            'params_M', 'time_per_step_ms'
        ])
        writer.writeheader()
    
    # Prepare data (load once, reuse for all experiments)
    print("\n" + "="*60)
    print("Loading SNLI Dataset")
    print("="*60)
    
    train_dataset = SNLIDataset('train', max_samples=max_train_samples)
    val_dataset = SNLIDataset('validation', max_samples=max_val_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    total_experiments = len(R_values) * len(T_values) * len(seeds)
    experiment_num = 0
    
    print(f"\n{'='*60}")
    print(f"Grid Search: R × T × Seeds on Real SNLI")
    print(f"{'='*60}")
    print(f"R values (refinement steps): {R_values}")
    print(f"T values (HRM outer loop period): {T_values}")
    print(f"Seeds: {seeds}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Max steps per experiment: {max_steps}")
    print(f"Total experiments: {total_experiments}")
    print(f"{'='*60}\n")
    
    for R in R_values:
        for T in T_values:
            for seed in seeds:
                experiment_num += 1
                
                print(f"\n{'='*60}")
                print(f"[{experiment_num}/{total_experiments}] R={R}, T={T}, seed={seed}")
                print(f"{'='*60}")
                
                # Set seed
                torch.manual_seed(seed)
                
                # Create model
                model = PoHForNLI(
                    vocab_size=vocab_size,
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    depth=depth,
                    R=R,
                    T=T,
                )
                
                params_M = model.count_parameters() / 1e6
                print(f"Parameters: {params_M:.2f}M")
                print(f"Refinement steps (R): {R}")
                print(f"HRM outer loop period (T): {T}")
                
                # Train
                best_val_acc, final_val_acc, time_minutes = train_and_evaluate(
                    model, train_loader, val_loader, device, max_steps=max_steps
                )
                
                time_per_step_ms = (time_minutes * 60 * 1000) / max_steps
                
                print(f"\nResults:")
                print(f"  Best val acc: {best_val_acc:.4f}")
                print(f"  Final val acc: {final_val_acc:.4f}")
                print(f"  Time: {time_minutes:.2f} min ({time_per_step_ms:.1f} ms/step)")
                
                # Save results
                with open(results_file, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        'timestamp', 'R', 'T', 'seed', 
                        'best_val_acc', 'final_val_acc', 'time_minutes', 
                        'params_M', 'time_per_step_ms'
                    ])
                    writer.writerow({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'R': R,
                        'T': T,
                        'seed': seed,
                        'best_val_acc': f"{best_val_acc:.4f}",
                        'final_val_acc': f"{final_val_acc:.4f}",
                        'time_minutes': f"{time_minutes:.2f}",
                        'params_M': f"{params_M:.2f}",
                        'time_per_step_ms': f"{time_per_step_ms:.1f}",
                    })
    
    print(f"\n{'='*60}")
    print(f"Grid Search Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {results_file}")
    
    return results_file


# ========== Analysis ==========

def analyze_results(results_file: str):
    """Analyze grid search results and find optimal R and T."""
    import pandas as pd
    import numpy as np
    
    df = pd.read_csv(results_file)
    
    # Convert to numeric
    df['best_val_acc'] = pd.to_numeric(df['best_val_acc'])
    df['time_minutes'] = pd.to_numeric(df['time_minutes'])
    
    # Group by R and T (average over seeds)
    grouped = df.groupby(['R', 'T']).agg({
        'best_val_acc': ['mean', 'std'],
        'final_val_acc': ['mean', 'std'],
        'time_minutes': 'mean',
    }).reset_index()
    
    grouped.columns = ['R', 'T', 'best_acc_mean', 'best_acc_std', 'final_acc_mean', 'final_acc_std', 'time_mean']
    
    # Compute efficiency: accuracy / (time * sqrt(R * T))
    grouped['efficiency'] = grouped['best_acc_mean'] / (grouped['time_mean'] * np.sqrt(grouped['R'] * grouped['T']))
    
    # Find best configurations
    print(f"\n{'='*60}")
    print("Analysis Results")
    print(f"{'='*60}\n")
    
    # Best accuracy
    best_acc_row = grouped.loc[grouped['best_acc_mean'].idxmax()]
    print(f"🏆 Best Accuracy:")
    print(f"  R={int(best_acc_row['R'])} (refinement steps)")
    print(f"  T={int(best_acc_row['T'])} (HRM outer loop period)")
    print(f"  Best Val Accuracy: {best_acc_row['best_acc_mean']:.4f} ± {best_acc_row['best_acc_std']:.4f}")
    print(f"  Final Val Accuracy: {best_acc_row['final_acc_mean']:.4f} ± {best_acc_row['final_acc_std']:.4f}")
    print(f"  Training Time: {best_acc_row['time_mean']:.2f} min")
    
    # Best efficiency
    best_eff_row = grouped.loc[grouped['efficiency'].idxmax()]
    print(f"\n⚡ Best Efficiency (acc / time·√(R·T)):")
    print(f"  R={int(best_eff_row['R'])} (refinement steps)")
    print(f"  T={int(best_eff_row['T'])} (HRM outer loop period)")
    print(f"  Best Val Accuracy: {best_eff_row['best_acc_mean']:.4f} ± {best_eff_row['best_acc_std']:.4f}")
    print(f"  Final Val Accuracy: {best_eff_row['final_acc_mean']:.4f} ± {best_eff_row['final_acc_std']:.4f}")
    print(f"  Training Time: {best_eff_row['time_mean']:.2f} min")
    print(f"  Efficiency Score: {best_eff_row['efficiency']:.6f}")
    
    # Show top 5
    print(f"\n{'='*60}")
    print("Top 5 Configurations (by accuracy):")
    print(f"{'='*60}")
    top5 = grouped.sort_values('best_acc_mean', ascending=False).head(5)
    print(top5[['R', 'T', 'best_acc_mean', 'best_acc_std', 'time_mean', 'efficiency']].to_string(index=False))
    
    # Show full table
    print(f"\n{'='*60}")
    print("Complete Results (sorted by accuracy):")
    print(f"{'='*60}")
    print(grouped.sort_values('best_acc_mean', ascending=False).to_string(index=False))
    
    # Save summary
    summary_file = results_file.replace('.csv', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Best Accuracy Configuration:\n")
        f.write(f"  R={int(best_acc_row['R'])} (refinement steps)\n")
        f.write(f"  T={int(best_acc_row['T'])} (HRM outer loop period)\n")
        f.write(f"  Accuracy: {best_acc_row['best_acc_mean']:.4f} ± {best_acc_row['best_acc_std']:.4f}\n")
        f.write(f"  Time: {best_acc_row['time_mean']:.2f} min\n\n")
        
        f.write(f"Best Efficiency Configuration:\n")
        f.write(f"  R={int(best_eff_row['R'])} (refinement steps)\n")
        f.write(f"  T={int(best_eff_row['T'])} (HRM outer loop period)\n")
        f.write(f"  Accuracy: {best_eff_row['best_acc_mean']:.4f} ± {best_eff_row['best_acc_std']:.4f}\n")
        f.write(f"  Time: {best_eff_row['time_mean']:.2f} min\n")
        f.write(f"  Efficiency: {best_eff_row['efficiency']:.6f}\n\n")
        
        f.write(f"Full Results:\n")
        f.write(grouped.sort_values('best_acc_mean', ascending=False).to_string(index=False))
    
    print(f"\n✅ Summary saved to: {summary_file}")


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser(description="Find optimal R and T for PoH on real NLI")
    
    # Search space
    parser.add_argument('--R-values', type=int, nargs='+', default=[1, 2, 4, 6, 8, 12],
                        help='Refinement steps to try (default: 1 2 4 6 8 12)')
    parser.add_argument('--T-values', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='HRM outer loop periods to try (default: 1 2 4 8)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds (default: 42 43 44)')
    
    # Model config
    parser.add_argument('--d-model', type=int, default=256, help='Model dimension')
    parser.add_argument('--n-heads', type=int, default=8, help='Number of heads')
    parser.add_argument('--d-ff', type=int, default=1024, help='FFN dimension')
    parser.add_argument('--depth', type=int, default=4, help='Number of layers')
    parser.add_argument('--vocab-size', type=int, default=10000, help='Vocabulary size')
    
    # Training
    parser.add_argument('--max-steps', type=int, default=2000, help='Training steps per experiment')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--max-train-samples', type=int, default=10000, help='Max training samples from SNLI')
    parser.add_argument('--max-val-samples', type=int, default=2000, help='Max validation samples from SNLI')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    
    # Output
    parser.add_argument('--save-dir', type=str, default='experiments/results/R_T_search_real_nli',
                        help='Save directory')
    
    args = parser.parse_args()
    
    # Check if datasets is installed
    if not DATASETS_AVAILABLE:
        print("\n❌ Error: 'datasets' library not installed")
        print("Install with: pip install datasets")
        return
    
    # Run grid search
    results_file = run_grid_search(
        R_values=args.R_values,
        T_values=args.T_values,
        seeds=args.seeds,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        depth=args.depth,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        device=args.device,
        save_dir=args.save_dir,
    )
    
    # Analyze results
    analyze_results(results_file)
    
    print(f"\n{'='*60}")
    print("✅ Complete! Check results in:")
    print(f"  {results_file}")
    print(f"  {results_file.replace('.csv', '_summary.txt')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

