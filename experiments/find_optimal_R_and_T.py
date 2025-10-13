#!/usr/bin/env python3
"""
Find Optimal R (Refinement Steps) and T (HRM Outer Loop Period)

Systematic grid search to find the best combination of:
- R: Number of refinement iterations (1, 2, 3, 4, 6, 8, 12, 16, 20)
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
from typing import Dict, List, Tuple
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pot.modules import PoHConfig, PoHStack, IterRefiner
from src.pot.core.hrm_controller import HRMPointerController


# ========== Synthetic NLI Dataset ==========

class SyntheticNLI(Dataset):
    """Quick synthetic NLI for hyperparameter search."""
    def __init__(self, vocab_size=1000, seq_len=32, size=5000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Random premise and hypothesis
        premise = torch.randint(0, self.vocab_size, (self.seq_len,))
        hypothesis = torch.randint(0, self.vocab_size, (self.seq_len,))
        
        # Random label (0=entailment, 1=neutral, 2=contradiction)
        label = torch.randint(0, 3, (1,)).item()
        
        return premise, hypothesis, label


# ========== Simple PoH-NLI Model ==========

class SimplePoHNLI(nn.Module):
    """Simple PoH model for NLI with configurable R and T."""
    def __init__(self, vocab_size, d_model, n_heads, depth, R, T):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.R = R
        self.T = T
        
        # Embeddings
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # PoH stack with HRM controller
        cfg = PoHConfig(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_model * 4,
            dropout=0.1,
            route_mode="soft",
            share_router=True,
            pos_encoding="absolute",
            max_seq_len=64,
        )
        
        # Create stack
        stack = PoHStack(cfg, depth=depth)
        
        # Override HRM controller T parameter
        # (This is a bit hacky but works for the experiment)
        for block in stack.blocks:
            if hasattr(block, 'router') and hasattr(block.router, 'T'):
                block.router.T = T
        
        # Iterative refiner with R refinement steps
        self.refiner = IterRefiner(stack, max_inner_iters=R)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 3)
        )
        
    def forward(self, premise, hypothesis):
        # Embed
        p_emb = self.embed(premise)  # [B, L, D]
        h_emb = self.embed(hypothesis)  # [B, L, D]
        
        # Process with PoH
        p_out, _ = self.refiner(p_emb)  # R refinement steps
        h_out, _ = self.refiner(h_emb)
        
        # Pool (mean pooling)
        p_pooled = p_out.mean(dim=1)  # [B, D]
        h_pooled = h_out.mean(dim=1)  # [B, D]
        
        # Concatenate and classify
        combined = torch.cat([p_pooled, h_pooled], dim=-1)  # [B, 2D]
        logits = self.classifier(combined)  # [B, 3]
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ========== Training ==========

def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    max_steps: int = 200,
    lr: float = 1e-3,
) -> Tuple[float, float, float]:
    """
    Train model and return (best_val_acc, final_val_acc, time_minutes).
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    final_val_acc = 0.0
    start_time = time.time()
    
    model.train()
    step = 0
    
    pbar = tqdm(total=max_steps, desc="Training", leave=False)
    
    while step < max_steps:
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
            
            step += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})
            
            # Evaluate periodically
            if step % 50 == 0 or step == max_steps:
                val_acc = evaluate(model, val_loader, device)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                final_val_acc = val_acc
                model.train()
            
            if step >= max_steps:
                break
    
    pbar.close()
    
    time_minutes = (time.time() - start_time) / 60
    return best_val_acc, final_val_acc, time_minutes


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    """Evaluate model and return accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for premise, hypothesis, labels in loader:
            premise = premise.to(device)
            hypothesis = hypothesis.to(device)
            labels = labels.to(device)
            
            logits = model(premise, hypothesis)
            preds = logits.argmax(dim=-1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total if total > 0 else 0.0


# ========== Grid Search ==========

def run_grid_search(
    R_values: List[int],
    T_values: List[int],
    seeds: List[int],
    vocab_size: int = 1000,
    d_model: int = 128,
    n_heads: int = 4,
    depth: int = 2,
    max_steps: int = 200,
    batch_size: int = 32,
    device: str = "cpu",
    save_dir: str = "experiments/results/R_T_search",
):
    """Run grid search over R and T values."""
    
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
    
    # Prepare data
    train_dataset = SyntheticNLI(vocab_size=vocab_size, size=5000)
    val_dataset = SyntheticNLI(vocab_size=vocab_size, size=1000)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    total_experiments = len(R_values) * len(T_values) * len(seeds)
    experiment_num = 0
    
    print(f"\n{'='*60}")
    print(f"Grid Search: R × T × Seeds")
    print(f"{'='*60}")
    print(f"R values: {R_values}")
    print(f"T values: {T_values}")
    print(f"Seeds: {seeds}")
    print(f"Total experiments: {total_experiments}")
    print(f"{'='*60}\n")
    
    for R in R_values:
        for T in T_values:
            for seed in seeds:
                experiment_num += 1
                
                print(f"\n[{experiment_num}/{total_experiments}] R={R}, T={T}, seed={seed}")
                print("-" * 60)
                
                # Set seed
                torch.manual_seed(seed)
                
                # Create model
                model = SimplePoHNLI(
                    vocab_size=vocab_size,
                    d_model=d_model,
                    n_heads=n_heads,
                    depth=depth,
                    R=R,
                    T=T,
                )
                
                params_M = model.count_parameters() / 1e6
                print(f"  Parameters: {params_M:.2f}M")
                
                # Train
                best_val_acc, final_val_acc, time_minutes = train_and_evaluate(
                    model, train_loader, val_loader, device, max_steps=max_steps
                )
                
                time_per_step_ms = (time_minutes * 60 * 1000) / max_steps
                
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
        'time_minutes': 'mean',
    }).reset_index()
    
    grouped.columns = ['R', 'T', 'acc_mean', 'acc_std', 'time_mean']
    
    # Compute efficiency: accuracy / (time * R * T)
    grouped['efficiency'] = grouped['acc_mean'] / (grouped['time_mean'] * grouped['R'] * grouped['T'])
    
    # Find best configurations
    print(f"\n{'='*60}")
    print("Analysis Results")
    print(f"{'='*60}\n")
    
    # Best accuracy
    best_acc_row = grouped.loc[grouped['acc_mean'].idxmax()]
    print(f"Best Accuracy:")
    print(f"  R={int(best_acc_row['R'])}, T={int(best_acc_row['T'])}")
    print(f"  Accuracy: {best_acc_row['acc_mean']:.4f} ± {best_acc_row['acc_std']:.4f}")
    print(f"  Time: {best_acc_row['time_mean']:.2f} min")
    
    # Best efficiency
    best_eff_row = grouped.loc[grouped['efficiency'].idxmax()]
    print(f"\nBest Efficiency (acc / time·R·T):")
    print(f"  R={int(best_eff_row['R'])}, T={int(best_eff_row['T'])}")
    print(f"  Accuracy: {best_eff_row['acc_mean']:.4f} ± {best_eff_row['acc_std']:.4f}")
    print(f"  Time: {best_eff_row['time_mean']:.2f} min")
    print(f"  Efficiency: {best_eff_row['efficiency']:.6f}")
    
    # Show full table
    print(f"\n{'='*60}")
    print("Complete Results (sorted by accuracy):")
    print(f"{'='*60}")
    print(grouped.sort_values('acc_mean', ascending=False).to_string(index=False))
    
    # Save summary
    summary_file = results_file.replace('.csv', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Best Accuracy Configuration:\n")
        f.write(f"  R={int(best_acc_row['R'])}, T={int(best_acc_row['T'])}\n")
        f.write(f"  Accuracy: {best_acc_row['acc_mean']:.4f} ± {best_acc_row['acc_std']:.4f}\n\n")
        
        f.write(f"Best Efficiency Configuration:\n")
        f.write(f"  R={int(best_eff_row['R'])}, T={int(best_eff_row['T'])}\n")
        f.write(f"  Accuracy: {best_eff_row['acc_mean']:.4f} ± {best_eff_row['acc_std']:.4f}\n\n")
        
        f.write(f"Full Results:\n")
        f.write(grouped.sort_values('acc_mean', ascending=False).to_string(index=False))
    
    print(f"\nSummary saved to: {summary_file}")


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser(description="Find optimal R and T for PoH")
    
    # Search space
    parser.add_argument('--R-values', type=int, nargs='+', default=[1, 2, 3, 4, 6, 8, 12],
                        help='Refinement steps to try (default: 1 2 3 4 6 8 12)')
    parser.add_argument('--T-values', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='HRM outer loop periods to try (default: 1 2 4 8)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds (default: 42 43 44)')
    
    # Model config
    parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n-heads', type=int, default=4, help='Number of heads')
    parser.add_argument('--depth', type=int, default=2, help='Number of layers')
    
    # Training
    parser.add_argument('--max-steps', type=int, default=200, help='Training steps')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    
    # Output
    parser.add_argument('--save-dir', type=str, default='experiments/results/R_T_search',
                        help='Save directory')
    
    args = parser.parse_args()
    
    # Run grid search
    results_file = run_grid_search(
        R_values=args.R_values,
        T_values=args.T_values,
        seeds=args.seeds,
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.depth,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        device=args.device,
        save_dir=args.save_dir,
    )
    
    # Analyze results
    analyze_results(results_file)


if __name__ == "__main__":
    main()

