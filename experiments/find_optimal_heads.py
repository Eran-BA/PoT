#!/usr/bin/env python3
"""
Find Optimal Number of Attention Heads for PoH

Systematic search over n_heads while keeping optimal R=4 and T=4 fixed.

Tests on sorting task (fast, clear signal).

Author: Eran Ben Artzy
Date: October 2025
"""

import os
import sys
import time
import csv
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pot.modules import PoHConfig, PoHStack, IterRefiner


# ========== Data Generation ==========

def generate_sort_data(num_samples: int, array_len: int, seed: int = 42):
    """Generate random sorting data."""
    np.random.seed(seed)
    arrays = []
    targets = []
    
    for _ in range(num_samples):
        values = np.random.randint(-100, 100, size=array_len)
        perm = np.argsort(values, kind='stable')
        
        arrays.append(values.reshape(-1, 1).astype(np.float32))
        targets.append(perm.astype(np.int64))
    
    return (
        torch.from_numpy(np.array(arrays)),
        torch.from_numpy(np.array(targets))
    )


# ========== Model ==========

class PoHSort(nn.Module):
    """PoH transformer for sorting with configurable heads."""
    
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        depth: int = 3,
        R: int = 4,
        T: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.R = R
        self.T = T
        
        # Input projection
        self.input_proj = nn.Linear(1, d_model)
        
        # PoH stack with HRM controller
        cfg = PoHConfig(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            route_mode="soft",
            share_router=True,
            pos_encoding="absolute",
            max_seq_len=100,
        )
        
        stack = PoHStack(cfg, depth=depth)
        
        # Set HRM period T in each block
        for block in stack.blocks:
            if hasattr(block, 'router'):
                if hasattr(block.router, 'hrm_controller'):
                    block.router.hrm_controller.T = T
                elif hasattr(block.router, 'T'):
                    block.router.T = T
        
        # Iterative refiner with R refinement steps
        self.refiner = IterRefiner(
            stack,
            max_inner_iters=R,
            outer_residual=True,
            rezero_init=True
        )
        
        # Output projection (pointer logits)
        self.output_proj = nn.Linear(d_model, 1)
    
    def forward(self, x):
        """
        Args:
            x: [B, L, 1] - input values
        Returns:
            logits: [B, L, L] - pointer logits
        """
        B, L, _ = x.shape
        
        # Embed
        h = self.input_proj(x)  # [B, L, D]
        
        # PoH encoding with R refinement iterations
        h, _ = self.refiner(h)  # [B, L, D]
        
        # Compute pointer logits
        queries = h  # [B, L, D]
        keys = h     # [B, L, D]
        
        logits = torch.bmm(queries, keys.transpose(1, 2))  # [B, L, L]
        logits = logits / np.sqrt(self.d_model)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ========== Training ==========

def train_epoch(model, data, optimizer, batch_size, device):
    """Train for one epoch."""
    model.train()
    arrays, targets = data
    num_samples = len(arrays)
    
    total_loss = 0.0
    num_batches = 0
    
    indices = torch.randperm(num_samples)
    
    for i in range(0, num_samples, batch_size):
        batch_idx = indices[i:i+batch_size]
        batch_arrays = arrays[batch_idx].to(device)
        batch_targets = targets[batch_idx].to(device)
        
        # Forward
        logits = model(batch_arrays)  # [B, L, L]
        
        # Pointer loss (cross-entropy at each position)
        B, L, _ = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * L, L),
            batch_targets.reshape(B * L)
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(model, data, batch_size, device):
    """Evaluate model on data."""
    model.eval()
    arrays, targets = data
    num_samples = len(arrays)
    
    total_correct = 0
    total_elements = 0
    kendall_taus = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_arrays = arrays[i:i+batch_size].to(device)
            batch_targets = targets[i:i+batch_size].to(device)
            
            logits = model(batch_arrays)
            preds = logits.argmax(dim=-1)
            
            # Accuracy
            correct = (preds == batch_targets).sum().item()
            total_correct += correct
            total_elements += batch_targets.numel()
            
            # Kendall's tau
            for j in range(len(batch_targets)):
                tau, _ = kendalltau(
                    preds[j].cpu().numpy(),
                    batch_targets[j].cpu().numpy()
                )
                kendall_taus.append(tau if not np.isnan(tau) else 0.0)
    
    return {
        'accuracy': total_correct / total_elements,
        'kendall_tau': np.mean(kendall_taus)
    }


def train_and_evaluate(
    n_heads,
    train_data,
    val_data,
    d_model=128,
    depth=3,
    R=4,
    T=4,
    epochs=50,
    batch_size=32,
    lr=1e-3,
    device='cpu',
    seed=42
):
    """Train and evaluate a model with specific n_heads."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Build model
    model = PoHSort(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_model * 4,
        depth=depth,
        R=R,
        T=T,
        dropout=0.1
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    best_tau = -1.0
    final_tau = -1.0
    
    pbar = tqdm(range(1, epochs + 1), desc=f"n_heads={n_heads}", leave=False)
    
    for epoch in pbar:
        # Train
        train_loss = train_epoch(model, train_data, optimizer, batch_size, device)
        
        # Evaluate every 5 epochs
        if epoch % 5 == 0 or epoch == epochs:
            val_metrics = evaluate(model, val_data, batch_size, device)
            
            if val_metrics['kendall_tau'] > best_tau:
                best_tau = val_metrics['kendall_tau']
            
            final_tau = val_metrics['kendall_tau']
            
            pbar.set_postfix({
                'loss': f"{train_loss:.3f}",
                'tau': f"{val_metrics['kendall_tau']:.3f}"
            })
    
    pbar.close()
    
    return {
        'n_heads': n_heads,
        'best_tau': best_tau,
        'final_tau': final_tau,
        'params_M': model.count_parameters() / 1e6
    }


# ========== Grid Search ==========

def run_head_search(
    n_heads_values,
    seeds,
    array_len=16,
    train_samples=2000,
    val_samples=500,
    d_model=128,
    depth=3,
    R=4,
    T=4,
    epochs=50,
    batch_size=32,
    device='cpu',
    save_dir='experiments/results/head_search'
):
    """Run grid search over n_heads."""
    
    os.makedirs(save_dir, exist_ok=True)
    results_file = os.path.join(save_dir, 'head_search_results.csv')
    
    # Create CSV
    with open(results_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'timestamp', 'n_heads', 'seed', 'd_model', 'depth', 'R', 'T',
            'best_tau', 'final_tau', 'params_M'
        ])
        writer.writeheader()
    
    # Generate data once
    print(f"\n{'='*80}")
    print("Generating Sorting Data")
    print(f"{'='*80}")
    print(f"  Array length: {array_len}")
    print(f"  Training samples: {train_samples:,}")
    print(f"  Validation samples: {val_samples:,}")
    
    train_data = generate_sort_data(train_samples, array_len, seed=42)
    val_data = generate_sort_data(val_samples, array_len, seed=43)
    
    total_experiments = len(n_heads_values) * len(seeds)
    experiment_num = 0
    
    print(f"\n{'='*80}")
    print(f"Grid Search: n_heads × Seeds")
    print(f"{'='*80}")
    print(f"n_heads values: {n_heads_values}")
    print(f"Seeds: {seeds}")
    print(f"Fixed: d_model={d_model}, depth={depth}, R={R}, T={T}")
    print(f"Total experiments: {total_experiments}")
    print(f"{'='*80}\n")
    
    all_results = []
    
    for n_heads in n_heads_values:
        for seed in seeds:
            experiment_num += 1
            
            print(f"\n[{experiment_num}/{total_experiments}] n_heads={n_heads}, seed={seed}")
            
            start_time = time.time()
            
            result = train_and_evaluate(
                n_heads=n_heads,
                train_data=train_data,
                val_data=val_data,
                d_model=d_model,
                depth=depth,
                R=R,
                T=T,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
                seed=seed
            )
            
            time_min = (time.time() - start_time) / 60
            
            print(f"  Best τ: {result['best_tau']:.4f}")
            print(f"  Final τ: {result['final_tau']:.4f}")
            print(f"  Params: {result['params_M']:.2f}M")
            print(f"  Time: {time_min:.2f} min")
            
            # Save result
            with open(results_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'n_heads', 'seed', 'd_model', 'depth', 'R', 'T',
                    'best_tau', 'final_tau', 'params_M'
                ])
                writer.writerow({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'n_heads': n_heads,
                    'seed': seed,
                    'd_model': d_model,
                    'depth': depth,
                    'R': R,
                    'T': T,
                    'best_tau': f"{result['best_tau']:.4f}",
                    'final_tau': f"{result['final_tau']:.4f}",
                    'params_M': f"{result['params_M']:.2f}"
                })
            
            all_results.append(result)
    
    print(f"\n{'='*80}")
    print("Grid Search Complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {results_file}")
    
    return results_file, all_results


def analyze_results(results_file):
    """Analyze head search results."""
    import pandas as pd
    
    df = pd.read_csv(results_file)
    
    # Convert to numeric
    df['best_tau'] = pd.to_numeric(df['best_tau'])
    df['final_tau'] = pd.to_numeric(df['final_tau'])
    df['params_M'] = pd.to_numeric(df['params_M'])
    
    # Group by n_heads (average over seeds)
    grouped = df.groupby('n_heads').agg({
        'best_tau': ['mean', 'std'],
        'final_tau': ['mean', 'std'],
        'params_M': 'mean'
    }).reset_index()
    
    grouped.columns = ['n_heads', 'best_tau_mean', 'best_tau_std', 
                       'final_tau_mean', 'final_tau_std', 'params_M']
    
    # Sort by best_tau
    grouped = grouped.sort_values('best_tau_mean', ascending=False)
    
    print(f"\n{'='*80}")
    print("Analysis Results")
    print(f"{'='*80}\n")
    
    # Best configuration
    best_row = grouped.iloc[0]
    print(f"🏆 Best Configuration:")
    print(f"  n_heads = {int(best_row['n_heads'])}")
    print(f"  Best Kendall's τ: {best_row['best_tau_mean']:.4f} ± {best_row['best_tau_std']:.4f}")
    print(f"  Final Kendall's τ: {best_row['final_tau_mean']:.4f} ± {best_row['final_tau_std']:.4f}")
    print(f"  Parameters: {best_row['params_M']:.2f}M")
    
    # Full table
    print(f"\n{'='*80}")
    print("All Configurations (sorted by performance):")
    print(f"{'='*80}")
    print(f"\n{grouped.to_string(index=False)}")
    
    # Save summary
    summary_file = results_file.replace('.csv', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Best Configuration:\n")
        f.write(f"  n_heads = {int(best_row['n_heads'])}\n")
        f.write(f"  Best τ: {best_row['best_tau_mean']:.4f} ± {best_row['best_tau_std']:.4f}\n")
        f.write(f"  Final τ: {best_row['final_tau_mean']:.4f} ± {best_row['final_tau_std']:.4f}\n")
        f.write(f"  Params: {best_row['params_M']:.2f}M\n\n")
        f.write(f"Full Results:\n")
        f.write(grouped.to_string(index=False))
    
    print(f"\n✅ Summary saved to: {summary_file}")


# ========== Main ==========

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Find optimal number of heads for PoH')
    
    # Search space
    parser.add_argument('--n-heads-values', type=int, nargs='+', 
                        default=[2, 4, 6, 8, 12, 16],
                        help='Number of heads to try (default: 2 4 6 8 12 16)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds (default: 42 43 44)')
    
    # Fixed architecture
    parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
    parser.add_argument('--depth', type=int, default=3, help='Number of layers')
    parser.add_argument('--R', type=int, default=4, help='Refinement steps (optimal)')
    parser.add_argument('--T', type=int, default=4, help='HRM period (optimal)')
    
    # Task
    parser.add_argument('--array-len', type=int, default=16, help='Sorting array length')
    parser.add_argument('--train-samples', type=int, default=2000, help='Training samples')
    parser.add_argument('--val-samples', type=int, default=500, help='Validation samples')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    
    # Output
    parser.add_argument('--save-dir', type=str, default='experiments/results/head_search',
                        help='Save directory')
    
    args = parser.parse_args()
    
    print("="*80)
    print("OPTIMAL HEAD COUNT SEARCH FOR PoH")
    print("="*80)
    print(f"\nSearching for optimal n_heads with fixed optimal R={args.R}, T={args.T}")
    print(f"Testing: {args.n_heads_values}")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Run search
    results_file, _ = run_head_search(
        n_heads_values=args.n_heads_values,
        seeds=args.seeds,
        array_len=args.array_len,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        d_model=args.d_model,
        depth=args.depth,
        R=args.R,
        T=args.T,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        save_dir=args.save_dir
    )
    
    # Analyze
    analyze_results(results_file)
    
    print(f"\n{'='*80}")
    print("✅ Complete! Check results:")
    print(f"  {results_file}")
    print(f"  {results_file.replace('.csv', '_summary.txt')}")
    print("="*80)


if __name__ == "__main__":
    main()

