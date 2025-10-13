#!/usr/bin/env python3
"""
Sorting A/B Test with Optimal R and T Parameters

Tests PoH with optimal hyperparameters (R=4, T=4) against baseline
on sorting tasks of varying difficulty.

Based on grid search results:
- Best R = 4 (refinement steps)
- Best T = 4 (HRM outer loop period)

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
        # Random values in [-100, 100]
        values = np.random.randint(-100, 100, size=array_len)
        
        # Target = argsort (indices that would sort the array)
        perm = np.argsort(values, kind='stable')
        
        arrays.append(values.reshape(-1, 1).astype(np.float32))
        targets.append(perm.astype(np.int64))
    
    return (
        torch.from_numpy(np.array(arrays)),
        torch.from_numpy(np.array(targets))
    )


# ========== Models ==========

class BaselineTransformerSort(nn.Module):
    """Standard transformer for sorting (baseline)."""
    
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        depth: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_emb = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        
        # Standard transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
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
        
        # Embed + positional
        h = self.input_proj(x)  # [B, L, D]
        h = h + self.pos_emb[:, :L, :]
        
        # Transform
        h = self.transformer(h)  # [B, L, D]
        
        # Compute pointer logits (query x key)
        queries = h  # [B, L, D]
        keys = h     # [B, L, D]
        
        logits = torch.bmm(queries, keys.transpose(1, 2))  # [B, L, L]
        logits = logits / np.sqrt(self.d_model)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PoHTransformerSort(nn.Module):
    """PoH transformer for sorting with optimal R and T."""
    
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
    perfect_sorts = 0
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
            
            # Perfect sorts
            perfect = (preds == batch_targets).all(dim=1).sum().item()
            perfect_sorts += perfect
            
            # Kendall's tau
            for j in range(len(batch_targets)):
                tau, _ = kendalltau(
                    preds[j].cpu().numpy(),
                    batch_targets[j].cpu().numpy()
                )
                kendall_taus.append(tau if not np.isnan(tau) else 0.0)
    
    return {
        'accuracy': total_correct / total_elements,
        'perfect': perfect_sorts / num_samples,
        'kendall_tau': np.mean(kendall_taus)
    }


def train_and_evaluate(
    model,
    model_name,
    train_data,
    test_data,
    epochs=100,
    batch_size=32,
    lr=1e-3,
    device='cpu'
):
    """Train and evaluate a model."""
    print(f"\nTraining {model_name}...")
    print(f"  Parameters: {model.count_parameters()/1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    start_time = time.time()
    best_tau = -1.0
    final_metrics = None
    
    pbar = tqdm(range(1, epochs + 1), desc=model_name)
    
    for epoch in pbar:
        # Train
        train_loss = train_epoch(model, train_data, optimizer, batch_size, device)
        
        # Evaluate every 10 epochs
        if epoch % 10 == 0 or epoch == epochs:
            test_metrics = evaluate(model, test_data, batch_size, device)
            
            if test_metrics['kendall_tau'] > best_tau:
                best_tau = test_metrics['kendall_tau']
            
            final_metrics = test_metrics
            
            pbar.set_postfix({
                'loss': f"{train_loss:.3f}",
                'tau': f"{test_metrics['kendall_tau']:.3f}",
                'perfect': f"{test_metrics['perfect']:.2f}"
            })
    
    pbar.close()
    
    time_minutes = (time.time() - start_time) / 60
    
    return {
        'best_tau': best_tau,
        'final_tau': final_metrics['kendall_tau'],
        'final_accuracy': final_metrics['accuracy'],
        'final_perfect': final_metrics['perfect'],
        'time_min': time_minutes,
        'params_M': model.count_parameters() / 1e6
    }


# ========== Main A/B Test ==========

def run_ab_test(array_len, train_samples=2000, test_samples=500, R=4, T=4, 
                epochs=100, seed=42):
    """Run A/B test for a specific array length."""
    
    print(f"\n{'='*80}")
    print(f"A/B Test: Array Length {array_len}")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Array length: {array_len}")
    print(f"  Training samples: {train_samples:,}")
    print(f"  Test samples: {test_samples:,}")
    print(f"  PoH R (refinement steps): {R}")
    print(f"  PoH T (HRM period): {T}")
    print(f"  Epochs: {epochs}")
    print(f"  Seed: {seed}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    # Generate data
    print(f"\nGenerating data...")
    train_data = generate_sort_data(train_samples, array_len, seed=seed)
    test_data = generate_sort_data(test_samples, array_len, seed=seed+1)
    
    # Build models
    print(f"\nBuilding models...")
    
    baseline = BaselineTransformerSort(
        d_model=128,
        n_heads=4,
        d_ff=512,
        depth=3,
        dropout=0.1
    ).to(device)
    
    poh = PoHTransformerSort(
        d_model=128,
        n_heads=4,
        d_ff=512,
        depth=3,
        R=R,
        T=T,
        dropout=0.1
    ).to(device)
    
    # Train baseline
    baseline_results = train_and_evaluate(
        baseline, "Baseline", train_data, test_data,
        epochs=epochs, device=device
    )
    
    # Train PoH
    poh_results = train_and_evaluate(
        poh, f"PoH (R={R}, T={T})", train_data, test_data,
        epochs=epochs, device=device
    )
    
    # Results
    print(f"\n{'='*80}")
    print("📊 RESULTS")
    print(f"{'='*80}")
    
    print(f"\n📚 Baseline (Standard Transformer)")
    print(f"  Parameters: {baseline_results['params_M']:.2f}M")
    print(f"  Best Kendall's τ: {baseline_results['best_tau']:.4f}")
    print(f"  Final Kendall's τ: {baseline_results['final_tau']:.4f}")
    print(f"  Final accuracy: {baseline_results['final_accuracy']:.4f}")
    print(f"  Final perfect sorts: {baseline_results['final_perfect']:.2%}")
    print(f"  Training time: {baseline_results['time_min']:.2f} min")
    
    print(f"\n🔬 PoH (R={R}, T={T})")
    print(f"  Parameters: {poh_results['params_M']:.2f}M")
    print(f"  Best Kendall's τ: {poh_results['best_tau']:.4f}")
    print(f"  Final Kendall's τ: {poh_results['final_tau']:.4f}")
    print(f"  Final accuracy: {poh_results['final_accuracy']:.4f}")
    print(f"  Final perfect sorts: {poh_results['final_perfect']:.2%}")
    print(f"  Training time: {poh_results['time_min']:.2f} min")
    
    # Comparison
    delta_tau = poh_results['final_tau'] - baseline_results['final_tau']
    delta_pct = (delta_tau / baseline_results['final_tau']) * 100 if baseline_results['final_tau'] > 0 else 0
    
    print(f"\n{'='*80}")
    print("📈 COMPARISON")
    print(f"{'='*80}")
    print(f"Kendall's τ delta: {delta_tau:+.4f} ({delta_pct:+.2f}%)")
    
    if delta_tau > 0.01:
        print(f"✅ PoH WINS by {delta_tau:.4f} ({delta_pct:.2f}%)")
        winner = "PoH"
    elif delta_tau < -0.01:
        print(f"❌ Baseline WINS by {-delta_tau:.4f} ({-delta_pct:.2f}%)")
        winner = "Baseline"
    else:
        print(f"⚖️  TIE (difference < 0.01)")
        winner = "Tie"
    
    return {
        'array_len': array_len,
        'baseline': baseline_results,
        'poh': poh_results,
        'delta_tau': delta_tau,
        'delta_pct': delta_pct,
        'winner': winner
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Sorting A/B Test with Optimal R and T')
    parser.add_argument('--lengths', type=int, nargs='+', default=[12, 16, 20],
                        help='Array lengths to test (default: 12 16 20)')
    parser.add_argument('--R', type=int, default=4, help='PoH refinement steps (default: 4)')
    parser.add_argument('--T', type=int, default=4, help='HRM period (default: 4)')
    parser.add_argument('--train-samples', type=int, default=2000, help='Training samples')
    parser.add_argument('--test-samples', type=int, default=500, help='Test samples')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save-dir', type=str, default='experiments/results/sorting_ab',
                        help='Save directory')
    
    args = parser.parse_args()
    
    print("="*80)
    print("SORTING A/B TEST WITH OPTIMAL PARAMETERS")
    print("="*80)
    print(f"\nOptimal parameters from grid search:")
    print(f"  R = {args.R} (refinement steps)")
    print(f"  T = {args.T} (HRM outer loop period)")
    print(f"\nTesting on array lengths: {args.lengths}")
    print("="*80)
    
    # Run tests for each length
    all_results = []
    
    for array_len in args.lengths:
        result = run_ab_test(
            array_len,
            train_samples=args.train_samples,
            test_samples=args.test_samples,
            R=args.R,
            T=args.T,
            epochs=args.epochs,
            seed=args.seed
        )
        all_results.append(result)
    
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    results_file = os.path.join(args.save_dir, f"ab_results_R{args.R}_T{args.T}.csv")
    
    with open(results_file, 'w', newline='') as f:
        fieldnames = [
            'timestamp', 'array_len', 'model', 'R', 'T',
            'best_tau', 'final_tau', 'final_accuracy', 'final_perfect',
            'time_min', 'params_M', 'delta_tau', 'winner'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for result in all_results:
            # Baseline row
            writer.writerow({
                'timestamp': timestamp,
                'array_len': result['array_len'],
                'model': 'Baseline',
                'R': '-',
                'T': '-',
                'best_tau': f"{result['baseline']['best_tau']:.4f}",
                'final_tau': f"{result['baseline']['final_tau']:.4f}",
                'final_accuracy': f"{result['baseline']['final_accuracy']:.4f}",
                'final_perfect': f"{result['baseline']['final_perfect']:.4f}",
                'time_min': f"{result['baseline']['time_min']:.2f}",
                'params_M': f"{result['baseline']['params_M']:.2f}",
                'delta_tau': "0.0000",
                'winner': ''
            })
            
            # PoH row
            writer.writerow({
                'timestamp': timestamp,
                'array_len': result['array_len'],
                'model': 'PoH',
                'R': args.R,
                'T': args.T,
                'best_tau': f"{result['poh']['best_tau']:.4f}",
                'final_tau': f"{result['poh']['final_tau']:.4f}",
                'final_accuracy': f"{result['poh']['final_accuracy']:.4f}",
                'final_perfect': f"{result['poh']['final_perfect']:.4f}",
                'time_min': f"{result['poh']['time_min']:.2f}",
                'params_M': f"{result['poh']['params_M']:.2f}",
                'delta_tau': f"{result['delta_tau']:+.4f}",
                'winner': result['winner']
            })
    
    # Summary
    print(f"\n\n{'='*80}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Length':<10} {'Baseline τ':<15} {'PoH τ':<15} {'Delta':<15} {'Winner':<15}")
    print("-"*80)
    
    for result in all_results:
        print(f"{result['array_len']:<10} "
              f"{result['baseline']['final_tau']:.4f}         "
              f"{result['poh']['final_tau']:.4f}         "
              f"{result['delta_tau']:+.4f}         "
              f"{result['winner']}")
    
    print(f"\n✅ Results saved to: {results_file}")
    print("="*80)


if __name__ == "__main__":
    main()

