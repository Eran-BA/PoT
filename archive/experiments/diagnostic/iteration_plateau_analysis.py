"""
Comprehensive Iteration Sweep: Find the Plateau Point

Systematically tests PoH with varying iteration counts to find:
1. Where performance plateaus (diminishing returns)
2. Optimal iteration count for different tasks
3. Task-specific convergence patterns

Iterations tested: 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import argparse
import csv
import json
import os
import random
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(__file__))
from sort_pointer_fixed import PointerDecoderSort


def generate_partial_obs_data(num_samples, array_len, mask_rate=0.5, seed=None):
    """Generate partial observability sorting data."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    arrays = []
    targets = []
    obs_masks = []
    
    n_visible = max(2, int(array_len * (1 - mask_rate)))
    
    for _ in range(num_samples):
        # Generate unique values
        values = rng.choice(np.arange(-100, 100), size=array_len, replace=False)
        
        # Randomly mask
        visible_mask = np.zeros(array_len, dtype=bool)
        visible_indices = rng.choice(array_len, size=n_visible, replace=False)
        visible_mask[visible_indices] = True
        
        # Masked input (-999 for invisible)
        masked_values = values.copy().astype(np.float32)
        masked_values[~visible_mask] = -999.0
        
        # Target: sort ALL values
        perm = np.argsort(values, kind="stable")
        
        arrays.append(masked_values.reshape(-1, 1))
        targets.append(perm.astype(np.int64))
        obs_masks.append(visible_mask.astype(np.float32))
    
    return (
        torch.from_numpy(np.array(arrays)),
        torch.from_numpy(np.array(targets)),
        torch.from_numpy(np.array(obs_masks)),
    )


def compute_kendall_tau(pred, target):
    """Kendall-τ correlation."""
    n = len(pred)
    concordant = 0
    discordant = 0
    
    pred_np = pred.cpu().numpy() if torch.is_tensor(pred) else pred
    target_np = target.cpu().numpy() if torch.is_tensor(target) else target
    
    for i in range(n):
        for j in range(i + 1, n):
            pred_order = np.sign(pred_np[i] - pred_np[j])
            target_order = np.sign(target_np[i] - target_np[j])
            if pred_order == target_order and pred_order != 0:
                concordant += 1
            elif pred_order == -target_order and pred_order != 0:
                discordant += 1
    
    total_pairs = n * (n - 1) / 2
    return (concordant - discordant) / total_pairs if total_pairs > 0 else 0.0


def train_epoch(model, data, optimizer, batch_size, device, clip_norm=1.0):
    """Train for one epoch."""
    model.train()
    arrays, targets, obs_masks = data
    num_samples = len(arrays)
    indices = torch.randperm(num_samples)

    total_loss = 0.0
    num_batches = 0

    for start_idx in range(0, num_samples, batch_size):
        batch_idx = indices[start_idx : start_idx + batch_size]
        batch_arrays = arrays[batch_idx].to(device)
        batch_targets = targets[batch_idx].to(device)

        optimizer.zero_grad()
        logits, loss = model(batch_arrays, batch_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def eval_epoch(model, data, batch_size, device):
    """Evaluate with comprehensive metrics."""
    model.eval()
    arrays, targets, obs_masks = data
    num_samples = len(arrays)

    all_preds = []
    all_targets = []
    perfect_sorts = 0

    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            batch_arrays = arrays[start_idx : start_idx + batch_size].to(device)
            batch_targets = targets[start_idx : start_idx + batch_size].to(device)

            logits, _ = model(batch_arrays, targets=None)
            preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_targets.append(batch_targets.cpu())

            perfect = (preds == batch_targets).all(dim=1).sum().item()
            perfect_sorts += perfect

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    total_correct = (all_preds == all_targets).sum().item()
    total_elements = all_targets.numel()
    accuracy = total_correct / total_elements
    perfect_rate = perfect_sorts / num_samples
    
    # Kendall-tau (average over batch)
    kendall_scores = []
    for i in range(len(all_preds)):
        tau = compute_kendall_tau(all_preds[i], all_targets[i])
        kendall_scores.append(tau)
    avg_kendall = np.mean(kendall_scores)
    
    # Hamming distance
    hamming = 1.0 - accuracy

    return {
        "accuracy": accuracy,
        "perfect": perfect_rate,
        "kendall": avg_kendall,
        "hamming": hamming,
    }


def run_single_experiment(
    args,
    seed: int,
    max_inner_iters: int,
    model_type: str = "pot"
) -> Dict:
    """Run single experiment with specified iteration count."""
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate data (same for all iteration counts)
    train_data = generate_partial_obs_data(
        args.train_samples, args.array_len, args.mask_rate, seed=seed
    )
    dev_data = generate_partial_obs_data(
        args.dev_samples, args.array_len, args.mask_rate, seed=seed + 1000
    )
    test_data = generate_partial_obs_data(
        args.test_samples, args.array_len, args.mask_rate, seed=seed + 2000
    )

    # Initialize model
    if model_type == "baseline":
        model = PointerDecoderSort(
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_model * 2,
            use_poh=False,
        ).to(device)
    else:  # pot
        model = PointerDecoderSort(
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_model * 2,
            max_inner_iters=max_inner_iters,
            use_poh=True,
            use_hrm=not args.use_full_bptt,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training
    best_dev_kendall = -1.0
    best_epoch = 0
    best_test_metrics = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_data, optimizer, args.batch_size, device, args.clip_norm
        )
        dev_metrics = eval_epoch(model, dev_data, args.batch_size, device)

        # Track best dev Kendall-τ
        if dev_metrics["kendall"] > best_dev_kendall:
            best_dev_kendall = dev_metrics["kendall"]
            best_epoch = epoch
            # Evaluate on test
            best_test_metrics = eval_epoch(model, test_data, args.batch_size, device)

    return {
        "seed": seed,
        "model": model_type,
        "max_inner_iters": max_inner_iters,
        "array_len": args.array_len,
        "mask_rate": args.mask_rate,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "best_dev_kendall": best_dev_kendall,
        "test_accuracy": best_test_metrics["accuracy"],
        "test_perfect": best_test_metrics["perfect"],
        "test_kendall": best_test_metrics["kendall"],
        "test_hamming": best_test_metrics["hamming"],
    }


def main():
    parser = argparse.ArgumentParser(description="Iteration Plateau Analysis")
    
    # Data
    parser.add_argument('--array_len', type=int, default=20,
                        help='Array length')
    parser.add_argument('--mask_rate', type=float, default=0.5,
                        help='Masking rate (0.5 = 50% masked)')
    parser.add_argument('--train_samples', type=int, default=1000,
                        help='Training samples')
    parser.add_argument('--dev_samples', type=int, default=200,
                        help='Dev samples')
    parser.add_argument('--test_samples', type=int, default=500,
                        help='Test samples')
    
    # Model
    parser.add_argument('--d_model', type=int, default=128,
                        help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='Number of heads')
    parser.add_argument('--use_full_bptt', action='store_true',
                        help='Use full BPTT (not HRM-style)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('--clip_norm', type=float, default=1.0,
                        help='Gradient clipping norm')
    
    # Experiment
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3],
                        help='Random seeds to run')
    parser.add_argument('--iterations', type=int, nargs='+',
                        default=[1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20],
                        help='Iteration counts to test')
    parser.add_argument('--output_csv', type=str,
                        default='experiments/results/iteration_plateau.csv',
                        help='Output CSV file')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ITERATION PLATEAU ANALYSIS")
    print("=" * 80)
    print(f"\nTask: Partial observability sorting")
    print(f"Array length: {args.array_len}")
    print(f"Mask rate: {args.mask_rate} ({int(args.mask_rate * 100)}%)")
    print(f"Training: {args.train_samples} samples, {args.epochs} epochs")
    print(f"Seeds: {args.seeds}")
    print(f"Iterations to test: {args.iterations}")
    print(f"\nTotal experiments: {len(args.seeds) * (len(args.iterations) + 1)}")
    print(f"  - Baseline: {len(args.seeds)} runs")
    print(f"  - PoH: {len(args.seeds)} × {len(args.iterations)} = {len(args.seeds) * len(args.iterations)} runs")
    print("=" * 80)
    print()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    # Results storage
    all_results = []
    
    # Run baseline (for reference)
    print("\n" + "=" * 80)
    print("BASELINE (Single-Pass, No PoH)")
    print("=" * 80)
    for seed in args.seeds:
        print(f"\n  Seed {seed}...", end=" ", flush=True)
        result = run_single_experiment(args, seed, max_inner_iters=1, model_type="baseline")
        all_results.append(result)
        print(f"✓ Kendall-τ: {result['test_kendall']:.4f}")
    
    # Run PoH with different iteration counts
    print("\n" + "=" * 80)
    print("POH WITH VARYING ITERATIONS")
    print("=" * 80)
    
    for iters in args.iterations:
        print(f"\n{'─' * 40}")
        print(f"Iterations: {iters}")
        print(f"{'─' * 40}")
        
        iter_results = []
        for seed in args.seeds:
            print(f"  Seed {seed}...", end=" ", flush=True)
            result = run_single_experiment(args, seed, max_inner_iters=iters, model_type="pot")
            all_results.append(result)
            iter_results.append(result['test_kendall'])
            print(f"✓ Kendall-τ: {result['test_kendall']:.4f}")
        
        # Summary for this iteration count
        mean_tau = np.mean(iter_results)
        std_tau = np.std(iter_results)
        print(f"  → Mean ± Std: {mean_tau:.4f} ± {std_tau:.4f}")
    
    # Save results to CSV
    print(f"\n{'=' * 80}")
    print("SAVING RESULTS")
    print(f"{'=' * 80}")
    
    with open(args.output_csv, 'w', newline='') as f:
        fieldnames = [
            'seed', 'model', 'max_inner_iters', 'array_len', 'mask_rate',
            'n_params', 'best_epoch', 'best_dev_kendall',
            'test_accuracy', 'test_perfect', 'test_kendall', 'test_hamming'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)
    
    print(f"✓ Results saved to: {args.output_csv}")
    
    # Compute and display summary statistics
    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 80}\n")
    
    # Group by iteration count
    from collections import defaultdict
    iter_groups = defaultdict(list)
    baseline_taus = []
    
    for result in all_results:
        if result['model'] == 'baseline':
            baseline_taus.append(result['test_kendall'])
        else:
            iter_groups[result['max_inner_iters']].append(result['test_kendall'])
    
    # Baseline stats
    baseline_mean = np.mean(baseline_taus)
    baseline_std = np.std(baseline_taus)
    print(f"Baseline (single-pass):")
    print(f"  Kendall-τ: {baseline_mean:.4f} ± {baseline_std:.4f}")
    print(f"  Runs: {len(baseline_taus)}")
    print()
    
    # PoH stats by iteration count
    print("PoH Results by Iteration Count:")
    print(f"{'─' * 60}")
    print(f"{'Iters':<8} {'Mean τ':<12} {'Std':<10} {'vs Base':<12} {'Improvement'}")
    print(f"{'─' * 60}")
    
    for iters in sorted(iter_groups.keys()):
        taus = iter_groups[iters]
        mean_tau = np.mean(taus)
        std_tau = np.std(taus)
        diff = mean_tau - baseline_mean
        pct = (diff / baseline_mean) * 100 if baseline_mean > 0 else 0
        
        status = "✅" if diff > 0.005 else "⚠️" if diff > 0 else "❌"
        
        print(f"{iters:<8} {mean_tau:.4f}      ±{std_tau:.4f}   "
              f"{diff:+.4f}      {pct:+.1f}% {status}")
    
    print(f"{'─' * 60}")
    
    # Find plateau point
    print("\n" + "=" * 80)
    print("PLATEAU ANALYSIS")
    print("=" * 80)
    
    sorted_iters = sorted(iter_groups.keys())
    if len(sorted_iters) >= 3:
        # Find where improvement becomes < 0.005 (0.5%)
        plateau_iter = None
        for i in range(1, len(sorted_iters)):
            curr_mean = np.mean(iter_groups[sorted_iters[i]])
            prev_mean = np.mean(iter_groups[sorted_iters[i-1]])
            improvement = curr_mean - prev_mean
            
            print(f"  {sorted_iters[i-1]:2d} → {sorted_iters[i]:2d} iters: "
                  f"Δτ = {improvement:+.4f} ({(improvement/prev_mean)*100:+.1f}%)")
            
            if improvement < 0.005 and plateau_iter is None:
                plateau_iter = sorted_iters[i-1]
        
        if plateau_iter:
            print(f"\n✓ Plateau detected at ~{plateau_iter} iterations")
            print(f"  (Further increases yield < 0.5% improvement)")
        else:
            print(f"\n⚠ No clear plateau found - consider testing more iterations")
    
    # Best configuration
    print("\n" + "=" * 80)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 80)
    
    best_iter = max(iter_groups.keys(), key=lambda k: np.mean(iter_groups[k]))
    best_mean = np.mean(iter_groups[best_iter])
    best_std = np.std(iter_groups[best_iter])
    
    print(f"\nBest iteration count: {best_iter}")
    print(f"  Mean Kendall-τ: {best_mean:.4f} ± {best_std:.4f}")
    print(f"  vs Baseline: {best_mean - baseline_mean:+.4f} ({((best_mean - baseline_mean)/baseline_mean)*100:+.1f}%)")
    
    print(f"\n{'=' * 80}")
    print("COMPLETE!")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()

