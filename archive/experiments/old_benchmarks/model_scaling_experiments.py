"""
Model Scaling Experiments: PoH vs Baseline Across Model Sizes

Tests whether PoH benefits more from increased model capacity than baseline.

Model sizes tested:
- Tiny:   d_model=64,  n_heads=2,  d_ff=128   (~15k params)
- Small:  d_model=128, n_heads=4,  d_ff=256   (~100k params)  [default]
- Medium: d_model=256, n_heads=8,  d_ff=512   (~800k params)
- Large:  d_model=512, n_heads=8,  d_ff=1024  (~3M params)
- XLarge: d_model=768, n_heads=12, d_ff=2048  (~10M params)

Hypothesis: PoH's advantage increases with model size
- Larger models have more capacity for head specialization
- More heads = more routing opportunities
- Controller can learn more sophisticated routing patterns

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
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(__file__))
from sort_pointer_fixed import PointerDecoderSort


# Model size configurations
MODEL_CONFIGS = {
    'tiny': {
        'd_model': 64,
        'n_heads': 2,
        'd_ff': 128,
        'description': 'Tiny (~15k params)',
    },
    'small': {
        'd_model': 128,
        'n_heads': 4,
        'd_ff': 256,
        'description': 'Small (~100k params)',
    },
    'medium': {
        'd_model': 256,
        'n_heads': 8,
        'd_ff': 512,
        'description': 'Medium (~800k params)',
    },
    'large': {
        'd_model': 512,
        'n_heads': 8,
        'd_ff': 1024,
        'description': 'Large (~3M params)',
    },
    'xlarge': {
        'd_model': 768,
        'n_heads': 12,
        'd_ff': 2048,
        'description': 'XLarge (~10M params)',
    },
}


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
    model_config: Dict,
    seed: int,
    use_poh: bool,
    max_inner_iters: int = 12,
) -> Dict:
    """Run single experiment with specified model size."""
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate data
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
    model = PointerDecoderSort(
        d_model=model_config['d_model'],
        n_heads=model_config['n_heads'],
        d_ff=model_config['d_ff'],
        max_inner_iters=max_inner_iters if use_poh else 1,
        use_poh=use_poh,
        use_hrm=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"    Model: {model_config['description']}")
    print(f"    Parameters: {n_params:,} ({n_trainable:,} trainable)")

    # Optimizer (adjust LR for larger models)
    base_lr = args.lr
    # Scale LR down slightly for very large models
    if model_config['d_model'] >= 512:
        lr = base_lr * 0.5
    elif model_config['d_model'] >= 256:
        lr = base_lr * 0.75
    else:
        lr = base_lr
    
    print(f"    Learning rate: {lr:.2e}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

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

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"      Epoch {epoch}/{args.epochs}: "
                  f"dev_τ={dev_metrics['kendall']:.4f}, "
                  f"best={best_dev_kendall:.4f}")

    print(f"    ✓ Best epoch: {best_epoch}, Test τ: {best_test_metrics['kendall']:.4f}")

    return {
        "seed": seed,
        "model_size": model_config['description'],
        "d_model": model_config['d_model'],
        "n_heads": model_config['n_heads'],
        "d_ff": model_config['d_ff'],
        "use_poh": use_poh,
        "max_inner_iters": max_inner_iters if use_poh else 1,
        "array_len": args.array_len,
        "mask_rate": args.mask_rate,
        "n_params": n_params,
        "n_trainable": n_trainable,
        "learning_rate": lr,
        "best_epoch": best_epoch,
        "best_dev_kendall": best_dev_kendall,
        "test_accuracy": best_test_metrics["accuracy"],
        "test_perfect": best_test_metrics["perfect"],
        "test_kendall": best_test_metrics["kendall"],
        "test_hamming": best_test_metrics["hamming"],
    }


def main():
    parser = argparse.ArgumentParser(description="Model Scaling Experiments")
    
    # Data
    parser.add_argument('--array_len', type=int, default=20,
                        help='Array length')
    parser.add_argument('--mask_rate', type=float, default=0.5,
                        help='Masking rate')
    parser.add_argument('--train_samples', type=int, default=1000,
                        help='Training samples')
    parser.add_argument('--dev_samples', type=int, default=200,
                        help='Dev samples')
    parser.add_argument('--test_samples', type=int, default=500,
                        help='Test samples')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Base learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('--clip_norm', type=float, default=1.0,
                        help='Gradient clipping norm')
    
    # Model sizes
    parser.add_argument('--sizes', type=str, nargs='+',
                        default=['small', 'medium', 'large'],
                        choices=['tiny', 'small', 'medium', 'large', 'xlarge'],
                        help='Model sizes to test')
    parser.add_argument('--max_inner_iters', type=int, default=12,
                        help='Max inner iterations for PoH')
    
    # Experiment
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3],
                        help='Random seeds')
    parser.add_argument('--output_csv', type=str,
                        default='experiments/results/model_scaling.csv',
                        help='Output CSV file')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MODEL SCALING EXPERIMENTS: PoH vs Baseline")
    print("=" * 80)
    print(f"\nTask: Partial observability sorting")
    print(f"Array length: {args.array_len}")
    print(f"Mask rate: {args.mask_rate} ({int(args.mask_rate * 100)}%)")
    print(f"Training: {args.train_samples} samples, {args.epochs} epochs")
    print(f"Seeds: {args.seeds}")
    print(f"\nModel sizes to test: {args.sizes}")
    for size in args.sizes:
        config = MODEL_CONFIGS[size]
        print(f"  - {size.capitalize()}: {config['description']}")
        print(f"      d_model={config['d_model']}, n_heads={config['n_heads']}, d_ff={config['d_ff']}")
    print(f"\nPoH iterations: {args.max_inner_iters}")
    print(f"\nTotal experiments: {len(args.sizes)} sizes × 2 models × {len(args.seeds)} seeds = {len(args.sizes) * 2 * len(args.seeds)}")
    print("=" * 80)
    print()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    # Results storage
    all_results = []
    
    # Run experiments for each model size
    for size_name in args.sizes:
        config = MODEL_CONFIGS[size_name]
        
        print("\n" + "=" * 80)
        print(f"MODEL SIZE: {size_name.upper()} - {config['description']}")
        print("=" * 80)
        
        # Run baseline for this size
        print(f"\n  Baseline (single-pass, no PoH)")
        print("  " + "─" * 40)
        for seed in args.seeds:
            print(f"\n  Seed {seed}:")
            result = run_single_experiment(
                args, config, seed, use_poh=False
            )
            all_results.append(result)
        
        # Run PoH for this size
        print(f"\n  PoH ({args.max_inner_iters} iterations)")
        print("  " + "─" * 40)
        for seed in args.seeds:
            print(f"\n  Seed {seed}:")
            result = run_single_experiment(
                args, config, seed, use_poh=True, max_inner_iters=args.max_inner_iters
            )
            all_results.append(result)
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    with open(args.output_csv, 'w', newline='') as f:
        fieldnames = [
            'seed', 'model_size', 'd_model', 'n_heads', 'd_ff',
            'use_poh', 'max_inner_iters', 'array_len', 'mask_rate',
            'n_params', 'n_trainable', 'learning_rate',
            'best_epoch', 'best_dev_kendall',
            'test_accuracy', 'test_perfect', 'test_kendall', 'test_hamming'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)
    
    print(f"✓ Results saved to: {args.output_csv}")
    
    # Compute summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    from collections import defaultdict
    
    # Group by model size
    size_groups = defaultdict(lambda: {'baseline': [], 'poh': []})
    
    for result in all_results:
        size = result['model_size']
        if result['use_poh']:
            size_groups[size]['poh'].append(result['test_kendall'])
        else:
            size_groups[size]['baseline'].append(result['test_kendall'])
    
    print()
    print(f"{'Model Size':<20} {'Baseline τ':<15} {'PoH τ':<15} {'Improvement':<15} {'Status'}")
    print("─" * 80)
    
    for size_name in args.sizes:
        config = MODEL_CONFIGS[size_name]
        size_desc = config['description']
        
        baseline_taus = size_groups[size_desc]['baseline']
        poh_taus = size_groups[size_desc]['poh']
        
        if baseline_taus and poh_taus:
            baseline_mean = np.mean(baseline_taus)
            baseline_std = np.std(baseline_taus)
            poh_mean = np.mean(poh_taus)
            poh_std = np.std(poh_taus)
            
            improvement = poh_mean - baseline_mean
            improvement_pct = (improvement / baseline_mean) * 100 if baseline_mean > 0 else 0
            
            status = "✅" if improvement > 0.01 else "⚠️" if improvement > 0 else "❌"
            
            print(f"{size_desc:<20} "
                  f"{baseline_mean:.4f}±{baseline_std:.4f}  "
                  f"{poh_mean:.4f}±{poh_std:.4f}  "
                  f"{improvement:+.4f} ({improvement_pct:+.1f}%)  "
                  f"{status}")
    
    print("─" * 80)
    
    # Scaling analysis
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS")
    print("=" * 80)
    print("\nDoes PoH benefit more from increased capacity?")
    print()
    
    improvements_by_size = []
    param_counts = []
    
    for size_name in args.sizes:
        config = MODEL_CONFIGS[size_name]
        size_desc = config['description']
        
        baseline_taus = size_groups[size_desc]['baseline']
        poh_taus = size_groups[size_desc]['poh']
        
        if baseline_taus and poh_taus:
            baseline_mean = np.mean(baseline_taus)
            poh_mean = np.mean(poh_taus)
            improvement = poh_mean - baseline_mean
            
            # Get param count from results
            for r in all_results:
                if r['model_size'] == size_desc and r['use_poh']:
                    param_counts.append(r['n_params'])
                    improvements_by_size.append(improvement)
                    break
    
    if len(improvements_by_size) >= 2:
        # Check if improvement increases with size
        correlation = np.corrcoef(param_counts, improvements_by_size)[0, 1]
        
        print(f"Correlation between model size and PoH improvement: {correlation:.3f}")
        
        if correlation > 0.5:
            print("✓ Strong positive correlation: PoH benefits MORE from larger models!")
        elif correlation > 0.2:
            print("⚠ Weak positive correlation: Some scaling benefit")
        elif correlation > -0.2:
            print("⚠ No clear correlation: PoH benefit independent of size")
        else:
            print("✗ Negative correlation: PoH may work better with smaller models")
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print()


if __name__ == '__main__':
    main()

