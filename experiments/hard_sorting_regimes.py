"""
Hard Sorting Regimes to Demonstrate PoH Advantages

Author: Eran Ben Artzy
License: Apache 2.0

Focus on regimes where PoH should show REAL gains:
1. Data-scarce: Very few training examples (50-200)
2. Noisy values: Gaussian noise added to integers
3. Heavy duplicates: 50% duplicate values
4. Long sequences with distractors: PAD tokens to ignore
"""

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from sort_pointer_fixed import PointerDecoderSort, train_epoch, eval_epoch


def generate_noisy_sort_data(num_samples, array_len, noise_std=0.1):
    """Generate sorting data with Gaussian noise."""
    arrays = []
    targets = []

    for _ in range(num_samples):
        # Base integers
        values = np.random.choice(
            np.arange(-50, 50), size=array_len, replace=False
        )
        # Add noise
        noisy_values = values + np.random.normal(0, noise_std * np.abs(values).max(), size=array_len)
        
        # Target is sort of ORIGINAL integers (stable)
        perm = np.argsort(values, kind="stable")

        arrays.append(noisy_values.reshape(-1, 1).astype(np.float32))
        targets.append(perm.astype(np.int64))

    return (
        torch.from_numpy(np.array(arrays)),
        torch.from_numpy(np.array(targets)),
    )


def generate_duplicate_sort_data(num_samples, array_len, duplicate_rate=0.5):
    """Generate sorting data with many duplicates."""
    arrays = []
    targets = []

    for _ in range(num_samples):
        # Limit value range to force duplicates
        n_unique = max(2, int(array_len * (1 - duplicate_rate)))
        values = np.random.choice(range(-20, 20), size=array_len, replace=True)
        
        # Stable argsort
        perm = np.argsort(values, kind="stable")

        arrays.append(values.reshape(-1, 1).astype(np.float32))
        targets.append(perm.astype(np.int64))

    return (
        torch.from_numpy(np.array(arrays)),
        torch.from_numpy(np.array(targets)),
    )


def generate_distractor_sort_data(num_samples, array_len, distractor_rate=0.3):
    """
    Generate sorting data with distractor tokens.
    Distractors should be placed at the END of the permutation (ignored).
    """
    arrays = []
    targets = []
    
    n_distractors = int(array_len * distractor_rate)
    n_real = array_len - n_distractors

    for _ in range(num_samples):
        # Real values
        real_values = np.random.choice(
            np.arange(-50, 50), size=n_real, replace=False
        )
        # Distractors (very large sentinel value)
        distractor_values = np.full(n_distractors, 999.0)
        
        # Randomly interleave
        all_values = np.concatenate([real_values, distractor_values])
        positions = np.random.permutation(array_len)
        shuffled = all_values[positions]
        
        # Target: sorted indices of real values, then distractor positions
        real_positions = positions[:n_real]
        distractor_positions = positions[n_real:]
        
        # Argsort of real values
        real_sorted = np.argsort(real_values, kind="stable")
        target_perm = real_positions[real_sorted].tolist() + distractor_positions.tolist()
        
        arrays.append(shuffled.reshape(-1, 1).astype(np.float32))
        targets.append(np.array(target_perm).astype(np.int64))

    return (
        torch.from_numpy(np.array(arrays)),
        torch.from_numpy(np.array(targets)),
    )


def evaluate_detailed(model, data, batch_size, device):
    """Evaluate with detailed metrics."""
    model.eval()
    arrays, targets = data
    num_samples = len(arrays)

    total_correct = 0
    total_elements = 0
    perfect_sorts = 0

    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            batch_arrays = arrays[start_idx : start_idx + batch_size].to(device)
            batch_targets = targets[start_idx : start_idx + batch_size].to(device)

            logits, _ = model(batch_arrays, targets=None)
            preds = logits.argmax(dim=-1)

            correct = (preds == batch_targets).sum().item()
            total_correct += correct
            total_elements += batch_targets.numel()

            perfect = (preds == batch_targets).all(dim=1).sum().item()
            perfect_sorts += perfect

    accuracy = total_correct / total_elements
    perfect_rate = perfect_sorts / num_samples
    
    return {
        "accuracy": accuracy,
        "perfect_sort": perfect_rate,
    }


def run_experiment(name, train_data, test_data, args):
    """Run single A/B experiment."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*80}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Baseline
    print("\nTraining Baseline...")
    baseline = PointerDecoderSort(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_model * 2,
        use_poh=False,
    ).to(device)
    opt_baseline = torch.optim.Adam(baseline.parameters(), lr=args.lr)
    
    for epoch in range(1, args.epochs + 1):
        train_epoch(baseline, train_data, opt_baseline, args.batch_size)
        if epoch % 10 == 0:
            metrics = evaluate_detailed(baseline, test_data, args.batch_size, device)
            print(f"  Epoch {epoch}: {metrics['accuracy']:.3f} acc, {metrics['perfect_sort']:.3f} perfect")
    
    baseline_metrics = evaluate_detailed(baseline, test_data, args.batch_size, device)
    
    # PoH
    print("\nTraining PoH...")
    poh = PointerDecoderSort(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_model * 2,
        max_inner_iters=args.max_inner_iters,
        use_poh=True,
        use_hrm=args.use_hrm,
    ).to(device)
    opt_poh = torch.optim.Adam(poh.parameters(), lr=args.lr)
    
    for epoch in range(1, args.epochs + 1):
        train_epoch(poh, train_data, opt_poh, args.batch_size)
        if epoch % 10 == 0:
            metrics = evaluate_detailed(poh, test_data, args.batch_size, device)
            print(f"  Epoch {epoch}: {metrics['accuracy']:.3f} acc, {metrics['perfect_sort']:.3f} perfect")
    
    poh_metrics = evaluate_detailed(poh, test_data, args.batch_size, device)
    
    # Results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Baseline: {baseline_metrics['accuracy']:.3f} acc, {baseline_metrics['perfect_sort']:.3f} perfect")
    print(f"PoH:      {poh_metrics['accuracy']:.3f} acc, {poh_metrics['perfect_sort']:.3f} perfect")
    
    delta_acc = poh_metrics['accuracy'] - baseline_metrics['accuracy']
    delta_perfect = poh_metrics['perfect_sort'] - baseline_metrics['perfect_sort']
    
    if delta_acc > 0.01:
        print(f"\n🎉 PoH wins: +{delta_acc:.1%} accuracy, +{delta_perfect:.1%} perfect sorts")
    elif delta_acc < -0.01:
        print(f"\n⚠️  Baseline wins: {-delta_acc:.1%} better accuracy")
    else:
        print(f"\n⚖️  Tied: <1% difference")
    
    return {
        "name": name,
        "baseline": baseline_metrics,
        "poh": poh_metrics,
        "delta_acc": delta_acc,
        "delta_perfect": delta_perfect,
    }


def main():
    parser = argparse.ArgumentParser(description="Hard sorting regimes for PoH")
    parser.add_argument("--array_len", type=int, default=12, help="Array length")
    parser.add_argument("--train_samples", type=int, default=100, help="Training samples (data-scarce)")
    parser.add_argument("--test_samples", type=int, default=500, help="Test samples")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--max_inner_iters", type=int, default=4, help="PoH iterations")
    parser.add_argument("--use_hrm", action="store_true", help="Use HRM-style gradients")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--regime",
        type=str,
        default="all",
        choices=["data_scarce", "noisy", "duplicates", "distractors", "all"],
        help="Which regime to test",
    )
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Array length: {args.array_len}")
    print(f"Training samples: {args.train_samples} (DATA-SCARCE)")
    print(f"PoH iterations: {args.max_inner_iters}")
    
    results = []

    # 1. Data-scarce with clean data
    if args.regime in ["data_scarce", "all"]:
        from sort_pointer_fixed import generate_sort_data
        train_data = generate_sort_data(args.train_samples, args.array_len, unique=True)
        test_data = generate_sort_data(args.test_samples, args.array_len, unique=True)
        result = run_experiment("Data-Scarce (Clean)", train_data, test_data, args)
        results.append(result)

    # 2. Noisy values
    if args.regime in ["noisy", "all"]:
        train_data = generate_noisy_sort_data(args.train_samples, args.array_len, noise_std=0.15)
        test_data = generate_noisy_sort_data(args.test_samples, args.array_len, noise_std=0.15)
        result = run_experiment("Noisy Values (15% noise)", train_data, test_data, args)
        results.append(result)

    # 3. Heavy duplicates
    if args.regime in ["duplicates", "all"]:
        train_data = generate_duplicate_sort_data(args.train_samples, args.array_len, duplicate_rate=0.5)
        test_data = generate_duplicate_sort_data(args.test_samples, args.array_len, duplicate_rate=0.5)
        result = run_experiment("Heavy Duplicates (50%)", train_data, test_data, args)
        results.append(result)

    # 4. Distractors
    if args.regime in ["distractors", "all"]:
        train_data = generate_distractor_sort_data(args.train_samples, args.array_len, distractor_rate=0.3)
        test_data = generate_distractor_sort_data(args.test_samples, args.array_len, distractor_rate=0.3)
        result = run_experiment("Distractors (30%)", train_data, test_data, args)
        results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    poh_wins = sum(1 for r in results if r["delta_acc"] > 0.01)
    baseline_wins = sum(1 for r in results if r["delta_acc"] < -0.01)
    ties = len(results) - poh_wins - baseline_wins
    
    print(f"\nPoH wins: {poh_wins}/{len(results)}")
    print(f"Baseline wins: {baseline_wins}/{len(results)}")
    print(f"Ties: {ties}/{len(results)}")
    
    for r in results:
        print(f"\n{r['name']}:")
        print(f"  PoH advantage: {r['delta_acc']:+.1%} accuracy, {r['delta_perfect']:+.1%} perfect sorts")


if __name__ == "__main__":
    main()

