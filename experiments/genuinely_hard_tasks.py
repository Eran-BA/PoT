"""
Genuinely Hard Tasks for Pointer Networks

Author: Eran Ben Artzy
License: Apache 2.0

Tasks that should be HARD and differentiate architectures:
1. Multi-hop reasoning: Sort by INDIRECT values (sort indices by values they point to)
2. Compositional: Sort by f(g(x)) where f,g are learned
3. Partial observability: Only see k out of n values, must infer rest
4. Adversarial: Train on sorted, test on reverse-sorted (distribution shift)
5. Long-range dependencies: Sort pairs by sum, then by first element
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


def generate_indirect_sort_data(num_samples, array_len):
    """
    Multi-hop reasoning: Given indices [0,1,2,...,n-1] and values [v0,v1,...,vn-1],
    sort the INDICES by the VALUES they point to.
    
    Example: indices=[0,1,2], values=[5,2,8] → sorted indices=[1,0,2] (because v1<v0<v2)
    
    This requires:
    1. Attention to fetch value for each index
    2. Compare fetched values
    3. Sort indices accordingly
    """
    arrays = []
    targets = []
    
    for _ in range(num_samples):
        # Generate random values
        values = np.random.randint(-50, 50, size=array_len)
        
        # Indices are just positions
        indices = np.arange(array_len)
        
        # Target: argsort of values (which index to pick at each output position)
        perm = np.argsort(values, kind="stable")
        
        # Input is CONCATENATION: [indices | values]
        # Shape: [array_len, 2] where [:,0] = indices, [:,1] = values
        input_array = np.stack([indices, values], axis=-1).astype(np.float32)
        
        arrays.append(input_array)
        targets.append(perm.astype(np.int64))
    
    return (
        torch.from_numpy(np.array(arrays)),
        torch.from_numpy(np.array(targets)),
    )


def generate_compositional_sort_data(num_samples, array_len):
    """
    Compositional: Sort by f(x) = (x mod 7) * 3 + (x // 7)
    
    Requires learning the composition, not just direct value comparison.
    """
    arrays = []
    targets = []
    
    for _ in range(num_samples):
        # Random integers
        values = np.random.randint(-30, 30, size=array_len)
        
        # Transform: f(x) = (x mod 7) * 3 + (x // 7)
        transformed = (values % 7) * 3 + (values // 7)
        
        # Sort by TRANSFORMED values
        perm = np.argsort(transformed, kind="stable")
        
        arrays.append(values.reshape(-1, 1).astype(np.float32))
        targets.append(perm.astype(np.int64))
    
    return (
        torch.from_numpy(np.array(arrays)),
        torch.from_numpy(np.array(targets)),
    )


def generate_partial_observable_sort_data(num_samples, array_len, visible_rate=0.5):
    """
    Partial observability: Only k values are visible, rest are masked (-999).
    Must infer full sort from partial information.
    
    This is GENUINELY HARD - requires reasoning about unseen values.
    """
    arrays = []
    targets = []
    
    n_visible = max(2, int(array_len * visible_rate))
    
    for _ in range(num_samples):
        # Full values
        values = np.random.choice(np.arange(-50, 50), size=array_len, replace=False)
        
        # Randomly mask some
        visible_mask = np.zeros(array_len, dtype=bool)
        visible_indices = np.random.choice(array_len, size=n_visible, replace=False)
        visible_mask[visible_indices] = True
        
        # Create input with masks
        masked_values = values.copy()
        masked_values[~visible_mask] = -999.0  # Sentinel for masked
        
        # Target: sort ALL values (including masked ones)
        perm = np.argsort(values, kind="stable")
        
        arrays.append(masked_values.reshape(-1, 1).astype(np.float32))
        targets.append(perm.astype(np.int64))
    
    return (
        torch.from_numpy(np.array(arrays)),
        torch.from_numpy(np.array(targets)),
    )


def generate_long_range_dependency_data(num_samples, array_len):
    """
    Long-range dependencies: Sort pairs (a,b) by sum a+b, then by a as tiebreaker.
    
    Requires:
    1. Computing pairwise sums (all-to-all attention)
    2. Comparing across positions
    3. Tiebreaking with second criterion
    """
    if array_len % 2 != 0:
        array_len += 1
    
    arrays = []
    targets = []
    
    for _ in range(num_samples):
        # Generate pairs
        values = np.random.randint(-20, 20, size=array_len)
        
        # Compute sort keys: (sum of each pair, first element)
        sort_keys = []
        for i in range(0, array_len, 2):
            pair_sum = values[i] + values[i+1]
            first = values[i]
            sort_keys.append((pair_sum, first, i))  # (sum, first, pair_index)
        
        # Sort by (sum, first)
        sort_keys.sort()
        
        # Build permutation: for each sorted pair, output its two indices
        perm = []
        for _, _, pair_idx in sort_keys:
            perm.extend([pair_idx, pair_idx + 1])
        
        arrays.append(values.reshape(-1, 1).astype(np.float32))
        targets.append(np.array(perm).astype(np.int64))
    
    return (
        torch.from_numpy(np.array(arrays)),
        torch.from_numpy(np.array(targets)),
    )


def generate_adversarial_shift_data(num_samples, array_len, is_train=True):
    """
    Distribution shift: Train on ascending sort, test on DESCENDING sort.
    
    Tests OOD generalization and whether the model learned the sorting primitive
    or just memorized the training distribution.
    """
    arrays = []
    targets = []
    
    for _ in range(num_samples):
        values = np.random.choice(np.arange(-50, 50), size=array_len, replace=False)
        
        if is_train:
            # Train: ascending sort
            perm = np.argsort(values, kind="stable")
        else:
            # Test: DESCENDING sort
            perm = np.argsort(-values, kind="stable")
        
        arrays.append(values.reshape(-1, 1).astype(np.float32))
        targets.append(perm.astype(np.int64))
    
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
    print("\n[BASELINE] Training...")
    baseline = PointerDecoderSort(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_model * 2,
        use_poh=False,
    ).to(device)
    opt_baseline = torch.optim.Adam(baseline.parameters(), lr=args.lr)
    
    baseline_best = 0.0
    for epoch in range(1, args.epochs + 1):
        train_epoch(baseline, train_data, opt_baseline, args.batch_size)
        if epoch % 10 == 0 or epoch == args.epochs:
            metrics = evaluate_detailed(baseline, test_data, args.batch_size, device)
            baseline_best = max(baseline_best, metrics['accuracy'])
            print(f"  Epoch {epoch}: {metrics['accuracy']:.3f} acc, {metrics['perfect_sort']:.3f} perfect (best: {baseline_best:.3f})")
    
    baseline_metrics = evaluate_detailed(baseline, test_data, args.batch_size, device)
    
    # PoH
    print("\n[PoH] Training...")
    poh = PointerDecoderSort(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_model * 2,
        max_inner_iters=args.max_inner_iters,
        use_poh=True,
        use_hrm=args.use_hrm,
    ).to(device)
    opt_poh = torch.optim.Adam(poh.parameters(), lr=args.lr)
    
    poh_best = 0.0
    for epoch in range(1, args.epochs + 1):
        train_epoch(poh, train_data, opt_poh, args.batch_size)
        if epoch % 10 == 0 or epoch == args.epochs:
            metrics = evaluate_detailed(poh, test_data, args.batch_size, device)
            poh_best = max(poh_best, metrics['accuracy'])
            print(f"  Epoch {epoch}: {metrics['accuracy']:.3f} acc, {metrics['perfect_sort']:.3f} perfect (best: {poh_best:.3f})")
    
    poh_metrics = evaluate_detailed(poh, test_data, args.batch_size, device)
    
    # Results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Baseline: {baseline_metrics['accuracy']:.3f} acc, {baseline_metrics['perfect_sort']:.3f} perfect (best: {baseline_best:.3f})")
    print(f"PoH:      {poh_metrics['accuracy']:.3f} acc, {poh_metrics['perfect_sort']:.3f} perfect (best: {poh_best:.3f})")
    
    delta_acc = poh_metrics['accuracy'] - baseline_metrics['accuracy']
    delta_best = poh_best - baseline_best
    
    if delta_acc > 0.02:
        print(f"\n🎉 PoH wins: +{delta_acc:.1%} accuracy (best: +{delta_best:.1%})")
    elif delta_acc < -0.02:
        print(f"\n⚠️  Baseline wins: {-delta_acc:.1%} better accuracy")
    else:
        print(f"\n⚖️  Tied: <2% difference")
    
    return {
        "name": name,
        "baseline_acc": baseline_metrics['accuracy'],
        "baseline_best": baseline_best,
        "poh_acc": poh_metrics['accuracy'],
        "poh_best": poh_best,
        "delta": delta_acc,
        "delta_best": delta_best,
    }


def main():
    parser = argparse.ArgumentParser(description="Genuinely hard pointer network tasks")
    parser.add_argument("--array_len", type=int, default=10, help="Array length")
    parser.add_argument("--train_samples", type=int, default=500, help="Training samples")
    parser.add_argument("--test_samples", type=int, default=200, help="Test samples")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--max_inner_iters", type=int, default=4, help="PoH iterations")
    parser.add_argument("--use_hrm", action="store_true", help="Use HRM-style gradients")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--task",
        type=str,
        default="compositional",
        choices=["indirect", "compositional", "partial", "long_range", "adversarial"],
        help="Which task to test",
    )
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Task: {args.task}")
    print(f"Array length: {args.array_len}")
    print(f"Training samples: {args.train_samples}")

    if args.task == "indirect":
        train_data = generate_indirect_sort_data(args.train_samples, args.array_len)
        test_data = generate_indirect_sort_data(args.test_samples, args.array_len)
        result = run_experiment("Indirect Sort (Multi-hop)", train_data, test_data, args)
        
    elif args.task == "compositional":
        train_data = generate_compositional_sort_data(args.train_samples, args.array_len)
        test_data = generate_compositional_sort_data(args.test_samples, args.array_len)
        result = run_experiment("Compositional f(x) = (x%7)*3 + x//7", train_data, test_data, args)
        
    elif args.task == "partial":
        train_data = generate_partial_observable_sort_data(args.train_samples, args.array_len, visible_rate=0.5)
        test_data = generate_partial_observable_sort_data(args.test_samples, args.array_len, visible_rate=0.5)
        result = run_experiment("Partial Observability (50% masked)", train_data, test_data, args)
        
    elif args.task == "long_range":
        train_data = generate_long_range_dependency_data(args.train_samples, args.array_len)
        test_data = generate_long_range_dependency_data(args.test_samples, args.array_len)
        result = run_experiment("Long-range Dependencies (pair sums)", train_data, test_data, args)
        
    elif args.task == "adversarial":
        train_data = generate_adversarial_shift_data(args.train_samples, args.array_len, is_train=True)
        test_data = generate_adversarial_shift_data(args.test_samples, args.array_len, is_train=False)
        result = run_experiment("Adversarial Shift (train asc, test desc)", train_data, test_data, args)

    print(f"\n{'='*80}")
    print(f"TASK: {args.task}")
    print(f"PoH advantage: {result['delta']:+.1%} accuracy (best: {result['delta_best']:+.1%})")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

