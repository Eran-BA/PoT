"""
Comprehensive Sorting Experiments to Demonstrate PoH Advantages

Author: Eran Ben Artzy
License: Apache 2.0

Tests:
1. Sample efficiency (200, 500, 1k, 2k, 5k samples)
2. Length generalization (train ≤12, test 16/24/32)
3. Duplicates & stable ties
4. PoH routing modes (concat, soft mixture, top-k)
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

# Import from our fixed implementation
import sys
sys.path.insert(0, os.path.dirname(__file__))
from sort_pointer_fixed import PointerDecoderSort, generate_sort_data, train_epoch, eval_epoch


def kendall_tau_score(preds, targets):
    """
    Compute average Kendall-τ correlation (simple implementation).
    Kendall-τ = (concordant pairs - discordant pairs) / total pairs
    """
    scores = []
    for pred, target in zip(preds, targets):
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        n = len(pred_np)
        
        concordant = 0
        discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                pred_order = np.sign(pred_np[i] - pred_np[j])
                target_order = np.sign(target_np[i] - target_np[j])
                if pred_order == target_order and pred_order != 0:
                    concordant += 1
                elif pred_order == -target_order and pred_order != 0:
                    discordant += 1
        
        total_pairs = n * (n - 1) / 2
        tau = (concordant - discordant) / total_pairs if total_pairs > 0 else 0.0
        scores.append(tau)
    
    return np.mean(scores)


def hamming_distance(preds, targets):
    """Compute average Hamming distance (proportion of disagreements)."""
    return (preds != targets).float().mean().item()


def eval_with_metrics(model, data, batch_size):
    """Evaluate with comprehensive metrics."""
    model.eval()
    arrays, targets = data
    num_samples = len(arrays)

    all_preds = []
    all_targets = []
    total_correct = 0
    total_elements = 0
    perfect_sorts = 0

    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            batch_arrays = arrays[start_idx : start_idx + batch_size].to(
                model.value_embed[0].weight.device
            )
            batch_targets = targets[start_idx : start_idx + batch_size].to(
                model.value_embed[0].weight.device
            )

            logits, _ = model(batch_arrays, targets=None)
            preds = logits.argmax(dim=-1)

            all_preds.append(preds)
            all_targets.append(batch_targets)

            correct = (preds == batch_targets).sum().item()
            total_correct += correct
            total_elements += batch_targets.numel()

            perfect = (preds == batch_targets).all(dim=1).sum().item()
            perfect_sorts += perfect

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    accuracy = total_correct / total_elements
    perfect_rate = perfect_sorts / num_samples
    kendall = kendall_tau_score(all_preds, all_targets)
    hamming = hamming_distance(all_preds, all_targets)

    return {
        "accuracy": accuracy,
        "perfect_sort": perfect_rate,
        "kendall_tau": kendall,
        "hamming_dist": hamming,
    }


def sample_efficiency_experiment(args):
    """Test 1: Sample efficiency curves."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: SAMPLE EFFICIENCY")
    print("=" * 80)

    sample_sizes = [200, 500, 1000, 2000, 5000]
    test_lengths = [12, 20]
    results = []

    for test_len in test_lengths:
        print(f"\n--- Testing on length {test_len} ---")
        test_data = generate_sort_data(500, test_len, unique=True)

        for n_samples in sample_sizes:
            print(f"\nTraining with {n_samples} samples...")
            train_data = generate_sort_data(n_samples, test_len, unique=True)

            # Baseline
            baseline = PointerDecoderSort(
                d_model=args.d_model,
                n_heads=args.n_heads,
                d_ff=args.d_model * 2,
                use_poh=False,
            ).to(args.device)
            opt_baseline = torch.optim.Adam(baseline.parameters(), lr=args.lr)

            for epoch in range(args.epochs):
                train_epoch(baseline, train_data, opt_baseline, args.batch_size)

            baseline_metrics = eval_with_metrics(baseline, test_data, args.batch_size)

            # PoH
            poh = PointerDecoderSort(
                d_model=args.d_model,
                n_heads=args.n_heads,
                d_ff=args.d_model * 2,
                max_inner_iters=args.max_inner_iters,
                use_poh=True,
            ).to(args.device)
            opt_poh = torch.optim.Adam(poh.parameters(), lr=args.lr)

            for epoch in range(args.epochs):
                train_epoch(poh, train_data, opt_poh, args.batch_size)

            poh_metrics = eval_with_metrics(poh, test_data, args.batch_size)

            result = {
                "test_length": test_len,
                "train_samples": n_samples,
                "baseline_acc": baseline_metrics["accuracy"],
                "baseline_perfect": baseline_metrics["perfect_sort"],
                "baseline_kendall": baseline_metrics["kendall_tau"],
                "poh_acc": poh_metrics["accuracy"],
                "poh_perfect": poh_metrics["perfect_sort"],
                "poh_kendall": poh_metrics["kendall_tau"],
            }
            results.append(result)

            print(f"  Baseline: {baseline_metrics['accuracy']:.3f} acc, {baseline_metrics['perfect_sort']:.3f} perfect")
            print(f"  PoH:      {poh_metrics['accuracy']:.3f} acc, {poh_metrics['perfect_sort']:.3f} perfect")

    return results


def length_generalization_experiment(args):
    """Test 2: Train on ≤12, test on longer sequences."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: LENGTH GENERALIZATION")
    print("=" * 80)

    train_len = 12
    test_lengths = [12, 16, 20, 24, 32]
    results = []

    print(f"\nTraining on length {train_len} with {args.train_samples} samples...")
    train_data = generate_sort_data(args.train_samples, train_len, unique=True)

    # Baseline
    print("\nTraining baseline...")
    baseline = PointerDecoderSort(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_model * 2,
        use_poh=False,
    ).to(args.device)
    opt_baseline = torch.optim.Adam(baseline.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_epoch(baseline, train_data, opt_baseline, args.batch_size)

    # PoH
    print("Training PoH...")
    poh = PointerDecoderSort(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_model * 2,
        max_inner_iters=args.max_inner_iters,
        use_poh=True,
    ).to(args.device)
    opt_poh = torch.optim.Adam(poh.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_epoch(poh, train_data, opt_poh, args.batch_size)

    # Test on various lengths
    print("\nTesting on various lengths...")
    for test_len in test_lengths:
        test_data = generate_sort_data(500, test_len, unique=True)

        baseline_metrics = eval_with_metrics(baseline, test_data, args.batch_size)
        poh_metrics = eval_with_metrics(poh, test_data, args.batch_size)

        result = {
            "train_length": train_len,
            "test_length": test_len,
            "baseline_acc": baseline_metrics["accuracy"],
            "baseline_perfect": baseline_metrics["perfect_sort"],
            "baseline_kendall": baseline_metrics["kendall_tau"],
            "poh_acc": poh_metrics["accuracy"],
            "poh_perfect": poh_metrics["perfect_sort"],
            "poh_kendall": poh_metrics["kendall_tau"],
        }
        results.append(result)

        print(f"\nLength {test_len}:")
        print(f"  Baseline: {baseline_metrics['accuracy']:.3f} acc, {baseline_metrics['perfect_sort']:.3f} perfect, {baseline_metrics['kendall_tau']:.3f} τ")
        print(f"  PoH:      {poh_metrics['accuracy']:.3f} acc, {poh_metrics['perfect_sort']:.3f} perfect, {poh_metrics['kendall_tau']:.3f} τ")

    return results


def duplicates_experiment(args):
    """Test 3: Stable sort with duplicates."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: DUPLICATES & STABLE SORT")
    print("=" * 80)

    test_len = 12
    results = []

    print(f"\nTraining on length {test_len} with duplicates...")
    train_data = generate_sort_data(args.train_samples, test_len, unique=False)
    test_data = generate_sort_data(500, test_len, unique=False)

    # Baseline
    print("\nTraining baseline...")
    baseline = PointerDecoderSort(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_model * 2,
        use_poh=False,
    ).to(args.device)
    opt_baseline = torch.optim.Adam(baseline.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_epoch(baseline, train_data, opt_baseline, args.batch_size)

    baseline_metrics = eval_with_metrics(baseline, test_data, args.batch_size)

    # PoH
    print("Training PoH...")
    poh = PointerDecoderSort(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_model * 2,
        max_inner_iters=args.max_inner_iters,
        use_poh=True,
    ).to(args.device)
    opt_poh = torch.optim.Adam(poh.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_epoch(poh, train_data, opt_poh, args.batch_size)

    poh_metrics = eval_with_metrics(poh, test_data, args.batch_size)

    result = {
        "test_length": test_len,
        "duplicates": True,
        "baseline_acc": baseline_metrics["accuracy"],
        "baseline_perfect": baseline_metrics["perfect_sort"],
        "baseline_kendall": baseline_metrics["kendall_tau"],
        "poh_acc": poh_metrics["accuracy"],
        "poh_perfect": poh_metrics["perfect_sort"],
        "poh_kendall": poh_metrics["kendall_tau"],
    }
    results.append(result)

    print(f"\nResults with duplicates:")
    print(f"  Baseline: {baseline_metrics['accuracy']:.3f} acc, {baseline_metrics['perfect_sort']:.3f} perfect")
    print(f"  PoH:      {poh_metrics['accuracy']:.3f} acc, {poh_metrics['perfect_sort']:.3f} perfect")

    return results


def save_results(all_results, output_dir):
    """Save results to CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for exp_name, results in all_results.items():
        # CSV
        csv_path = os.path.join(output_dir, f"{exp_name}_{timestamp}.csv")
        if results:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"\n✓ Saved {exp_name} results to {csv_path}")

        # JSON
        json_path = os.path.join(output_dir, f"{exp_name}_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive sorting experiments for PoH vs Baseline"
    )
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--max_inner_iters", type=int, default=4, help="PoH iterations")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--train_samples", type=int, default=2000, help="Training samples"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/results",
        help="Output directory",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["sample_efficiency", "length_gen", "duplicates"],
        choices=["sample_efficiency", "length_gen", "duplicates"],
        help="Which experiments to run",
    )
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {args.device}")
    print(f"Experiments: {', '.join(args.experiments)}")

    all_results = {}

    if "sample_efficiency" in args.experiments:
        all_results["sample_efficiency"] = sample_efficiency_experiment(args)

    if "length_gen" in args.experiments:
        all_results["length_gen"] = length_generalization_experiment(args)

    if "duplicates" in args.experiments:
        all_results["duplicates"] = duplicates_experiment(args)

    # Save all results
    save_results(all_results, args.output_dir)

    print("\n" + "=" * 80)
    print("✓ ALL EXPERIMENTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

