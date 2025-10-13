"""
Fair A/B Comparison: Baseline vs PoT on Partial Observability

Clean protocol:
- Same data/task/split
- Parameter-matched models
- Identical training (epochs, LR, optimizer, clip)
- PoT: max_inner_iters=2, grad_mode=last (HRM-style)
- Report: best dev Kendall-τ → test metrics with mean±95% CI

Author: Eran Ben Artzy
License: Apache 2.0
"""

import argparse
import csv
import json
import os
import random
from datetime import datetime

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
    
    return (
        torch.from_numpy(np.array(arrays)),
        torch.from_numpy(np.array(targets)),
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
    arrays, targets = data
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
    arrays, targets = data
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


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_single_seed(args, seed):
    """Run single seed experiment."""
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate data (same for both models)
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
    if args.model == "baseline":
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
            max_inner_iters=args.max_inner_iters,
            use_poh=True,
            use_hrm=not args.use_full_bptt,  # HRM by default, full BPTT if flag set
        ).to(device)

    n_params = count_parameters(model)
    print(f"  Seed {seed}, {args.model.upper()}: {n_params:,} parameters")

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

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"    Epoch {epoch}: train_loss={train_loss:.3f}, dev_kendall={dev_metrics['kendall']:.3f}")

    print(f"  Best epoch: {best_epoch}, dev_kendall={best_dev_kendall:.3f}")
    print(f"  Test: kendall={best_test_metrics['kendall']:.3f}, perfect={best_test_metrics['perfect']:.3f}, "
          f"hamming={best_test_metrics['hamming']:.3f}")

    return {
        "seed": seed,
        "model": args.model,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "best_dev_kendall": best_dev_kendall,
        "test_kendall": best_test_metrics["kendall"],
        "test_perfect": best_test_metrics["perfect"],
        "test_accuracy": best_test_metrics["accuracy"],
        "test_hamming": best_test_metrics["hamming"],
    }


def compute_stats(results):
    """Compute mean and 95% CI."""
    values = np.array(results)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    ci = 1.96 * std / np.sqrt(len(values))  # 95% CI
    return mean, ci


def main():
    parser = argparse.ArgumentParser(description="Fair A/B: Baseline vs PoT")
    parser.add_argument("--model", type=str, required=True, choices=["baseline", "pot"],
                        help="Model type")
    parser.add_argument("--array_len", type=int, default=12, help="Array length")
    parser.add_argument("--mask_rate", type=float, default=0.5, help="Masking rate")
    parser.add_argument("--train_samples", type=int, default=1000, help="Training samples")
    parser.add_argument("--dev_samples", type=int, default=300, help="Dev samples")
    parser.add_argument("--test_samples", type=int, default=300, help="Test samples")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--clip_norm", type=float, default=1.0, help="Gradient clip norm")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--max_inner_iters", type=int, default=2, help="PoT iterations")
    parser.add_argument("--use_full_bptt", action="store_true", 
                        help="Use full BPTT (default: HRM last-iterate)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="Random seeds")
    parser.add_argument("--output_csv", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    print("=" * 80)
    print(f"FAIR A/B COMPARISON: {args.model.upper()}")
    print("=" * 80)
    print(f"Task: Partial Observability Sorting")
    print(f"Array length: {args.array_len}, Mask rate: {args.mask_rate}")
    print(f"Training: {args.train_samples} samples, {args.epochs} epochs")
    print(f"Model: d_model={args.d_model}, n_heads={args.n_heads}")
    if args.model == "pot":
        grad_mode = "full BPTT" if args.use_full_bptt else "last (HRM)"
        print(f"PoT: max_inner_iters={args.max_inner_iters}, grad_mode={grad_mode}")
    print(f"Seeds: {args.seeds}")
    print()

    # Run all seeds
    all_results = []
    for seed in args.seeds:
        result = run_single_seed(args, seed)
        all_results.append(result)
        print()

    # Compute statistics
    print("=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)

    kendall_mean, kendall_ci = compute_stats([r["test_kendall"] for r in all_results])
    perfect_mean, perfect_ci = compute_stats([r["test_perfect"] for r in all_results])
    hamming_mean, hamming_ci = compute_stats([r["test_hamming"] for r in all_results])
    accuracy_mean, accuracy_ci = compute_stats([r["test_accuracy"] for r in all_results])

    print(f"\n{args.model.upper()} Results (n={len(args.seeds)} seeds):")
    print(f"  Kendall-τ:    {kendall_mean:.3f} ± {kendall_ci:.3f}")
    print(f"  Perfect sort: {perfect_mean:.3f} ± {perfect_ci:.3f} ({perfect_mean*100:.1f}%)")
    print(f"  Accuracy:     {accuracy_mean:.3f} ± {accuracy_ci:.3f}")
    print(f"  Hamming dist: {hamming_mean:.3f} ± {hamming_ci:.3f}")

    # Save results
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n✓ Results saved to {args.output_csv}")

    # Also save summary
    summary = {
        "model": args.model,
        "task": "partial_obs",
        "array_len": args.array_len,
        "mask_rate": args.mask_rate,
        "n_seeds": len(args.seeds),
        "kendall_mean": float(kendall_mean),
        "kendall_ci": float(kendall_ci),
        "perfect_mean": float(perfect_mean),
        "perfect_ci": float(perfect_ci),
        "accuracy_mean": float(accuracy_mean),
        "accuracy_ci": float(accuracy_ci),
        "hamming_mean": float(hamming_mean),
        "hamming_ci": float(hamming_ci),
    }

    if args.output_csv:
        summary_path = args.output_csv.replace(".csv", "_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

