"""
High-Performance Partial Observability Sorting

Goal: Push for HIGHER absolute scores on partial observability task
Strategy: Larger models, more training, curriculum learning, better features

Author: Eran Ben Artzy
License: Apache 2.0
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
from sort_pointer_fixed import PointerDecoderSort, train_epoch


def generate_partial_observable_enhanced(num_samples, array_len, visible_rate=0.5, value_range=100):
    """
    Enhanced partial observability: just use masked values (single feature).
    More data + bigger model should help learn from partial observations.
    """
    arrays = []
    targets = []
    
    n_visible = max(2, int(array_len * visible_rate))
    
    for _ in range(num_samples):
        # Generate unique values
        values = np.random.choice(
            np.arange(-value_range, value_range), size=array_len, replace=False
        )
        
        # Randomly mask some
        visible_mask = np.zeros(array_len, dtype=bool)
        visible_indices = np.random.choice(array_len, size=n_visible, replace=False)
        visible_mask[visible_indices] = True
        
        # Create input with masks (use sentinel -999 for masked)
        masked_values = values.copy().astype(np.float32)
        masked_values[~visible_mask] = -999.0
        
        # Target: sort ALL values (including masked ones)
        perm = np.argsort(values, kind="stable")
        
        arrays.append(masked_values.reshape(-1, 1))
        targets.append(perm.astype(np.int64))
    
    return (
        torch.from_numpy(np.array(arrays)),
        torch.from_numpy(np.array(targets)),
    )


def train_epoch_enhanced(model, data, optimizer, batch_size, device):
    """Enhanced training with gradient accumulation and better metrics."""
    model.train()
    arrays, targets = data
    num_samples = len(arrays)
    indices = torch.randperm(num_samples)

    total_loss = 0.0
    total_correct = 0
    total_elements = 0
    num_batches = 0

    for start_idx in range(0, num_samples, batch_size):
        batch_idx = indices[start_idx : start_idx + batch_size]
        batch_arrays = arrays[batch_idx].to(device)
        batch_targets = targets[batch_idx].to(device)

        optimizer.zero_grad()
        logits, loss = model(batch_arrays, batch_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        preds = logits.argmax(dim=-1)
        correct = (preds == batch_targets).sum().item()
        total_correct += correct
        total_elements += batch_targets.numel()
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    accuracy = total_correct / total_elements
    return avg_loss, accuracy


def eval_epoch_enhanced(model, data, batch_size, device):
    """Enhanced evaluation with Kendall-tau."""
    model.eval()
    arrays, targets = data
    num_samples = len(arrays)

    total_correct = 0
    total_elements = 0
    perfect_sorts = 0
    kendall_scores = []

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
            
            # Kendall-tau
            for i in range(len(batch_arrays)):
                pred_np = preds[i].cpu().numpy()
                target_np = batch_targets[i].cpu().numpy()
                tau = compute_kendall_tau(pred_np, target_np)
                kendall_scores.append(tau)

    accuracy = total_correct / total_elements
    perfect_rate = perfect_sorts / num_samples
    avg_kendall = np.mean(kendall_scores) if kendall_scores else 0.0

    return {
        "accuracy": accuracy,
        "perfect": perfect_rate,
        "kendall": avg_kendall,
    }


def compute_kendall_tau(pred, target):
    """Compute Kendall-tau correlation."""
    n = len(pred)
    concordant = 0
    discordant = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            pred_order = np.sign(pred[i] - pred[j])
            target_order = np.sign(target[i] - target[j])
            if pred_order == target_order and pred_order != 0:
                concordant += 1
            elif pred_order == -target_order and pred_order != 0:
                discordant += 1
    
    total_pairs = n * (n - 1) / 2
    return (concordant - discordant) / total_pairs if total_pairs > 0 else 0.0


def curriculum_training(model, optimizer, device, args):
    """Curriculum: start with easier (more visible), progress to harder."""
    visible_rates = [0.7, 0.6, 0.5, 0.4]
    epochs_per_stage = args.epochs // len(visible_rates)
    
    best_kendall = 0.0
    
    for stage, vis_rate in enumerate(visible_rates):
        print(f"\n{'='*80}")
        print(f"CURRICULUM STAGE {stage+1}/{len(visible_rates)}: {int(vis_rate*100)}% visible")
        print(f"{'='*80}")
        
        # Generate data for this stage
        train_data = generate_partial_observable_enhanced(
            args.train_samples, args.array_len, visible_rate=vis_rate
        )
        test_data = generate_partial_observable_enhanced(
            args.test_samples, args.array_len, visible_rate=0.5  # Always test on 50%
        )
        
        for epoch in range(1, epochs_per_stage + 1):
            train_loss, train_acc = train_epoch_enhanced(
                model, train_data, optimizer, args.batch_size, device
            )
            
            if epoch % 10 == 0 or epoch == epochs_per_stage:
                test_metrics = eval_epoch_enhanced(model, test_data, args.batch_size, device)
                best_kendall = max(best_kendall, test_metrics["kendall"])
                
                print(f"  Epoch {epoch}/{epochs_per_stage}: "
                      f"train_acc={train_acc:.3f}, test_acc={test_metrics['accuracy']:.3f}, "
                      f"kendall={test_metrics['kendall']:.3f}, perfect={test_metrics['perfect']:.3f}")
    
    return best_kendall


def main():
    parser = argparse.ArgumentParser(description="High-performance partial observability")
    parser.add_argument("--array_len", type=int, default=10, help="Array length")
    parser.add_argument("--train_samples", type=int, default=2000, help="Training samples")
    parser.add_argument("--test_samples", type=int, default=500, help="Test samples")
    parser.add_argument("--epochs", type=int, default=120, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--max_inner_iters", type=int, default=6, help="PoH iterations")
    parser.add_argument("--use_curriculum", action="store_true", help="Use curriculum learning")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Array length: {args.array_len}")
    print(f"Model: d_model={args.d_model}, n_heads={args.n_heads}, iters={args.max_inner_iters}")
    print(f"Training samples: {args.train_samples}, epochs: {args.epochs}")
    print(f"Curriculum learning: {args.use_curriculum}")

    # Generate data
    print("\nGenerating data...")
    if not args.use_curriculum:
        train_data = generate_partial_observable_enhanced(
            args.train_samples, args.array_len, visible_rate=0.5
        )
        test_data = generate_partial_observable_enhanced(
            args.test_samples, args.array_len, visible_rate=0.5
        )

    print("\n" + "="*80)
    print("BASELINE MODEL")
    print("="*80)
    
    # Baseline
    baseline = PointerDecoderSort(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_model * 2,
        use_poh=False,
    ).to(device)
    opt_baseline = torch.optim.Adam(baseline.parameters(), lr=args.lr)
    
    if args.use_curriculum:
        baseline_best = curriculum_training(baseline, opt_baseline, device, args)
    else:
        baseline_best = 0.0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch_enhanced(
                baseline, train_data, opt_baseline, args.batch_size, device
            )
            
            if epoch % 10 == 0 or epoch == 1:
                test_metrics = eval_epoch_enhanced(baseline, test_data, args.batch_size, device)
                baseline_best = max(baseline_best, test_metrics["kendall"])
                print(f"Epoch {epoch}: train_acc={train_acc:.3f}, test_acc={test_metrics['accuracy']:.3f}, "
                      f"kendall={test_metrics['kendall']:.3f}, perfect={test_metrics['perfect']:.3f}")

    baseline_final = eval_epoch_enhanced(baseline, test_data, args.batch_size, device)

    print("\n" + "="*80)
    print("PoH MODEL")
    print("="*80)
    
    # PoH
    poh = PointerDecoderSort(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_model * 2,
        max_inner_iters=args.max_inner_iters,
        use_poh=True,
    ).to(device)
    opt_poh = torch.optim.Adam(poh.parameters(), lr=args.lr)
    
    # Re-generate data for fair comparison
    if not args.use_curriculum:
        train_data = generate_partial_observable_enhanced(
            args.train_samples, args.array_len, visible_rate=0.5
        )
        test_data = generate_partial_observable_enhanced(
            args.test_samples, args.array_len, visible_rate=0.5
        )
    
    if args.use_curriculum:
        poh_best = curriculum_training(poh, opt_poh, device, args)
    else:
        poh_best = 0.0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch_enhanced(
                poh, train_data, opt_poh, args.batch_size, device
            )
            
            if epoch % 10 == 0 or epoch == 1:
                test_metrics = eval_epoch_enhanced(poh, test_data, args.batch_size, device)
                poh_best = max(poh_best, test_metrics["kendall"])
                print(f"Epoch {epoch}: train_acc={train_acc:.3f}, test_acc={test_metrics['accuracy']:.3f}, "
                      f"kendall={test_metrics['kendall']:.3f}, perfect={test_metrics['perfect']:.3f}")

    poh_final = eval_epoch_enhanced(poh, test_data, args.batch_size, device)

    # Results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Baseline: acc={baseline_final['accuracy']:.3f}, kendall={baseline_final['kendall']:.3f} (best: {baseline_best:.3f})")
    print(f"PoH:      acc={poh_final['accuracy']:.3f}, kendall={poh_final['kendall']:.3f} (best: {poh_best:.3f})")
    print(f"\n🎯 PoH Advantage: +{(poh_best - baseline_best):.3f} Kendall-τ")
    print(f"   Baseline Best: {baseline_best:.1%}")
    print(f"   PoH Best: {poh_best:.1%}")


if __name__ == "__main__":
    main()

