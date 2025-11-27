#!/usr/bin/env python3
"""
Test PoH on various sequence prediction tasks.

Tasks:
- Reverse: Predict reverse order
- Max position: Find position of maximum element
- Median position: Find position of median element

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import argparse
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import models from sort_array_test
import sys
sys.path.insert(0, "/Users/rnbnrzy/Desktop/PoT")
from experiments.sort_array_test import PointerSortModel, BaselineSortModel


def generate_reverse_data(n_samples: int, array_len: int) -> List[Tuple[List[float], List[int]]]:
    """Generate arrays and their reverse order targets."""
    data = []
    for _ in range(n_samples):
        arr = [random.uniform(0, 100) for _ in range(array_len)]
        target = list(range(array_len - 1, -1, -1))  # Reverse: [N-1, N-2, ..., 1, 0]
        data.append((arr, target))
    return data


def generate_max_position_data(n_samples: int, array_len: int) -> List[Tuple[List[float], List[int]]]:
    """Generate arrays and max element position (one-hot style)."""
    data = []
    for _ in range(n_samples):
        arr = [random.uniform(0, 100) for _ in range(array_len)]
        max_pos = arr.index(max(arr))
        # Target: all positions point to max position
        target = [max_pos] * array_len
        data.append((arr, target))
    return data


def generate_sorted_indices_data(n_samples: int, array_len: int) -> List[Tuple[List[float], List[int]]]:
    """Generate arrays and indices that would sort them (argsort)."""
    data = []
    for _ in range(n_samples):
        arr = [random.uniform(0, 100) for _ in range(array_len)]
        # argsort: indices that would sort the array
        sorted_indices = sorted(range(array_len), key=lambda i: arr[i])
        data.append((arr, sorted_indices))
    return data


def train_and_eval(task_name, train_data, test_data, args, device):
    """Train and evaluate models on a task."""
    print(f"\n{'='*80}")
    print(f"Task: {task_name}")
    print(f"{'='*80}\n")

    # Initialize models
    poh_model = PointerSortModel(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_model * 2,
        max_inner_iters=args.max_inner_iters,
    ).to(device)

    baseline_model = BaselineSortModel(
        d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_model * 2
    ).to(device)

    # Optimizers
    poh_optimizer = torch.optim.Adam(poh_model.parameters(), lr=args.lr)
    baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=args.lr)

    # Training loop
    best_poh_acc = 0.0
    best_baseline_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # Train PoH
        poh_model.train()
        for i in range(0, len(train_data), args.batch_size):
            batch = train_data[i : i + args.batch_size]
            arrays = [arr for arr, _ in batch]
            targets = [tgt for _, tgt in batch]

            x = torch.tensor(arrays, dtype=torch.float32).unsqueeze(-1).to(device)
            y = torch.tensor(targets, dtype=torch.long).to(device)

            logits = poh_model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            poh_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(poh_model.parameters(), 1.0)
            poh_optimizer.step()

        # Train Baseline
        baseline_model.train()
        for i in range(0, len(train_data), args.batch_size):
            batch = train_data[i : i + args.batch_size]
            arrays = [arr for arr, _ in batch]
            targets = [tgt for _, tgt in batch]

            x = torch.tensor(arrays, dtype=torch.float32).unsqueeze(-1).to(device)
            y = torch.tensor(targets, dtype=torch.long).to(device)

            logits = baseline_model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            baseline_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(baseline_model.parameters(), 1.0)
            baseline_optimizer.step()

        # Evaluate every 10 epochs
        if epoch % 10 == 0 or epoch == args.epochs:
            poh_acc = evaluate(poh_model, test_data, device, args.batch_size)
            baseline_acc = evaluate(baseline_model, test_data, device, args.batch_size)

            best_poh_acc = max(best_poh_acc, poh_acc)
            best_baseline_acc = max(best_baseline_acc, baseline_acc)

            print(f"Epoch {epoch:3d}: PoH {poh_acc:.3f} | Baseline {baseline_acc:.3f}")

    # Final results
    print(f"\n{task_name} Results:")
    print(f"  PoH:      {best_poh_acc:.4f}")
    print(f"  Baseline: {best_baseline_acc:.4f}")
    print(f"  Advantage: {(best_poh_acc - best_baseline_acc)*100:+.2f}%")

    return best_poh_acc, best_baseline_acc


def evaluate(model, data, device, batch_size):
    """Evaluate model accuracy."""
    model.eval()
    total_correct = 0
    total_items = 0

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            arrays = [arr for arr, _ in batch]
            targets = [tgt for _, tgt in batch]

            x = torch.tensor(arrays, dtype=torch.float32).unsqueeze(-1).to(device)
            y = torch.tensor(targets, dtype=torch.long).to(device)

            logits = model(x)
            preds = logits.argmax(dim=-1)

            correct = (preds == y).sum().item()
            total_correct += correct
            total_items += y.numel()

    return total_correct / total_items


def main():
    parser = argparse.ArgumentParser(description="Sequence Tasks A/B Test")
    parser.add_argument("--array_len", type=int, default=8, help="Array length")
    parser.add_argument("--train_samples", type=int, default=2000, help="Training samples")
    parser.add_argument("--test_samples", type=int, default=200, help="Test samples")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--max_inner_iters", type=int, default=4, help="PoH iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["reverse", "argsort", "max"],
        help="Tasks to run: reverse, argsort, max",
    )
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Array length: {args.array_len}")
    print(f"Tasks: {', '.join(args.tasks)}")

    results = {}

    # Task 1: Reverse
    if "reverse" in args.tasks:
        train_data = generate_reverse_data(args.train_samples, args.array_len)
        test_data = generate_reverse_data(args.test_samples, args.array_len)
        poh_acc, baseline_acc = train_and_eval(
            "REVERSE", train_data, test_data, args, device
        )
        results["reverse"] = (poh_acc, baseline_acc)

    # Task 2: ArgSort (find indices that sort)
    if "argsort" in args.tasks:
        train_data = generate_sorted_indices_data(args.train_samples, args.array_len)
        test_data = generate_sorted_indices_data(args.test_samples, args.array_len)
        poh_acc, baseline_acc = train_and_eval(
            "ARGSORT", train_data, test_data, args, device
        )
        results["argsort"] = (poh_acc, baseline_acc)

    # Task 3: Find Max
    if "max" in args.tasks:
        train_data = generate_max_position_data(args.train_samples, args.array_len)
        test_data = generate_max_position_data(args.test_samples, args.array_len)
        poh_acc, baseline_acc = train_and_eval(
            "FIND MAX", train_data, test_data, args, device
        )
        results["max"] = (poh_acc, baseline_acc)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    for task, (poh, baseline) in results.items():
        advantage = (poh - baseline) * 100
        print(
            f"{task.upper():10s}: PoH {poh:.3f} | Baseline {baseline:.3f} | Δ {advantage:+.2f}%"
        )


if __name__ == "__main__":
    main()

