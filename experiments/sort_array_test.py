#!/usr/bin/env python3
"""
Array Sorting A/B Test: PoH vs Baseline

Tests whether the Pointer-over-Heads mechanism can learn to sort arrays
better than a vanilla transformer baseline.

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


class PointerSortModel(nn.Module):
    """PoH model adapted for array sorting."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        max_inner_iters: int = 3,
        use_hrm_style: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_inner_iters = max_inner_iters
        self.use_hrm_style = use_hrm_style

        # Embedding for array values (+ positional encoding)
        self.value_embed = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.pos_embed = nn.Embedding(100, d_model)  # Support up to 100 positions

        # Controller (routes between heads)
        self.controller = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_heads),
        )

        # Multi-head attention
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model),
        )

        # Biaffine pointer head
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.biaffine = nn.Parameter(torch.randn(d_model, d_model) * 0.01)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, 1] - array values

        Returns:
            logits: [B, N, N] - pointer logits
        """
        B, N, _ = x.shape

        # Embed values + positions
        positions = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.value_embed(x) + self.pos_embed(positions)  # [B, N, d_model]

        # Iterative refinement with routing
        for iter_idx in range(self.max_inner_iters):
            # HRM-style: detach gradients for all but last iteration
            if self.use_hrm_style and iter_idx < self.max_inner_iters - 1:
                h = h.detach()

            # Controller decides routing
            alphas = F.softmax(self.controller(h), dim=-1)  # [B, N, n_heads]

            # Self-attention
            attn_out, _ = self.attn(h, h, h)

            # Residual + norm
            h = self.ln1(h + attn_out)
            h = self.ln2(h + self.ffn(h))

        # Biaffine pointer logits
        query = self.query_proj(h)  # [B, N, d_model]
        key = self.key_proj(h)  # [B, N, d_model]

        # Biaffine: query @ W @ key^T
        logits = torch.einsum("bnd,de,bme->bnm", query, self.biaffine, key)

        return logits


class BaselineSortModel(nn.Module):
    """Vanilla transformer baseline for sorting."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, d_ff: int = 256):
        super().__init__()
        self.d_model = d_model

        # Embedding with positional encoding
        self.value_embed = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.pos_embed = nn.Embedding(100, d_model)

        # Self-attention
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model),
        )

        # Biaffine pointer
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.biaffine = nn.Parameter(torch.randn(d_model, d_model) * 0.01)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, 1]

        Returns:
            logits: [B, N, N]
        """
        B, N, _ = x.shape

        # Embed with positions
        positions = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.value_embed(x) + self.pos_embed(positions)

        # Single pass transformer
        attn_out, _ = self.attn(h, h, h)
        h = self.ln1(h + attn_out)
        h = self.ln2(h + self.ffn(h))

        # Biaffine pointer logits
        query = self.query_proj(h)
        key = self.key_proj(h)
        logits = torch.einsum("bnd,de,bme->bnm", query, self.biaffine, key)

        return logits


def generate_sort_data(
    n_samples: int, array_len: int, value_range: Tuple[int, int] = (0, 100)
) -> List[Tuple[List[float], List[int]]]:
    """Generate random arrays and their sort target positions.

    Args:
        n_samples: Number of samples
        array_len: Length of each array
        value_range: Range of values (min, max)

    Returns:
        List of (array, target_positions) tuples
        target_positions[i] = position where element i should go in sorted array

    Example:
        arr = [5, 2, 8, 1]
        sorted: [1, 2, 5, 8]
        target_positions = [2, 1, 3, 0]  # where each input element should go
        - Element 0 (value 5) goes to position 2
        - Element 1 (value 2) goes to position 1
        - Element 2 (value 8) goes to position 3
        - Element 3 (value 1) goes to position 0
    """
    data = []
    for _ in range(n_samples):
        # Random array
        arr = [random.uniform(value_range[0], value_range[1]) for _ in range(array_len)]

        # Get the rank (position in sorted array) of each element
        # sorted_indices[i] tells us what position element i should have
        indexed_arr = list(enumerate(arr))
        sorted_arr = sorted(indexed_arr, key=lambda x: x[1])

        # Create reverse mapping: for each original index, what's its position in sorted array
        target_positions = [0] * array_len
        for new_pos, (orig_idx, _) in enumerate(sorted_arr):
            target_positions[orig_idx] = new_pos

        data.append((arr, target_positions))

    return data


def train_epoch(model, data, optimizer, device, batch_size=32):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    steps = 0

    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]

        # Prepare batch
        arrays = [arr for arr, _ in batch]
        targets = [tgt for _, tgt in batch]

        # To tensors
        x = torch.tensor(arrays, dtype=torch.float32).unsqueeze(-1).to(device)  # [B, N, 1]
        y = torch.tensor(targets, dtype=torch.long).to(device)  # [B, N]

        # Forward
        logits = model(x)  # [B, N, N]

        # Loss: cross-entropy for each position
        B, N, _ = logits.shape
        loss = F.cross_entropy(logits.view(B * N, N), y.view(B * N))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Metrics
        preds = logits.argmax(dim=-1)  # [B, N]
        correct = (preds == y).sum().item()

        total_loss += loss.item()
        total_correct += correct
        total_items += B * N
        steps += 1

    return {
        "loss": total_loss / steps,
        "accuracy": total_correct / total_items,
    }


def eval_model(model, data, device, batch_size=32):
    """Evaluate model."""
    model.eval()
    total_correct = 0
    total_items = 0
    total_perfect = 0

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
            perfect = (preds == y).all(dim=1).sum().item()

            total_correct += correct
            total_items += y.numel()
            total_perfect += perfect

    return {
        "accuracy": total_correct / total_items,
        "perfect_sort_rate": total_perfect / len(data),
    }


def main():
    parser = argparse.ArgumentParser(description="Array Sorting A/B Test")
    parser.add_argument("--array_len", type=int, default=10, help="Array length")
    parser.add_argument("--train_samples", type=int, default=5000, help="Training samples")
    parser.add_argument("--test_samples", type=int, default=500, help="Test samples")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--max_inner_iters", type=int, default=3, help="PoH iterations")
    parser.add_argument(
        "--use_hrm",
        action="store_true",
        help="Use HRM-style last-iterate gradients (backprop only through last iteration)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Array length: {args.array_len}")
    print(f"Training samples: {args.train_samples}")
    print("=" * 80)

    # Generate data
    print("\nGenerating data...")
    train_data = generate_sort_data(args.train_samples, args.array_len)
    test_data = generate_sort_data(args.test_samples, args.array_len)

    # Initialize models
    print("\nInitializing models...")
    print(f"HRM-style gradients: {'Yes' if args.use_hrm else 'No'}")
    poh_model = PointerSortModel(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_model * 2,
        max_inner_iters=args.max_inner_iters,
        use_hrm_style=args.use_hrm,
    ).to(device)

    baseline_model = BaselineSortModel(
        d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_model * 2
    ).to(device)

    poh_params = sum(p.numel() for p in poh_model.parameters())
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    print(f"PoH parameters: {poh_params:,}")
    print(f"Baseline parameters: {baseline_params:,}")

    # Optimizers
    poh_optimizer = torch.optim.Adam(poh_model.parameters(), lr=args.lr)
    baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=args.lr)

    # Training
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    best_poh_acc = 0.0
    best_baseline_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # Train PoH
        poh_metrics = train_epoch(poh_model, train_data, poh_optimizer, device, args.batch_size)

        # Train Baseline
        baseline_metrics = train_epoch(
            baseline_model, train_data, baseline_optimizer, device, args.batch_size
        )

        # Evaluate
        if epoch % 5 == 0 or epoch == args.epochs:
            poh_test = eval_model(poh_model, test_data, device, args.batch_size)
            baseline_test = eval_model(baseline_model, test_data, device, args.batch_size)

            best_poh_acc = max(best_poh_acc, poh_test["accuracy"])
            best_baseline_acc = max(best_baseline_acc, baseline_test["accuracy"])

            print(f"\nEpoch {epoch}/{args.epochs}")
            print(
                f"  PoH:      Train Acc: {poh_metrics['accuracy']:.3f}, "
                f"Test Acc: {poh_test['accuracy']:.3f}, "
                f"Perfect Sort: {poh_test['perfect_sort_rate']:.3f}"
            )
            print(
                f"  Baseline: Train Acc: {baseline_metrics['accuracy']:.3f}, "
                f"Test Acc: {baseline_test['accuracy']:.3f}, "
                f"Perfect Sort: {baseline_test['perfect_sort_rate']:.3f}"
            )

    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    final_poh = eval_model(poh_model, test_data, device, args.batch_size)
    final_baseline = eval_model(baseline_model, test_data, device, args.batch_size)

    print(f"\nPoH Model:")
    print(f"  Test Accuracy:      {final_poh['accuracy']:.4f}")
    print(f"  Perfect Sort Rate:  {final_poh['perfect_sort_rate']:.4f}")
    print(f"  Best Accuracy:      {best_poh_acc:.4f}")

    print(f"\nBaseline Model:")
    print(f"  Test Accuracy:      {final_baseline['accuracy']:.4f}")
    print(f"  Perfect Sort Rate:  {final_baseline['perfect_sort_rate']:.4f}")
    print(f"  Best Accuracy:      {best_baseline_acc:.4f}")

    improvement = (final_poh["accuracy"] - final_baseline["accuracy"]) * 100
    print(
        f"\n{'🎉' if improvement > 0 else '📊'} PoH vs Baseline: "
        f"{improvement:+.2f}% accuracy difference"
    )

    if final_poh["perfect_sort_rate"] > final_baseline["perfect_sort_rate"]:
        print(
            f"✅ PoH achieves {final_poh['perfect_sort_rate']:.1%} perfect sorts "
            f"vs {final_baseline['perfect_sort_rate']:.1%} for baseline"
        )


if __name__ == "__main__":
    main()
