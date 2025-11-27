#!/usr/bin/env python3
"""
Visualize routing patterns during array sorting.

Shows how the PoH controller routes attention across heads during iterative refinement.

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerSortModelWithTracking(nn.Module):
    """PoH model with routing tracking for visualization."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        max_inner_iters: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_inner_iters = max_inner_iters

        # Embedding
        self.value_embed = nn.Sequential(
            nn.Linear(1, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, d_model)
        )
        self.pos_embed = nn.Embedding(100, d_model)

        # Controller
        self.controller = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_heads),
        )

        # Attention
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

    def forward(self, x: torch.Tensor, track_routing: bool = False):
        """Forward with optional routing tracking."""
        B, N, _ = x.shape

        # Embed
        positions = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.value_embed(x) + self.pos_embed(positions)

        routing_history = []

        # Iterative refinement
        for iter_idx in range(self.max_inner_iters):
            # Controller routing
            alphas = F.softmax(self.controller(h), dim=-1)  # [B, N, n_heads]

            if track_routing:
                routing_history.append(alphas.detach().cpu().numpy())

            # Attention
            attn_out, _ = self.attn(h, h, h)

            # Residual + norm
            h = self.ln1(h + attn_out)
            h = self.ln2(h + self.ffn(h))

        # Pointer logits
        query = self.query_proj(h)
        key = self.key_proj(h)
        logits = torch.einsum("bnd,de,bme->bnm", query, self.biaffine, key)

        if track_routing:
            return logits, routing_history
        return logits


def visualize_routing(
    model, array, target, routing_history, save_path="routing_visualization.png"
):
    """Visualize routing patterns across iterations."""
    n_iters = len(routing_history)
    n_heads = routing_history[0].shape[-1]
    array_len = len(array)

    fig, axes = plt.subplots(n_iters + 1, n_heads + 1, figsize=(16, 4 * n_iters))

    # Adjust for single iteration case
    if n_iters == 1:
        axes = axes.reshape(1, -1)

    # Plot input array and target
    for iter_idx in range(n_iters):
        ax = axes[iter_idx, 0]
        ax.barh(range(array_len), array, color="skyblue")
        ax.set_ylabel(f"Iteration {iter_idx + 1}")
        ax.set_xlabel("Value")
        ax.set_yticks(range(array_len))
        ax.set_yticklabels([f"Pos {i}" for i in range(array_len)])
        ax.invert_yaxis()
        ax.set_title("Input Array")
        ax.grid(axis="x", alpha=0.3)

        # Plot routing weights for each head
        routing = routing_history[iter_idx][0]  # [N, n_heads]

        for head_idx in range(n_heads):
            ax = axes[iter_idx, head_idx + 1]
            weights = routing[:, head_idx]

            # Heatmap
            im = ax.imshow(
                weights.reshape(-1, 1), cmap="YlOrRd", aspect="auto", vmin=0, vmax=1
            )
            ax.set_yticks(range(array_len))
            ax.set_yticklabels([f"Pos {i}" for i in range(array_len)])
            ax.set_xticks([])
            ax.set_title(f"Head {head_idx + 1}")

            # Add values as text
            for i in range(array_len):
                ax.text(
                    0,
                    i,
                    f"{weights[i]:.2f}",
                    ha="center",
                    va="center",
                    color="black" if weights[i] < 0.5 else "white",
                    fontsize=9,
                )

    # Plot final predictions
    ax = axes[-1, 0]
    sorted_indices = sorted(range(array_len), key=lambda i: array[i])
    sorted_array = [array[i] for i in sorted_indices]
    ax.barh(range(array_len), sorted_array, color="lightgreen")
    ax.set_ylabel("Target")
    ax.set_xlabel("Value")
    ax.set_yticks(range(array_len))
    ax.set_yticklabels([f"Pos {i}" for i in range(array_len)])
    ax.invert_yaxis()
    ax.set_title("Sorted Array")
    ax.grid(axis="x", alpha=0.3)

    # Plot target positions
    for head_idx in range(n_heads):
        ax = axes[-1, head_idx + 1]
        ax.barh(range(array_len), target, color="mediumseagreen")
        ax.set_xlabel("Target Position")
        ax.set_yticks(range(array_len))
        ax.set_yticklabels([f"Pos {i}" for i in range(array_len)])
        ax.invert_yaxis()
        ax.set_title("Target Positions")
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved visualization to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize routing during sorting")
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to trained model"
    )
    parser.add_argument("--array_len", type=int, default=8, help="Array length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")

    # Create model
    model = PointerSortModelWithTracking(
        d_model=256, n_heads=4, d_ff=512, max_inner_iters=4
    ).to(device)

    # Load trained weights if provided
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"✅ Loaded model from {args.model_path}")

    model.eval()

    # Generate sample array
    array = [random.uniform(0, 100) for _ in range(args.array_len)]
    print(f"\nInput array: {[f'{x:.1f}' for x in array]}")

    # Compute target positions
    indexed_arr = list(enumerate(array))
    sorted_arr = sorted(indexed_arr, key=lambda x: x[1])
    target = [0] * args.array_len
    for new_pos, (orig_idx, _) in enumerate(sorted_arr):
        target[orig_idx] = new_pos

    print(f"Target positions: {target}")
    print(f"Sorted array: {[f'{array[i]:.1f}' for i in sorted([j for j, _ in indexed_arr], key=lambda k: array[k])]}")

    # Forward pass with tracking
    x = torch.tensor(array, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

    with torch.no_grad():
        logits, routing_history = model(x, track_routing=True)
        preds = logits.argmax(dim=-1).squeeze().cpu().numpy()

    print(f"Predictions: {preds.tolist()}")
    print(f"Accuracy: {(preds == np.array(target)).sum() / len(target) * 100:.1f}%")

    # Visualize
    visualize_routing(model, array, target, routing_history)


if __name__ == "__main__":
    main()

