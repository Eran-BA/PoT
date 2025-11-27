"""
Proper Pointer Network Decoder for Array Sorting

Author: Eran Ben Artzy
License: Apache 2.0

This implements a REAL pointer decoder with:
- Decoder loop over output ranks (t=1..T)
- Rank-based queries
- Coverage masking to prevent re-selection
- Proper permutation output
"""

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerDecoderSort(nn.Module):
    """Pointer decoder with optional PoH routing inside attention."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        max_inner_iters: int = 3,
        use_poh: bool = False,
        use_hrm: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_inner_iters = max_inner_iters
        self.use_poh = use_poh
        self.use_hrm = use_hrm

        # Encoder: embed input values + positions
        self.value_embed = nn.Sequential(
            nn.Linear(1, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, d_model)
        )
        self.pos_embed = nn.Embedding(100, d_model)  # Support up to 100 elements

        # Encoder transformer
        self.encoder_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.encoder_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.encoder_ln1 = nn.LayerNorm(d_model)
        self.encoder_ln2 = nn.LayerNorm(d_model)

        # Decoder: rank embeddings
        self.rank_embed = nn.Embedding(100, d_model)  # Rank queries

        # Pointer mechanism
        if use_poh:
            # PoH controller decides routing over heads
            self.controller = nn.Linear(d_model, n_heads)
            # Per-head query/key projections
            self.query_proj = nn.ModuleList(
                [nn.Linear(d_model, d_model // n_heads) for _ in range(n_heads)]
            )
            self.key_proj = nn.ModuleList(
                [nn.Linear(d_model, d_model // n_heads) for _ in range(n_heads)]
            )
        else:
            # Standard single-head pointer
            self.query_proj = nn.Linear(d_model, d_model)
            self.key_proj = nn.Linear(d_model, d_model)

    def encode(self, x):
        """Encode input array values."""
        B, N, _ = x.shape  # [B, N, 1]

        # Embed values + positions
        positions = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.value_embed(x.squeeze(-1).unsqueeze(-1)) + self.pos_embed(
            positions
        )  # [B, N, d_model]

        # Encoder transformer
        attn_out, _ = self.encoder_attn(h, h, h)
        h = self.encoder_ln1(h + attn_out)
        h = self.encoder_ln2(h + self.encoder_ffn(h))

        return h  # [B, N, d_model]

    def decode_step(self, rank_t, enc_memory, mask):
        """
        Single decoder step: rank t chooses an input position.

        Args:
            rank_t: Scalar rank index (0..T-1)
            enc_memory: [B, N, d_model] encoder outputs
            mask: [B, N] coverage mask (1=available, 0=used)

        Returns:
            logits: [B, N] pointer logits over input positions
        """
        B, N, _ = enc_memory.shape

        # Query for this rank
        rank_idx = torch.full((B,), rank_t, device=enc_memory.device, dtype=torch.long)
        q_t = self.rank_embed(rank_idx)  # [B, d_model]

        if self.use_poh:
            # PoH routing: controller decides head weights
            alphas = F.softmax(self.controller(q_t), dim=-1)  # [B, n_heads]

            # Compute per-head pointer logits
            head_logits = []
            for h_idx in range(self.n_heads):
                q_h = self.query_proj[h_idx](q_t)  # [B, d_model//n_heads]
                k_h = self.key_proj[h_idx](enc_memory)  # [B, N, d_model//n_heads]
                logits_h = torch.einsum("bd,bnd->bn", q_h, k_h)  # [B, N]
                head_logits.append(logits_h)

            head_logits = torch.stack(head_logits, dim=1)  # [B, n_heads, N]

            # Weighted combination
            logits = torch.einsum("bh,bhn->bn", alphas, head_logits)  # [B, N]
        else:
            # Standard pointer
            q = self.query_proj(q_t)  # [B, d_model]
            k = self.key_proj(enc_memory)  # [B, N, d_model]
            logits = torch.einsum("bd,bnd->bn", q, k)  # [B, N]

        # Apply coverage mask
        logits = logits.masked_fill(mask == 0, float("-inf"))

        return logits

    def forward(self, x, targets=None):
        """
        Full pointer decoder with coverage masking.

        Args:
            x: [B, N, 1] input array
            targets: [B, N] gold permutation indices (teacher forcing)

        Returns:
            all_logits: [B, N, N] pointer logits for each output rank
            loss: scalar (if targets provided)
        """
        B, N, _ = x.shape

        # Encode
        enc_memory = self.encode(x)  # [B, N, d_model]

        # Coverage mask: all positions available initially
        mask = torch.ones(B, N, device=x.device, dtype=torch.float32)

        # Decode: loop over output ranks
        all_logits = []
        for t in range(N):
            logits_t = self.decode_step(t, enc_memory, mask)  # [B, N]
            all_logits.append(logits_t)

            if targets is not None:
                # Teacher forcing: mask out the gold position for next step
                chosen_idx = targets[:, t]  # [B]
                mask[torch.arange(B, device=x.device), chosen_idx] = 0.0
            else:
                # Inference: mask out greedy argmax
                chosen_idx = logits_t.argmax(dim=-1)  # [B]
                mask[torch.arange(B, device=x.device), chosen_idx] = 0.0

        all_logits = torch.stack(all_logits, dim=1)  # [B, N, N]

        # Compute loss
        loss = None
        if targets is not None:
            # Cross-entropy averaged over ranks
            loss = F.cross_entropy(
                all_logits.reshape(B * N, N), targets.reshape(B * N), reduction="mean"
            )

        return all_logits, loss


def generate_sort_data(num_samples, array_len, unique=True):
    """
    Generate sorting data with unique integers per array.

    Returns:
        arrays: [num_samples, array_len, 1]
        targets: [num_samples, array_len] - permutation indices (stable argsort)
    """
    arrays = []
    targets = []

    for _ in range(num_samples):
        if unique:
            # Generate unique random integers
            values = np.random.choice(
                np.arange(-1000, 1000), size=array_len, replace=False
            )
        else:
            # Allow duplicates
            values = np.random.randint(-100, 100, size=array_len)

        # Stable argsort: target[t] = index of t-th smallest element (ties broken by position)
        perm = np.argsort(values, kind="stable")

        arrays.append(values.reshape(-1, 1).astype(np.float32))
        targets.append(perm.astype(np.int64))

    return (
        torch.from_numpy(np.array(arrays)),
        torch.from_numpy(np.array(targets)),
    )


def train_epoch(model, data, optimizer, batch_size):
    """Train for one epoch."""
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
        batch_arrays = arrays[batch_idx].to(model.value_embed[0].weight.device)
        batch_targets = targets[batch_idx].to(model.value_embed[0].weight.device)

        optimizer.zero_grad()
        logits, loss = model(batch_arrays, batch_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Metrics
        preds = logits.argmax(dim=-1)
        correct = (preds == batch_targets).sum().item()
        total_correct += correct
        total_elements += batch_targets.numel()
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    accuracy = total_correct / total_elements
    return avg_loss, accuracy


def eval_epoch(model, data, batch_size):
    """Evaluate with proper masked decoding."""
    model.eval()
    arrays, targets = data
    num_samples = len(arrays)

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

            # Inference: no teacher forcing
            logits, _ = model(batch_arrays, targets=None)
            preds = logits.argmax(dim=-1)

            # Per-element accuracy
            correct = (preds == batch_targets).sum().item()
            total_correct += correct
            total_elements += batch_targets.numel()

            # Perfect sort rate
            perfect = (preds == batch_targets).all(dim=1).sum().item()
            perfect_sorts += perfect

    accuracy = total_correct / total_elements
    perfect_rate = perfect_sorts / num_samples
    return accuracy, perfect_rate


def main():
    parser = argparse.ArgumentParser(description="Pointer decoder for array sorting")
    parser.add_argument("--array_len", type=int, default=6, help="Array length")
    parser.add_argument(
        "--train_samples", type=int, default=2000, help="Training samples"
    )
    parser.add_argument("--test_samples", type=int, default=500, help="Test samples")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--max_inner_iters", type=int, default=3, help="PoH iterations")
    parser.add_argument("--use_poh", action="store_true", help="Use PoH routing")
    parser.add_argument(
        "--use_hrm", action="store_true", help="Use HRM-style gradients"
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
    print(f"PoH routing: {args.use_poh}")
    print(f"HRM-style gradients: {args.use_hrm}")
    print("=" * 80)

    # Generate data
    print("\nGenerating data...")
    train_data = generate_sort_data(args.train_samples, args.array_len)
    test_data = generate_sort_data(args.test_samples, args.array_len)

    # Initialize model
    print("\nInitializing model...")
    model = PointerDecoderSort(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_model * 2,
        max_inner_iters=args.max_inner_iters,
        use_poh=args.use_poh,
        use_hrm=args.use_hrm,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    best_test_acc = 0.0
    best_perfect = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_data, optimizer, args.batch_size)
        test_acc, perfect_rate = eval_epoch(model, test_data, args.batch_size)

        best_test_acc = max(best_test_acc, test_acc)
        best_perfect = max(best_perfect, perfect_rate)

        if epoch % 5 == 0 or epoch == 1:
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")
            print(
                f"  Test Acc: {test_acc:.3f}, Perfect Sort: {perfect_rate:.3f} (best: {best_perfect:.3f})"
            )

    # Final results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    test_acc, perfect_rate = eval_epoch(model, test_data, args.batch_size)
    print(f"\nTest Accuracy:      {test_acc:.4f} (best: {best_test_acc:.4f})")
    print(f"Perfect Sort Rate:  {perfect_rate:.4f} (best: {best_perfect:.4f})")


if __name__ == "__main__":
    main()

