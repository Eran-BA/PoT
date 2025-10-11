"""
Proper Pointer Network Decoder for Array Sorting with PoH Integration

Author: Eran Ben Artzy
License: Apache 2.0

Key fixes:
1. No detach between decoder ranks (only inside PoH inner loop if HRM)
2. Refine the latent z (pre-FFN), not post-FFN output
3. Stable argsort for ties
"""

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoHBlock(nn.Module):
    """Single PoH transformer block with iterative refinement."""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, max_iters: int, use_hrm: bool
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_iters = max_iters
        self.use_hrm = use_hrm

        # Controller for routing
        self.controller = nn.Linear(d_model, n_heads)

        # Per-head attention projections
        self.q_proj = nn.ModuleList(
            [nn.Linear(d_model, d_model // n_heads) for _ in range(n_heads)]
        )
        self.k_proj = nn.ModuleList(
            [nn.Linear(d_model, d_model // n_heads) for _ in range(n_heads)]
        )
        self.v_proj = nn.ModuleList(
            [nn.Linear(d_model, d_model // n_heads) for _ in range(n_heads)]
        )
        self.out_proj = nn.Linear(d_model, d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, z):
        """
        Iteratively refine latent z.
        Returns the refined latent (pre-FFN).
        """
        for iter_idx in range(self.max_iters):
            # HRM-style: detach for all but last iteration
            if self.use_hrm and iter_idx < self.max_iters - 1:
                z = z.detach()

            # Controller routing
            alphas = F.softmax(self.controller(z), dim=-1)  # [B, T, n_heads]

            # Per-head attention
            head_outs = []
            for h_idx in range(self.n_heads):
                q = self.q_proj[h_idx](z)  # [B, T, d_model//n_heads]
                k = self.k_proj[h_idx](z)
                v = self.v_proj[h_idx](z)

                scores = torch.einsum("btd,bsd->bts", q, k) / (
                    (self.d_model // self.n_heads) ** 0.5
                )
                attn = F.softmax(scores, dim=-1)
                out = torch.einsum("bts,bsd->btd", attn, v)
                head_outs.append(out)

            # Stack and concat all heads
            head_outs = torch.cat(head_outs, dim=-1)  # [B, T, d_model]
            attn_out = self.out_proj(head_outs)

            # Residual + norm
            z = self.ln1(z + attn_out)
            z_refined = z  # Store latent before FFN

            # FFN (only applied at the end, not used for next iteration's input)
            z = self.ln2(z + self.ffn(z))

        return z_refined  # Return pre-FFN latent


class PointerDecoderSort(nn.Module):
    """Pointer decoder with optional PoH routing."""

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
        self.use_poh = use_poh

        # Encoder: embed input values + positions
        self.value_embed = nn.Sequential(
            nn.Linear(1, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, d_model)
        )
        self.pos_embed = nn.Embedding(100, d_model)

        # Encoder transformer
        self.encoder_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.encoder_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.encoder_ln1 = nn.LayerNorm(d_model)
        self.encoder_ln2 = nn.LayerNorm(d_model)

        # Decoder: rank embeddings
        self.rank_embed = nn.Embedding(100, d_model)

        # PoH block for iterative refinement (if enabled)
        if use_poh:
            self.poh_block = PoHBlock(d_model, n_heads, d_ff, max_inner_iters, use_hrm)

        # Pointer mechanism
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)

    def encode(self, x):
        """Encode input array values."""
        B, N, _ = x.shape

        # Embed values + positions
        positions = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.value_embed(x.squeeze(-1).unsqueeze(-1)) + self.pos_embed(positions)

        # Encoder transformer
        attn_out, _ = self.encoder_attn(h, h, h)
        h = self.encoder_ln1(h + attn_out)
        h = self.encoder_ln2(h + self.encoder_ffn(h))

        return h

    def decode_step(self, rank_t, z, mask):
        """
        Single decoder step: rank t chooses an input position.
        
        Args:
            rank_t: Scalar rank index
            z: [B, T, d_model] current latent (refined by PoH)
            mask: [B, T] coverage mask
        """
        B, T, _ = z.shape

        # Query for this rank
        rank_idx = torch.full((B,), rank_t, device=z.device, dtype=torch.long)
        q_t = self.rank_embed(rank_idx)  # [B, d_model]

        # Pointer attention over refined latent
        q = self.query_proj(q_t)  # [B, d_model]
        k = self.key_proj(z)  # [B, T, d_model]
        logits = torch.einsum("bd,btd->bt", q, k)  # [B, T]

        # Apply coverage mask
        logits = logits.masked_fill(mask == 0, float("-inf"))

        return logits

    def forward(self, x, targets=None):
        """
        Full pointer decoder with coverage masking.
        
        Args:
            x: [B, N, 1] input array
            targets: [B, N] gold permutation indices
        """
        B, N, _ = x.shape

        # Encode
        z = self.encode(x)  # [B, N, d_model]

        # Optionally refine with PoH (inner iterations happen HERE)
        if self.use_poh:
            z = self.poh_block(z)  # Refine latent, returns pre-FFN z

        # Coverage mask: all positions available initially
        mask = torch.ones(B, N, device=x.device, dtype=torch.float32)

        # Decode: loop over output ranks (NO detach between ranks)
        all_logits = []
        for t in range(N):
            logits_t = self.decode_step(t, z, mask)  # [B, N]
            all_logits.append(logits_t)

            if targets is not None:
                # Teacher forcing
                chosen_idx = targets[:, t]
                mask[torch.arange(B, device=x.device), chosen_idx] = 0.0
            else:
                # Greedy inference
                chosen_idx = logits_t.argmax(dim=-1)
                mask[torch.arange(B, device=x.device), chosen_idx] = 0.0

        all_logits = torch.stack(all_logits, dim=1)  # [B, N, N]

        # Compute loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                all_logits.reshape(B * N, N),
                targets.reshape(B * N),
                reduction="mean",
            )

        return all_logits, loss


def generate_sort_data(num_samples, array_len, unique=True):
    """Generate sorting data with stable argsort."""
    arrays = []
    targets = []

    for _ in range(num_samples):
        if unique:
            # Generate unique integers
            values = np.random.choice(
                np.arange(-1000, 1000), size=array_len, replace=False
            )
        else:
            values = np.random.randint(-100, 100, size=array_len)

        # Stable argsort
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

            logits, _ = model(batch_arrays, targets=None)
            preds = logits.argmax(dim=-1)

            correct = (preds == batch_targets).sum().item()
            total_correct += correct
            total_elements += batch_targets.numel()

            perfect = (preds == batch_targets).all(dim=1).sum().item()
            perfect_sorts += perfect

    accuracy = total_correct / total_elements
    perfect_rate = perfect_sorts / num_samples
    return accuracy, perfect_rate


def main():
    parser = argparse.ArgumentParser(
        description="Fixed pointer decoder with proper PoH integration"
    )
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
        "--use_hrm", action="store_true", help="Use HRM-style gradients (inside PoH)"
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
    if args.use_poh:
        print(f"  Max inner iterations: {args.max_inner_iters}")
        print(f"  HRM-style (inside PoH): {args.use_hrm}")
    print("=" * 80)

    # Generate data
    print("\nGenerating data (unique integers, stable argsort)...")
    train_data = generate_sort_data(args.train_samples, args.array_len, unique=True)
    test_data = generate_sort_data(args.test_samples, args.array_len, unique=True)

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
        train_loss, train_acc = train_epoch(
            model, train_data, optimizer, args.batch_size
        )
        test_acc, perfect_rate = eval_epoch(model, test_data, args.batch_size)

        best_test_acc = max(best_test_acc, test_acc)
        best_perfect = max(best_perfect, perfect_rate)

        if epoch % 5 == 0 or epoch == 1:
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")
            print(
                f"  Test Acc: {test_acc:.3f}, Perfect: {perfect_rate:.3f} (best: {best_perfect:.3f})"
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

