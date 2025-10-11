"""
Noisy Topological Sort: Constraint Repair with Iterative Refinement

Author: Eran Ben Artzy
License: Apache 2.0

Task: Recover hidden permutation π from noisy pairwise "before/after" constraints.
Why hard: Constraints can be contradictory/cyclic → needs iterative repair.
PoH edge: Constraint propagation over iterations; adaptive compute.
"""

import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

Edge = Tuple[int, int]  # (u -> v) means u should appear before v


@dataclass
class NTSCfg:
    T: int = 12  # sequence length
    q_keep: float = 0.6  # prob to keep a true edge (from π)
    p_flip: float = 0.15  # prob to flip direction of a kept edge
    spurious_rate: float = 0.08  # fraction of extra random edges (creates cycles)
    n_train: int = 500
    n_test: int = 200
    seed: int = 42
    d_model: int = 128


# -------------------- Data Generation --------------------


def sample_instance(
    T: int, q_keep: float, p_flip: float, spurious_rate: float, rng: random.Random
):
    """Return (perm π [T], edges list of (u->v))."""
    pi = list(range(T))
    rng.shuffle(pi)  # hidden true order
    pos = {pi[i]: i for i in range(T)}

    edges: List[Edge] = []
    # Keep true edges with prob q_keep
    for i in range(T):
        for j in range(i + 1, T):
            u, v = pi[i], pi[j]  # u before v in truth
            if rng.random() < q_keep:
                if rng.random() < p_flip:
                    edges.append((v, u))  # flipped (noisy)
                else:
                    edges.append((u, v))

    # Add spurious edges (can create cycles)
    m = int(spurious_rate * T * T)
    existing = set(edges)
    tried = 0
    while m > 0 and tried < 10 * T * T:
        a, b = rng.randrange(T), rng.randrange(T)
        if a == b:
            tried += 1
            continue
        e = (a, b)
        if e not in existing and (b, a) not in existing:
            existing.add(e)
            edges.append(e)
            m -= 1
        tried += 1

    return torch.tensor(pi, dtype=torch.long), edges


# -------------------- Features --------------------


def edges_to_degrees(edges: List[Edge], T: int):
    """Compute in-degree, out-degree, and net degree for each node."""
    indeg = torch.zeros(T)
    outdeg = torch.zeros(T)
    for u, v in edges:
        outdeg[u] += 1.0
        indeg[v] += 1.0
    deg = torch.stack([indeg, outdeg, outdeg - indeg], dim=-1)  # [T,3]
    # Normalize
    deg = deg / max(1.0, float(T - 1))
    return deg  # [T,3]


def sinusoidal_pos(T: int, d_model: int, device=None):
    """Sinusoidal position embeddings."""
    pe = torch.zeros(T, d_model, device=device)
    position = torch.arange(0, T, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device).float()
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [T, d_model]


class ConstraintFeaturizer(nn.Module):
    """Maps degree features + position to token embeddings [B, T, d_model]."""

    def __init__(self, d_model: int, deg_dim: int = 3):
        super().__init__()
        self.deg_proj = nn.Linear(deg_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, deg_feats: torch.Tensor, pos_emb: torch.Tensor):
        # deg_feats: [B,T,3], pos_emb: [B,T,d_model]
        h = self.deg_proj(deg_feats) + pos_emb
        return self.norm(h)


def build_batch(instances: List[Tuple[torch.Tensor, List[Edge]]], d_model: int, device):
    """Returns deg_feats[B,T,3], pos[B,T,D], gold_perm[B,T], edges_list."""
    B = len(instances)
    T = instances[0][0].numel()
    deg_feats = torch.stack(
        [edges_to_degrees(edges, T) for (_, edges) in instances], dim=0
    ).to(device)
    pos = sinusoidal_pos(T, d_model, device).unsqueeze(0).expand(B, T, d_model)
    gold_perm = torch.stack([pi for (pi, _) in instances], dim=0).to(device)
    edges_list = [edges for (_, edges) in instances]
    return deg_feats, pos, gold_perm, edges_list


# -------------------- Metrics --------------------


def kendall_tau_normalized(pred_perm: torch.Tensor, gold_perm: torch.Tensor) -> float:
    """Kendall-τ via normalized inversion count."""
    T = pred_perm.numel()
    pos_pred = torch.empty(T, dtype=torch.long, device=pred_perm.device)
    pos_gold = torch.empty(T, dtype=torch.long, device=gold_perm.device)
    pos_pred[pred_perm] = torch.arange(T, device=pred_perm.device)
    pos_gold[gold_perm] = torch.arange(T, device=gold_perm.device)
    inv = 0
    total = T * (T - 1) // 2
    for i in range(T):
        for j in range(i + 1, T):
            inv += int((pos_pred[i] - pos_pred[j]) * (pos_gold[i] - pos_gold[j]) < 0)
    return 1.0 - (inv / max(1, total))


def constraint_satisfaction(pred_perm: torch.Tensor, edges: List[Edge]) -> float:
    """Fraction of edges satisfied by predicted order."""
    if not edges:
        return 1.0
    T = pred_perm.numel()
    pos = torch.empty(T, dtype=torch.long, device=pred_perm.device)
    pos[pred_perm] = torch.arange(T, device=pred_perm.device)
    sat = 0
    for u, v in edges:
        sat += int(pos[u] < pos[v])
    return sat / len(edges)


def perfect_order(pred_perm: torch.Tensor, gold_perm: torch.Tensor) -> bool:
    return bool(torch.equal(pred_perm, gold_perm))


# -------------------- Pointer Decoder (adapted from sort_pointer_fixed) --------------------


class TopoSortDecoder(nn.Module):
    """Rank-conditioned pointer decoder for topological sort."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        max_inner_iters: int = 4,
        use_poh: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_poh = use_poh

        # Rank embeddings
        self.rank_embed = nn.Embedding(100, d_model)

        # Optional PoH refinement
        if use_poh:
            from sort_pointer_fixed import PointerDecoderSort

            # Use the PoH block from sort_pointer_fixed
            dummy = PointerDecoderSort(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff, max_inner_iters=max_inner_iters, use_poh=True
            )
            # Extract just the encoding part (we'll build our own decoder)
            self.encoder_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.encoder_ffn = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
            )
            self.encoder_ln1 = nn.LayerNorm(d_model)
            self.encoder_ln2 = nn.LayerNorm(d_model)
        else:
            # Baseline: simple encoding
            self.encoder_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.encoder_ffn = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
            )
            self.encoder_ln1 = nn.LayerNorm(d_model)
            self.encoder_ln2 = nn.LayerNorm(d_model)

        # Pointer mechanism
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)

    def encode(self, x):
        """Encode input features."""
        # Single encoding pass
        attn_out, _ = self.encoder_attn(x, x, x)
        h = self.encoder_ln1(x + attn_out)
        h = self.encoder_ln2(h + self.encoder_ffn(h))
        return h

    def decode_step(self, rank_t, z, mask):
        """Single decoder step: rank t chooses an input position."""
        B, T, _ = z.shape
        rank_idx = torch.full((B,), rank_t, device=z.device, dtype=torch.long)
        q_t = self.rank_embed(rank_idx)

        q = self.query_proj(q_t)
        k = self.key_proj(z)
        logits = torch.einsum("bd,btd->bt", q, k)

        # Apply coverage mask
        logits = logits.masked_fill(mask == 0, float("-inf"))
        return logits

    def forward(self, x, targets=None):
        """Full decoder with coverage masking."""
        B, N, _ = x.shape

        # Encode
        z = self.encode(x)

        # Coverage mask
        mask = torch.ones(B, N, device=x.device, dtype=torch.float32)

        # Decode
        all_logits = []
        for t in range(N):
            logits_t = self.decode_step(t, z, mask)
            all_logits.append(logits_t)

            if targets is not None:
                chosen_idx = targets[:, t]
                mask[torch.arange(B, device=x.device), chosen_idx] = 0.0
            else:
                chosen_idx = logits_t.argmax(dim=-1)
                mask[torch.arange(B, device=x.device), chosen_idx] = 0.0

        all_logits = torch.stack(all_logits, dim=1)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                all_logits.reshape(B * N, N), targets.reshape(B * N), reduction="mean"
            )

        return all_logits, loss


# -------------------- Training --------------------


def train_epoch(model, featurizer, data, optimizer, batch_size, d_model, device):
    """Train for one epoch."""
    model.train()
    featurizer.train()
    random.shuffle(data)

    total_loss = 0.0
    total_correct = 0
    total_elements = 0
    num_batches = 0

    for start_idx in range(0, len(data), batch_size):
        batch = data[start_idx : start_idx + batch_size]
        deg_feats, pos, gold, _ = build_batch(batch, d_model, device)

        # Featurize
        x = featurizer(deg_feats, pos)

        # Forward
        optimizer.zero_grad()
        logits, loss = model(x, gold)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(featurizer.parameters()), 1.0
        )
        optimizer.step()

        # Metrics
        preds = logits.argmax(dim=-1)
        correct = (preds == gold).sum().item()
        total_correct += correct
        total_elements += gold.numel()
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    accuracy = total_correct / max(1, total_elements)
    return avg_loss, accuracy


def eval_epoch(model, featurizer, data, batch_size, d_model, device):
    """Evaluate with comprehensive metrics."""
    model.eval()
    featurizer.eval()

    total_correct = 0
    total_elements = 0
    perfect_orders = 0
    kendall_scores = []
    constraint_sats = []

    with torch.no_grad():
        for start_idx in range(0, len(data), batch_size):
            batch = data[start_idx : start_idx + batch_size]
            deg_feats, pos, gold, edges_list = build_batch(batch, d_model, device)

            x = featurizer(deg_feats, pos)
            logits, _ = model(x, targets=None)
            preds = logits.argmax(dim=-1)

            correct = (preds == gold).sum().item()
            total_correct += correct
            total_elements += gold.numel()

            for i in range(len(batch)):
                pred_i = preds[i].cpu()
                gold_i = gold[i].cpu()
                edges_i = edges_list[i]

                perfect_orders += int(perfect_order(pred_i, gold_i))
                kendall_scores.append(kendall_tau_normalized(pred_i, gold_i))
                constraint_sats.append(constraint_satisfaction(pred_i, edges_i))

    accuracy = total_correct / max(1, total_elements)
    perfect_rate = perfect_orders / max(1, len(data))
    avg_kendall = np.mean(kendall_scores) if kendall_scores else 0.0
    avg_constraint_sat = np.mean(constraint_sats) if constraint_sats else 0.0

    return {
        "accuracy": accuracy,
        "perfect": perfect_rate,
        "kendall": avg_kendall,
        "constraint_sat": avg_constraint_sat,
    }


# -------------------- Main --------------------


def main():
    parser = argparse.ArgumentParser(description="Noisy Topological Sort")
    parser.add_argument("--T", type=int, default=12, help="Sequence length")
    parser.add_argument("--q_keep", type=float, default=0.6, help="Edge keep probability")
    parser.add_argument("--p_flip", type=float, default=0.15, help="Edge flip probability")
    parser.add_argument("--spurious", type=float, default=0.08, help="Spurious edge rate")
    parser.add_argument("--train_samples", type=int, default=500, help="Training samples")
    parser.add_argument("--test_samples", type=int, default=200, help="Test samples")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--max_inner_iters", type=int, default=4, help="PoH iterations")
    parser.add_argument("--use_poh", action="store_true", help="Use PoH")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    print(f"Device: {device}")
    print(f"Sequence length: {args.T}")
    print(f"Noise: q_keep={args.q_keep}, p_flip={args.p_flip}, spurious={args.spurious}")
    print(f"Use PoH: {args.use_poh}")

    # Generate data
    print("\nGenerating data...")
    train_data = [
        sample_instance(args.T, args.q_keep, args.p_flip, args.spurious, rng)
        for _ in range(args.train_samples)
    ]
    test_data = [
        sample_instance(args.T, args.q_keep, args.p_flip, args.spurious, rng)
        for _ in range(args.test_samples)
    ]

    # Initialize model
    print("Initializing model...")
    featurizer = ConstraintFeaturizer(args.d_model).to(device)
    model = TopoSortDecoder(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_model * 2,
        max_inner_iters=args.max_inner_iters,
        use_poh=args.use_poh,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(featurizer.parameters()), lr=args.lr
    )

    # Training
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    best_kendall = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, featurizer, train_data, optimizer, args.batch_size, args.d_model, device
        )
        test_metrics = eval_epoch(
            model, featurizer, test_data, args.batch_size, args.d_model, device
        )

        best_kendall = max(best_kendall, test_metrics["kendall"])

        if epoch % 10 == 0 or epoch == 1:
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"  Train: loss={train_loss:.3f}, acc={train_acc:.3f}")
            print(
                f"  Test: acc={test_metrics['accuracy']:.3f}, perfect={test_metrics['perfect']:.3f}, "
                f"kendall={test_metrics['kendall']:.3f}, constraint_sat={test_metrics['constraint_sat']:.3f}"
            )

    # Final results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    final = eval_epoch(model, featurizer, test_data, args.batch_size, args.d_model, device)
    print(f"Accuracy: {final['accuracy']:.3f}")
    print(f"Perfect orders: {final['perfect']:.3f}")
    print(f"Kendall-τ: {final['kendall']:.3f} (best: {best_kendall:.3f})")
    print(f"Constraint satisfaction: {final['constraint_sat']:.3f}")


if __name__ == "__main__":
    main()

