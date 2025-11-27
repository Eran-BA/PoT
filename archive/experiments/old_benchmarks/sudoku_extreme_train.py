#!/usr/bin/env python3
"""
Sudoku-Extreme training (TRM-style options, PoH/TRM-latent path, O(1) grads default).

Mirrors TRM recipes:
  - MLP-t encoder (no positional encodings)
  - Attention encoder (post-norm default)
  - PoH + TRM-style latent recursion (constant input reinjection)

Dataset: data/sudoku-extreme-1k-aug-1000 (from vendor TinyRecursiveModels builder)
Task: 9x9 -> 9x9 grid-to-grid (seq_len=81), vocab_size=11 (0=PAD, 1..10 tokens)
"""

from __future__ import annotations

import os
import json
import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.pot.data.sudoku_extreme import SudokuExtremeDataset
from src.pot.models.hrm_layers import RMSNorm, SwiGLU
from src.pot.core.hrm_controller import HRMPointerController
from src.pot.models.puzzle_embedding import PuzzleEmbedding


def select_device() -> torch.device:
    force = os.getenv("FORCE_DEVICE")
    if force in {"cpu", "cuda", "mps"}:
        return torch.device(force)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MLPTokenEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.ln = nn.LayerNorm(d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h = self.mlp(h) + h
        return self.ln(h)


class AttnEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, depth: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                "attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                "ffn": SwiGLU(d_model, d_ff, dropout),
                "n1": RMSNorm(d_model),
                "n2": RMSNorm(d_model),
                "drop": nn.Dropout(dropout),
            }))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            attn_out, _ = layer["attn"](x, x, x, need_weights=False)
            x = layer["n1"](x + layer["drop"](attn_out))
            f = layer["ffn"](x)
            x = layer["n2"](x + layer["drop"](f))
        return x


class PoHTRMEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, depth: int, T: int, latent_len: int, latent_k: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.latent_len = latent_len
        self.latent_k = latent_k
        self.pos = nn.Parameter(torch.randn(1, 81 + latent_len, d_model) * 0.02)
        self.hrm = HRMPointerController(d_model=d_model, n_heads=n_heads, T=T, dropout=dropout)
        self.layers = AttnEncoder(d_model, n_heads, d_ff, depth, dropout)
        self.lat_q = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.lat_ff = SwiGLU(d_model, d_ff, dropout)
        self.lat_n1 = RMSNorm(d_model)
        self.lat_n2 = RMSNorm(d_model)
        self.lat_drop = nn.Dropout(dropout)
        self.latent_init = nn.Parameter(torch.randn(1, latent_len, d_model) * 0.02)
        self.pre = nn.LayerNorm(d_model)
    def forward_once(self, x: torch.Tensor, hrm_state=None) -> Tuple[torch.Tensor, object]:
        # One refinement iteration with HRM routing across stacked layers
        route, hrm_state, _ = self.hrm(x, state=hrm_state)
        h = x
        for layer in self.layers.layers:
            attn: nn.MultiheadAttention = layer["attn"]
            attn_out, _ = attn(h, h, h, need_weights=False)
            B, T, D = h.shape
            d_head = D // attn.num_heads
            attn_heads = attn_out.view(B, T, attn.num_heads, d_head)
            rw = route.unsqueeze(1).unsqueeze(-1)
            routed = (attn_heads * rw).view(B, T, D)
            h = layer["n1"](h + layer["drop"](routed))
            f = layer["ffn"](h)
            h = layer["n2"](h + layer["drop"](f))
        return h, hrm_state
    def forward(self, x_grid_ref: torch.Tensor, steps: int, o1_grads: bool = True):
        B = x_grid_ref.size(0)
        latent = self.latent_init.expand(B, -1, -1)
        hrm_state = self.hrm.init_state(B, x_grid_ref.device)
        x_out = None
        lat = latent
        for _ in range(steps):
            for _k in range(self.latent_k):
                ctx = x_grid_ref
                lat_attn, _ = self.lat_q(lat, ctx, ctx, need_weights=False)
                lat = self.lat_n1(lat + self.lat_drop(lat_attn))
                lat_ff = self.lat_ff(lat)
                lat = self.lat_n2(lat + self.lat_drop(lat_ff))
            x_step = torch.cat([lat, x_grid_ref], dim=1)
            x_step = self.pre(x_step + self.pos[:, :x_step.size(1), :])
            x_out, hrm_state = self.forward_once(x_step, hrm_state)
            if o1_grads:
                lat = lat.detach(); x_out = x_out.detach()
                if hasattr(hrm_state, "z_L") and torch.is_tensor(hrm_state.z_L): hrm_state.z_L = hrm_state.z_L.detach()
                if hasattr(hrm_state, "z_H") and torch.is_tensor(hrm_state.z_H): hrm_state.z_H = hrm_state.z_H.detach()
        return x_out


class Solver(nn.Module):
    def __init__(self, arch: str, vocab_size: int, d_model: int, n_heads: int, d_ff: int, depth: int, T: int, latent_len: int, latent_k: int, mlp_t: bool, pos_none: bool):
        super().__init__()
        self.vocab_size = vocab_size
        if arch == "mlp_t":
            self.tok = MLPTokenEncoder(vocab_size, d_model)
            self.pos = None if pos_none else nn.Parameter(torch.randn(1, 81, d_model) * 0.02)
            self.enc = nn.Identity()
            self.proj = nn.Linear(d_model, vocab_size)
        elif arch == "attn":
            self.tok = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.pos = None if pos_none else nn.Parameter(torch.randn(1, 81, d_model) * 0.02)
            self.enc = AttnEncoder(d_model, n_heads, d_ff, depth)
            self.proj = nn.Linear(d_model, vocab_size)
        else:  # poh_trm
            self.tok = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.pos = None  # handled in PoHTRMEncoder
            self.enc = PoHTRMEncoder(d_model, n_heads, d_ff, depth, T=T, latent_len=latent_len, latent_k=latent_k)
            self.proj = nn.Linear(d_model, vocab_size)
        # Optional puzzle embeddings (disabled by default; reserved hook)
        self.puzzle_emb = PuzzleEmbedding(num_puzzles=1, d_model=d_model, init_std=0.02)
        self.pre = nn.LayerNorm(d_model)
        self.arch = arch
    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        if self.arch == "poh_trm":
            x_grid = self.tok(x_ids)
            x_grid = self.pre(x_grid)
            h = self.enc(x_grid, steps=4, o1_grads=True)  # modest default steps
        else:
            h = self.tok(x_ids)
            if self.pos is not None:
                h = h + self.pos[:, :h.size(1), :]
            h = self.pre(h)
            h = self.enc(h)
        return self.proj(h)


def train_epoch(model: nn.Module, loader: DataLoader, device: torch.device, lr: float) -> float:
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr)
    tot = 0.0; n = 0
    for batch in loader:
        x = batch["input"].to(device); y = batch["label"].to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1), ignore_index=0)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tot += float(loss.item()); n += 1
    return tot / max(1, n)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval(); tot = 0.0; correct = 0; total = 0
    for batch in loader:
        x = batch["input"].to(device); y = batch["label"].to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1), ignore_index=0)
        tot += float(loss.item())
        preds = logits.argmax(dim=-1)
        mask = (y != 0)
        correct += ((preds == y) & mask).sum().item(); total += mask.sum().item()
    return tot / max(1, len(loader)), 100.0 * correct / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=str, required=True)
    ap.add_argument('--output', type=str, default='experiments/results/sudoku_extreme')
    ap.add_argument('--arch', type=str, choices=['mlp_t', 'attn', 'poh_trm'], default='mlp_t')
    ap.add_argument('--d-model', type=int, default=256)
    ap.add_argument('--n-heads', type=int, default=8)
    ap.add_argument('--n-layers', type=int, default=4)
    ap.add_argument('--d-ff', type=int, default=1024)
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--eval-interval', type=int, default=5000)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--puzzle-emb-lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=1.0)
    ap.add_argument('--puzzle-emb-weight-decay', type=float, default=1.0)
    ap.add_argument('--mlp-t', action='store_true')
    ap.add_argument('--pos-enc', type=str, choices=['none', 'learned'], default='none')
    ap.add_argument('--L-layers', type=int, default=2)
    ap.add_argument('--H-cycles', type=int, default=3)
    ap.add_argument('--L-cycles', type=int, default=6)
    ap.add_argument('--latent-len', type=int, default=16)
    ap.add_argument('--latent-k', type=int, default=3)
    ap.add_argument('--T', type=int, default=4)
    args = ap.parse_args()

    device = select_device()
    print("Device:", device)

    train_ds = SudokuExtremeDataset(args.data_dir, 'train')
    test_ds = SudokuExtremeDataset(args.data_dir, 'test')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    vocab_size = int(train_ds.meta.get('vocab_size', 11))
    pos_none = (args.pos_enc == 'none')
    arch = args.arch
    if args.mlp_t:
        arch = 'mlp_t'

    model = Solver(
        arch=arch,
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        depth=args.n_layers,
        T=args.T,
        latent_len=args.latent_len,
        latent_k=args.latent_k,
        mlp_t=args.mlp_t,
        pos_none=pos_none,
    ).to(device)

    os.makedirs(args.output, exist_ok=True)

    best_acc = 0.0
    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, device, lr=args.lr)
        if ep % max(1, args.eval_interval // max(1, len(train_loader))) == 0 or ep == args.epochs:
            te_loss, te_acc = evaluate(model, test_loader, device)
            print(f"Epoch {ep}/{args.epochs} | train_loss={tr_loss:.4f} | test_loss={te_loss:.4f} | test_acc={te_acc:.2f}%")
            if te_acc > best_acc:
                best_acc = te_acc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': ep,
                    'test_acc': te_acc,
                }, os.path.join(args.output, f"{arch}_best.pt"))
                with open(os.path.join(args.output, f"{arch}_best.json"), 'w') as f:
                    json.dump({'epoch': ep, 'test_acc': te_acc}, f, indent=2)

    print(f"Best test acc: {best_acc:.2f}%")


if __name__ == '__main__':
    main()






