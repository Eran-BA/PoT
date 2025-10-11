#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Pointer-over-Heads Transformer (PoH-Transformer)
# A Transformer block that routes over attention heads using a pointer-style controller.
# Features:
#   - Static (feed-forward) or recurrent (feedback) controller
#   - Soft routing or hard top-k routing via Gumbel-Softmax (straight-through)
#   - Adaptive inner-loop halting: 'fixed' | 'entropy' | 'halting' (ACT-style)
#   - Combination modes: 'mask_concat' (vanilla-like) or 'mixture'
#   - Clean aux telemetry for analysis and logging
#
# Author: Eran (concept), implementation compiled by ChatGPT

import math
import argparse
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Utilities
# -------------------------
def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Mean categorical entropy from logits. Shape: [..., H] -> scalar."""
    p = F.softmax(logits, dim=-1)
    lp = F.log_softmax(logits, dim=-1)
    ent = -(p * lp).sum(dim=-1)
    return ent.mean()


def gumbel_softmax_topk(
    logits: torch.Tensor,
    tau: float = 1.0,
    topk: int = 1,
    hard: bool = True
) -> torch.Tensor:
    """
    Top-k straight-through Gumbel-Softmax. Returns [*, H] k-hot (convex-normalized).
    """
    g = -torch.empty_like(logits).exponential_().log()  # ~ Gumbel(0,1)
    y = (logits + g) / max(tau, 1e-6)
    y_soft = F.softmax(y, dim=-1)
    if not hard:
        return y_soft
    k = min(topk, y_soft.size(-1))
    vals, idx = torch.topk(y_soft, k=k, dim=-1)
    mask = torch.zeros_like(y_soft)
    mask.scatter_(-1, idx, 1.0)
    # Straight-through estimator
    y_hard = mask + (y_soft - y_soft.detach())
    # Normalize k-hot to sum to 1 (convex comb)
    y_hard = y_hard / (y_hard.sum(dim=-1, keepdim=True) + 1e-8)
    return y_hard


# -------------------------
# Core MHA (per-head outputs)
# -------------------------
class MultiHeadSelfAttention(nn.Module):
    """
    Standard MHA that returns per-head outputs BEFORE concatenation,
    enabling head-level routing.

    Inputs:
      x: [B, T, D]
      attn_mask (optional): broadcastable to [B, H, T, T] (additive mask; use -inf where masked)

    Returns:
      heads_out: [B, T, H, Dh]
      attn_probs: [B, H, T, T]
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        bias: bool = True,
        attn_dropout: float = 0.0
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q = nn.Linear(d_model, d_model, bias=bias)
        self.k = nn.Linear(d_model, d_model, bias=bias)
        self.v = nn.Linear(d_model, d_model, bias=bias)
        self.out = nn.Linear(d_model, d_model, bias=bias)
        self.drop = nn.Dropout(attn_dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head

        q = self.q(x).view(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]
        k = self.k(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v(x).view(B, T, H, Dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)  # [B, H, T, T]
        if attn_mask is not None:
            scores = scores + attn_mask  # additive mask: 0 allowed, -inf masked

        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)
        heads = torch.matmul(attn, v)  # [B, H, T, Dh]

        heads_out = heads.transpose(1, 2).contiguous()  # [B, T, H, Dh]
        return heads_out, attn


# -------------------------
# Pointer Controller (over heads)
# -------------------------
class PointerOverHeadsController(nn.Module):
    """
    Computes routing logits over H heads, per token.
    - If recurrent=True, includes a summary of provisional head outputs as feedback.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        recurrent: bool = False,
        summary: str = "mean",         # 'mean' | 'max'
        hidden_factor: float = 0.5
    ):
        super().__init__()
        self.recurrent = recurrent
        self.summary = summary
        self.n_heads = n_heads

        hid = max(32, int(d_model * hidden_factor))
        self.ln = nn.LayerNorm(d_model)
        self.mlp_static = nn.Sequential(
            nn.Linear(d_model, hid), nn.GELU(), nn.Linear(hid, n_heads)
        )
        if recurrent:
            self.mlp_recur = nn.Sequential(
                nn.Linear(d_model + n_heads, hid), nn.GELU(), nn.Linear(hid, n_heads)
            )

    def summarize_heads(self, heads_out: torch.Tensor) -> torch.Tensor:
        # heads_out: [B, T, H, Dh] -> [B, T, H]
        if self.summary == "mean":
            return heads_out.mean(dim=-1)
        if self.summary == "max":
            return heads_out.amax(dim=-1)
        raise ValueError("summary must be 'mean' or 'max'")

    def forward(
        self,
        token_ctx: torch.Tensor,                 # [B, T, D]
        provisional_heads: Optional[torch.Tensor] = None  # [B, T, H, Dh]
    ) -> torch.Tensor:
        h = self.ln(token_ctx)
        logits = self.mlp_static(h)             # [B, T, H]
        if self.recurrent:
            assert provisional_heads is not None, "recurrent=True requires provisional_heads"
            s = self.summarize_heads(provisional_heads)  # [B, T, H]
            logits = logits + self.mlp_recur(torch.cat([h, s], dim=-1))
        return logits


# -------------------------
# Pointer-over-Heads Transformer Block w/ Adaptive Halting
# -------------------------
class PointerMoHTransformerBlock(nn.Module):
    """
    Pointer-over-Heads Transformer block with adaptive inner-loop halting.

    halting_mode: 'fixed'   -> run exactly max_inner_iters
                  'entropy' -> early-stop when mean entropy < ent_threshold (after min_inner_iters)
                  'halting' -> learned ACT-style halting head + ponder cost (hard-capped by max_inner_iters)

    combination:  'mask_concat' (scale each head output by alpha, concat, W_o)
                  'mixture'     (convex sum across heads, small linear to D)
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        *,
        # attention & FFN
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        use_pre_norm: bool = True,
        # routing
        routing_tau: float = 0.7,
        routing_topk: int = 0,                    # 0 = soft; >0 = hard top-k
        controller_recurrent: bool = True,
        controller_summary: str = "mean",
        combination: str = "mask_concat",         # 'mask_concat' | 'mixture'
        # halting
        halting_mode: str = "fixed",              # 'fixed' | 'entropy' | 'halting'
        max_inner_iters: int = 2,
        min_inner_iters: int = 1,
        ent_threshold: float = 0.7,               # for 'entropy'
        ponder_coef: float = 0.001,               # for 'halting'
    ):
        super().__init__()
        assert combination in ("mask_concat", "mixture")
        assert halting_mode in ("fixed", "entropy", "halting")

        self.d_model = d_model
        self.n_heads = n_heads
        self.use_pre_norm = use_pre_norm

        self.routing_tau = routing_tau
        self.routing_topk = routing_topk
        self.combination = combination

        self.halting_mode = halting_mode
        self.max_inner_iters = max(1, max_inner_iters)
        self.min_inner_iters = max(1, min_inner_iters)
        self.ent_threshold = ent_threshold
        self.ponder_coef = ponder_coef

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.mha = MultiHeadSelfAttention(d_model, n_heads, attn_dropout=attn_dropout)
        self.controller = PointerOverHeadsController(
            d_model=d_model,
            n_heads=n_heads,
            recurrent=(controller_recurrent or self.max_inner_iters > 1),
            summary=controller_summary,
        )

        # Optional halting head (ACT-style)
        if self.halting_mode == "halting":
            self.halt_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 1),
                nn.Sigmoid()
            )

        # FFN
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(ff_dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(ff_dropout),
        )

        # Small projection used by 'mixture' combination
        self._mixture_proj = nn.Linear(d_model // n_heads, d_model)

    # --- helpers ---
    def _route(self, logits: torch.Tensor) -> torch.Tensor:
        if self.routing_topk and self.routing_topk > 0:
            return gumbel_softmax_topk(
                logits, tau=self.routing_tau, topk=self.routing_topk, hard=True
            )
        return F.softmax(logits / max(self.routing_tau, 1e-6), dim=-1)

    def _combine_heads(self, heads_out: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
        # heads_out: [B, T, H, Dh], alphas: [B, T, H]
        B, T, H, Dh = heads_out.shape
        if self.combination == "mask_concat":
            scaled = heads_out * alphas.unsqueeze(-1)  # [B, T, H, Dh]
            concat = scaled.reshape(B, T, H * Dh)      # [B, T, D]
            return self.mha.out(concat)
        else:  # 'mixture'
            mixed = torch.einsum("bthd,bth->btd", heads_out, alphas)  # [B, T, Dh]
            return self._mixture_proj(mixed)                          # [B, T, D]

    # --- main ---
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_aux: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        aux: Dict[str, torch.Tensor] = {}

        h = self.ln1(x) if self.use_pre_norm else x
        token_ctx = h

        it_used = 0
        ponder_cost = torch.zeros((), device=x.device)
        alphas_hist, logits_hist = [], []
        last_attn = None

        for it in range(self.max_inner_iters):
            heads_out, last_attn = self.mha(token_ctx, attn_mask)       # [B,T,H,Dh], [B,H,T,T]
            logits = self.controller(token_ctx, provisional_heads=heads_out)  # [B,T,H]
            alphas = self._route(logits)                                 # [B,T,H]
            routed = self._combine_heads(heads_out, alphas)              # [B,T,D]

            alphas_hist.append(alphas)
            logits_hist.append(logits)
            token_ctx = routed
            it_used = it + 1

            # Halting rules
            if it + 1 < self.min_inner_iters:
                continue

            if self.halting_mode == "fixed":
                pass  # run full max_inner_iters
            elif self.halting_mode == "entropy":
                with torch.no_grad():
                    ent = entropy_from_logits(logits)  # scalar
                if ent.item() < self.ent_threshold:
                    break
            elif self.halting_mode == "halting":
                p_halt = self.halt_head(token_ctx).mean()  # scalar
                ponder_cost = ponder_cost + p_halt
                if p_halt.item() > 0.5:  # threshold; tune
                    break

        # Attention residual
        attn_out = token_ctx + (x if self.use_pre_norm else 0.0)
        if not self.use_pre_norm:
            attn_out = self.ln1(attn_out)

        # FFN + residual
        y_in = self.ln2(attn_out) if self.use_pre_norm else attn_out
        y = attn_out + self.ff(y_in)
        if not self.use_pre_norm:
            y = self.ln2(y)

        if return_aux:
            aux["alphas"] = torch.stack(alphas_hist, dim=1)            # [B, iters, T, H]
            aux["logits"] = torch.stack(logits_hist, dim=1)            # [B, iters, T, H]
            aux["attn_probs"] = last_attn                              # [B, H, T, T]
            aux["inner_iters_used"] = torch.tensor(it_used, device=x.device)
            if self.halting_mode == "halting":
                aux["ponder_cost"] = self.ponder_coef * ponder_cost    # scalar
        return y, aux


# -------------------------
# Simple demo / smoke test
# -------------------------
def demo(
    B: int = 2, T: int = 16, D: int = 256,
    n_heads: int = 8, d_ff: int = 1024,
    halting_mode: str = "entropy",
    max_inner_iters: int = 4,
    min_inner_iters: int = 1,
    ent_threshold: float = 0.8,
    routing_topk: int = 2,
    combination: str = "mask_concat",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    torch.manual_seed(0)
    x = torch.randn(B, T, D, device=device)

    block = PointerMoHTransformerBlock(
        d_model=D, n_heads=n_heads, d_ff=d_ff,
        attn_dropout=0.0, ff_dropout=0.1, use_pre_norm=True,
        routing_tau=0.7, routing_topk=routing_topk,
        controller_recurrent=True, controller_summary="mean",
        combination=combination,
        halting_mode=halting_mode,
        max_inner_iters=max_inner_iters, min_inner_iters=min_inner_iters,
        ent_threshold=ent_threshold, ponder_coef=1e-3,
    ).to(device)

    y, aux = block(x, attn_mask=None, return_aux=True)
    print("y:", tuple(y.shape))
    print("iters used:", int(aux["inner_iters_used"].item()))
    print("alphas:", tuple(aux["alphas"].shape))     # [B, iters, T, H]
    print("attn_probs:", tuple(aux["attn_probs"].shape))  # [B, H, T, T]
    if "ponder_cost" in aux:
        print("ponder_cost:", float(aux["ponder_cost"].item()))


def build_argparser():
    p = argparse.ArgumentParser(description="Pointer-over-Heads Transformer demo")
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--D", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--halting_mode", type=str, default="entropy", choices=["fixed", "entropy", "halting"])
    p.add_argument("--max_inner_iters", type=int, default=4)
    p.add_argument("--min_inner_iters", type=int, default=1)
    p.add_argument("--ent_threshold", type=float, default=0.8)
    p.add_argument("--routing_topk", type=int, default=2, help="0 for soft routing; >0 for hard top-k")
    p.add_argument("--combination", type=str, default="mask_concat", choices=["mask_concat", "mixture"])
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    demo(
        B=args.B,
        T=args.T,
        D=args.D,
        n_heads=args.heads,
        d_ff=args.d_ff,
        halting_mode=args.halting_mode,
        max_inner_iters=args.max_inner_iters,
        min_inner_iters=args.min_inner_iters,
        ent_threshold=args.ent_threshold,
        routing_topk=args.routing_topk,
        combination=args.combination,
        device=device,
    )
