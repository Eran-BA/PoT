"""
HRM-style Pointer Controller (Task-Agnostic)

Two-timescale hierarchical controller for dynamic attention head routing.
Compatible with any transformer-based architecture.

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HRMState:
    """Persistent controller state across iterations."""
    z_L: torch.Tensor  # [B, d_ctrl] Low-level (fast) state
    z_H: torch.Tensor  # [B, d_ctrl] High-level (slow) state
    step: torch.Tensor  # [B] or scalar long; iteration counter


class HRMPointerController(nn.Module):
    """
    HRM-style two-timescale controller for routing over attention heads.

    Architecture:
    - Low-level module f_L: updates EVERY iteration (fast timescale)
    - High-level module f_H: updates every T iterations (slow timescale)
    - f_H provides context/guidance to f_L via cross-conditioning
    
    Output: Routing weights (alphas) over n_heads

    Note on routing granularity:
        This controller produces per-SEQUENCE routing weights [B, n_heads] that are
        broadcast to all tokens. For true per-TOKEN routing [B, T, H], use
        PointerOverHeadsController from src.models.layers instead.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads to route over
        d_ctrl: Controller hidden dimension (default: d_model)
        T: Period for H-module updates (slow timescale, default: 4)
        topk: Optional sparsification (route to top-k heads only)
        temperature_init: Initial softmax temperature (default: 2.0)
        temperature_min: Minimum temperature (default: 0.7)
        entropy_reg: Entropy regularization coefficient (default: 1e-3)
        use_layernorm: Apply LayerNorm to states (default: True)
        dropout: Dropout probability (default: 0.0)
        detach_H_update: If True, block gradients through slow-timescale H updates
                        for memory efficiency. If False (default), H-module is fully
                        differentiable as in the original HRM paper.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ctrl: Optional[int] = None,
        *,
        T: int = 4,
        topk: Optional[int] = None,
        temperature_init: float = 2.0,
        temperature_min: float = 0.7,
        entropy_reg: float = 1e-3,
        use_layernorm: bool = True,
        dropout: float = 0.0,
        detach_H_update: bool = False,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ctrl = d_ctrl or d_model
        self.T = int(T)
        self.topk = topk
        self.temperature_init = temperature_init
        self.temperature_min = temperature_min
        self.entropy_reg = entropy_reg
        self.detach_H_update = detach_H_update

        # Input projection: map task representation to controller space
        self.inp_proj = nn.Linear(d_model, self.d_ctrl)

        # HRM core: two-timescale recurrent modules
        self.f_L = nn.GRUCell(input_size=self.d_ctrl * 2, hidden_size=self.d_ctrl)  # Fast
        self.f_H = nn.GRUCell(input_size=self.d_ctrl, hidden_size=self.d_ctrl)  # Slow

        # Optional normalization
        self.ln_L = nn.LayerNorm(self.d_ctrl) if use_layernorm else nn.Identity()
        self.ln_H = nn.LayerNorm(self.d_ctrl) if use_layernorm else nn.Identity()

        # Head routing: map L-state to logits over heads
        self.router = nn.Linear(self.d_ctrl, n_heads)

        # Cross-conditioning: H modulates L
        self.mix_gate = nn.Linear(self.d_ctrl, self.d_ctrl, bias=False)

        self.drop = nn.Dropout(dropout)

        # Temperature (scheduled externally)
        self.log_temperature = nn.Parameter(
            torch.tensor(self._to_logtemp(temperature_init)),
            requires_grad=False
        )

    def _to_logtemp(self, T: float) -> float:
        return float(torch.log(torch.tensor(T)))

    def set_temperature(self, T: float):
        """Update softmax temperature (called by trainer for annealing)."""
        T = max(self.temperature_min, float(T))
        with torch.no_grad():
            self.log_temperature.copy_(torch.log(torch.tensor(T)))

    def init_state(self, batch_size: int, device: torch.device) -> HRMState:
        """Initialize zero state."""
        z0 = torch.zeros(batch_size, self.d_ctrl, device=device)
        step = torch.zeros(batch_size, dtype=torch.long, device=device)
        return HRMState(z_L=z0, z_H=z0, step=step)

    def _maybe_update_H(self, x_ctrl: torch.Tensor, state: HRMState) -> HRMState:
        """Update H-module only when (step % T) == 0.

        If detach_H_update=True, gradients are blocked through this update for memory efficiency.
        If detach_H_update=False (default), the H-module is fully differentiable.
        """
        needs = (state.step % self.T) == 0
        if needs.any():
            ctx = torch.no_grad() if self.detach_H_update else nullcontext()
            with ctx:
                z_H_new = self.f_H(x_ctrl, state.z_H)
                state = HRMState(z_L=state.z_L, z_H=self.ln_H(z_H_new), step=state.step)
        return state

    def forward(
        self,
        x: torch.Tensor,  # [B, L, d_model] or [B, d_model]
        head_outputs: Optional[torch.Tensor] = None,  # [B, n_heads, ...] (unused in routing, kept for API compat)
        *,
        state: Optional[HRMState] = None,
        per_token_pool: str = "mean",
        mask: Optional[torch.Tensor] = None,  # [B, L]
        return_aux: bool = True
    ) -> Tuple[torch.Tensor, HRMState, Dict[str, Any]]:
        """
        Compute routing weights over heads.

        Args:
            x: Input representation [B, L, d_model] or pooled [B, d_model]
            head_outputs: Precomputed head features (not used for routing, optional)
            state: Previous HRMState (or None to initialize)
            per_token_pool: How to pool sequence ("mean" or "cls")
            mask: Attention mask [B, L] for masked pooling
            return_aux: Return auxiliary metrics

        Returns:
            alphas: Routing weights [B, n_heads]
            new_state: Updated HRMState
            aux: Dict with 'entropy', 'logits', 'temperature', etc.
        """
        B = x.size(0)
        device = x.device
        
        if state is None:
            state = self.init_state(B, device)

        # Pool sequence to fixed-size representation
        if x.dim() == 3:  # [B, L, d_model]
            if mask is not None:
                # Masked mean
                m = mask.float().unsqueeze(-1)
                x_pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-6)
            else:
                x_pooled = x.mean(dim=1) if per_token_pool == "mean" else x[:, 0]
        else:
            x_pooled = x  # Already [B, d_model]

        x_ctrl = self.inp_proj(self.drop(x_pooled))  # [B, d_ctrl]

        # HRM updates: slow H, then fast L conditioned on H
        state = self._maybe_update_H(x_ctrl, state)

        l_inp = torch.cat([x_ctrl, state.z_H], dim=-1)  # [B, 2*d_ctrl]
        z_L_new = self.f_L(l_inp, state.z_L)
        z_L_new = self.ln_L(z_L_new)

        # Cross-condition: H modulates L (FiLM-style)
        z_L_cond = z_L_new + self.mix_gate(state.z_H)

        # Route to heads
        logits = self.router(self.drop(z_L_cond))  # [B, n_heads]
        T = torch.exp(self.log_temperature).clamp(min=self.temperature_min)
        probs = F.softmax(logits / T, dim=-1)

        # Optional top-k sparsification
        topk_idx = None
        if self.topk is not None and self.topk < self.n_heads:
            topk_vals, topk_idx = probs.topk(self.topk, dim=-1)
            mask_topk = torch.zeros_like(probs)
            mask_topk.scatter_(dim=-1, index=topk_idx, value=1.0)
            probs = probs * mask_topk
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # Entropy (for regularization)
        entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1).mean()

        # Update state
        new_step = state.step + 1
        new_state = HRMState(z_L=z_L_new, z_H=state.z_H, step=new_step)

        alphas = probs  # [B, n_heads]

        aux = {}
        if return_aux:
            aux = {
                "router_logits": logits.detach(),
                "alphas": alphas.detach(),
                "entropy": entropy.detach(),
                "temperature": float(T.detach().cpu()),
                "features": z_L_cond,  # [B, d_ctrl] - injectable controller feature
            }
            if topk_idx is not None:
                aux["topk_idx"] = topk_idx.detach()

        return alphas, new_state, aux

