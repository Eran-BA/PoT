"""
HRM Controller with ACT-style Adaptive Halting.

Extends HRMPointerController with learned halting for H-module updates.
Instead of fixed T-step updates, learns when to update H based on state.

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import HRMState


@dataclass
class ACTHRMState(HRMState):
    """Extended HRM state with ACT ponder tracking."""
    acc_R: torch.Tensor  # [B] accumulated ponder (halting probability)
    n_updates: torch.Tensor  # [B] count of H-updates so far


class ACTHRMPointerController(nn.Module):
    """
    HRM-style controller with ACT (Adaptive Computation Time) halting.
    
    Instead of fixed T-step H-updates, learns a halting probability per step.
    When accumulated probability >= 1.0, triggers H-module update.
    
    Key differences from standard HRM:
    - Learned halting gate: p_t = sigmoid(w^T z_H + b)
    - Ponder cost: tau * sum(p_t) added to loss
    - More flexible: adapts update frequency per sample/timestep
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ctrl: Optional[int] = None,
        *,
        # ACT-specific
        ponder_tau: float = 0.01,         # Ponder cost weight
        halt_epsilon: float = 0.01,       # Early stopping threshold
        max_ponders: int = 20,            # Max ponder steps (safety)
        # Standard HRM params
        topk: Optional[int] = None,
        temperature_init: float = 2.0,
        temperature_min: float = 0.7,
        entropy_reg: float = 1e-3,
        use_layernorm: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ctrl = d_ctrl or d_model
        self.topk = topk
        self.temperature_init = temperature_init
        self.temperature_min = temperature_min
        self.entropy_reg = entropy_reg
        
        # ACT params
        self.ponder_tau = ponder_tau
        self.halt_epsilon = halt_epsilon
        self.max_ponders = max_ponders
        
        # Input adapters
        self.inp_proj = nn.Linear(d_model, self.d_ctrl)
        
        # HRM core
        self.f_L = nn.GRUCell(input_size=self.d_ctrl * 2, hidden_size=self.d_ctrl)
        self.f_H = nn.GRUCell(input_size=self.d_ctrl, hidden_size=self.d_ctrl)
        
        self.ln_L = nn.LayerNorm(self.d_ctrl) if use_layernorm else nn.Identity()
        self.ln_H = nn.LayerNorm(self.d_ctrl) if use_layernorm else nn.Identity()
        
        # Router
        self.router = nn.Linear(self.d_ctrl, n_heads)
        self.mix_gate = nn.Linear(self.d_ctrl, self.d_ctrl, bias=False)
        
        # ACT halting head
        self.halt_head = nn.Linear(self.d_ctrl, 1)
        
        self.drop = nn.Dropout(dropout)
        
        # Temperature
        self.log_temperature = nn.Parameter(
            torch.tensor(self._to_logtemp(temperature_init)),
            requires_grad=False
        )
    
    def _to_logtemp(self, T: float) -> float:
        return float(torch.log(torch.tensor(T)))
    
    def set_temperature(self, T: float):
        T = max(self.temperature_min, float(T))
        with torch.no_grad():
            self.log_temperature.copy_(torch.log(torch.tensor(T)))
    
    def init_state(self, batch_size: int, device: torch.device) -> ACTHRMState:
        z0 = torch.zeros(batch_size, self.d_ctrl, device=device)
        step = torch.zeros(batch_size, dtype=torch.long, device=device)
        acc_R = torch.zeros(batch_size, device=device)
        n_updates = torch.zeros(batch_size, dtype=torch.long, device=device)
        return ACTHRMState(z_L=z0, z_H=z0, step=step, acc_R=acc_R, n_updates=n_updates)
    
    def _should_update_H(
        self,
        z_H: torch.Tensor,
        acc_R: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute halting probability and decide whether to update H.
        
        Returns:
            update_mask: [B] boolean, True where H should update
            new_acc_R: [B] updated accumulated ponder
            ponder_prob: [B] halting probability for this step
        """
        # Compute halting probability
        ponder_logit = self.halt_head(z_H).squeeze(-1)  # [B]
        ponder_prob = torch.sigmoid(ponder_logit)       # [B]
        
        # Accumulate ponder
        new_acc_R = torch.clamp(acc_R + ponder_prob, max=1.0)
        
        # Update when accumulated ponder >= threshold
        update_mask = (new_acc_R >= (1.0 - self.halt_epsilon))
        
        return update_mask, new_acc_R, ponder_prob
    
    def forward(
        self,
        x: torch.Tensor,
        head_outputs: torch.Tensor,
        *,
        state: Optional[ACTHRMState] = None,
        per_token_pool: str = "mean",
        mask: Optional[torch.Tensor] = None,
        return_aux: bool = True
    ) -> Tuple[torch.Tensor, ACTHRMState, Dict[str, Any]]:
        """
        Forward with ACT halting.
        
        Returns:
            alphas: [B, n_heads] routing weights
            new_state: updated ACTHRMState
            aux: dict with ponder cost, halting info, etc.
        """
        B = x.size(0)
        device = x.device
        
        if state is None:
            state = self.init_state(B, device)
        
        # Summarize x
        if x.dim() == 3:
            if mask is not None:
                m = mask.float().unsqueeze(-1)
                x_sum = (x * m).sum(dim=1)
                x_den = m.sum(dim=1).clamp_min(1e-6)
                x_pooled = x_sum / x_den
            else:
                if per_token_pool == "mean":
                    x_pooled = x.mean(dim=1)
                elif per_token_pool == "cls":
                    x_pooled = x[:, 0]
                else:
                    x_pooled = x.mean(dim=1)
        else:
            x_pooled = x
        
        x_ctrl = self.inp_proj(self.drop(x_pooled))
        
        # ACT halting logic
        update_mask, new_acc_R, ponder_prob = self._should_update_H(
            state.z_H, state.acc_R
        )
        
        # Update H where needed
        if update_mask.any():
            # Run f_H only for samples that need update
            z_H_new = state.z_H.clone()
            
            # Update subset
            if update_mask.all():
                z_H_updated = self.f_H(x_ctrl, state.z_H)
                z_H_new = self.ln_H(z_H_updated)
            else:
                # Selective update (more efficient)
                x_ctrl_update = x_ctrl[update_mask]
                z_H_update = state.z_H[update_mask]
                z_H_updated = self.f_H(x_ctrl_update, z_H_update)
                z_H_new[update_mask] = self.ln_H(z_H_updated)
            
            # Reset ponder for updated samples
            new_acc_R = new_acc_R * (~update_mask).float()
            n_updates = state.n_updates + update_mask.long()
        else:
            z_H_new = state.z_H
            n_updates = state.n_updates
        
        # L-module (always updates)
        l_inp = torch.cat([x_ctrl, z_H_new], dim=-1)
        z_L_new = self.f_L(l_inp, state.z_L)
        z_L_new = self.ln_L(z_L_new)
        
        # Cross-conditioning
        z_L_cond = z_L_new + self.mix_gate(z_H_new)
        
        # Routing
        logits = self.router(self.drop(z_L_cond))
        T = torch.exp(self.log_temperature).clamp(min=self.temperature_min)
        probs = F.softmax(logits / T, dim=-1)
        
        # Top-k
        topk_idx = None
        if self.topk is not None and self.topk < self.n_heads:
            topk_vals, topk_idx = probs.topk(self.topk, dim=-1)
            mask_topk = torch.zeros_like(probs)
            mask_topk.scatter_(dim=-1, index=topk_idx, value=1.0)
            probs = probs * mask_topk
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        
        # Entropy
        entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1).mean()
        
        # Ponder cost (for loss)
        ponder_cost = ponder_prob.mean()
        
        # Update state
        new_step = state.step + 1
        new_state = ACTHRMState(
            z_L=z_L_new,
            z_H=z_H_new,
            step=new_step,
            acc_R=new_acc_R,
            n_updates=n_updates
        )
        
        alphas = probs
        
        aux = {}
        if return_aux:
            aux = {
                "router_logits": logits.detach(),
                "alphas": alphas.detach(),
                "entropy": entropy.detach(),
                "temperature": float(T.detach().cpu()),
                "ponder_cost": ponder_cost.detach(),
                "ponder_prob": ponder_prob.detach().mean(),
                "n_H_updates": n_updates.float().mean().detach(),
                "acc_R": new_acc_R.mean().detach(),
            }
            if topk_idx is not None:
                aux["topk_idx"] = topk_idx.detach()
        
        return alphas, new_state, aux

