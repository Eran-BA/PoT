"""
Mamba Depth Controller

A Mamba-style controller using Selective State Space Models (SSMs) for efficient
depth-wise routing with O(N) linear time complexity. Unlike traditional attention
which has O(N²) complexity, Mamba uses input-dependent state transitions.

Key Features:
- Linear O(N) complexity via selective state space models
- Input-dependent transitions: A, B, C, D matrices depend on input (selective scan)
- Efficient recurrent processing across depth axis
- Memory-efficient compared to Transformer controllers

Reference: Gu & Dao (2024) "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
https://arxiv.org/abs/2312.00752

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# State Class
# =============================================================================

@dataclass
class MambaDepthState:
    """State for Mamba depth controller.
    
    Attributes:
        h: [B, d_ctrl] hidden state (controller output)
        ssm_state: [B, d_ctrl, d_state] SSM expanded state
        step: Current depth step counter
    """
    h: torch.Tensor
    ssm_state: torch.Tensor
    step: int


# =============================================================================
# Mamba Depth Controller
# =============================================================================

class MambaDepthController(nn.Module):
    """
    Mamba-based depth controller with Selective State Space Models.
    
    Operates across the DEPTH axis (refinement iterations), not across tokens.
    Uses selective SSM which provides O(N) complexity through input-dependent
    state transitions, making it more efficient than Transformer controllers.
    
    Architecture:
        X^(t) [B, S, d_model] → Pool → x_ctrl [B, d_ctrl]
        SelectiveSSM(x_ctrl, ssm_state) → h^(t), ssm_state^(t)
        Router(h^(t), X^(t)) → α^(t) [B, S, H]
    
    Selective SSM Recurrence:
        Δ = softplus(Linear(x))           # Input-dependent discretization
        A_bar = exp(Δ * A)                # Discretized transition
        B_bar = Δ * B(x)                  # Input-dependent input matrix
        h' = A_bar * h + B_bar * x        # State update
        y = C(x) * h' + D * x             # Output
    
    Args:
        d_model: Model dimension (token representation size)
        n_heads: Number of attention heads to route over (H)
        d_ctrl: Controller hidden dimension (default: d_model)
        d_state: SSM state expansion factor (default: 16)
        dt_rank: Rank for delta projection (default: d_ctrl // 16)
        dropout: Dropout probability (default: 0.0)
        token_conditioned: If True, α depends on both token x_i and SSM state
        temperature: Softmax temperature for routing (default: 1.0)
        topk: Optional top-k sparsification (default: None)
        use_layernorm: Apply LayerNorm to states (default: True)
        entropy_reg: Entropy regularization coefficient (default: 1e-3)
        dt_min: Minimum delta value for stability (default: 0.001)
        dt_max: Maximum delta value (default: 0.1)
        dt_init: Initialization strategy for delta ("random" or "constant")
        dt_scale: Scale for delta initialization (default: 1.0)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ctrl: Optional[int] = None,
        d_state: int = 16,
        dt_rank: Optional[int] = None,
        dropout: float = 0.0,
        token_conditioned: bool = True,
        temperature: float = 1.0,
        topk: Optional[int] = None,
        use_layernorm: bool = True,
        entropy_reg: float = 1e-3,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ctrl = d_ctrl or d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, self.d_ctrl // 16)
        self.temperature = float(temperature)
        self.token_conditioned = token_conditioned
        self.topk = topk
        self.entropy_reg = entropy_reg
        self.dt_min = dt_min
        self.dt_max = dt_max
        
        if topk is not None:
            assert 1 <= topk <= n_heads, "topk must be in [1, n_heads]"
        
        # Input projection and pooling
        self.pool_ln = nn.LayerNorm(d_model)
        self.inp_proj = nn.Linear(d_model, self.d_ctrl)
        
        # SSM Parameters
        # A: State transition matrix (initialized as negative, will be exp'd)
        # Following Mamba, A is initialized with log of HiPPO-like matrix
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_ctrl, 1)
        self.A_log = nn.Parameter(torch.log(A))  # [d_ctrl, d_state]
        
        # D: Skip connection (direct input to output)
        self.D = nn.Parameter(torch.ones(self.d_ctrl))
        
        # Input-dependent projections for B, C, and delta
        # Combined projection for efficiency
        self.x_proj = nn.Linear(self.d_ctrl, self.dt_rank + d_state * 2, bias=False)
        
        # Delta (discretization step) projection
        self.dt_proj = nn.Linear(self.dt_rank, self.d_ctrl, bias=True)
        
        # Initialize dt_proj bias for proper delta range
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        else:
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # Initialize bias to produce dt in [dt_min, dt_max] range
        dt = torch.exp(
            torch.rand(self.d_ctrl) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_min)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # Inverse of softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_ctrl, self.d_ctrl)
        
        # Optional layer norm
        self.ln_h = nn.LayerNorm(self.d_ctrl) if use_layernorm else nn.Identity()
        
        # Router
        self.drop = nn.Dropout(dropout)
        if token_conditioned:
            self.router = nn.Sequential(
                nn.LayerNorm(d_model + self.d_ctrl),
                nn.Linear(d_model + self.d_ctrl, self.d_ctrl),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.d_ctrl, n_heads),
            )
        else:
            self.router = nn.Sequential(
                nn.LayerNorm(self.d_ctrl),
                nn.Linear(self.d_ctrl, self.d_ctrl),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.d_ctrl, n_heads),
            )
    
    def init_state(self, batch_size: int, device: torch.device) -> MambaDepthState:
        """Initialize zero state."""
        h0 = torch.zeros(batch_size, self.d_ctrl, device=device)
        ssm_state0 = torch.zeros(batch_size, self.d_ctrl, self.d_state, device=device)
        return MambaDepthState(h=h0, ssm_state=ssm_state0, step=0)
    
    def _pool(self, X: torch.Tensor) -> torch.Tensor:
        """Pool tokens to single vector per batch."""
        Xn = self.pool_ln(X)
        return Xn.mean(dim=1)  # [B, d_model]
    
    def _topk_mask_renorm(self, alpha: torch.Tensor) -> torch.Tensor:
        """Apply top-k masking and renormalize."""
        if self.topk is None or self.topk >= alpha.size(-1):
            return alpha
        
        topv, topi = torch.topk(alpha, k=self.topk, dim=-1)
        masked = torch.zeros_like(alpha)
        masked.scatter_(-1, topi, topv)
        denom = masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return masked / denom
    
    def _compute_entropy(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute routing entropy."""
        return -(alpha * alpha.clamp_min(1e-12).log()).sum(dim=-1).mean()
    
    def set_temperature(self, T: float):
        """Update softmax temperature."""
        self.temperature = max(0.1, float(T))
    
    def _selective_ssm_step(
        self,
        x: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selective SSM step with input-dependent discretization.
        
        This is the core Mamba mechanism: state transitions depend on input,
        allowing the model to selectively remember or forget information.
        
        Args:
            x: [B, d_ctrl] input
            ssm_state: [B, d_ctrl, d_state] current SSM state
            
        Returns:
            y: [B, d_ctrl] output
            ssm_state_new: [B, d_ctrl, d_state] updated SSM state
        """
        B = x.shape[0]
        
        # Get A (discretized will be computed)
        A = -torch.exp(self.A_log.float())  # [d_ctrl, d_state]
        
        # Project x to get delta, B, C
        x_dbl = self.x_proj(x)  # [B, dt_rank + d_state * 2]
        
        # Split into delta, B, C
        delta_proj = x_dbl[:, :self.dt_rank]  # [B, dt_rank]
        B_proj = x_dbl[:, self.dt_rank:self.dt_rank + self.d_state]  # [B, d_state]
        C_proj = x_dbl[:, self.dt_rank + self.d_state:]  # [B, d_state]
        
        # Compute delta (discretization step) - input-dependent
        delta = F.softplus(self.dt_proj(delta_proj))  # [B, d_ctrl]
        delta = delta.clamp(min=self.dt_min, max=self.dt_max)
        
        # Discretize A and B using delta
        # A_bar = exp(delta * A)
        # For numerical stability, we compute this element-wise
        delta_A = delta.unsqueeze(-1) * A.unsqueeze(0)  # [B, d_ctrl, d_state]
        A_bar = torch.exp(delta_A)  # [B, d_ctrl, d_state]
        
        # B_bar = delta * B (simplified discretization)
        delta_B = delta.unsqueeze(-1) * B_proj.unsqueeze(1)  # [B, d_ctrl, d_state]
        
        # SSM recurrence: h' = A_bar * h + B_bar * x
        # x needs to be expanded: [B, d_ctrl] -> [B, d_ctrl, 1]
        x_expanded = x.unsqueeze(-1)  # [B, d_ctrl, 1]
        
        ssm_state_new = A_bar * ssm_state + delta_B * x_expanded  # [B, d_ctrl, d_state]
        
        # Output: y = C * h' + D * x
        # C_proj: [B, d_state], ssm_state_new: [B, d_ctrl, d_state]
        # We need to compute sum over d_state: [B, d_ctrl]
        y = torch.einsum('bd,bcd->bc', C_proj, ssm_state_new)  # [B, d_ctrl]
        y = y + self.D * x  # Skip connection
        
        return y, ssm_state_new
    
    def step(
        self,
        X: torch.Tensor,
        state: Optional[MambaDepthState] = None,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, MambaDepthState, Dict[str, Any]]:
        """
        One refinement step.
        
        Args:
            X: [B, S, d_model] token representations
            state: Previous MambaDepthState (or None to initialize)
            return_aux: If True, return auxiliary metrics
            
        Returns:
            alpha: [B, S, H] routing weights
            state: Updated MambaDepthState
            aux: Dict with metrics
        """
        B, S, D = X.shape
        device = X.device
        
        if state is None:
            state = self.init_state(B, device)
        
        # Pool and project input
        g = self._pool(X)  # [B, d_model]
        x_ctrl = self.inp_proj(self.drop(g))  # [B, d_ctrl]
        
        # Selective SSM step
        y, ssm_state_new = self._selective_ssm_step(x_ctrl, state.ssm_state)
        
        # Output projection and normalization
        h_new = self.out_proj(y)
        h_new = self.ln_h(h_new)
        
        # Route to heads
        if self.token_conditioned:
            h_tok = h_new[:, None, :].expand(B, S, self.d_ctrl)
            logits = self.router(torch.cat([X, h_tok], dim=-1))  # [B, S, H]
        else:
            logits_global = self.router(h_new)  # [B, H]
            logits = logits_global[:, None, :].expand(B, S, self.n_heads)
        
        alpha = F.softmax(logits / self.temperature, dim=-1)
        alpha = self._topk_mask_renorm(alpha)
        
        entropy = self._compute_entropy(alpha)
        
        new_state = MambaDepthState(h=h_new, ssm_state=ssm_state_new, step=state.step + 1)
        
        aux: Dict[str, Any] = {}
        if return_aux:
            aux = {
                "router_logits": logits.detach(),
                "alphas": alpha.detach(),
                "entropy": entropy.detach(),
                "temperature": self.temperature,
                "depth_step": state.step,
            }
        
        return alpha, new_state, aux
    
    def forward(
        self,
        x: torch.Tensor,
        head_outputs: Optional[torch.Tensor] = None,
        *,
        state: Optional[MambaDepthState] = None,
        per_token_pool: str = "mean",
        mask: Optional[torch.Tensor] = None,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, MambaDepthState, Dict[str, Any]]:
        """Forward pass compatible with HRMPointerController API."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        alpha, state, aux = self.step(x, state, return_aux)
        
        if alpha.size(1) == 1:
            alpha = alpha.squeeze(1)
        
        return alpha, state, aux

