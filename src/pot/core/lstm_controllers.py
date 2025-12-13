"""
LSTM and xLSTM Depth Controllers

Alternative recurrent controllers that operate across the depth axis (refinement
iterations). These provide different capacity/efficiency trade-offs compared to
the default GRU-based controller.

Controllers included:
- LSTMDepthController: Standard LSTM with stronger gating than GRU
- xLSTMDepthController: Extended LSTM with exponential gating (sLSTM style)

All controllers operate across DEPTH (refinement iterations), not across the
input sequence length. Each token maintains its own independent controller
state that evolves as the model iterates through reasoning steps.

References:
- LSTM: Hochreiter & Schmidhuber (1997)
- xLSTM: Beck et al. (2024) https://arxiv.org/abs/2405.04517

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
# State Classes
# =============================================================================

@dataclass
class LSTMDepthState:
    """State for LSTM depth controller."""
    h: torch.Tensor  # [B, d_ctrl] hidden state
    c: torch.Tensor  # [B, d_ctrl] cell state
    step: int        # current depth step


@dataclass
class xLSTMDepthState:
    """State for xLSTM depth controller (sLSTM variant with exponential gating)."""
    h: torch.Tensor  # [B, d_ctrl] hidden state
    c: torch.Tensor  # [B, d_ctrl] cell state
    n: torch.Tensor  # [B, d_ctrl] normalizer state (for exponential gating)
    m: torch.Tensor  # [B] max state (for numerical stability)
    step: int        # current depth step


# =============================================================================
# LSTM Depth Controller
# =============================================================================

class LSTMDepthController(nn.Module):
    """
    LSTM-based depth controller for routing over attention heads.
    
    Operates across the DEPTH axis (refinement iterations), not across tokens.
    Uses standard LSTM which provides stronger gating than GRU through separate
    input, forget, and output gates plus a cell state.
    
    Architecture:
        X^(t) [B, S, d_model] → Pool → x_ctrl [B, d_ctrl]
        LSTM(x_ctrl, (h, c)) → h^(t), c^(t)
        Router(h^(t), X^(t)) → α^(t) [B, S, H]
    
    Args:
        d_model: Model dimension (token representation size)
        n_heads: Number of attention heads to route over (H)
        d_ctrl: Controller hidden dimension (default: d_model)
        dropout: Dropout probability (default: 0.0)
        token_conditioned: If True, α depends on both token x_i and LSTM state
        temperature: Softmax temperature for routing (default: 1.0)
        topk: Optional top-k sparsification (default: None)
        use_layernorm: Apply LayerNorm to states (default: True)
        entropy_reg: Entropy regularization coefficient (default: 1e-3)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ctrl: Optional[int] = None,
        dropout: float = 0.0,
        token_conditioned: bool = True,
        temperature: float = 1.0,
        topk: Optional[int] = None,
        use_layernorm: bool = True,
        entropy_reg: float = 1e-3,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ctrl = d_ctrl or d_model
        self.temperature = float(temperature)
        self.token_conditioned = token_conditioned
        self.topk = topk
        self.entropy_reg = entropy_reg
        
        if topk is not None:
            assert 1 <= topk <= n_heads, "topk must be in [1, n_heads]"
        
        # Input projection and pooling
        self.pool_ln = nn.LayerNorm(d_model)
        self.inp_proj = nn.Linear(d_model, self.d_ctrl)
        
        # LSTM cell
        self.lstm_cell = nn.LSTMCell(
            input_size=self.d_ctrl,
            hidden_size=self.d_ctrl,
        )
        
        # Optional layer norm for states
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
    
    def init_state(self, batch_size: int, device: torch.device) -> LSTMDepthState:
        """Initialize zero state."""
        h0 = torch.zeros(batch_size, self.d_ctrl, device=device)
        c0 = torch.zeros(batch_size, self.d_ctrl, device=device)
        return LSTMDepthState(h=h0, c=c0, step=0)
    
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
    
    def step(
        self,
        X: torch.Tensor,
        state: Optional[LSTMDepthState] = None,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, LSTMDepthState, Dict[str, Any]]:
        """
        One refinement step.
        
        Args:
            X: [B, S, d_model] token representations
            state: Previous LSTMDepthState (or None to initialize)
            return_aux: If True, return auxiliary metrics
            
        Returns:
            alpha: [B, S, H] routing weights
            state: Updated LSTMDepthState
            aux: Dict with metrics
        """
        B, S, D = X.shape
        device = X.device
        
        if state is None:
            state = self.init_state(B, device)
        
        # Pool and project input
        g = self._pool(X)  # [B, d_model]
        x_ctrl = self.inp_proj(self.drop(g))  # [B, d_ctrl]
        
        # LSTM step
        h_new, c_new = self.lstm_cell(x_ctrl, (state.h, state.c))
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
        
        new_state = LSTMDepthState(h=h_new, c=c_new, step=state.step + 1)
        
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
        state: Optional[LSTMDepthState] = None,
        per_token_pool: str = "mean",
        mask: Optional[torch.Tensor] = None,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, LSTMDepthState, Dict[str, Any]]:
        """Forward pass compatible with HRMPointerController API."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        alpha, state, aux = self.step(x, state, return_aux)
        
        if alpha.size(1) == 1:
            alpha = alpha.squeeze(1)
        
        return alpha, state, aux


# =============================================================================
# xLSTM Depth Controller (sLSTM variant with exponential gating)
# =============================================================================

class xLSTMDepthController(nn.Module):
    """
    xLSTM-based depth controller with exponential gating (sLSTM variant).
    
    Implements the scalar LSTM (sLSTM) variant from xLSTM paper with:
    - Exponential gating for input and forget gates
    - Normalizer state for numerical stability
    - Enhanced memory capacity compared to standard LSTM
    
    Operates across the DEPTH axis (refinement iterations), not across tokens.
    
    Reference: Beck et al. (2024) "xLSTM: Extended Long Short-Term Memory"
    https://arxiv.org/abs/2405.04517
    
    Architecture:
        X^(t) [B, S, d_model] → Pool → x_ctrl [B, d_ctrl]
        sLSTM(x_ctrl, state) → h^(t), c^(t), n^(t), m^(t)
        Router(h^(t), X^(t)) → α^(t) [B, S, H]
    
    Args:
        d_model: Model dimension (token representation size)
        n_heads: Number of attention heads to route over (H)
        d_ctrl: Controller hidden dimension (default: d_model)
        dropout: Dropout probability (default: 0.0)
        token_conditioned: If True, α depends on both token x_i and xLSTM state
        temperature: Softmax temperature for routing (default: 1.0)
        topk: Optional top-k sparsification (default: None)
        use_layernorm: Apply LayerNorm to states (default: True)
        entropy_reg: Entropy regularization coefficient (default: 1e-3)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ctrl: Optional[int] = None,
        dropout: float = 0.0,
        token_conditioned: bool = True,
        temperature: float = 1.0,
        topk: Optional[int] = None,
        use_layernorm: bool = True,
        entropy_reg: float = 1e-3,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ctrl = d_ctrl or d_model
        self.temperature = float(temperature)
        self.token_conditioned = token_conditioned
        self.topk = topk
        self.entropy_reg = entropy_reg
        
        if topk is not None:
            assert 1 <= topk <= n_heads, "topk must be in [1, n_heads]"
        
        # Input projection and pooling
        self.pool_ln = nn.LayerNorm(d_model)
        self.inp_proj = nn.Linear(d_model, self.d_ctrl)
        
        # sLSTM gates (all computed from input + hidden)
        # i: input gate, f: forget gate, o: output gate, z: cell input
        self.W_i = nn.Linear(self.d_ctrl, self.d_ctrl)
        self.U_i = nn.Linear(self.d_ctrl, self.d_ctrl, bias=False)
        
        self.W_f = nn.Linear(self.d_ctrl, self.d_ctrl)
        self.U_f = nn.Linear(self.d_ctrl, self.d_ctrl, bias=False)
        
        self.W_o = nn.Linear(self.d_ctrl, self.d_ctrl)
        self.U_o = nn.Linear(self.d_ctrl, self.d_ctrl, bias=False)
        
        self.W_z = nn.Linear(self.d_ctrl, self.d_ctrl)
        self.U_z = nn.Linear(self.d_ctrl, self.d_ctrl, bias=False)
        
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
    
    def init_state(self, batch_size: int, device: torch.device) -> xLSTMDepthState:
        """Initialize zero state."""
        h0 = torch.zeros(batch_size, self.d_ctrl, device=device)
        c0 = torch.zeros(batch_size, self.d_ctrl, device=device)
        n0 = torch.ones(batch_size, self.d_ctrl, device=device)  # normalizer starts at 1
        m0 = torch.zeros(batch_size, device=device)  # max tracker for stability
        return xLSTMDepthState(h=h0, c=c0, n=n0, m=m0, step=0)
    
    def _pool(self, X: torch.Tensor) -> torch.Tensor:
        """Pool tokens to single vector per batch."""
        Xn = self.pool_ln(X)
        return Xn.mean(dim=1)
    
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
    
    def _slstm_step(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        n: torch.Tensor,
        m: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        sLSTM step with exponential gating.
        
        Key difference from standard LSTM: exponential gates for better gradient flow.
        Uses log-space computation for numerical stability.
        
        Args:
            x: [B, d_ctrl] input
            h: [B, d_ctrl] hidden state
            c: [B, d_ctrl] cell state
            n: [B, d_ctrl] normalizer state
            m: [B] max state for stability
            
        Returns:
            h_new, c_new, n_new, m_new
        """
        # Compute gate pre-activations
        i_tilde = self.W_i(x) + self.U_i(h)  # [B, d_ctrl]
        f_tilde = self.W_f(x) + self.U_f(h)  # [B, d_ctrl]
        o_pre = self.W_o(x) + self.U_o(h)    # [B, d_ctrl]
        z_tilde = self.W_z(x) + self.U_z(h)  # [B, d_ctrl]
        
        # Exponential gating with numerical stability
        # m_new = max(m + f_tilde.max, i_tilde.max)
        f_max = f_tilde.max(dim=-1, keepdim=False)[0]  # [B]
        i_max = i_tilde.max(dim=-1, keepdim=False)[0]  # [B]
        m_new = torch.maximum(m + f_max, i_max)  # [B]
        
        # Compute stabilized exponential gates
        # exp(f_tilde - m_new) and exp(i_tilde - m_new)
        m_new_expanded = m_new[:, None]  # [B, 1]
        m_expanded = m[:, None]  # [B, 1]
        
        i_gate = torch.exp(i_tilde - m_new_expanded)  # [B, d_ctrl]
        f_gate = torch.exp(f_tilde + m_expanded - m_new_expanded)  # [B, d_ctrl]
        o_gate = torch.sigmoid(o_pre)  # Output gate uses sigmoid
        
        # Cell input
        z = torch.tanh(z_tilde)
        
        # Update cell state and normalizer
        c_new = f_gate * c + i_gate * z
        n_new = f_gate * n + i_gate
        
        # Compute hidden state (normalized cell output)
        h_new = o_gate * (c_new / n_new.clamp_min(1e-6))
        
        return h_new, c_new, n_new, m_new
    
    def step(
        self,
        X: torch.Tensor,
        state: Optional[xLSTMDepthState] = None,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, xLSTMDepthState, Dict[str, Any]]:
        """
        One refinement step.
        
        Args:
            X: [B, S, d_model] token representations
            state: Previous xLSTMDepthState (or None to initialize)
            return_aux: If True, return auxiliary metrics
            
        Returns:
            alpha: [B, S, H] routing weights
            state: Updated xLSTMDepthState
            aux: Dict with metrics
        """
        B, S, D = X.shape
        device = X.device
        
        if state is None:
            state = self.init_state(B, device)
        
        # Pool and project input
        g = self._pool(X)
        x_ctrl = self.inp_proj(self.drop(g))
        
        # sLSTM step
        h_new, c_new, n_new, m_new = self._slstm_step(
            x_ctrl, state.h, state.c, state.n, state.m
        )
        h_new = self.ln_h(h_new)
        
        # Route to heads
        if self.token_conditioned:
            h_tok = h_new[:, None, :].expand(B, S, self.d_ctrl)
            logits = self.router(torch.cat([X, h_tok], dim=-1))
        else:
            logits_global = self.router(h_new)
            logits = logits_global[:, None, :].expand(B, S, self.n_heads)
        
        alpha = F.softmax(logits / self.temperature, dim=-1)
        alpha = self._topk_mask_renorm(alpha)
        
        entropy = self._compute_entropy(alpha)
        
        new_state = xLSTMDepthState(
            h=h_new, c=c_new, n=n_new, m=m_new, step=state.step + 1
        )
        
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
        state: Optional[xLSTMDepthState] = None,
        per_token_pool: str = "mean",
        mask: Optional[torch.Tensor] = None,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, xLSTMDepthState, Dict[str, Any]]:
        """Forward pass compatible with HRMPointerController API."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        alpha, state, aux = self.step(x, state, return_aux)
        
        if alpha.size(1) == 1:
            alpha = alpha.squeeze(1)
        
        return alpha, state, aux


# =============================================================================
# minGRU Depth Controller (Simplified GRU variant)
# =============================================================================

class minGRUDepthController(nn.Module):
    """
    Minimal GRU depth controller with simplified gating.
    
    Implements a minimal GRU variant with reduced parameters:
    - Single gate (linear interpolation between old and new)
    - No reset gate
    - Fewer parameters, faster computation
    
    Operates across the DEPTH axis (refinement iterations), not across tokens.
    
    Reference: Inspired by minGRU/minLSTM efficiency research
    
    Args:
        d_model: Model dimension (token representation size)
        n_heads: Number of attention heads to route over (H)
        d_ctrl: Controller hidden dimension (default: d_model)
        dropout: Dropout probability (default: 0.0)
        token_conditioned: If True, α depends on both token x_i and state
        temperature: Softmax temperature for routing (default: 1.0)
        topk: Optional top-k sparsification (default: None)
        use_layernorm: Apply LayerNorm to states (default: True)
        entropy_reg: Entropy regularization coefficient (default: 1e-3)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ctrl: Optional[int] = None,
        dropout: float = 0.0,
        token_conditioned: bool = True,
        temperature: float = 1.0,
        topk: Optional[int] = None,
        use_layernorm: bool = True,
        entropy_reg: float = 1e-3,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ctrl = d_ctrl or d_model
        self.temperature = float(temperature)
        self.token_conditioned = token_conditioned
        self.topk = topk
        self.entropy_reg = entropy_reg
        
        if topk is not None:
            assert 1 <= topk <= n_heads, "topk must be in [1, n_heads]"
        
        # Input projection and pooling
        self.pool_ln = nn.LayerNorm(d_model)
        self.inp_proj = nn.Linear(d_model, self.d_ctrl)
        
        # minGRU: single gate + candidate
        self.W_z = nn.Linear(self.d_ctrl, self.d_ctrl)  # gate
        self.W_h = nn.Linear(self.d_ctrl, self.d_ctrl)  # candidate
        
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
    
    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize zero state."""
        return torch.zeros(batch_size, self.d_ctrl, device=device)
    
    def _pool(self, X: torch.Tensor) -> torch.Tensor:
        """Pool tokens to single vector per batch."""
        Xn = self.pool_ln(X)
        return Xn.mean(dim=1)
    
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
    
    def step(
        self,
        X: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        One refinement step.
        
        Args:
            X: [B, S, d_model] token representations
            state: [B, d_ctrl] previous hidden state (or None to initialize)
            return_aux: If True, return auxiliary metrics
            
        Returns:
            alpha: [B, S, H] routing weights
            state: [B, d_ctrl] updated hidden state
            aux: Dict with metrics
        """
        B, S, D = X.shape
        device = X.device
        
        if state is None:
            state = self.init_state(B, device)
        
        # Pool and project input
        g = self._pool(X)
        x_ctrl = self.inp_proj(self.drop(g))
        
        # minGRU step: z = sigmoid(W_z @ x), h_new = (1-z) * h + z * tanh(W_h @ x)
        z = torch.sigmoid(self.W_z(x_ctrl))
        h_candidate = torch.tanh(self.W_h(x_ctrl))
        h_new = (1 - z) * state + z * h_candidate
        h_new = self.ln_h(h_new)
        
        # Route to heads
        if self.token_conditioned:
            h_tok = h_new[:, None, :].expand(B, S, self.d_ctrl)
            logits = self.router(torch.cat([X, h_tok], dim=-1))
        else:
            logits_global = self.router(h_new)
            logits = logits_global[:, None, :].expand(B, S, self.n_heads)
        
        alpha = F.softmax(logits / self.temperature, dim=-1)
        alpha = self._topk_mask_renorm(alpha)
        
        entropy = self._compute_entropy(alpha)
        
        aux: Dict[str, Any] = {}
        if return_aux:
            aux = {
                "router_logits": logits.detach(),
                "alphas": alpha.detach(),
                "entropy": entropy.detach(),
                "temperature": self.temperature,
            }
        
        return alpha, h_new, aux
    
    def forward(
        self,
        x: torch.Tensor,
        head_outputs: Optional[torch.Tensor] = None,
        *,
        state: Optional[torch.Tensor] = None,
        per_token_pool: str = "mean",
        mask: Optional[torch.Tensor] = None,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Forward pass compatible with HRMPointerController API."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        alpha, state, aux = self.step(x, state, return_aux)
        
        if alpha.size(1) == 1:
            alpha = alpha.squeeze(1)
        
        return alpha, state, aux
