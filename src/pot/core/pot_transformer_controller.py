"""
PoT Depth Transformer Controller (Nested PoT)

A Transformer-based controller that uses GATED multi-head attention internally,
similar to the PoT block. This creates a "nested PoT" architecture where:
- The outer PoT block uses α weights to gate its attention heads
- The inner depth controller ALSO uses gated MHA to process depth history

This is more expressive than the standard CausalDepthTransformerRouter which
uses standard (non-gated) MHA internally.

Key Features:
- Causal attention over depth: step t can only see steps 0..t
- GATED multi-head attention (PoT-style) within the controller
- Pooled controller input for efficiency
- Token-conditioned routing for per-token expressivity
- Optional top-k sparsification

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .depth_transformer_controller import DepthControllerCache


class GatedMHA(nn.Module):
    """
    Gated Multi-Head Attention - PoT-style attention with learnable head gating.
    
    Instead of concatenating head outputs, this module learns α weights
    that control how much each head contributes to the final output.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Output projection (per head, not concatenated)
        self.out_proj = nn.Linear(self.head_dim, d_model)
        
        # Gate network: produces α weights for each head
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_heads),
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with gated head mixing.
        
        Args:
            x: Input tensor [T, B, d_model] or [B, T, d_model]
            mask: Optional attention mask [T, T] where True = masked out
            
        Returns:
            output: Gated attention output [same shape as x]
            gate_weights: α weights used [B, n_heads]
        """
        # Handle different input formats
        if x.dim() == 3 and x.size(0) != x.size(1):
            # Likely [T, B, D] format, convert to [B, T, D]
            x = x.transpose(0, 1)
            was_time_first = True
        else:
            was_time_first = False
            
        B, T, D = x.shape
        H = self.n_heads
        head_dim = self.head_dim
        
        # Compute Q, K, V
        Q = self.q_proj(x).view(B, T, H, head_dim).transpose(1, 2)  # [B, H, T, head_dim]
        K = self.k_proj(x).view(B, T, H, head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, T, T]
        
        # Apply mask if provided
        if mask is not None:
            # mask: [T, T] where True = masked out
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        head_outputs = torch.matmul(attn_weights, V)  # [B, H, T, head_dim]
        
        # Project each head output to d_model
        head_outputs = head_outputs.transpose(1, 2)  # [B, T, H, head_dim]
        head_outputs_proj = self.out_proj(head_outputs)  # [B, T, H, d_model]
        
        # Compute gate weights (α) - pool over time dimension for global gating
        x_pooled = x.mean(dim=1)  # [B, d_model]
        gate_logits = self.gate(x_pooled)  # [B, n_heads]
        gate_weights = F.softmax(gate_logits, dim=-1)  # [B, n_heads]
        
        # Apply gating: weighted sum of head outputs
        # gate_weights: [B, n_heads] -> [B, 1, n_heads, 1]
        gate_weights_exp = gate_weights.unsqueeze(1).unsqueeze(-1) * H  # Scale by H for proper magnitude
        output = (head_outputs_proj * gate_weights_exp).sum(dim=2)  # [B, T, d_model]
        
        if was_time_first:
            output = output.transpose(0, 1)  # Back to [T, B, D]
            
        return output, gate_weights


class PoTTransformerLayer(nn.Module):
    """
    A single PoT-style Transformer layer with gated MHA.
    
    This is similar to a standard TransformerEncoderLayer but uses
    GatedMHA instead of standard MultiheadAttention.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Pre-norm architecture
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Gated MHA
        self.gated_mha = GatedMHA(d_model, n_heads, dropout)
        
        # FFN (SwiGLU-style for better performance)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input [T, B, d_model]
            mask: Causal mask [T, T]
            
        Returns:
            output: Processed tensor [T, B, d_model]
            gate_weights: α weights from attention [B, n_heads]
        """
        # Pre-norm + Gated MHA + residual
        attn_out, gate_weights = self.gated_mha(self.norm1(x), mask)
        x = x + attn_out
        
        # Pre-norm + FFN + residual
        x = x + self.ffn(self.norm2(x))
        
        return x, gate_weights


class PoTDepthTransformerRouter(nn.Module):
    """
    PoT-style Causal Depth Transformer Controller (Nested PoT).
    
    This controller uses GATED multi-head attention internally, creating
    a nested PoT architecture. The controller's own attention heads are
    dynamically weighted, and the output is used to produce α weights
    for the main model's PoT blocks.
    
    Architecture:
        ┌─────────────────────────────────────────────────────────────┐
        │  PoT Depth Transformer Controller (Nested PoT)             │
        │                                                             │
        │  Input: X^(t) [B, S, d_model] → Pool → u^(t) [B, d_ctrl]   │
        │                                                             │
        │  Depth sequence U^(0:t) → PoT Transformer (causal mask)    │
        │  └── n_ctrl_layers of PoTTransformerLayer                  │
        │      └── GatedMHA with learnable head gating (nested PoT)  │
        │                                                             │
        │  Output y^(t) → Router → α^(t) [B, S, H] routing weights   │
        └─────────────────────────────────────────────────────────────┘
    
    This is more expressive than CausalDepthTransformerRouter because
    the internal attention is also gated, allowing the controller to
    learn which of its own heads are important for different depth steps.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ctrl: int = 256,
        n_ctrl_layers: int = 2,
        n_ctrl_heads: int = 4,
        dropout: float = 0.0,
        max_depth: int = 32,
        token_conditioned: bool = True,
        temperature: float = 1.0,
        topk: Optional[int] = None,
        entropy_reg: float = 1e-3,
    ):
        super().__init__()
        assert d_ctrl % n_ctrl_heads == 0, "d_ctrl must be divisible by n_ctrl_heads"
        assert temperature > 0.0, "temperature must be positive"
        if topk is not None:
            assert 1 <= topk <= n_heads, "topk must be in [1, n_heads]"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ctrl = d_ctrl
        self.n_ctrl_layers = n_ctrl_layers
        self.n_ctrl_heads = n_ctrl_heads
        self.temperature = float(temperature)
        self.token_conditioned = bool(token_conditioned)
        self.topk = topk
        self.max_depth = max_depth
        self.entropy_reg = entropy_reg

        # Pool token states -> controller input
        self.pool_ln = nn.LayerNorm(d_model)
        self.to_ctrl = nn.Sequential(
            nn.Linear(d_model, d_ctrl),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ctrl, d_ctrl),
        )

        # Learned depth positional embeddings
        self.depth_pos = nn.Parameter(torch.zeros(max_depth, d_ctrl))
        nn.init.normal_(self.depth_pos, mean=0.0, std=0.02)

        # PoT-style Transformer layers (with gated MHA)
        self.layers = nn.ModuleList([
            PoTTransformerLayer(
                d_model=d_ctrl,
                n_heads=n_ctrl_heads,
                d_ff=4 * d_ctrl,
                dropout=dropout,
            )
            for _ in range(n_ctrl_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_ctrl)

        # Router head logits
        if self.token_conditioned:
            self.router = nn.Sequential(
                nn.LayerNorm(d_model + d_ctrl),
                nn.Linear(d_model + d_ctrl, d_ctrl),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ctrl, n_heads),
            )
        else:
            self.router = nn.Sequential(
                nn.LayerNorm(d_ctrl),
                nn.Linear(d_ctrl, d_ctrl),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ctrl, n_heads),
            )

    def init_cache(self) -> DepthControllerCache:
        """Initialize empty cache for a new forward pass."""
        return DepthControllerCache(u_list=[])

    def init_state(self, batch_size: int, device: torch.device) -> DepthControllerCache:
        """Initialize state for async batching (API compatibility)."""
        return self.init_cache()

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        """Create causal mask where True means masked out."""
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def _pool(self, X_t: torch.Tensor) -> torch.Tensor:
        """Pool tokens to a single vector per batch."""
        Xn = self.pool_ln(X_t)
        return Xn.mean(dim=1)

    def _topk_mask_renorm(self, alpha: torch.Tensor) -> torch.Tensor:
        """Apply top-k masking and renormalize."""
        k = self.topk
        if k is None or k >= alpha.size(-1):
            return alpha

        topv, topi = torch.topk(alpha, k=k, dim=-1)
        masked = torch.zeros_like(alpha)
        masked.scatter_(-1, topi, topv)
        denom = masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return masked / denom

    def _compute_entropy(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute routing entropy for regularization."""
        return -(alpha * alpha.clamp_min(1e-12).log()).sum(dim=-1).mean()

    def set_temperature(self, T: float):
        """Update softmax temperature."""
        self.temperature = max(0.1, float(T))

    def step(
        self,
        X_t: torch.Tensor,
        t: int,
        cache: Optional[DepthControllerCache] = None,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, DepthControllerCache, Dict[str, Any]]:
        """
        One refinement step with nested PoT processing.
        
        Args:
            X_t: [B, S, d_model] token representations at step t
            t: Current refinement step index (0-indexed)
            cache: Previous DepthControllerCache (or None to initialize)
            return_aux: If True, return auxiliary metrics

        Returns:
            alpha_t: [B, S, H] routing weights over heads for each token
            cache: Updated DepthControllerCache
            aux: Dict with auxiliary metrics including internal gate weights
        """
        if t < 0 or t >= self.max_depth:
            raise ValueError(
                f"t={t} out of range for max_depth={self.max_depth}. "
                "Increase max_depth in controller init."
            )
        
        if cache is None:
            cache = self.init_cache()

        B, S, D = X_t.shape
        assert D == self.d_model, f"Expected d_model={self.d_model}, got {D}"

        # (1) Build controller input u^(t) from pooled token states
        g_t = self._pool(X_t)                            # [B, d_model]
        u_t = self.to_ctrl(g_t) + self.depth_pos[t]      # [B, d_ctrl]
        cache.u_list.append(u_t)

        # (2) Run PoT-style depth Transformer over U^(0..t)
        U = torch.stack(cache.u_list, dim=0)  # [T, B, d_ctrl]
        T_depth = U.size(0)
        attn_mask = self._causal_mask(T_depth, device=U.device)

        # Process through PoT layers (with gated MHA)
        all_gate_weights = []
        x = U
        for layer in self.layers:
            x, gate_weights = layer(x, attn_mask)
            all_gate_weights.append(gate_weights)
        
        x = self.final_norm(x)
        r_t = x[-1]  # [B, d_ctrl] - take last (current) step

        # (3) Produce routing weights alpha_t
        if self.token_conditioned:
            r_tok = r_t[:, None, :].expand(B, S, self.d_ctrl)
            logits = self.router(torch.cat([X_t, r_tok], dim=-1))
        else:
            logits_global = self.router(r_t)
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
                "depth_step": t,
                "internal_gate_weights": [gw.detach() for gw in all_gate_weights],
                "features": r_t,  # [B, d_ctrl] - injectable controller feature
            }

        return alpha, cache, aux

    def forward(
        self,
        x: torch.Tensor,
        head_outputs: Optional[torch.Tensor] = None,
        *,
        state: Optional[DepthControllerCache] = None,
        step: int = 0,
        per_token_pool: str = "mean",
        mask: Optional[torch.Tensor] = None,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, DepthControllerCache, Dict[str, Any]]:
        """Forward pass compatible with HRMPointerController API."""
        if x.dim() == 2:
            x = x.unsqueeze(1)

        alpha, cache, aux = self.step(x, step, state, return_aux)

        if alpha.size(1) == 1:
            alpha = alpha.squeeze(1)

        return alpha, cache, aux
