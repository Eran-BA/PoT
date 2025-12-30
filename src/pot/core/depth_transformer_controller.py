"""
Causal Depth Transformer Controller

A Transformer-based controller that operates across the depth axis (refinement iterations),
providing an alternative to GRU-based controllers. Unlike GRUs which only have implicit
access to past states through compressed hidden states, this depth Transformer can
explicitly attend to *any* relevant previous refinement step.

Key Features:
- Causal attention over depth: step t can only see steps 0..t
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


@dataclass
class DepthControllerCache:
    """Holds the depth-sequence inputs u^(0..t) used by the causal depth Transformer."""
    u_list: List[torch.Tensor]  # each: [B, d_ctrl]


class CausalDepthTransformerRouter(nn.Module):
    """
    Causal depth-Transformer controller producing per-token head routing weights alpha.

    This controller operates across the DEPTH axis (refinement iterations), not across
    the input sequence length. Each token maintains its own independent controller state
    that evolves as the model iterates through reasoning steps.

    Architecture:
        ┌─────────────────────────────────────────────────────────────┐
        │  Causal Depth Transformer Controller                        │
        │                                                             │
        │  Input: X^(t) [B, S, d_model] → Pool → u^(t) [B, d_ctrl]   │
        │                                                             │
        │  Depth sequence U^(0:t) → DepthTransformer (causal mask)   │
        │  └── n_ctrl_layers layers, d_ctrl width                    │
        │                                                             │
        │  Output y^(t) → Router → α^(t) [B, S, H] routing weights   │
        └─────────────────────────────────────────────────────────────┘

    API:
        alpha_t, cache = controller.step(X_t, t, cache)

    Inputs:
        X_t: [B, S, d_model]  (token reps at refinement step t)
        t: int               (current refinement index)
        cache: DepthControllerCache or None

    Output:
        alpha_t: [B, S, H]   (routing weights over H attention heads for each token)
        cache: DepthControllerCache (updated)

    Args:
        d_model: Model dimension (token representation size)
        n_heads: Number of attention heads to route over (H)
        d_ctrl: Controller hidden dimension (default: 256)
        n_ctrl_layers: Number of depth-transformer layers (default: 2)
        n_ctrl_heads: Number of attention heads in depth-transformer (default: 4)
        dropout: Dropout probability (default: 0.0)
        max_depth: Maximum refinement steps K (default: 32)
        token_conditioned: If True, α depends on both x_i and depth state (recommended)
        temperature: Softmax temperature for routing (default: 1.0)
        topk: Optional top-k sparsification after softmax (default: None)
        entropy_reg: Entropy regularization coefficient (default: 1e-3)
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

        # Causal depth Transformer over sequence length = (t+1)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_ctrl,
            nhead=n_ctrl_heads,
            dim_feedforward=4 * d_ctrl,
            dropout=dropout,
            activation="gelu",
            batch_first=False,  # expects [T, B, D]
            norm_first=True,    # pre-LN (more stable)
        )
        self.depth_tx = nn.TransformerEncoder(enc_layer, num_layers=n_ctrl_layers)

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
        """Initialize state for async batching (API compatibility with HRM controllers).
        
        Args:
            batch_size: Batch size (not used, cache is batch-agnostic)
            device: Device (not used, tensors are created lazily)
            
        Returns:
            Empty DepthControllerCache
        """
        return self.init_cache()

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for depth Transformer.
        
        Returns: [T, T] bool tensor where True means "masked out" for nn.Transformer
        """
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def _pool(self, X_t: torch.Tensor) -> torch.Tensor:
        """Pool tokens to a single vector per batch.
        
        Default: mean pool (after LayerNorm).
        
        Args:
            X_t: [B, S, d_model] token representations
            
        Returns:
            [B, d_model] pooled representation
        """
        Xn = self.pool_ln(X_t)
        return Xn.mean(dim=1)

    def _topk_mask_renorm(self, alpha: torch.Tensor) -> torch.Tensor:
        """Apply top-k masking and renormalize.
        
        Args:
            alpha: [B, S, H] softmax probabilities
            
        Returns:
            [B, S, H] masked and renormalized probabilities
        """
        k = self.topk
        if k is None or k >= alpha.size(-1):
            return alpha

        # Get top-k indices
        topv, topi = torch.topk(alpha, k=k, dim=-1)  # [B, S, k]
        masked = torch.zeros_like(alpha)
        masked.scatter_(-1, topi, topv)

        # Renormalize (avoid divide-by-zero)
        denom = masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return masked / denom

    def _compute_entropy(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute routing entropy for regularization.
        
        Args:
            alpha: [B, S, H] routing probabilities
            
        Returns:
            Scalar entropy (averaged over batch and tokens)
        """
        return -(alpha * alpha.clamp_min(1e-12).log()).sum(dim=-1).mean()

    def set_temperature(self, T: float):
        """Update softmax temperature (for annealing schedules)."""
        self.temperature = max(0.1, float(T))

    def step(
        self,
        X_t: torch.Tensor,
        t: int,
        cache: Optional[DepthControllerCache] = None,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, DepthControllerCache, Dict[str, Any]]:
        """
        One refinement step: update depth cache, run causal depth Transformer over prefix,
        produce alpha_t over heads for each token.

        Args:
            X_t: [B, S, d_model] token representations at step t
            t: Current refinement step index (0-indexed)
            cache: Previous DepthControllerCache (or None to initialize)
            return_aux: If True, return auxiliary metrics

        Returns:
            alpha_t: [B, S, H] routing weights over heads for each token
            cache: Updated DepthControllerCache
            aux: Dict with 'entropy', 'logits', 'temperature', etc.
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

        # (2) Run causal depth Transformer over U^(0..t)
        # Stack -> [T, B, d_ctrl] (batch_first=False)
        U = torch.stack(cache.u_list, dim=0)
        T_depth = U.size(0)
        attn_mask = self._causal_mask(T_depth, device=U.device)  # [T, T], bool

        Y = self.depth_tx(U, mask=attn_mask)  # [T, B, d_ctrl]
        r_t = Y[-1]                           # [B, d_ctrl] - take last (current) step

        # (3) Produce routing weights alpha_t
        if self.token_conditioned:
            # Expand r_t to match token dimension and concatenate with X_t
            r_tok = r_t[:, None, :].expand(B, S, self.d_ctrl)         # [B, S, d_ctrl]
            logits = self.router(torch.cat([X_t, r_tok], dim=-1))     # [B, S, H]
        else:
            # Global routing: same for all tokens
            logits_global = self.router(r_t)                          # [B, H]
            logits = logits_global[:, None, :].expand(B, S, self.n_heads)

        alpha = F.softmax(logits / self.temperature, dim=-1)          # [B, S, H]
        alpha = self._topk_mask_renorm(alpha)

        # Compute entropy for regularization
        entropy = self._compute_entropy(alpha)

        # Build auxiliary info
        aux: Dict[str, Any] = {}
        if return_aux:
            aux = {
                "router_logits": logits.detach(),
                "alphas": alpha.detach(),
                "entropy": entropy.detach(),
                "temperature": self.temperature,
                "depth_step": t,
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
        """
        Forward pass compatible with HRMPointerController API.

        This is a convenience wrapper around step() for drop-in compatibility
        with existing code that uses the HRM controller interface.

        Args:
            x: Input representation [B, L, d_model] or [B, d_model]
            head_outputs: Precomputed head features (not used, kept for API compat)
            state: Previous DepthControllerCache (or None to initialize)
            step: Current iteration step (default: 0)
            per_token_pool: Pooling method (unused, mean pooling always used)
            mask: Attention mask (unused in this implementation)
            return_aux: Return auxiliary metrics

        Returns:
            alphas: Routing weights [B, S, H] or [B, H] depending on input
            new_state: Updated DepthControllerCache
            aux: Dict with auxiliary metrics
        """
        # Handle 2D input (already pooled)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, d_model]

        alpha, cache, aux = self.step(x, step, state, return_aux)

        # If input was 2D, squeeze output to match
        if alpha.size(1) == 1:
            alpha = alpha.squeeze(1)  # [B, H]

        return alpha, cache, aux
