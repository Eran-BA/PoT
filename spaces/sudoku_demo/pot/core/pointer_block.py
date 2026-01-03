"""
Generic Pointer-over-Heads block (task-agnostic).

Combines multi-head attention with dynamic routing via HRM controller.

.. deprecated::
    This module is deprecated. Use :class:`src.models.pointer_block.PointerMoHTransformerBlock`
    instead, which correctly applies routing to per-head outputs before projection.

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import warnings

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple

from .hrm_controller import HRMPointerController, HRMState

# Emit deprecation warning on import
warnings.warn(
    "src.pot.core.pointer_block is deprecated. "
    "Use src.models.pointer_block.PointerMoHTransformerBlock instead, "
    "which correctly applies routing to per-head outputs before projection.",
    DeprecationWarning,
    stacklevel=2
)


class PointerBlock(nn.Module):
    """
    Task-agnostic Pointer-over-Heads block.

    .. deprecated::
        This class is deprecated because it does not correctly apply routing weights
        to attention outputs. The routing alphas are computed but never used.
        Use :class:`src.models.pointer_block.PointerMoHTransformerBlock` instead.

    Architecture:
    1. Multi-head self-attention (produces n_heads outputs)
    2. HRM controller routes over heads (produces routing weights)
    3. Mix heads according to routing weights
    4. Residual connection + feedforward

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feedforward dimension (default: 4 * d_model)
        **controller_kwargs: Passed to HRMPointerController
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        **controller_kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff or (4 * d_model)

        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # HRM controller for routing
        self.controller = HRMPointerController(
            d_model=d_model,
            n_heads=n_heads,
            **controller_kwargs
        )

        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, d_model),
            nn.Dropout(dropout)
        )

        # Layer norms
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  # [B, L, d_model]
        mask: Optional[torch.Tensor] = None,  # [B, L]
        state: Optional[HRMState] = None,
        iters: int = 1,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, Optional[HRMState], Dict[str, Any]]:
        """
        Forward pass with iterative refinement.

        Args:
            x: Input [B, L, d_model]
            mask: Attention mask [B, L]
            state: Previous HRMState
            iters: Number of refinement iterations
            return_aux: Return auxiliary outputs

        Returns:
            x: Output [B, L, d_model]
            state: Updated HRMState
            aux: Dict with routing info, entropy, etc.
        """
        B, L, D = x.shape

        # Initialize state
        if state is None:
            state = self.controller.init_state(B, x.device)

        aux_history = []

        for t in range(iters):
            # Multi-head attention
            attn_out, attn_weights = self.attn(
                x, x, x,
                key_padding_mask=(~mask if mask is not None else None),
                need_weights=True,
                average_attn_weights=False  # Get per-head weights
            )

            # Route over heads via HRM controller
            alphas, state, aux = self.controller(
                x=x,
                head_outputs=None,  # Not needed for routing
                state=state,
                mask=mask,
                return_aux=True
            )

            # Mix attention outputs (simplified: weight entire output)
            # In a true per-head mixing, you'd compute separate head outputs
            # For now, use alphas to modulate attention
            x_attn = attn_out  # [B, L, D]

            # Residual + norm
            x = self.ln1(x + self.dropout(x_attn))

            # Feedforward
            x_ff = self.ff(x)
            x = self.ln2(x + x_ff)

            if return_aux:
                aux['iter'] = t
                aux_history.append(aux)

        # Aggregate aux
        final_aux = {}
        if return_aux and aux_history:
            final_aux = {
                'entropy_mean': torch.tensor([a['entropy'] for a in aux_history]).mean().item(),
                'temperature': aux_history[-1]['temperature'],
                'alphas_final': aux_history[-1]['alphas'],
                'all_aux': aux_history
            }

        return x, state, final_aux

