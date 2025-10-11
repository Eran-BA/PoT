"""
Pointer-over-Heads Transformer Block with Adaptive Halting.

This module implements the core PoH transformer block, which dynamically
routes over attention heads using a pointer-style controller. Supports
multiple halting modes, routing strategies, and gradient computation modes.

Classes:
    PointerMoHTransformerBlock: Main transformer block with head routing

Key Features:
    - Adaptive halting: fixed, entropy-based, or ACT-style learned halting
    - Routing strategies: soft mixture or hard top-k selection
    - Gradient modes: full BPTT or HRM-style last-iterate
    - Deep supervision support for iterative refinement
    - TRM-style outer supervision steps

Example:
    >>> from src.models.pointer_block import PointerMoHTransformerBlock
    >>> block = PointerMoHTransformerBlock(
    ...     d_model=768,
    ...     n_heads=8,
    ...     d_ff=2048,
    ...     halting_mode="entropy",
    ...     max_inner_iters=3
    ... )
    >>> x = torch.randn(2, 10, 768)
    >>> y, aux = block(x)
    >>> print(f"Used {aux['inner_iters_used']} iterations")

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

from typing import Optional, Tuple, Dict
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import (
    MultiHeadSelfAttention,
    PointerOverHeadsController,
    entropy_from_logits,
    gumbel_softmax_topk
)


class PointerMoHTransformerBlock(nn.Module):
    """Pointer-over-Heads Transformer block with adaptive inner-loop halting.
    
    This block performs iterative refinement by routing over multiple attention
    heads using a learnable controller. Unlike standard transformers that use all
    heads equally, PoH dynamically selects which heads to attend to at each step.
    
    The block supports three halting modes:
    - 'fixed': Run exactly max_inner_iters iterations
    - 'entropy': Stop early when routing entropy drops below threshold
    - 'halting': Learned ACT-style halting with ponder cost
    
    And two combination modes for routing:
    - 'mask_concat': Scale each head by routing weight, concat, project
    - 'mixture': Compute convex combination of heads directly
    
    Args:
        d_model: Model hidden dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward network hidden dimension
        attn_dropout: Dropout probability for attention weights
        ff_dropout: Dropout probability for feed-forward layers
        use_pre_norm: If True, use pre-LayerNorm (recommended)
        routing_tau: Temperature for Gumbel-Softmax routing
        routing_topk: Number of heads to select (0 = soft routing)
        controller_recurrent: If True, controller uses head feedback
        controller_summary: How to summarize heads ('mean' or 'max')
        combination: Head combination mode ('mask_concat' or 'mixture')
        halting_mode: Stopping criterion ('fixed', 'entropy', or 'halting')
        max_inner_iters: Maximum number of inner iterations
        min_inner_iters: Minimum number of inner iterations
        ent_threshold: Entropy threshold for early stopping (entropy mode)
        ponder_coef: Coefficient for ponder cost (halting mode)
        grad_mode: Gradient mode ('full' for full BPTT, 'last' for HRM-style)
        
    Attributes:
        mha: Multi-head self-attention layer
        controller: Pointer controller for head routing
        ff: Feed-forward network
        halt_head: Optional ACT-style halting head
        ln1, ln2: LayerNorm layers
        
    Example:
        >>> # Basic fixed-iteration block
        >>> block = PointerMoHTransformerBlock(
        ...     d_model=768, n_heads=8, d_ff=2048,
        ...     halting_mode="fixed", max_inner_iters=2
        ... )
        
        >>> # Entropy-based adaptive halting with hard top-2 routing
        >>> block = PointerMoHTransformerBlock(
        ...     d_model=768, n_heads=8, d_ff=2048,
        ...     halting_mode="entropy", ent_threshold=0.7,
        ...     routing_topk=2  # Hard top-2 selection
        ... )
        
        >>> # ACT-style learned halting with soft routing
        >>> block = PointerMoHTransformerBlock(
        ...     d_model=768, n_heads=8, d_ff=2048,
        ...     halting_mode="halting", ponder_coef=0.001,
        ...     routing_topk=0  # Soft mixture
        ... )
        
        >>> # HRM-style last-iterate gradients (memory efficient)
        >>> block = PointerMoHTransformerBlock(
        ...     d_model=768, n_heads=8, d_ff=2048,
        ...     grad_mode="last"  # Only backprop through last iteration
        ... )
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        *,
        # Attention & FFN
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        use_pre_norm: bool = True,
        # Routing
        routing_tau: float = 0.7,
        routing_topk: int = 0,  # 0 = soft; >0 = hard top-k
        controller_recurrent: bool = True,
        controller_summary: str = "mean",
        combination: str = "mask_concat",  # 'mask_concat' | 'mixture'
        # Halting
        halting_mode: str = "fixed",  # 'fixed' | 'entropy' | 'halting'
        max_inner_iters: int = 2,
        min_inner_iters: int = 1,
        ent_threshold: float = 0.7,  # For 'entropy'
        ponder_coef: float = 0.001,  # For 'halting'
        grad_mode: str = "full",  # 'full' = full BPTT | 'last' = HRM-style
    ):
        super().__init__()
        assert combination in ("mask_concat", "mixture"), \
            f"combination must be 'mask_concat' or 'mixture', got {combination}"
        assert halting_mode in ("fixed", "entropy", "halting"), \
            f"halting_mode must be 'fixed', 'entropy', or 'halting', got {halting_mode}"
        assert grad_mode in ("full", "last"), \
            f"grad_mode must be 'full' or 'last', got {grad_mode}"

        self.d_model = d_model
        self.n_heads = n_heads
        self.use_pre_norm = use_pre_norm

        # Routing configuration
        self.routing_tau = routing_tau
        self.routing_topk = routing_topk
        self.combination = combination

        # Halting configuration
        self.halting_mode = halting_mode
        self.max_inner_iters = max(1, max_inner_iters)
        self.min_inner_iters = max(1, min_inner_iters)
        self.ent_threshold = ent_threshold
        self.ponder_coef = ponder_coef
        self.grad_mode = grad_mode

        # Core components
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.mha = MultiHeadSelfAttention(d_model, n_heads, attn_dropout=attn_dropout)
        self.controller = PointerOverHeadsController(
            d_model=d_model,
            n_heads=n_heads,
            recurrent=(controller_recurrent or self.max_inner_iters > 1),
            summary=controller_summary,
        )

        # Optional halting head for ACT-style learned halting
        if self.halting_mode == "halting":
            # Per-sequence halting: output shape [B, 1]
            self.halt_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 1)  # Logit (sigmoid applied later)
            )

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(ff_dropout),
        )

        # Small projection for 'mixture' combination mode
        self._mixture_proj = nn.Linear(d_model // n_heads, d_model)

    def _route(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert routing logits to weights (soft or hard top-k).
        
        Args:
            logits: Routing logits [B, T, H]
            
        Returns:
            Routing weights [B, T, H] (probabilities summing to 1)
        """
        if self.routing_topk and self.routing_topk > 0:
            # Hard top-k routing with straight-through gradient
            return gumbel_softmax_topk(
                logits, tau=self.routing_tau, topk=self.routing_topk, hard=True
            )
        # Soft routing (weighted mixture of all heads)
        return F.softmax(logits / max(self.routing_tau, 1e-6), dim=-1)

    def _combine_heads(
        self,
        heads_out: torch.Tensor,
        alphas: torch.Tensor
    ) -> torch.Tensor:
        """Combine per-head outputs using routing weights.
        
        Args:
            heads_out: Per-head outputs [B, T, H, Dh]
            alphas: Routing weights [B, T, H]
            
        Returns:
            Combined output [B, T, D]
            
        Note:
            Two combination modes:
            - 'mask_concat': Scale heads by weights, concat, project (like vanilla MHA)
            - 'mixture': Compute weighted average of heads directly (more efficient)
        """
        B, T, H, Dh = heads_out.shape
        
        if self.combination == "mask_concat":
            # Scale each head by its routing weight
            scaled = heads_out * alphas.unsqueeze(-1)  # [B, T, H, Dh]
            # Concatenate and project (uses MHA output projection)
            concat = scaled.reshape(B, T, H * Dh)  # [B, T, D]
            return self.mha.out(concat)
        else:  # 'mixture'
            # Compute convex combination of heads directly
            mixed = torch.einsum("bthd,bth->btd", heads_out, alphas)  # [B, T, Dh]
            return self._mixture_proj(mixed)  # [B, T, D]

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_aux: bool = True,
        collect_all: bool = False,  # For deep supervision
        return_final_z: bool = False  # For TRM-style outer supervision
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with iterative head routing and adaptive halting.
        
        The block performs up to max_inner_iters iterations of:
        1. Compute attention with all heads
        2. Use controller to decide routing weights
        3. Combine heads according to routing
        4. Check halting criterion
        
        Args:
            x: Input representations [B, T, D]
            attn_mask: Optional attention mask [B, H, T, T]
            return_aux: Whether to return auxiliary information
            collect_all: If True, collect all iteration states (for deep supervision).
                        Disables early stopping to maintain full gradient graph.
            return_final_z: If True, return final latent before residual/FFN
                          (for TRM-style outer supervision steps)
                          
        Returns:
            Tuple containing:
            - y: Output representations [B, T, D]
            - aux: Dictionary with auxiliary information:
                - alphas: Routing weights per iteration [B, iters, T, H]
                - logits: Routing logits per iteration [B, iters, T, H]
                - attn_probs: Attention probabilities [B, H, T, T] (last iteration)
                - inner_iters_used: Number of iterations executed
                - (if collect_all) routed: Per-iteration states [B, iters, T, D]
                - (if collect_all & halting) halt_logits: Halting logits [B, iters]
                - (if return_final_z) z_final: Final latent [B, T, D]
                - (if halting mode) ponder_cost: ACT-style ponder cost
                
        Example:
            >>> block = PointerMoHTransformerBlock(d_model=768, n_heads=8, d_ff=2048)
            >>> x = torch.randn(2, 10, 768)
            
            >>> # Standard forward pass
            >>> y, aux = block(x)
            >>> print(f"Iterations used: {aux['inner_iters_used']}")
            
            >>> # Deep supervision mode (collects all intermediate states)
            >>> y, aux = block(x, collect_all=True)
            >>> print(f"Routed states shape: {aux['routed'].shape}")  # [2, iters, 10, 768]
            
            >>> # TRM-style outer supervision
            >>> y, aux = block(x, return_final_z=True)
            >>> z_final = aux['z_final']  # Use for next supervision step
        """
        aux: Dict[str, torch.Tensor] = {}

        # Pre-normalization (if enabled)
        h = self.ln1(x) if self.use_pre_norm else x
        token_ctx = h

        # Iteration tracking
        it_used = 0
        ponder_cost = torch.zeros((), device=x.device)
        
        # History tracking
        alphas_hist, logits_hist = [], []
        routed_hist = []  # For deep supervision
        halt_logits_hist = []  # For ACT-style differentiable halting
        last_attn = None

        # Run iterations
        max_iters = self.max_inner_iters

        for it in range(max_iters):
            # 1. Compute attention with all heads
            heads_out, last_attn = self.mha(token_ctx, attn_mask)  # [B,T,H,Dh], [B,H,T,T]
            
            # 2. Controller decides routing over heads
            logits = self.controller(token_ctx, provisional_heads=heads_out)  # [B,T,H]
            alphas = self._route(logits)  # [B,T,H]
            
            # 3. Combine heads according to routing
            routed = self._combine_heads(heads_out, alphas)  # [B,T,D]

            # Track history
            alphas_hist.append(alphas)
            logits_hist.append(logits)
            routed_hist.append(routed)
            
            # 4. Prepare context for next iteration
            token_ctx_next = routed
            
            # HRM-style gradient mode: cut gradient graph between iterations
            # This keeps memory constant by only backpropagating through the last iteration
            if self.grad_mode == "last" and it < max_iters - 1:
                token_ctx_next = token_ctx_next.detach()  # Stop gradients here
            
            token_ctx = token_ctx_next
            it_used = it + 1

            # Collect halting logits for differentiable ACT (if in collect_all mode)
            if self.halting_mode == "halting" and collect_all:
                halt_logit = self.halt_head(token_ctx).mean(dim=1)  # [B, 1] -> [B]
                halt_logits_hist.append(halt_logit)

            # 5. Check halting criterion (skip if collect_all=True for gradient flow)
            if not collect_all:
                # Respect minimum iterations
                if it + 1 < self.min_inner_iters:
                    continue

                # Check halting mode
                if self.halting_mode == "fixed":
                    pass  # Always run full max_inner_iters
                    
                elif self.halting_mode == "entropy":
                    # Stop when routing becomes confident (low entropy)
                    with torch.no_grad():
                        ent = entropy_from_logits(logits)  # Scalar
                    if ent.item() < self.ent_threshold:
                        break
                        
                elif self.halting_mode == "halting":
                    # ACT-style learned halting
                    p_halt = self.halt_head(token_ctx).mean()  # Scalar probability
                    ponder_cost = ponder_cost + p_halt
                    if p_halt.item() > 0.5:  # Threshold (can be tuned)
                        break

        # Save final latent BEFORE residual/FFN (for TRM outer supervision)
        z_final = token_ctx
        
        # Apply residual connection for attention
        attn_out = token_ctx + (x if self.use_pre_norm else 0.0)
        if not self.use_pre_norm:
            attn_out = self.ln1(attn_out)

        # Feed-forward network with residual
        y_in = self.ln2(attn_out) if self.use_pre_norm else attn_out
        y = attn_out + self.ff(y_in)
        if not self.use_pre_norm:
            y = self.ln2(y)

        # Collect auxiliary outputs
        if return_aux:
            aux["alphas"] = torch.stack(alphas_hist, dim=1)  # [B, iters, T, H]
            aux["logits"] = torch.stack(logits_hist, dim=1)  # [B, iters, T, H]
            aux["attn_probs"] = last_attn  # [B, H, T, T]
            aux["inner_iters_used"] = torch.tensor(it_used, device=x.device)
            
            # Per-iteration routed states for deep supervision
            if collect_all and routed_hist:
                aux["routed"] = torch.stack(routed_hist, dim=1)  # [B, iters, T, D]
            
            # Halting logits for differentiable ACT
            if collect_all and halt_logits_hist:
                aux["halt_logits"] = torch.stack(halt_logits_hist, dim=1)  # [B, iters]
            
            # Final latent for TRM-style outer supervision
            if return_final_z:
                aux["z_final"] = z_final  # [B, T, D]
            
            # Ponder cost for ACT halting mode
            if self.halting_mode == "halting" and not collect_all:
                aux["ponder_cost"] = self.ponder_coef * ponder_cost
                
        return y, aux

