"""
Production-ready PoH block architecture.

Drop-in replacement for TransformerEncoderLayer with:
- Head-wise routing (soft or top-k)
- Inner-loop refinement
- Optional ACT halting
- Parameter parity with baseline (<1% delta)

Clean hierarchy:
  PoHBlock → PoHStack → IterRefiner

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========== Routing Primitives ==========

def topk_route(scores: torch.Tensor, k: int) -> torch.Tensor:
    """
    Top-k routing mask.
    
    Args:
        scores: [B, T, H] unnormalized routing logits per head
        k: Number of heads to select
    
    Returns:
        One-hot top-k mask [B, T, H]
    """
    if k >= scores.size(-1):  # No routing
        return torch.ones_like(scores)
    
    topk_vals, topk_idx = scores.topk(k, dim=-1)
    mask = torch.zeros_like(scores)
    mask.scatter_(-1, topk_idx, 1.0)
    
    return mask


def soft_route(scores: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Soft routing via temperature-annealed softmax.
    
    Args:
        scores: [B, T, H] routing logits
        temperature: Softmax temperature (lower = sharper)
    
    Returns:
        Routing weights [B, T, H] (sum to 1 over H)
    """
    return F.softmax(scores / max(1e-6, temperature), dim=-1)


# ========== Configuration ==========

@dataclass
class PoHConfig:
    """Configuration for PoH architecture."""
    
    # Model architecture
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    
    # Routing
    route_mode: str = "soft"          # ["soft", "topk"]
    route_topk: int = 2               # For topk mode
    route_temp: float = 1.0           # For soft mode
    share_router: bool = True         # Share router across layers
    
    # ACT halting
    act_halting: bool = False
    act_threshold: float = 0.99
    act_penalty: float = 0.01
    
    # Normalization
    norm_type: str = "pre"            # ["pre", "post"]
    
    # Parameter parity
    param_match_baseline: bool = True  # Keep <1% delta vs baseline
    
    # Positional encoding
    pos_encoding: str = "absolute"    # ["none", "absolute", "rotary"]
    max_seq_len: int = 512            # Used for absolute mode
    
    # GPT/autoregressive mode
    is_causal: bool = False           # Enable causal masking (GPT-style)
    depth: int = 6                    # Number of blocks (for PoHGPT)
    max_inner_iters: int = 1          # REFINEMENT iterations (NOT HRM inner loop!)
                                      # HRM inner loop = f_L (fast), HRM outer loop = f_H (slow)
                                      # This param = how many times to refine the representation
    outer_residual: bool = False      # Residual across refinement steps (not HRM loops)
    rezero_init: bool = False         # ReZero initialization (for PoHGPT)


# ========== Router Head ==========

class HeadRouter(nn.Module):
    """
    Produces per-token, per-head routing logits: [B, T, H].
    
    Lightweight by default to keep param parity with baseline.
    """
    
    def __init__(self, d_model: int, n_heads: int, share_proj: bool = True):
        super().__init__()
        
        # Small hidden dim to minimize params
        hid = max(32, d_model // 4) if share_proj else d_model
        
        self.proj = nn.Sequential(
            nn.Linear(d_model, hid, bias=True),
            nn.ReLU(),
            nn.Linear(hid, n_heads, bias=False),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
        
        Returns:
            Routing logits [B, T, H]
        """
        return self.proj(x)


# ========== PoH Block ==========

class PoHBlock(nn.Module):
    """
    A Transformer-style block with head-wise routing.
    
    Routing scales/filters per-head attention outputs before residual.
    Drop-in replacement for nn.TransformerEncoderLayer.
    """
    
    def __init__(self, cfg: PoHConfig, router: Optional[HeadRouter] = None):
        super().__init__()
        self.cfg = cfg
        
        # Standard components
        self.self_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True
        )
        
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        
        # Router: either shared across stack or per-block
        self.router = router or HeadRouter(cfg.d_model, cfg.n_heads, share_proj=cfg.share_router)
        
        # Gating: one learned scalar per head (minimal params)
        self.head_gain = nn.Parameter(torch.ones(cfg.n_heads))
        
        # Parameter parity: strip bias to keep params tight
        if cfg.param_match_baseline:
            for m in self.ff:
                if isinstance(m, nn.Linear):
                    m.bias = None
    
    def route_mask(self, route_logits: torch.Tensor) -> torch.Tensor:
        """Compute routing weights from logits."""
        if self.cfg.route_mode == "topk":
            return topk_route(route_logits, self.cfg.route_topk)
        return soft_route(route_logits, self.cfg.route_temp)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with head-wise routing.
        
        Args:
            x: [B, T, d_model]
            attn_mask: Optional attention mask
        
        Returns:
            out: [B, T, d_model]
            stats: Dict with routing/entropy statistics
        """
        stats = {}
        
        # Pre/post norm
        y = x
        if self.cfg.norm_type == "pre":
            y = self.norm1(y)
        
        # --- Attention ---
        attn_out, attn_weights = self.self_attn(
            y, y, y,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False  # Get per-head weights [B, H, T, T]
        )
        
        # Routing: compute per-token per-head weights
        route_logits = self.router(y)  # [B, T, H]
        route = self.route_mask(route_logits)  # [B, T, H]
        
        stats["route_logits"] = route_logits
        stats["route"] = route
        
        # Reshape attn_out to heads to apply head-wise gain * route
        B, T, D = attn_out.shape
        H = self.cfg.n_heads
        d_head = D // H
        attn_out_h = attn_out.view(B, T, H, d_head)  # [B, T, H, d_head]
        
        # Scale each head by (learned gain) * (route weight per token, head)
        gain = self.head_gain.view(1, 1, H, 1)  # [1, 1, H, 1]
        routed = attn_out_h * gain * route.unsqueeze(-1)  # Broadcast over d_head
        attn_out = routed.view(B, T, D)
        
        # Residual
        x2 = x + self.dropout(attn_out)
        if self.cfg.norm_type == "post":
            x2 = self.norm1(x2)
        
        # --- Feedforward ---
        y2 = x2
        if self.cfg.norm_type == "pre":
            y2 = self.norm2(y2)
        
        y2 = self.ff(y2)
        out = x2 + self.dropout(y2)
        
        if self.cfg.norm_type == "post":
            out = self.norm2(out)
        
        # Statistics for analysis (no grad)
        with torch.no_grad():
            p = route.clamp_min(1e-12)
            stats["route_entropy_mean"] = float(
                (-(p * p.log()).sum(dim=-1)).mean().item()
            )
            
            attn_p = attn_weights.clamp_min(1e-12)
            stats["attn_entropy_mean"] = float(
                (-(attn_p * attn_p.log()).sum(dim=-1)).mean().item()
            )
        
        return out, stats


# ========== Stack ==========

class PoHStack(nn.Module):
    """
    Stack of PoHBlocks with GPT-style residual chaining.
    
    Each block already has internal residuals (MHA + FFN); this wrapper
    simply feeds output → next block, preserving standard transformer skip semantics.
    
    Includes optional positional encoding (absolute/rotary/none).
    """
    
    def __init__(self, cfg: PoHConfig, depth: int):
        super().__init__()
        self.cfg = cfg
        
        # Positional encoding
        from .positional import PositionalEncoding
        self.pos_encoder = PositionalEncoding(cfg)
        
        # Shared router if configured
        shared_router = (
            HeadRouter(cfg.d_model, cfg.n_heads, share_proj=True)
            if cfg.share_router
            else None
        )
        
        self.blocks = nn.ModuleList([
            PoHBlock(cfg, router=shared_router)
            for _ in range(depth)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Forward through all blocks with GPT-style residual chaining.
        
        Args:
            x: [B, T, d_model]
            attn_mask: Optional attention mask
        
        Returns:
            x: [B, T, d_model]
            stats_all: List of stats dicts (one per block)
        """
        # Apply positional encoding before first block
        x, _ = self.pos_encoder(x)
        
        stats_all = []
        
        for blk in self.blocks:
            # GPT-style: residual already handled inside each block
            x, stats = blk(x, attn_mask=attn_mask)
            stats_all.append(stats)
        
        return x, stats_all


# ========== Iterative Refiner ==========

class IterRefiner(nn.Module):
    """
    Wraps a PoHStack and applies R refinement steps.
    
    TERMINOLOGY CLARIFICATION:
    - "Refinement iterations" (this module) = apply stack R times per forward pass
    - "HRM inner loop" (controller) = f_L updates every step (fast timescale)
    - "HRM outer loop" (controller) = f_H updates every T steps (slow timescale)
    
    Optionally adds residual connections across refinement steps (ReZero-style).
    GPT-style residuals are already present within each block.
    
    This is the top-level module for multi-step refinement.
    """
    
    def __init__(
        self,
        stack: PoHStack,
        max_inner_iters: int = 1,  # R = number of refinement steps
        outer_residual: bool = False,  # Residual across refinement steps
        rezero_init: bool = False,
        act: bool = False,
        threshold: Optional[float] = None,
        penalty: Optional[float] = None,
    ):
        super().__init__()
        self.stack = stack
        self.R = max_inner_iters  # R = refinement steps (avoiding "K" to prevent confusion)
        
        # Outer residual settings
        self.outer_residual = outer_residual
        self.rezero_init = rezero_init
        
        # ReZero-style learnable gain for outer residual
        if self.outer_residual and rezero_init:
            self.alpha = nn.Parameter(torch.zeros(1))  # ReZero: start as identity
        elif self.outer_residual:
            self.alpha = nn.Parameter(torch.ones(1))   # Plain residual scaling
        
        # ACT settings (use stack's config as default)
        self.act = act or stack.cfg.act_halting
        self.threshold = threshold if threshold is not None else stack.cfg.act_threshold
        self.penalty = penalty if penalty is not None else stack.cfg.act_penalty
        
        # ACT halt projection (one scalar per token)
        if self.act:
            self.halt_proj = nn.Linear(stack.cfg.d_model, 1)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_inner_stats: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, any]]]]:
        """
        Apply R refinement iterations (NOT to be confused with HRM inner/outer loops!).
        
        Args:
            x: [B, T, d_model]
            attn_mask: Optional attention mask
            return_inner_stats: Return per-refinement-step statistics
        
        Returns:
            out: [B, T, d_model]
            refinement_stats: List of stats dicts (one per refinement step) if requested
        """
        B, T, D = x.size()
        inner_stats = [] if return_inner_stats else None
        
        # TRM-style constant-input injection: keep a reference to the original input
        x_ref = x

        # --- No ACT: simple R refinement steps ---
        if not self.act:
            h = x
            for t in range(self.R):
                h_prev = h
                # Inject original input each iteration as reference (TRM-style)
                h_in = h + x_ref
                h, stats = self.stack(h_in, attn_mask=attn_mask)
                
                # Optional outer residual across iterations
                if self.outer_residual:
                    # GPT-style within blocks, plus outer residual across iterations
                    h = h_prev + self.alpha * (h - h_prev)  # ReZero-stabilized residual
                
                if return_inner_stats:
                    inner_stats.append(self._pack_stats(stats, t))
            
            return (h, inner_stats) if return_inner_stats else (h, None)
        
        # --- ACT variant: adaptive halting ---
        halting_prob = torch.zeros(B, T, device=x.device)
        rema_prob = torch.ones(B, T, device=x.device)
        weighted_sum = torch.zeros_like(x)
        h = x
        
        for t in range(self.R):
            h_prev = h
            # Inject original input each iteration as reference (TRM-style)
            h_in = h + x_ref
            h, stats = self.stack(h_in, attn_mask=attn_mask)
            
            # Optional outer residual across iterations
            if self.outer_residual:
                h = h_prev + self.alpha * (h - h_prev)
            
            # Halting probability
            p = torch.sigmoid(self.halt_proj(h)).squeeze(-1)  # [B, T]
            
            new_halt = (halting_prob + p * rema_prob >= self.threshold).float() * (rema_prob > 0)
            
            # Portion to add at this step
            add_prob = torch.where(
                new_halt.bool(),
                self.threshold - halting_prob,
                p * rema_prob
            )
            
            halting_prob = halting_prob + add_prob
            rema_prob = rema_prob - add_prob
            
            # Weighted accumulation
            weighted_sum = weighted_sum + (add_prob.unsqueeze(-1) * h)
            
            if return_inner_stats:
                s = self._pack_stats(stats, t)
                s["halted_frac"] = float((halting_prob >= self.threshold).float().mean().item())
                inner_stats.append(s)
            
            # Early stop if everyone halted
            if (halting_prob >= self.threshold).all():
                break
        
        # ACT ponder cost: penalty * expected steps
        ponder_cost = self.penalty * halting_prob.mean()
        
        if return_inner_stats and inner_stats:
            inner_stats[-1]["ponder_cost"] = float(ponder_cost.item())
        
        return (weighted_sum, inner_stats) if return_inner_stats else (weighted_sum, None)
    
    @staticmethod
    def _pack_stats(block_stats_list: List[Dict], t: int) -> Dict[str, any]:
        """Aggregate per-block stats for logging."""
        out = {"inner_step": t + 1}
        
        if not block_stats_list:
            return out
        
        # Average entropy across blocks
        ent_route = [s.get("route_entropy_mean") for s in block_stats_list if "route_entropy_mean" in s]
        ent_attn = [s.get("attn_entropy_mean") for s in block_stats_list if "attn_entropy_mean" in s]
        
        if ent_route:
            out["route_entropy_mean"] = float(sum(ent_route) / len(ent_route))
        if ent_attn:
            out["attn_entropy_mean"] = float(sum(ent_attn) / len(ent_attn))
        
        return out

