"""
Swin Transformer-Style Hierarchical Depth Controller

A general-purpose depth controller inspired by Swin Transformer, using:
- Local window attention for efficiency
- Shifted windows for cross-window connectivity
- Hierarchical patch merging for multi-scale reasoning
- Depth-wise evolution tracking

This controller is more general than task-specific region pooling because
it doesn't require prior knowledge of the task structure.

Key Features:
- O(S) complexity per window instead of O(S²) global attention
- Shifted windows enable information flow between local regions
- Hierarchical stages progressively reduce resolution
- Compatible with existing PoT controller APIs

References:
- Liu et al., "Swin Transformer: Hierarchical Vision Transformer using 
  Shifted Windows", ICCV 2021

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# Cache for Depth State
# =============================================================================

@dataclass
class SwinDepthCache:
    """
    Cache for Swin depth controller state across refinement steps.
    
    Stores per-stage feature pyramids to enable depth-wise attention.
    """
    # Per-stage feature lists: stage_features[stage_idx][depth_step] = [B, S_stage, D_stage]
    stage_features: List[List[torch.Tensor]] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.stage_features:
            self.stage_features = []


# =============================================================================
# Window Utilities
# =============================================================================

def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, int, int]:
    """
    Partition input into non-overlapping windows.
    
    Args:
        x: [B, H, W, C] input tensor
        window_size: Size of each window (assumes square windows)
        
    Returns:
        windows: [B * num_windows, window_size, window_size, C]
        H: Original height
        W: Original width
    """
    B, H, W, C = x.shape
    
    # Pad if needed to be divisible by window_size
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    
    Hp, Wp = H + pad_h, W + pad_w
    
    # Reshape to windows: [B, H//ws, ws, W//ws, ws, C]
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    
    # Permute to [B, H//ws, W//ws, ws, ws, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    
    # Flatten to [B * num_windows, ws, ws, C]
    num_windows = (Hp // window_size) * (Wp // window_size)
    windows = x.view(B * num_windows, window_size, window_size, C)
    
    return windows, Hp, Wp


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int, 
                   Hp: int, Wp: int) -> torch.Tensor:
    """
    Reverse window partition back to original shape.
    
    Args:
        windows: [B * num_windows, window_size, window_size, C]
        window_size: Size of each window
        H: Original height (before padding)
        W: Original width (before padding)
        Hp: Padded height
        Wp: Padded width
        
    Returns:
        x: [B, H, W, C]
    """
    B = windows.shape[0] // ((Hp // window_size) * (Wp // window_size))
    
    # Reshape to [B, H//ws, W//ws, ws, ws, C]
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    
    # Permute to [B, H//ws, ws, W//ws, ws, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    
    # Reshape to [B, Hp, Wp, C]
    x = x.view(B, Hp, Wp, -1)
    
    # Remove padding
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    
    return x


# =============================================================================
# Window Attention
# =============================================================================

class WindowAttention(nn.Module):
    """
    Window-based Multi-head Self-Attention with relative position bias.
    
    Computes attention within local windows, with learnable relative
    position bias to encode spatial relationships.
    
    Args:
        dim: Input dimension
        window_size: Size of attention window (assumes square)
        num_heads: Number of attention heads
        qkv_bias: If True, add bias to QKV projections
        attn_drop: Attention dropout probability
        proj_drop: Output projection dropout probability
    """
    
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Relative position bias table
        # (2*window_size-1) x (2*window_size-1) possible relative positions
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        # Compute relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # [2, ws, ws]
        coords_flatten = coords.flatten(1)  # [2, ws*ws]
        
        # Relative coords: [2, ws*ws, ws*ws]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [ws*ws, ws*ws, 2]
        
        # Shift to start from 0
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        
        # Multiply by (2*window_size - 1) to get unique index
        relative_coords[:, :, 0] *= 2 * window_size - 1
        
        # Sum to get single index
        relative_position_index = relative_coords.sum(-1)  # [ws*ws, ws*ws]
        self.register_buffer("relative_position_index", relative_position_index)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [B * num_windows, window_size * window_size, C]
            mask: [num_windows, ws*ws, ws*ws] attention mask for shifted windows
            
        Returns:
            [B * num_windows, window_size * window_size, C]
        """
        B_, N, C = x.shape  # B_ = B * num_windows, N = ws*ws
        
        # QKV projection: [B_, N, 3*C] -> [B_, N, 3, num_heads, head_dim]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B_, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # [B_, num_heads, N, N]
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)  # [N, N, num_heads]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # Apply mask for shifted windows
        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(B_ // num_windows, num_windows, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


# =============================================================================
# Swin Block
# =============================================================================

class SwinBlock(nn.Module):
    """
    Swin Transformer Block with window attention and optional shifting.
    
    Each block consists of:
    1. (Shifted) Window Multi-head Self-Attention
    2. LayerNorm
    3. MLP (Feed-Forward Network)
    4. LayerNorm
    
    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        window_size: Size of attention window
        shift_size: Shift size for shifted window attention (0 = no shift)
        mlp_ratio: MLP hidden dim = dim * mlp_ratio
        qkv_bias: If True, add bias to QKV projections
        drop: Dropout probability
        attn_drop: Attention dropout probability
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        assert 0 <= self.shift_size < self.window_size, "shift_size must be < window_size"
        
        # Attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        
        # MLP
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )
        
        # Attention mask for shifted windows (computed dynamically)
        self.register_buffer("attn_mask", None, persistent=False)
    
    def _compute_attn_mask(self, H: int, W: int, device: torch.device) -> Optional[torch.Tensor]:
        """Compute attention mask for shifted window attention."""
        if self.shift_size == 0:
            return None
        
        # Calculate padded dimensions
        Hp = int(math.ceil(H / self.window_size)) * self.window_size
        Wp = int(math.ceil(W / self.window_size)) * self.window_size
        
        # Create mask image
        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
        
        # Slice indices for different regions
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        
        # Assign different values to different regions
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        
        # Partition into windows
        mask_windows, _, _ = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        
        # Compute attention mask
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        
        return attn_mask
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [B, H*W, C] flattened spatial input
            H: Height of spatial grid
            W: Width of spatial grid
            
        Returns:
            [B, H*W, C] output with same shape
        """
        B, L, C = x.shape
        assert L == H * W, f"Input size mismatch: L={L}, H*W={H*W}"
        
        shortcut = x
        x = self.norm1(x)
        
        # Reshape to spatial grid
        x = x.view(B, H, W, C)
        
        # Cyclic shift for shifted window attention
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        
        # Partition into windows
        x_windows, Hp, Wp = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # Compute attention mask if needed
        attn_mask = self._compute_attn_mask(H, W, x.device)
        
        # Window attention
        attn_windows = self.attn(x_windows, mask=attn_mask)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W, Hp, Wp)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        
        # Flatten back and residual
        x = x.view(B, H * W, C)
        x = shortcut + x
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


# =============================================================================
# Patch Merging (Downsampling)
# =============================================================================

class PatchMerging(nn.Module):
    """
    Patch Merging Layer for hierarchical downsampling.
    
    Concatenates 2x2 neighboring patches and projects to 2*dim.
    This reduces spatial resolution by 2x while doubling channel dimension.
    
    Args:
        dim: Input dimension
        norm_layer: Normalization layer class
    """
    
    def __init__(self, dim: int, norm_layer: type = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        """
        Forward pass.
        
        Args:
            x: [B, H*W, C] input
            H: Height
            W: Width
            
        Returns:
            x: [B, H//2 * W//2, 2*C] downsampled output
            new_H: H // 2
            new_W: W // 2
        """
        B, L, C = x.shape
        assert L == H * W, f"Input size mismatch: L={L}, H*W={H*W}"
        
        # Pad if needed
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = x.view(B, H, W, C)
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w
            x = x.view(B, H * W, C)
        
        x = x.view(B, H, W, C)
        
        # Extract 2x2 patches
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        
        # Concatenate
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2 * W/2, 4*C]
        
        # Normalize and project
        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2 * W/2, 2*C]
        
        return x, H // 2, W // 2


# =============================================================================
# Patch Expanding (Upsampling)
# =============================================================================

class PatchExpanding(nn.Module):
    """
    Patch Expanding Layer for hierarchical upsampling.
    
    Expands spatial resolution by 2x while halving channel dimension.
    
    Args:
        dim: Input dimension
        norm_layer: Normalization layer class
    """
    
    def __init__(self, dim: int, norm_layer: type = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        """
        Forward pass.
        
        Args:
            x: [B, H*W, C] input
            H: Height
            W: Width
            
        Returns:
            x: [B, 2H * 2W, C//2] upsampled output
            new_H: 2 * H
            new_W: 2 * W
        """
        B, L, C = x.shape
        assert L == H * W, f"Input size mismatch: L={L}, H*W={H*W}"
        
        x = self.expand(x)  # [B, H*W, 2*C]
        x = x.view(B, H, W, 2 * C)
        
        # Rearrange to [B, 2H, 2W, C//2]
        x = x.view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, 2 * H, 2 * W, C // 2)
        x = x.view(B, -1, C // 2)
        
        x = self.norm(x)
        
        return x, 2 * H, 2 * W


# =============================================================================
# Swin Stage
# =============================================================================

class SwinStage(nn.Module):
    """
    A single Swin Transformer stage containing multiple blocks.
    
    Each stage has a configurable number of Swin blocks, alternating
    between regular and shifted window attention.
    
    Args:
        dim: Input dimension
        depth: Number of Swin blocks in this stage
        num_heads: Number of attention heads
        window_size: Size of attention window
        mlp_ratio: MLP expansion ratio
        qkv_bias: If True, add bias to QKV projections
        drop: Dropout probability
        attn_drop: Attention dropout probability
        downsample: If True, apply PatchMerging at the end
    """
    
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        downsample: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        
        # Build blocks alternating between regular and shifted attention
        self.blocks = nn.ModuleList([
            SwinBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
            )
            for i in range(depth)
        ])
        
        # Downsampling
        self.downsample = PatchMerging(dim) if downsample else None
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        """
        Forward pass.
        
        Args:
            x: [B, H*W, C] input
            H: Height
            W: Width
            
        Returns:
            x: Output tensor
            new_H: Output height
            new_W: Output width
        """
        for block in self.blocks:
            x = block(x, H, W)
        
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        
        return x, H, W


# =============================================================================
# Swin Depth Controller
# =============================================================================

class SwinDepthController(nn.Module):
    """
    Swin Transformer-style Hierarchical Depth Controller.
    
    A general-purpose controller that uses local window attention with
    shifting for cross-window connectivity, and hierarchical stages for
    multi-scale reasoning.
    
    Architecture:
        Input X^(t) [B, S, d_model]
            ↓
        Reshape to 2D grid [B, H, W, d_model]
            ↓
        Stage 1: Window Attention + Shifted Window Attention
            ↓ (PatchMerging: 2x downsample)
        Stage 2: Window Attention + Shifted Window Attention
            ↓ (PatchMerging: 2x downsample)
        Stage 3: Window Attention
            ↓
        Global Pool → Depth Transformer (causal over steps 0..t)
            ↓
        Upsample + Router → α [B, S, H]
    
    Args:
        d_model: Input token dimension
        n_heads: Number of routing heads (output)
        d_ctrl: Controller dimension (default: d_model)
        window_size: Size of attention windows (default: 7)
        n_stages: Number of hierarchical stages (default: 3)
        stage_depths: Number of blocks per stage (default: [2, 2, 2])
        n_ctrl_heads: Number of attention heads per stage (default: [4, 8, 16])
        mlp_ratio: MLP expansion ratio (default: 4.0)
        dropout: Dropout probability (default: 0.0)
        max_depth: Maximum refinement steps (default: 32)
        token_conditioned: If True, routing varies per token (default: True)
        temperature: Softmax temperature (default: 1.0)
        topk: Optional top-k sparsification (default: None)
        entropy_reg: Entropy regularization coefficient (default: 1e-3)
        depth_skip: If True, add residual skip connection across depth iterations.
            This helps gradient flow and preserves information from each refinement
            step directly. Recommended for iterative refinement tasks. (default: True)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ctrl: Optional[int] = None,
        window_size: int = 7,
        n_stages: int = 3,
        stage_depths: Optional[List[int]] = None,
        n_ctrl_heads: Optional[List[int]] = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_depth: int = 32,
        token_conditioned: bool = True,
        temperature: float = 1.0,
        topk: Optional[int] = None,
        entropy_reg: float = 1e-3,
        depth_skip: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ctrl = d_ctrl or d_model
        self.window_size = window_size
        self.n_stages = n_stages
        self.max_depth = max_depth
        self.token_conditioned = token_conditioned
        self.temperature = float(temperature)
        self.topk = topk
        self.entropy_reg = entropy_reg
        self.depth_skip = depth_skip
        
        # Default stage depths
        if stage_depths is None:
            stage_depths = [2] * n_stages
        assert len(stage_depths) == n_stages, "stage_depths must match n_stages"
        self.stage_depths = stage_depths
        
        # Default attention heads per stage (doubling at each stage)
        if n_ctrl_heads is None:
            n_ctrl_heads = [min(4 * (2 ** i), 32) for i in range(n_stages)]
        assert len(n_ctrl_heads) == n_stages, "n_ctrl_heads must match n_stages"
        self.n_ctrl_heads = n_ctrl_heads
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.d_ctrl),
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        dim = self.d_ctrl
        for i in range(n_stages):
            stage = SwinStage(
                dim=dim,
                depth=stage_depths[i],
                num_heads=n_ctrl_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=dropout,
                attn_drop=dropout,
                downsample=(i < n_stages - 1),  # No downsample on last stage
            )
            self.stages.append(stage)
            if i < n_stages - 1:
                dim *= 2  # Dimension doubles after each downsample
        
        self.final_dim = dim
        
        # Depth transformer for cross-depth attention
        self.depth_proj = nn.Linear(self.final_dim, self.d_ctrl)
        self.depth_pos = nn.Parameter(torch.zeros(max_depth, self.d_ctrl))
        nn.init.normal_(self.depth_pos, std=0.02)
        
        depth_layer = nn.TransformerEncoderLayer(
            d_model=self.d_ctrl,
            nhead=4,
            dim_feedforward=4 * self.d_ctrl,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
            norm_first=True,
        )
        self.depth_transformer = nn.TransformerEncoder(depth_layer, num_layers=2)
        
        # Router
        self.final_norm = nn.LayerNorm(self.d_ctrl)
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
    
    def init_cache(self) -> SwinDepthCache:
        """Initialize empty cache."""
        return SwinDepthCache(stage_features=[[] for _ in range(self.n_stages)])
    
    def init_state(self, batch_size: int, device: torch.device) -> SwinDepthCache:
        """Initialize state (API compatibility)."""
        return self.init_cache()
    
    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for depth attention."""
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
    
    def _hierarchical_causal_mask(
        self, T_depth: int, S_reduced: int, device: torch.device
    ) -> torch.Tensor:
        """
        Create hierarchical causal mask for depth attention with spatial positions.
        
        Position (t, s) can attend to position (t', s') if t' <= t.
        This allows all spatial positions at step t to see all spatial
        positions at earlier steps, enabling spatial-temporal reasoning.
        
        Args:
            T_depth: Number of depth steps
            S_reduced: Number of spatial positions after Swin stages
            device: Torch device
            
        Returns:
            [T*S, T*S] bool mask where True means "masked out"
        """
        total_len = T_depth * S_reduced
        
        # Create block-wise causal structure
        # Each block of S_reduced positions corresponds to one depth step
        mask = torch.ones(total_len, total_len, device=device, dtype=torch.bool)
        
        for t in range(T_depth):
            # Positions at depth t can attend to all positions at depths <= t
            row_start = t * S_reduced
            row_end = (t + 1) * S_reduced
            col_end = (t + 1) * S_reduced  # Can see up to and including current step
            
            mask[row_start:row_end, :col_end] = False
        
        return mask
    
    def _infer_grid_size(self, S: int) -> Tuple[int, int]:
        """
        Infer 2D grid size from sequence length.
        
        Tries to find the most square-like factorization.
        """
        sqrt_s = int(math.sqrt(S))
        for h in range(sqrt_s, 0, -1):
            if S % h == 0:
                return h, S // h
        return 1, S  # Fallback to 1D
    
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
        """Compute routing entropy for regularization."""
        return -(alpha * alpha.clamp_min(1e-12).log()).sum(dim=-1).mean()
    
    def set_temperature(self, T: float):
        """Update softmax temperature."""
        self.temperature = max(0.1, float(T))
    
    def step(
        self,
        X_t: torch.Tensor,
        t: int,
        cache: Optional[SwinDepthCache] = None,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, SwinDepthCache, Dict[str, Any]]:
        """
        One refinement step with hierarchical spatial-temporal depth tracking.
        
        Instead of pooling to a single vector, this method preserves spatial
        structure by tracking all positions at the reduced resolution from
        the final Swin stage across depth steps.
        
        Args:
            X_t: [B, S, d_model] token representations at step t
            t: Current refinement step index (0-indexed)
            cache: Previous SwinDepthCache (or None to initialize)
            return_aux: If True, return auxiliary metrics
            
        Returns:
            alpha_t: [B, S, H] routing weights for each token
            cache: Updated SwinDepthCache
            aux: Dict with metrics
        """
        if t < 0 or t >= self.max_depth:
            raise ValueError(f"t={t} out of range for max_depth={self.max_depth}")
        
        if cache is None:
            cache = self.init_cache()
        
        B, S, D = X_t.shape
        device = X_t.device
        
        # Infer grid size
        H, W = self._infer_grid_size(S)
        
        # Project input
        x = self.input_proj(X_t)  # [B, S, d_ctrl]
        
        # Process through hierarchical stages
        for i, stage in enumerate(self.stages):
            x, H, W = stage(x, H, W)
        
        # x is now [B, S_reduced, final_dim] where S_reduced = H * W after all stages
        S_reduced = x.size(1)
        
        # Project to depth controller space (per spatial position)
        x_depth = self.depth_proj(x)  # [B, S_reduced, d_ctrl]
        
        # Add depth positional embedding to all spatial positions
        x_depth = x_depth + self.depth_pos[t].unsqueeze(0).unsqueeze(0)
        
        # Store in cache - keep ALL spatial positions, not just pooled
        if len(cache.stage_features) == 0:
            cache.stage_features = [[]]
        cache.stage_features[0].append(x_depth)  # Append [B, S_reduced, d_ctrl]
        
        # Stack depth history: [T, B, S_reduced, d_ctrl]
        depth_stack = torch.stack(cache.stage_features[0], dim=0)
        T_depth = depth_stack.size(0)
        
        # Reshape for depth transformer: [T * S_reduced, B, d_ctrl]
        # Each spatial position attends causally over depth history
        # AND can attend to other spatial positions at the same or earlier depths
        depth_seq = depth_stack.permute(0, 2, 1, 3)  # [T, S_reduced, B, d_ctrl]
        depth_seq = depth_seq.reshape(T_depth * S_reduced, B, self.d_ctrl)
        
        # Create hierarchical causal mask
        depth_mask = self._hierarchical_causal_mask(T_depth, S_reduced, device)
        
        y = self.depth_transformer(depth_seq, mask=depth_mask)  # [T*S_reduced, B, d_ctrl]
        
        # Take the last depth step's spatial features: last S_reduced positions
        y_t = y[-S_reduced:]  # [S_reduced, B, d_ctrl]
        y_t = y_t.permute(1, 0, 2)  # [B, S_reduced, d_ctrl]
        
        # Depth skip connection: add current step's input directly to output
        # This helps gradient flow and preserves information across refinement iterations
        if self.depth_skip:
            y_t = y_t + x_depth
        
        y_t = self.final_norm(y_t)
        
        # Pool the spatial-aware depth features for routing
        y_pooled = y_t.mean(dim=1)  # [B, d_ctrl]
        
        # Routing
        if self.token_conditioned:
            y_exp = y_pooled.unsqueeze(1).expand(-1, S, -1)  # [B, S, d_ctrl]
            logits = self.router(torch.cat([X_t, y_exp], dim=-1))  # [B, S, H]
        else:
            logits_global = self.router(y_pooled)  # [B, H]
            logits = logits_global.unsqueeze(1).expand(-1, S, -1)  # [B, S, H]
        
        alpha = F.softmax(logits / self.temperature, dim=-1)
        alpha = self._topk_mask_renorm(alpha)
        
        entropy = self._compute_entropy(alpha)
        
        # Auxiliary info
        aux: Dict[str, Any] = {}
        if return_aux:
            aux = {
                "router_logits": logits.detach(),
                "alphas": alpha.detach(),
                "entropy": entropy.detach(),
                "temperature": self.temperature,
                "depth_step": t,
                "grid_size": (H, W),
                "reduced_spatial_size": S_reduced,
                "features": y_pooled,  # [B, d_ctrl] - injectable controller feature
            }
        
        return alpha, cache, aux
    
    def forward(
        self,
        x: torch.Tensor,
        head_outputs: Optional[torch.Tensor] = None,
        *,
        state: Optional[SwinDepthCache] = None,
        step: int = 0,
        per_token_pool: str = "mean",
        mask: Optional[torch.Tensor] = None,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, SwinDepthCache, Dict[str, Any]]:
        """
        Forward pass compatible with existing controller APIs.
        
        Args:
            x: Input representation [B, S, d_model] or [B, d_model]
            head_outputs: Precomputed head features (not used)
            state: Previous SwinDepthCache (or None to initialize)
            step: Current iteration step (default: 0)
            per_token_pool: Pooling method (unused)
            mask: Attention mask (unused)
            return_aux: Return auxiliary metrics
            
        Returns:
            alphas: Routing weights [B, S, H] or [B, H]
            new_state: Updated SwinDepthCache
            aux: Dict with auxiliary metrics
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, d_model]
        
        alpha, cache, aux = self.step(x, step, state, return_aux)
        
        if alpha.size(1) == 1:
            alpha = alpha.squeeze(1)  # [B, H]
        
        return alpha, cache, aux

