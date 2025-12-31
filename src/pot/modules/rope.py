"""
Rotary Position Embeddings (RoPE).

Implementation based on HRM's layers.py:
https://github.com/sapientinc/HRM

RoPE provides relative position information by rotating Q and K vectors
in the attention mechanism, without adding positional embeddings to input.

Key benefits over learned/sinusoidal positional embeddings:
- Better length generalization
- Relative position awareness
- No additional parameters to learn

Author: Eran Ben Artzy (adapted from HRM)
Year: 2025
License: Apache 2.0
"""

import torch
import torch.nn as nn
from typing import Tuple


CosSin = Tuple[torch.Tensor, torch.Tensor]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dims of the input.
    
    For input [..., d], returns [..., d] where:
    - First half becomes negative of second half
    - Second half becomes first half
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.
    
    Args:
        q: Query tensor [batch, seq_len, num_heads, head_dim]
        k: Key tensor [batch, seq_len, num_heads, head_dim]
        cos: Cosine embeddings [seq_len, head_dim]
        sin: Sine embeddings [seq_len, head_dim]
        
    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as inputs
    """
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)
    
    # cos, sin: [seq_len, head_dim] -> need to broadcast to [1, seq_len, 1, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Creates and caches cos/sin embeddings for rotary position encoding.
    Based on the paper "RoFormer: Enhanced Transformer with Rotary Position Embedding".
    
    Args:
        dim: Dimension of the embedding (typically head_dim = d_model // n_heads)
        max_position_embeddings: Maximum sequence length
        base: Base for the frequency computation (default: 10000.0)
        device: Device to create tensors on
        
    Example:
        >>> rope = RotaryEmbedding(dim=64, max_position_embeddings=900)
        >>> cos, sin = rope()
        >>> q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    """
    
    def __init__(
        self, 
        dim: int, 
        max_position_embeddings: int, 
        base: float = 10000.0,
        device: torch.device = None
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        
        # Compute position-frequency outer product
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        
        # Create embeddings: concat(freqs, freqs) to match head_dim
        # Different from paper but uses different permutation for same result
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Cache as buffers (not parameters - not learned)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
    
    def forward(self, seq_len: int = None) -> CosSin:
        """
        Get cached cos/sin embeddings.
        
        Args:
            seq_len: Optional sequence length (uses full cache if None)
            
        Returns:
            Tuple of (cos, sin) tensors, each [seq_len, dim]
        """
        if seq_len is None:
            return self.cos_cached, self.sin_cached
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]
    
    def extend_if_needed(self, seq_len: int, device: torch.device = None):
        """
        Extend the cached embeddings if sequence length exceeds current cache.
        
        Args:
            seq_len: Required sequence length
            device: Device for new tensors
        """
        if seq_len <= self.max_position_embeddings:
            return
        
        # Recompute for longer sequence
        self.max_position_embeddings = seq_len
        device = device or self.cos_cached.device
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

