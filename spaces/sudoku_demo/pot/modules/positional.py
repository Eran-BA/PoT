"""
Config-switchable positional encoding for PoH.

Supports:
- "none": No positional information (permutation-invariant)
- "absolute": Learned positional embeddings (GPT-2 style)
- "rotary": Rotary Position Embedding (RoPE, LLaMA style)

Author: Eran Ben Artzy
Year: 2025
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

# Optional: RoPE support
try:
    from rotary_embedding_torch import RotaryEmbedding
    ROTARY_AVAILABLE = True
except ImportError:
    RotaryEmbedding = None
    ROTARY_AVAILABLE = False


class PositionalEncoding(nn.Module):
    """
    Config-switchable positional encoding.
    
    Modes:
    - "none": No position info (for permutation-invariant tasks)
    - "absolute": Learned positional embeddings [max_seq_len, d_model]
    - "rotary": RoPE (rotary position embeddings in Q/K space)
    
    Usage:
        cfg = PoHConfig(pos_encoding="absolute", max_seq_len=512)
        pos_enc = PositionalEncoding(cfg)
        x_with_pos = pos_enc(x)  # [B, T, d_model]
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.mode = cfg.pos_encoding
        self.d_model = cfg.d_model
        self.max_seq_len = cfg.max_seq_len
        self.n_heads = cfg.n_heads
        
        if self.mode == "absolute":
            # Learned positional embeddings (GPT-2 style)
            self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        
        elif self.mode == "rotary":
            # RoPE: rotates Q/K in attention
            if not ROTARY_AVAILABLE:
                raise ImportError(
                    "RoPE requires rotary-embedding-torch. Install with:\n"
                    "  pip install rotary-embedding-torch"
                )
            # Rotate per head dimension
            self.rope = RotaryEmbedding(cfg.d_model // cfg.n_heads)
        
        elif self.mode == "none":
            # No positional encoding
            pass
        
        else:
            raise ValueError(
                f"Unknown pos_encoding: {self.mode}. "
                f"Choose from: 'none', 'absolute', 'rotary'"
            )
    
    def forward(
        self,
        x: torch.Tensor,
        attn_module: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Optional[nn.Module]]:
        """
        Apply positional encoding.
        
        Args:
            x: [B, T, d_model] input tensor
            attn_module: Optional MultiheadAttention for RoPE injection
        
        Returns:
            x_with_pos: [B, T, d_model] with positions applied
            attn_module: Optionally modified attention module (for RoPE)
        """
        if self.mode == "none":
            return x, attn_module
        
        if self.mode == "absolute":
            B, T, D = x.shape
            if T > self.max_seq_len:
                raise ValueError(
                    f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}. "
                    f"Increase max_seq_len in config."
                )
            
            # Add learned positional embeddings
            pos_ids = torch.arange(T, device=x.device)
            pos_emb = self.pos_emb(pos_ids).unsqueeze(0)  # [1, T, D]
            return x + pos_emb, attn_module
        
        if self.mode == "rotary":
            # RoPE is applied inside attention (Q/K rotation)
            # Mark the attention module for rotation
            if attn_module is not None:
                attn_module.rope = self.rope
            return x, attn_module
        
        raise ValueError(f"Unknown pos_encoding: {self.mode}")
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        if self.mode == "absolute":
            return f"mode={self.mode}, max_seq_len={self.max_seq_len}"
        return f"mode={self.mode}"


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al. 2017).
    
    Fixed (non-learned) encoding using sin/cos functions.
    Alternative to learned absolute embeddings.
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute sinusoidal encodings
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
        
        Returns:
            x + pe: [B, T, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def apply_rotary_pos_emb(q, k, rope):
    """
    Apply RoPE to query and key tensors.
    
    Args:
        q: [B, num_heads, T, head_dim]
        k: [B, num_heads, T, head_dim]
        rope: RotaryEmbedding instance
    
    Returns:
        q_rot, k_rot: Rotated query and key
    """
    if rope is None:
        return q, k
    
    # rope expects [B, T, head_dim] or [B, num_heads, T, head_dim]
    q_rot = rope.rotate_queries_or_keys(q)
    k_rot = rope.rotate_queries_or_keys(k)
    
    return q_rot, k_rot

