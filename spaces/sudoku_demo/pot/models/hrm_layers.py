"""
HRM Architecture Components
===========================

Port of HRM's architecture improvements:
- SwiGLU activation (Gated Linear Units with Swish)
- RMSNorm (Root Mean Square Layer Normalization)
- Post-norm architecture

Based on the HRM paper and official implementation.

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    More stable than LayerNorm for deep networks.
    Used in LLaMA, PaLM, and HRM.
    
    Args:
        d_model: Model dimension
        eps: Small constant for numerical stability
    
    Reference:
        "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
        https://arxiv.org/abs/1910.07467
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] input tensor
        
        Returns:
            normalized: [B, L, D] RMS normalized tensor
        """
        # Compute RMS: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        x_norm = x / rms
        
        return self.weight * x_norm


class SwiGLU(nn.Module):
    """
    SwiGLU: Swish-Gated Linear Unit.
    
    Combines Swish activation with gating mechanism.
    Used in PaLM, LLaMA, and HRM for better performance than ReLU.
    
    Architecture:
        SwiGLU(x) = Swish(W1 * x) ⊙ (W2 * x)
        where Swish(x) = x * sigmoid(x)
              ⊙ is element-wise multiplication
    
    Args:
        d_model: Input dimension
        d_ff: Hidden dimension (typically 4 * d_model)
        dropout: Dropout probability
    
    Reference:
        "GLU Variants Improve Transformer" (Shazeer, 2020)
        https://arxiv.org/abs/2002.05202
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Two linear projections for gating
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # Gate
        self.w2 = nn.Linear(d_model, d_ff, bias=False)  # Value
        
        # Output projection
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_model] input tensor
        
        Returns:
            output: [B, L, d_model] transformed tensor
        """
        # Swish activation: x * sigmoid(x)
        swish_gate = F.silu(self.w1(x))  # silu = swish
        
        # Value path
        value = self.w2(x)
        
        # Gated multiplication
        hidden = swish_gate * value
        
        # Output projection
        output = self.w3(self.dropout(hidden))
        
        return output
    
    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, d_ff={self.d_ff}'


class PostNormTransformerLayer(nn.Module):
    """
    Post-norm Transformer layer with SwiGLU.
    
    Architecture:
        x = x + Attention(x)          # Add THEN norm
        x = RMSNorm(x)
        x = x + SwiGLU(x)             # Add THEN norm
        x = RMSNorm(x)
    
    This differs from Pre-norm (standard) where norm comes BEFORE:
        x = x + Attention(RMSNorm(x))  # Norm THEN add
        x = x + SwiGLU(RMSNorm(x))
    
    Post-norm provides better gradient flow for deep networks.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # SwiGLU FFN (replaces standard ReLU FFN)
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        
        # RMSNorm (replaces LayerNorm)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Post-norm forward pass.
        
        Args:
            x: [B, L, d_model] input tensor
        
        Returns:
            output: [B, L, d_model] transformed tensor
        """
        # Post-norm attention
        attn_out, _ = self.attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)  # Norm AFTER residual
        
        # Post-norm FFN
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)  # Norm AFTER residual
        
        return x

