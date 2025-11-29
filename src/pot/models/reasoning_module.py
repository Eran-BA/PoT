"""
ReasoningModule - Transformer stack with PoT head routing.

This module implements a transformer block that uses Pointer over Heads (PoT)
for dynamic attention head routing. It follows HRM's post-norm structure.

Used as H_level or L_level in the HybridHRM architecture.

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import torch
import torch.nn as nn
from typing import Optional

from src.pot.core.hrm_controller import HRMPointerController, HRMState
from src.pot.models.hrm_layers import RMSNorm, SwiGLU


class ReasoningModule(nn.Module):
    """
    Transformer stack with PoT head routing.
    
    Used as H_level or L_level in the hybrid architecture.
    Follows HRM's post-norm structure but adds PoT head routing.
    
    Key features:
    - Input injection: hidden = hidden + injection before processing
    - PoT head routing: dynamically weight attention heads
    - Post-norm architecture with RMSNorm
    - SwiGLU feedforward layers
    - No dropout (to match HRM)
    
    Args:
        d_model: Hidden dimension size
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feedforward hidden dimension
        dropout: Dropout rate (default 0.0 to match HRM)
        T: HRM period for pointer controller
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.0,  # HRM doesn't use dropout
        T: int = 4,  # HRM period for pointer controller
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # PoT Pointer Controller for head routing
        self.pointer_controller = HRMPointerController(
            d_model=d_model,
            n_heads=n_heads,
            T=T,
            dropout=dropout
        )
        
        # Transformer layers (no dropout to match HRM)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            SwiGLU(d_model, d_ff, dropout=0.0) for _ in range(n_layers)
        ])
        self.norm1_layers = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.norm2_layers = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
    
    def forward(
        self, 
        hidden: torch.Tensor,      # [B, seq_len, d_model] - current hidden state
        injection: torch.Tensor,   # [B, seq_len, d_model] - input to inject
        ptr_state: Optional[HRMState] = None
    ):
        """
        Forward pass with input injection and PoT head routing.
        
        HRM-style: hidden = hidden + injection before processing (post-norm)
        
        Args:
            hidden: Current hidden state [B, seq_len, d_model]
            injection: Input to inject (added to hidden) [B, seq_len, d_model]
            ptr_state: Optional pointer controller state for recurrence
            
        Returns:
            output: Processed hidden state [B, seq_len, d_model]
            ptr_state: Updated pointer controller state
        """
        B, T, D = hidden.shape
        
        # Input injection (HRM key insight)
        x = hidden + injection
        
        # Get routing weights from PoT controller
        route_weights, ptr_state, _ = self.pointer_controller(x, state=ptr_state)
        
        # CRITICAL: Scale routing weights by n_heads to preserve magnitude
        # Softmax weights sum to 1, so without scaling we'd reduce output by n_heads factor
        route_weights_scaled = route_weights * self.n_heads  # [B, H] -> sums to n_heads
        
        for attn, ffn, norm1, norm2 in zip(
            self.attn_layers, self.ffn_layers,
            self.norm1_layers, self.norm2_layers
        ):
            # Attention with PoT head routing
            attn_out, _ = attn(x, x, x, need_weights=False)
            d_head = D // self.n_heads
            attn_out_heads = attn_out.view(B, T, self.n_heads, d_head)
            route_exp = route_weights_scaled.unsqueeze(1).unsqueeze(-1)  # [B, 1, H, 1]
            attn_out_routed = (attn_out_heads * route_exp).view(B, T, D)
            
            # Post-norm (like HRM)
            x = norm1(x + attn_out_routed)
            x = norm2(x + ffn(x))
        
        return x, ptr_state

