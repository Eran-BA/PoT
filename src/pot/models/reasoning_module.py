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
from typing import Optional, Literal, Any

from src.pot.core.hrm_controller import HRMPointerController, HRMState
from src.pot.core.controller_factory import create_controller, CONTROLLER_TYPES
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
    - Configurable dropout for regularization
    - Switchable controller type (GRU, LSTM, xLSTM, minGRU, Transformer)
    
    Args:
        d_model: Hidden dimension size
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feedforward hidden dimension
        dropout: Dropout rate (default 0.0 to match HRM)
        T: HRM period for pointer controller
        controller_type: Type of controller ("gru", "lstm", "xlstm", "mingru", "transformer")
        controller_kwargs: Additional kwargs for controller creation
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.0,  # HRM doesn't use dropout
        T: int = 4,  # HRM period for pointer controller
        controller_type: str = "gru",  # Default to GRU for backward compatibility
        controller_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.controller_type = controller_type
        
        # PoT Pointer Controller for head routing
        ctrl_kwargs = controller_kwargs or {}
        self.pointer_controller = create_controller(
            controller_type=controller_type,
            d_model=d_model,
            n_heads=n_heads,
            T=T,
            dropout=dropout,
            **ctrl_kwargs,
        )
        
        # Transformer layers with configurable dropout
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            SwiGLU(d_model, d_ff, dropout=dropout) for _ in range(n_layers)
        ])
        self.norm1_layers = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.norm2_layers = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
    
    def forward(
        self, 
        hidden: torch.Tensor,      # [B, seq_len, d_model] - current hidden state
        injection: torch.Tensor,   # [B, seq_len, d_model] - input to inject
        ptr_state: Optional[Any] = None,
        depth_step: int = 0,  # For transformer controller
    ):
        """
        Forward pass with input injection and PoT head routing.
        
        HRM-style: hidden = hidden + injection before processing (post-norm)
        
        Args:
            hidden: Current hidden state [B, seq_len, d_model]
            injection: Input to inject (added to hidden) [B, seq_len, d_model]
            ptr_state: Optional pointer controller state for recurrence
            depth_step: Current depth step (used by transformer controller)
            
        Returns:
            output: Processed hidden state [B, seq_len, d_model]
            ptr_state: Updated pointer controller state
        """
        B, T, D = hidden.shape
        
        # Input injection (HRM key insight)
        x = hidden + injection
        
        # Get routing weights from PoT controller
        # Different controllers have different APIs, but all support forward()
        if self.controller_type == "gru":
            # HRMPointerController uses state= kwarg
            route_weights, ptr_state, _ = self.pointer_controller(x, state=ptr_state)
        elif self.controller_type == "transformer":
            # CausalDepthTransformerRouter uses step() with cache
            route_weights, ptr_state, _ = self.pointer_controller.step(
                x, t=depth_step, cache=ptr_state
            )
        else:
            # LSTM, xLSTM, minGRU use step() with state
            route_weights, ptr_state, _ = self.pointer_controller.step(x, state=ptr_state)
        
        # Handle different output shapes
        # GRU returns [B, H], others return [B, S, H]
        if route_weights.dim() == 2:
            # [B, H] -> scale and expand for broadcasting
            route_weights_scaled = route_weights * self.n_heads  # [B, H]
            route_exp = route_weights_scaled.unsqueeze(1).unsqueeze(-1)  # [B, 1, H, 1]
        else:
            # [B, S, H] -> scale, need to handle per-token routing
            route_weights_scaled = route_weights * self.n_heads  # [B, S, H]
            route_exp = route_weights_scaled.unsqueeze(-1)  # [B, S, H, 1]
        
        for attn, ffn, norm1, norm2 in zip(
            self.attn_layers, self.ffn_layers,
            self.norm1_layers, self.norm2_layers
        ):
            # Attention with PoT head routing
            attn_out, _ = attn(x, x, x, need_weights=False)
            d_head = D // self.n_heads
            attn_out_heads = attn_out.view(B, T, self.n_heads, d_head)
            
            if route_weights.dim() == 2:
                # Global routing: [B, 1, H, 1] broadcast
                attn_out_routed = (attn_out_heads * route_exp).view(B, T, D)
            else:
                # Per-token routing: [B, S, H, 1]
                attn_out_routed = (attn_out_heads * route_exp).view(B, T, D)
            
            # Post-norm (like HRM)
            x = norm1(x + attn_out_routed)
            x = norm2(x + ffn(x))
        
        return x, ptr_state

