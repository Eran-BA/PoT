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
import torch.nn.functional as F
from typing import Optional, Tuple, Any, Dict

from src.pot.core.hrm_controller import HRMPointerController, HRMState
from src.pot.core.controller_factory import create_controller, CONTROLLER_TYPES
from src.pot.core.feature_injection import FeatureInjector, INJECTION_MODES
from src.pot.models.hrm_layers import RMSNorm, SwiGLU
from src.pot.modules.rope import RotaryEmbedding, apply_rotary_pos_emb, CosSin

# Try to import Flash Attention for faster long-sequence attention
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    try:
        from flash_attn_interface import flash_attn_func
        FLASH_ATTN_AVAILABLE = True
    except ImportError:
        FLASH_ATTN_AVAILABLE = False
        flash_attn_func = None


class RoPEMultiheadAttention(nn.Module):
    """
    Multi-head attention with Rotary Position Embeddings (RoPE).
    
    Based on HRM's Attention class, this applies RoPE to Q and K vectors
    before computing attention. Uses Flash Attention when available for
    O(N) memory and faster computation on long sequences.
    
    Args:
        d_model: Hidden dimension size
        n_heads: Number of attention heads
        dropout: Dropout rate (default 0.0)
        batch_first: If True, input is [B, T, D] (default True)
        use_flash_attn: If True, use Flash Attention when available (default True)
        
    Note:
        Unlike nn.MultiheadAttention, this requires cos_sin tuple from RotaryEmbedding.
        Pass via forward(q, k, v, cos_sin=...).
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        batch_first: bool = True,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.batch_first = batch_first
        self.dropout = dropout
        self.use_flash_attn = use_flash_attn and FLASH_ATTN_AVAILABLE
        
        # QKV projection (fused for efficiency)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Dropout for attention weights (only used in non-Flash path)
        self.attn_dropout = nn.Dropout(dropout)
    
    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor, 
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Flash Attention path - O(N) memory, fused CUDA kernels.
        
        Args:
            q, k, v: [B, T, n_heads, head_dim]
            
        Returns:
            output: [B, T, n_heads, head_dim]
        """
        # Flash attention expects [B, T, n_heads, head_dim]
        # and handles softmax scaling internally
        attn_output = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            causal=False,  # Non-causal for bidirectional reasoning
        )
        # Handle different flash_attn versions
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]
        return attn_output
    
    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard PyTorch attention path - O(N²) memory.
        
        Args:
            q, k, v: [B, T, n_heads, head_dim]
            attn_mask: Optional attention mask
            
        Returns:
            output: [B, T, n_heads, head_dim]
            attn_weights: [B, n_heads, T, T]
        """
        B, T, H, D = q.shape
        
        # Reshape for attention: [B, n_heads, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scale = D ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, n_heads, T, T]
        
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights_dropped = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights_dropped, v)  # [B, n_heads, T, head_dim]
        
        # Reshape back to [B, T, n_heads, head_dim]
        attn_output = attn_output.transpose(1, 2)
        
        return attn_output, attn_weights
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cos_sin: Optional[CosSin] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with RoPE and optional Flash Attention.
        
        Args:
            query: Query tensor [B, T, D] (batch_first=True)
            key: Key tensor [B, T, D]
            value: Value tensor [B, T, D]
            cos_sin: Tuple of (cos, sin) from RotaryEmbedding, or None to skip RoPE
            need_weights: Whether to return attention weights (default False)
                         Note: Flash Attention doesn't support returning weights
            attn_mask: Optional attention mask (only used in standard path)
            
        Returns:
            output: [B, T, D]
            attn_weights: Optional attention weights if need_weights=True (None if Flash)
        """
        # Self-attention: query == key == value
        B, T, D = query.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(query)  # [B, T, 3*D]
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # Each [B, T, n_heads, head_dim]
        
        # Apply RoPE to Q and K
        if cos_sin is not None:
            cos, sin = cos_sin
            # Slice to sequence length
            cos = cos[:T]
            sin = sin[:T]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Choose attention implementation
        if self.use_flash_attn and not need_weights and attn_mask is None:
            # Flash Attention path (fast, O(N) memory)
            attn_output = self._flash_attention(q, k, v)
            attn_weights = None
        else:
            # Standard attention path (supports weights and masks)
            attn_output, attn_weights = self._standard_attention(q, k, v, attn_mask)
            if not need_weights:
                attn_weights = None
        
        # Reshape to [B, T, D]
        attn_output = attn_output.contiguous().view(B, T, D)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output, attn_weights


class ReasoningModule(nn.Module):
    """
    Transformer stack with PoT head routing and optional feature injection.
    
    Used as H_level or L_level in the hybrid architecture.
    Follows HRM's post-norm structure but adds PoT head routing.
    
    Key features:
    - Input injection: hidden = hidden + injection before processing
    - PoT head routing: dynamically weight attention heads
    - Feature injection: optionally inject controller knowledge back into tokens
    - Post-norm architecture with RMSNorm
    - SwiGLU feedforward layers
    - Configurable dropout for regularization
    - Switchable controller type (GRU, LSTM, xLSTM, minGRU, Transformer, etc.)
    - Optional RoPE (Rotary Position Embeddings) for position-aware attention
    
    Args:
        d_model: Hidden dimension size
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feedforward hidden dimension
        dropout: Dropout rate (default 0.0 to match HRM)
        T: HRM period for pointer controller
        controller_type: Type of controller ("gru", "lstm", "xlstm", "mingru", "transformer", etc.)
        controller_kwargs: Additional kwargs for controller creation
        injection_mode: Feature injection mode ("none", "broadcast", "film", "depth_token", "cross_attn")
        injection_kwargs: Additional kwargs for FeatureInjector
        use_rope: Whether to use Rotary Position Embeddings (default False)
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
        injection_mode: str = "none",  # Feature injection mode
        injection_kwargs: Optional[dict] = None,
        use_rope: bool = False,  # Use Rotary Position Embeddings
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.controller_type = controller_type
        self.injection_mode = injection_mode
        self.use_rope = use_rope
        
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
        
        # Get controller's d_ctrl for feature injection
        d_ctrl = getattr(self.pointer_controller, 'd_ctrl', d_model)
        
        # Feature injector (optional)
        inj_kwargs = injection_kwargs or {}
        self.injector = FeatureInjector(
            d_model=d_model,
            d_ctrl=d_ctrl,
            mode=injection_mode,
            dropout=dropout,
            **inj_kwargs,
        )
        
        # Transformer layers with configurable dropout
        # Use RoPE attention if requested, otherwise standard nn.MultiheadAttention
        if use_rope:
            self.attn_layers = nn.ModuleList([
                RoPEMultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
                for _ in range(n_layers)
            ])
        else:
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
        injection_memory: Optional[torch.Tensor] = None,  # For cross_attn mode
        cos_sin: Optional[CosSin] = None,  # For RoPE: (cos, sin) from RotaryEmbedding
    ) -> Tuple[torch.Tensor, Any, Optional[torch.Tensor]]:
        """
        Forward pass with input injection and PoT head routing.
        
        HRM-style: hidden = hidden + injection before processing (post-norm)
        
        Args:
            hidden: Current hidden state [B, seq_len, d_model]
            injection: Input to inject (added to hidden) [B, seq_len, d_model]
            ptr_state: Optional pointer controller state for recurrence
            depth_step: Current depth step (used by transformer controller)
            injection_memory: Memory for cross_attn injection mode [B, T, d_model]
            cos_sin: Optional (cos, sin) tuple from RotaryEmbedding for RoPE attention
            
        Returns:
            output: Processed hidden state [B, seq_len, d_model]
            ptr_state: Updated pointer controller state
            injection_memory: Updated injection memory (for cross_attn mode)
        """
        B, T, D = hidden.shape
        
        # Input injection (HRM key insight)
        x = hidden + injection
        
        # Get routing weights AND features from PoT controller
        # Different controllers have different APIs, but all support forward()
        if self.controller_type == "gru":
            # HRMPointerController uses state= kwarg
            route_weights, ptr_state, aux = self.pointer_controller(x, state=ptr_state)
        elif self.controller_type in ("transformer", "pot_transformer", "swin"):
            # CausalDepthTransformerRouter, PoTDepthTransformerRouter, and SwinDepthController use step() with cache
            route_weights, ptr_state, aux = self.pointer_controller.step(
                x, t=depth_step, cache=ptr_state
            )
        elif self.controller_type in ("mamba", "diffusion"):
            # MambaDepthController and DiffusionDepthController use step() with state
            route_weights, ptr_state, aux = self.pointer_controller.step(x, state=ptr_state)
        else:
            # LSTM, xLSTM, minGRU use step() with state
            route_weights, ptr_state, aux = self.pointer_controller.step(x, state=ptr_state)
        
        # Extract controller features for injection
        features = aux.get("features")  # [B, d_ctrl] or None
        
        # Handle different output shapes for routing
        # GRU returns [B, H], others return [B, S, H]
        if route_weights.dim() == 2:
            # [B, H] -> scale and expand for broadcasting
            route_weights_scaled = route_weights * self.n_heads  # [B, H]
            route_exp = route_weights_scaled.unsqueeze(1).unsqueeze(-1)  # [B, 1, H, 1]
        else:
            # [B, S, H] -> scale, need to handle per-token routing
            route_weights_scaled = route_weights * self.n_heads  # [B, S, H]
            route_exp = route_weights_scaled.unsqueeze(-1)  # [B, S, H, 1]
        
        # Apply feature injection BEFORE transformer layers
        # (so injected features participate in attention)
        if self.injection_mode == "depth_token" and features is not None:
            # Depth token mode: prepend token, need to handle routing
            x, injection_memory = self.injector(x, features, injection_memory, alpha=route_weights)
            # x is now [B, S+1, D], route_exp needs adjustment
            if route_weights.dim() == 3:
                # Per-token routing: prepend a routing weight for depth token
                # Use mean routing for depth token
                depth_route = route_exp.mean(dim=1, keepdim=True)  # [B, 1, H, 1]
                route_exp = torch.cat([depth_route, route_exp], dim=1)  # [B, S+1, H, 1]
            T = T + 1  # Update sequence length
        elif features is not None:
            # Pass alpha for alpha_gated mode (ignored by other modes)
            x, injection_memory = self.injector(x, features, injection_memory, alpha=route_weights)
        
        for attn, ffn, norm1, norm2 in zip(
            self.attn_layers, self.ffn_layers,
            self.norm1_layers, self.norm2_layers
        ):
            # Attention with PoT head routing
            # Use RoPE if available (RoPEMultiheadAttention accepts cos_sin)
            if self.use_rope:
                attn_out, _ = attn(x, x, x, cos_sin=cos_sin, need_weights=False)
            else:
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
        
        # Remove depth token if it was added
        if self.injection_mode == "depth_token":
            x = self.injector.remove_depth_token(x)
        
        return x, ptr_state, injection_memory
