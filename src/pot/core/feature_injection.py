"""
Feature Injection Module

Provides configurable mechanisms for injecting controller knowledge back into
token embeddings, beyond routing-only. This enables ablation studies comparing
routing-only vs. feature injection approaches.

Modes:
- "none": Routing-only (no injection) - backward compatible default
- "broadcast": Gated broadcast of controller features to all tokens
- "film": FiLM modulation (scale and shift)
- "depth_token": Prepend learnable depth token that participates in attention
- "cross_attn": Cross-attention to depth memory bank

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

from __future__ import annotations

from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


INJECTION_MODES = ["none", "broadcast", "film", "depth_token", "cross_attn"]
InjectionMode = Literal["none", "broadcast", "film", "depth_token", "cross_attn"]


class GatedBroadcastInjection(nn.Module):
    """
    Broadcast controller features to all tokens with learned gating.
    
    Formula:
        gate = sigmoid(r @ W_g)
        x_out = x + gate * broadcast(r @ W_r)
    
    The gate prevents the injection from overpowering token features.
    
    Args:
        d_model: Token embedding dimension
        d_ctrl: Controller state dimension
    """
    
    def __init__(self, d_model: int, d_ctrl: int):
        super().__init__()
        self.proj = nn.Linear(d_ctrl, d_model)
        self.gate = nn.Linear(d_ctrl, 1)
    
    def forward(
        self,
        x: torch.Tensor,  # [B, S, D]
        r: torch.Tensor,  # [B, d_ctrl]
    ) -> torch.Tensor:
        """Inject controller features into tokens."""
        # Project controller state to token space
        r_proj = self.proj(r)  # [B, D]
        
        # Compute gate (scalar per batch)
        gate = torch.sigmoid(self.gate(r))  # [B, 1]
        
        # Broadcast and add with gating
        injection = gate.unsqueeze(1) * r_proj.unsqueeze(1)  # [B, 1, D]
        return x + injection


class FiLMInjection(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) injection.
    
    Formula:
        gamma, beta = MLP(r).split()
        x_out = gamma * x + beta
    
    Acts as "context conditioning" - modulates existing features rather than
    adding new information directly.
    
    Args:
        d_model: Token embedding dimension
        d_ctrl: Controller state dimension
    """
    
    def __init__(self, d_model: int, d_ctrl: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_ctrl, d_ctrl),
            nn.GELU(),
            nn.Linear(d_ctrl, d_model * 2),  # gamma and beta
        )
        
        # Initialize to identity transform (gamma=1, beta=0)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.mlp[-1].bias.data[:d_model] = 1.0  # gamma = 1
    
    def forward(
        self,
        x: torch.Tensor,  # [B, S, D]
        r: torch.Tensor,  # [B, d_ctrl]
    ) -> torch.Tensor:
        """Apply FiLM modulation."""
        D = x.size(-1)
        
        # Get gamma and beta from controller state
        params = self.mlp(r)  # [B, 2*D]
        gamma, beta = params.split(D, dim=-1)  # [B, D] each
        
        # Expand for broadcasting
        gamma = gamma.unsqueeze(1)  # [B, 1, D]
        beta = beta.unsqueeze(1)    # [B, 1, D]
        
        return gamma * x + beta


class DepthTokenInjection(nn.Module):
    """
    Prepend a learnable depth token to the sequence.
    
    The depth token is derived from controller state and participates in
    self-attention, allowing tokens to selectively attend to depth knowledge.
    
    This is the "Transformer-native" way - preserves attention semantics
    and avoids blindly broadcasting the same vector to all tokens.
    
    Args:
        d_model: Token embedding dimension
        d_ctrl: Controller state dimension
    """
    
    def __init__(self, d_model: int, d_ctrl: int):
        super().__init__()
        self.proj = nn.Linear(d_ctrl, d_model)
        self.ln = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,  # [B, S, D]
        r: torch.Tensor,  # [B, d_ctrl]
    ) -> torch.Tensor:
        """Prepend depth token to sequence."""
        # Project controller state to depth token
        depth_token = self.proj(r)  # [B, D]
        depth_token = self.ln(depth_token)
        depth_token = depth_token.unsqueeze(1)  # [B, 1, D]
        
        # Prepend to sequence
        return torch.cat([depth_token, x], dim=1)  # [B, S+1, D]
    
    def remove_depth_token(self, x: torch.Tensor) -> torch.Tensor:
        """Remove the prepended depth token after attention."""
        return x[:, 1:, :]


class CrossAttentionInjection(nn.Module):
    """
    Cross-attention to a depth memory bank.
    
    Stores controller state summaries across depth steps and lets tokens
    cross-attend to retrieve relevant past depth information.
    
    Formula:
        x_out = x + CrossAttn(Q=x, K=memory, V=memory)
    
    This is the most expressive mode - different tokens can pull different
    past depth information.
    
    Args:
        d_model: Token embedding dimension
        d_ctrl: Controller state dimension
        memory_size: Maximum number of states to store
        n_heads: Number of attention heads
        dropout: Attention dropout
    """
    
    def __init__(
        self,
        d_model: int,
        d_ctrl: int,
        memory_size: int = 16,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.memory_size = memory_size
        self.d_ctrl = d_ctrl
        
        # Project controller state to memory space
        self.memory_proj = nn.Linear(d_ctrl, d_model)
        
        # Cross-attention: tokens attend to memory
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,  # [B, S, D]
        r: torch.Tensor,  # [B, d_ctrl]
        memory: Optional[torch.Tensor] = None,  # [B, T, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-attend to memory and update it.
        
        Returns:
            x_out: [B, S, D] tokens with injected features
            memory: [B, T+1, D] updated memory (capped at memory_size)
        """
        B, S, D = x.shape
        device = x.device
        
        # Project current controller state
        r_proj = self.memory_proj(r)  # [B, D]
        r_proj = r_proj.unsqueeze(1)  # [B, 1, D]
        
        # Initialize or update memory
        if memory is None:
            memory = r_proj
        else:
            memory = torch.cat([memory, r_proj], dim=1)
            # Keep only last memory_size entries
            if memory.size(1) > self.memory_size:
                memory = memory[:, -self.memory_size:, :]
        
        # Cross-attention: tokens query the memory
        attn_out, _ = self.cross_attn(
            query=x,
            key=memory,
            value=memory,
            need_weights=False,
        )
        
        # Residual connection with LayerNorm
        x_out = self.ln(x + attn_out)
        
        return x_out, memory


class FeatureInjector(nn.Module):
    """
    Feature injection from controller state into token embeddings.
    
    Provides 4 configurable modes for injecting controller knowledge back
    into tokens, enabling ablation studies of routing-only vs. feature injection.
    
    Modes:
    - "none": Routing-only (no injection) - backward compatible
    - "broadcast": Gated broadcast of r to all tokens
    - "film": FiLM modulation (gamma * x + beta)
    - "depth_token": Prepend learnable depth token
    - "cross_attn": Cross-attention to depth memory bank
    
    Args:
        d_model: Token embedding dimension
        d_ctrl: Controller state dimension
        mode: Injection mode (default: "none")
        memory_size: For cross_attn mode, max memory entries
        n_heads: For cross_attn mode, number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        d_model: int,
        d_ctrl: int,
        mode: InjectionMode = "none",
        memory_size: int = 16,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mode = mode
        self.d_model = d_model
        self.d_ctrl = d_ctrl
        
        if mode == "none":
            self.injector = None
        elif mode == "broadcast":
            self.injector = GatedBroadcastInjection(d_model, d_ctrl)
        elif mode == "film":
            self.injector = FiLMInjection(d_model, d_ctrl)
        elif mode == "depth_token":
            self.injector = DepthTokenInjection(d_model, d_ctrl)
        elif mode == "cross_attn":
            self.injector = CrossAttentionInjection(
                d_model, d_ctrl,
                memory_size=memory_size,
                n_heads=n_heads,
                dropout=dropout,
            )
        else:
            raise ValueError(
                f"Unknown injection mode: '{mode}'. "
                f"Valid options: {INJECTION_MODES}"
            )
    
    def forward(
        self,
        x: torch.Tensor,  # [B, S, D] tokens
        r: Optional[torch.Tensor] = None,  # [B, d_ctrl] controller feature
        memory: Optional[torch.Tensor] = None,  # [B, T, D] for cross_attn
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply feature injection.
        
        Args:
            x: Token embeddings [B, S, D]
            r: Controller feature vector [B, d_ctrl]
            memory: Memory bank for cross_attn mode [B, T, D]
        
        Returns:
            x_out: Injected token embeddings
            memory: Updated memory (only for cross_attn mode)
        """
        if self.mode == "none" or r is None:
            return x, memory
        
        if self.mode == "cross_attn":
            return self.injector(x, r, memory)
        else:
            x_out = self.injector(x, r)
            return x_out, memory
    
    def has_depth_token(self) -> bool:
        """Check if this mode prepends a depth token."""
        return self.mode == "depth_token"
    
    def remove_depth_token(self, x: torch.Tensor) -> torch.Tensor:
        """Remove depth token if present."""
        if self.has_depth_token():
            return self.injector.remove_depth_token(x)
        return x
    
    def reset_memory(self):
        """Reset cross-attention memory (call at start of new sequence)."""
        # Memory is passed externally, so this is a no-op
        # Kept for API consistency
        pass

