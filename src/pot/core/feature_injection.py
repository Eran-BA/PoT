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
- "alpha_gated": Alpha-modulated broadcast (injection strength follows routing)

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

from __future__ import annotations

from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


INJECTION_MODES = ["none", "broadcast", "broadcast_memory", "film", "depth_token", "cross_attn", "alpha_gated"]
InjectionMode = Literal["none", "broadcast", "broadcast_memory", "film", "depth_token", "cross_attn", "alpha_gated"]


class GatedBroadcastInjection(nn.Module):
    """
    Broadcast controller features to all tokens with learned gating.
    
    Formula:
        gate = sigmoid(r @ W_g)
        x_out = x + gate * broadcast(LayerNorm(r @ W_r))
    
    The gate prevents the injection from overpowering token features.
    
    Args:
        d_model: Token embedding dimension
        d_ctrl: Controller state dimension
        use_layernorm: If True, apply LayerNorm to projected features (default: True)
    """
    
    def __init__(self, d_model: int, d_ctrl: int, use_layernorm: bool = True):
        super().__init__()
        self.proj = nn.Linear(d_ctrl, d_model)
        self.gate = nn.Linear(d_ctrl, 1)
        self.ln = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,  # [B, S, D]
        r: torch.Tensor,  # [B, d_ctrl]
        alpha: torch.Tensor = None,  # Unused, for API compatibility
    ) -> torch.Tensor:
        """Inject controller features into tokens."""
        # Project controller state to token space and normalize
        r_proj = self.ln(self.proj(r))  # [B, D]
        
        # Compute gate (scalar per batch)
        gate = torch.sigmoid(self.gate(r))  # [B, 1]
        
        # Broadcast and add with gating
        injection = gate.unsqueeze(1) * r_proj.unsqueeze(1)  # [B, 1, D]
        return x + injection


class GatedBroadcastWithMemoryInjection(nn.Module):
    """
    Gated broadcast injection with memory accumulation.
    
    Like GatedBroadcastInjection, but maintains a sliding-window memory bank
    of past controller states across depth steps (and ACT steps when memory
    is preserved in ACTCarry). At each step:
    
    1. Append current controller state to memory bank
    2. Cap memory at memory_size entries
    3. Compute a summary of memory via a learned attention query
    4. Gate and broadcast the summary to all tokens
    
    This combines broadcast's simplicity with cross_attn's ability to
    accumulate context over multiple reasoning iterations.
    
    Args:
        d_model: Token embedding dimension
        d_ctrl: Controller state dimension
        memory_size: Maximum number of states to store in memory
        n_heads: Number of attention heads for memory aggregation
        use_layernorm: If True, apply LayerNorm to projected features
        dropout: Attention dropout
    """
    
    def __init__(
        self,
        d_model: int,
        d_ctrl: int,
        memory_size: int = 16,
        n_heads: int = 4,
        use_layernorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.memory_size = memory_size
        
        # Project controller state to memory space
        self.memory_proj = nn.Linear(d_ctrl, d_model)
        
        # Learned query for attending over memory bank
        self.summary_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Attention over memory to produce summary
        self.mem_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Gate to control injection strength
        self.gate = nn.Linear(d_model, 1)
        
        # Layer norm for output
        self.ln = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,  # [B, S, D]
        r: torch.Tensor,  # [B, d_ctrl]
        memory: Optional[torch.Tensor] = None,  # [B, T, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Accumulate controller state in memory, summarize, and broadcast.
        
        Returns:
            x_out: [B, S, D] tokens with injected memory summary
            memory: [B, T+1, D] updated memory bank (capped at memory_size)
        """
        B = x.size(0)
        
        # Project current controller state to memory space
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
        
        # Attend over memory with learned query to get summary
        query = self.summary_query.expand(B, -1, -1)  # [B, 1, D]
        summary, _ = self.mem_attn(
            query=query,
            key=memory,
            value=memory,
            need_weights=False,
        )  # [B, 1, D]
        
        summary = self.ln(summary)  # [B, 1, D]
        
        # Compute gate from summary
        gate = torch.sigmoid(self.gate(summary))  # [B, 1, 1]
        
        # Broadcast gated summary to all tokens
        injection = gate * summary  # [B, 1, D]
        x_out = x + injection  # Broadcasts to [B, S, D]
        
        return x_out, memory


class AlphaGatedInjection(nn.Module):
    """
    Alpha-modulated broadcast injection.
    
    Formula:
        alpha_scalar = mean(alpha)  # Aggregate routing weights
        x_out = x + alpha_scalar * LayerNorm(r @ W_r)
    
    The routing weights (alpha) modulate the injection strength, creating
    coherence between head routing and feature injection.
    
    Configurable aggregation modes:
    - "mean": Use mean of alpha across heads
    - "max": Use max of alpha across heads  
    - "entropy": Use (1 - normalized_entropy) - inject more when routing is confident
    
    Args:
        d_model: Token embedding dimension
        d_ctrl: Controller state dimension
        alpha_aggregation: How to aggregate alpha ("mean", "max", "entropy")
        use_learned_gate: If True, combine alpha with a learned gate
        use_layernorm: If True, apply LayerNorm to projected features (default: True)
    """
    
    def __init__(
        self,
        d_model: int,
        d_ctrl: int,
        alpha_aggregation: str = "mean",
        use_learned_gate: bool = True,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.proj = nn.Linear(d_ctrl, d_model)
        self.alpha_aggregation = alpha_aggregation
        self.use_learned_gate = use_learned_gate
        self.ln = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        
        if use_learned_gate:
            self.gate = nn.Linear(d_ctrl, 1)
    
    def _aggregate_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Aggregate routing weights to a scalar per batch.
        
        Args:
            alpha: Routing weights [B, H] or [B, S, H]
            
        Returns:
            Scalar per batch [B, 1]
        """
        # Flatten to [B, -1] if needed
        if alpha.dim() == 3:
            alpha = alpha.mean(dim=1)  # [B, H]
        
        if self.alpha_aggregation == "mean":
            return alpha.mean(dim=-1, keepdim=True)  # [B, 1]
        elif self.alpha_aggregation == "max":
            return alpha.max(dim=-1, keepdim=True)[0]  # [B, 1]
        elif self.alpha_aggregation == "entropy":
            # Low entropy = confident routing = stronger injection
            # H = -sum(p * log(p)), normalized by log(n_heads)
            eps = 1e-8
            log_alpha = (alpha + eps).log()
            entropy = -(alpha * log_alpha).sum(dim=-1, keepdim=True)  # [B, 1]
            max_entropy = alpha.size(-1)  # log would give max entropy
            normalized_entropy = entropy / (max_entropy + eps)
            return 1.0 - normalized_entropy  # [B, 1] - high when confident
        else:
            raise ValueError(f"Unknown alpha_aggregation: {self.alpha_aggregation}")
    
    def forward(
        self,
        x: torch.Tensor,  # [B, S, D]
        r: torch.Tensor,  # [B, d_ctrl]
        alpha: torch.Tensor,  # [B, H] or [B, S, H] routing weights
    ) -> torch.Tensor:
        """Inject controller features modulated by routing weights."""
        if alpha is None:
            # Fallback to uniform if no alpha provided
            alpha_scalar = torch.ones(x.size(0), 1, device=x.device)
        else:
            alpha_scalar = self._aggregate_alpha(alpha)  # [B, 1]
        
        # Project controller state to token space and normalize
        r_proj = self.ln(self.proj(r))  # [B, D]
        
        # Optionally combine with learned gate
        if self.use_learned_gate:
            learned_gate = torch.sigmoid(self.gate(r))  # [B, 1]
            gate = alpha_scalar * learned_gate
        else:
            gate = alpha_scalar
        
        # Broadcast and add with alpha-modulated gating
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
    
    Provides configurable modes for injecting controller knowledge back
    into tokens, enabling ablation studies of routing-only vs. feature injection.
    
    Modes:
    - "none": Routing-only (no injection) - backward compatible
    - "broadcast": Gated broadcast of r to all tokens
    - "broadcast_memory": Gated broadcast with memory bank accumulation
    - "film": FiLM modulation (gamma * x + beta)
    - "depth_token": Prepend learnable depth token
    - "cross_attn": Cross-attention to depth memory bank
    - "alpha_gated": Alpha-modulated broadcast (injection follows routing)
    
    Args:
        d_model: Token embedding dimension
        d_ctrl: Controller state dimension
        mode: Injection mode (default: "none")
        memory_size: For cross_attn mode, max memory entries
        n_heads: For cross_attn mode, number of attention heads
        dropout: Dropout probability
        alpha_aggregation: For alpha_gated mode, how to aggregate alpha ("mean", "max", "entropy")
        use_learned_gate: For alpha_gated mode, combine alpha with learned gate
        use_layernorm: For broadcast/alpha_gated modes, apply LayerNorm (default: True)
    """
    
    def __init__(
        self,
        d_model: int,
        d_ctrl: int,
        mode: InjectionMode = "none",
        memory_size: int = 16,
        n_heads: int = 4,
        dropout: float = 0.0,
        alpha_aggregation: str = "mean",
        use_learned_gate: bool = True,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.mode = mode
        self.d_model = d_model
        self.d_ctrl = d_ctrl
        
        if mode == "none":
            self.injector = None
        elif mode == "broadcast":
            self.injector = GatedBroadcastInjection(d_model, d_ctrl, use_layernorm=use_layernorm)
        elif mode == "broadcast_memory":
            self.injector = GatedBroadcastWithMemoryInjection(
                d_model, d_ctrl,
                memory_size=memory_size,
                n_heads=n_heads,
                dropout=dropout,
                use_layernorm=use_layernorm,
            )
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
        elif mode == "alpha_gated":
            self.injector = AlphaGatedInjection(
                d_model, d_ctrl,
                alpha_aggregation=alpha_aggregation,
                use_learned_gate=use_learned_gate,
                use_layernorm=use_layernorm,
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
        alpha: Optional[torch.Tensor] = None,  # [B, H] or [B, S, H] routing weights
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply feature injection.
        
        Args:
            x: Token embeddings [B, S, D]
            r: Controller feature vector [B, d_ctrl]
            memory: Memory bank for cross_attn mode [B, T, D]
            alpha: Routing weights for alpha_gated mode [B, H] or [B, S, H]
        
        Returns:
            x_out: Injected token embeddings
            memory: Updated memory (for cross_attn and broadcast_memory modes)
        """
        if self.mode == "none" or r is None:
            return x, memory
        
        if self.mode in ("cross_attn", "broadcast_memory"):
            return self.injector(x, r, memory)
        elif self.mode == "alpha_gated":
            x_out = self.injector(x, r, alpha)
            return x_out, memory
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

