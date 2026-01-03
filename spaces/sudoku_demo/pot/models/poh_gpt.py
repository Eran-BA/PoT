"""
PoHGPT: GPT-style autoregressive model using PoH blocks.

This module provides a complete GPT-like architecture with:
- Causal (autoregressive) masking
- Token embeddings + positional encoding
- PoH stack with iterative refinement
- Language modeling head
- Generation capabilities

Author: Eran Ben Artzy
Year: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from ..modules import PoHConfig, PoHStack, IterRefiner, PositionalEncoding


class PoHGPT(nn.Module):
    """
    GPT-style autoregressive model using PoH blocks.
    
    Combines token embeddings, positional encoding, PoHStack (with iterative
    refinement), and a language modeling head for next-token prediction.
    
    Features:
    - Causal masking for autoregressive generation
    - Switchable positional encoding (absolute/rotary/none)
    - Iterative refinement with optional outer residual
    - ACT halting for adaptive computation
    - KV-cache support for efficient generation (TODO)
    
    Example:
        >>> cfg = PoHConfig(d_model=512, n_heads=8, is_causal=True)
        >>> model = PoHGPT(vocab_size=32000, cfg=cfg)
        >>> x = torch.randint(0, 32000, (2, 10))  # [batch, seq_len]
        >>> logits = model(x)  # [2, 10, 32000]
        >>> generated = model.generate(x, max_new_tokens=20)
    """
    
    def __init__(
        self,
        vocab_size: int,
        cfg: PoHConfig,
        depth: Optional[int] = None,
        max_inner_iters: Optional[int] = None,
        outer_residual: Optional[bool] = None,
        rezero_init: Optional[bool] = None,
    ):
        """
        Initialize PoHGPT.
        
        Args:
            vocab_size: Size of vocabulary
            cfg: PoH configuration (must have is_causal=True for GPT mode)
            depth: Number of PoH blocks (overrides cfg if provided)
            max_inner_iters: Number of refinement iterations (overrides cfg)
            outer_residual: Enable outer residual (overrides cfg)
            rezero_init: Use ReZero initialization (overrides cfg)
        """
        super().__init__()
        
        # Store config and vocab size
        self.cfg = cfg
        self.vocab_size = vocab_size
        
        # Extract depth and refinement settings (with fallbacks)
        self.depth = depth if depth is not None else getattr(cfg, "depth", 6)
        self.max_inner_iters = max_inner_iters if max_inner_iters is not None else getattr(cfg, "max_inner_iters", 1)
        self.outer_residual = outer_residual if outer_residual is not None else getattr(cfg, "outer_residual", False)
        self.rezero_init = rezero_init if rezero_init is not None else getattr(cfg, "rezero_init", False)
        
        # Token embedding
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        
        # Positional encoding (switchable: absolute/rotary/none)
        # Note: PoHStack already includes positional encoding internally
        # But for GPT-style we apply it to embeddings directly
        
        # PoH stack (note: positional encoding is applied inside PoHStack)
        self.stack = PoHStack(cfg, depth=self.depth)
        
        # Iterative refiner
        self.refiner = IterRefiner(
            self.stack,
            max_inner_iters=self.max_inner_iters,
            outer_residual=self.outer_residual,
            rezero_init=self.rezero_init,
        )
        
        # Language modeling head (tied embeddings optional)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        
        # GPT-style weight initialization
        self.apply(self._init_weights)
        
        # Optional: Tie input and output embeddings (GPT-2 style)
        # Uncomment if desired:
        # self.lm_head.weight = self.embed.weight
    
    def _init_weights(self, module):
        """Initialize weights following GPT-2 style."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        return_inner_stats: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [B, T]
            return_inner_stats: Return per-iteration statistics
        
        Returns:
            logits: Next-token logits [B, T, vocab_size]
            inner_stats: Optional list of per-iteration stats (if requested)
        """
        # Token embedding
        x = self.embed(input_ids)  # [B, T, d_model]
        
        # Create causal mask if needed
        attn_mask = None
        if self.cfg.is_causal:
            T = input_ids.size(1)
            # Upper triangular mask: position i cannot attend to j > i
            attn_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1
            )
        
        # Apply PoH stack with iterative refinement
        y, inner_stats = self.refiner(x, attn_mask=attn_mask, return_inner_stats=return_inner_stats)
        
        # Language modeling head
        logits = self.lm_head(y)  # [B, T, vocab_size]
        
        return (logits, inner_stats) if return_inner_stats else (logits, None)
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation loop.
        
        Args:
            input_ids: Initial tokens [B, T]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top-k tokens (if specified)
            top_p: Nucleus sampling probability threshold (if specified)
            eos_token_id: Stop generation if this token is generated
        
        Returns:
            Generated token sequence [B, T + max_new_tokens]
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Forward pass (automatically handles causal masking)
            logits, _ = self.forward(input_ids)
            
            # Get logits for next token (last position)
            next_token_logits = logits[:, -1, :] / temperature  # [B, vocab_size]
            
            # Top-k filtering
            if top_k is not None:
                v, ix = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")
            
            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter back to original indices
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float("Inf")
            
            # Sample from distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if EOS token generated
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return input_ids
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        
        Args:
            non_embedding: Exclude embedding parameters (GPT convention)
        
        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed.weight.numel()
        return n_params


class BaselineGPT(nn.Module):
    """
    Baseline GPT for comparison (standard transformer decoder).
    
    Implements a vanilla GPT with the same interface as PoHGPT for
    fair parameter-matched comparisons.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        depth: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        """
        Initialize baseline GPT.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            depth: Number of layers
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Token + position embeddings
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN (GPT-2 style)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # LM head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following GPT-2 style."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, None]:
        """Forward pass."""
        B, T = input_ids.shape
        
        # Embeddings
        tok_emb = self.embed(input_ids)
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        pos_emb = self.pos_embed(pos_ids)
        x = tok_emb + pos_emb
        
        # Causal mask
        attn_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        
        # Transformer
        y = self.transformer(x, mask=attn_mask, is_causal=True)
        
        # LM head
        logits = self.lm_head(y)
        
        return logits, None
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, **kwargs):
        """Generate tokens (same interface as PoHGPT)."""
        self.eval()
        
        for _ in range(max_new_tokens):
            logits, _ = self.forward(input_ids)
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def get_num_params(self, non_embedding=True):
        """Return number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed.weight.numel()
            n_params -= self.pos_embed.weight.numel()
        return n_params

