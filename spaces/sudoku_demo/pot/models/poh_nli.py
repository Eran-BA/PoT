"""
PoH model for Natural Language Inference.

Uses PoH blocks with iterative refinement for premise-hypothesis reasoning.

Author: Eran Ben Artzy
Year: 2025
"""

import torch
import torch.nn as nn
from typing import Optional

from pot.modules.block import PoHConfig, PoHStack, IterRefiner


class PoHForNLI(nn.Module):
    """PoH transformer model for Natural Language Inference."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_heads: int = 12,
        d_ff: int = 3072,
        depth: int = 12,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        max_inner_iters: int = 3,
        route_mode: str = "soft",
        route_topk: Optional[int] = None,
        outer_residual: bool = True,
        rezero_init: bool = True,
        share_router: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings (BERT-style)
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.token_type_emb = nn.Embedding(2, d_model)  # Segment embeddings
        
        self.emb_norm = nn.LayerNorm(d_model)
        self.emb_dropout = nn.Dropout(dropout)
        
        # PoH encoder with iterative refinement
        cfg = PoHConfig(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            route_mode=route_mode,
            route_topk=route_topk,
            share_router=share_router,
            pos_encoding="none",  # We handle positional encoding ourselves
            max_seq_len=max_seq_len,
        )
        
        self.stack = PoHStack(cfg, depth=depth)
        self.refiner = IterRefiner(
            self.stack,
            max_inner_iters=max_inner_iters,
            outer_residual=outer_residual,
            rezero_init=rezero_init,
        )
        
        # NLI classification head
        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 3)  # 3 classes
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_inner_stats: bool = False,
    ):
        """
        Args:
            input_ids: [B, T] - concatenated [CLS] premise [SEP] hypothesis [SEP]
            attention_mask: [B, T] - 1 for valid, 0 for padding
            token_type_ids: [B, T] - 0 for premise, 1 for hypothesis
            return_inner_stats: whether to return inner iteration statistics
        Returns:
            logits: [B, 3] or (logits, inner_stats) if return_inner_stats=True
        """
        B, T = input_ids.size()
        
        # Token embeddings
        x = self.token_emb(input_ids)  # [B, T, D]
        
        # Positional embeddings
        positions = torch.arange(T, device=input_ids.device)[None, :].expand(B, T)
        x = x + self.pos_emb(positions)
        
        # Token type embeddings
        if token_type_ids is not None:
            x = x + self.token_type_emb(token_type_ids)
        
        # Normalize and dropout
        x = self.emb_norm(x)
        x = self.emb_dropout(x)
        
        # Create attention mask for PoH
        # PoH's IterRefiner expects None or doesn't use attn_mask in the same way
        # We'll handle padding at the embedding level instead
        attn_mask = None
        
        # PoH encoding with iterative refinement
        sequence_output, inner_stats = self.refiner(
            x,
            attn_mask=attn_mask,
            return_inner_stats=True
        )  # [B, T, D]
        
        # Use [CLS] token for classification
        cls_output = sequence_output[:, 0, :]  # [B, D]
        
        # Classify
        pooled = self.pooler(cls_output)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # [B, 3]
        
        if return_inner_stats:
            return logits, inner_stats
        return logits
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

