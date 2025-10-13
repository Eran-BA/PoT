"""
BERT baseline for NLI task.

Standard BERT-base encoder with classification head for fair comparison.

Author: Eran Ben Artzy
Year: 2025
"""

import torch
import torch.nn as nn
from typing import Optional


class BERTBlock(nn.Module):
    """Standard BERT transformer block."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            mask: [B, T] - attention mask (1 for valid, 0 for padding)
        Returns:
            [B, T, D]
        """
        # Self-attention with residual
        residual = x
        x = self.ln1(x)
        
        # Convert mask to attention mask format if provided
        attn_mask = None
        if mask is not None:
            # Create attention mask: [B, T] -> [B, 1, T] for broadcasting
            # We want to mask out padding tokens
            attn_mask = (mask == 0)  # True where we want to mask
        
        attn_out, _ = self.attn(
            x, x, x,
            key_padding_mask=attn_mask,
            need_weights=False
        )
        x = residual + attn_out
        
        # Feed-forward with residual
        residual = x
        x = self.ln2(x)
        x = residual + self.ff(x)
        
        return x


class BERTEncoder(nn.Module):
    """BERT-style encoder."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_heads: int = 12,
        d_ff: int = 3072,
        depth: int = 12,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.token_type_emb = nn.Embedding(2, d_model)  # For segment embeddings
        
        self.emb_norm = nn.LayerNorm(d_model)
        self.emb_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            BERTBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(depth)
        ])
        
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
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [B, T]
            attention_mask: [B, T] - 1 for valid tokens, 0 for padding
            token_type_ids: [B, T] - segment IDs (0 for premise, 1 for hypothesis)
        Returns:
            [B, T, D] - contextualized representations
        """
        B, T = input_ids.size()
        
        # Token embeddings
        x = self.token_emb(input_ids)  # [B, T, D]
        
        # Positional embeddings
        positions = torch.arange(T, device=input_ids.device)[None, :].expand(B, T)
        x = x + self.pos_emb(positions)
        
        # Token type embeddings (if provided)
        if token_type_ids is not None:
            x = x + self.token_type_emb(token_type_ids)
        
        # Normalize and dropout
        x = self.emb_norm(x)
        x = self.emb_dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask=attention_mask)
        
        return x


class BERTForNLI(nn.Module):
    """BERT model for Natural Language Inference."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_heads: int = 12,
        d_ff: int = 3072,
        depth: int = 12,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.encoder = BERTEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            depth=depth,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        
        # NLI classification head
        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 3)  # 3 classes: entailment, neutral, contradiction
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [B, T] - concatenated [CLS] premise [SEP] hypothesis [SEP]
            attention_mask: [B, T]
            token_type_ids: [B, T] - 0 for premise, 1 for hypothesis
        Returns:
            logits: [B, 3] - class logits
        """
        # Encode
        sequence_output = self.encoder(input_ids, attention_mask, token_type_ids)  # [B, T, D]
        
        # Use [CLS] token for classification
        cls_output = sequence_output[:, 0, :]  # [B, D]
        
        # Classify
        pooled = self.pooler(cls_output)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # [B, 3]
        
        return logits
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

