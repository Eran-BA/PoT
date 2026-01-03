"""
HybridPoHHRM for Natural Language Inference.

Two-timescale reasoning architecture for NLI classification.
Uses the same HybridHRMBase as Sudoku but with text embeddings
and 3-class classification output.

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import torch
import torch.nn as nn
from typing import Optional

from pot.models.hybrid_hrm import HybridHRMBase
from pot.models.hrm_layers import RMSNorm


class HybridPoHHRMForNLI(HybridHRMBase):
    """
    HybridPoHHRM model for Natural Language Inference.
    
    Extends HybridHRMBase with NLI-specific embeddings and classification head.
    
    Architecture:
    - Token + Position + Segment embeddings (BERT-style)
    - Two-timescale reasoning (L_level fast + H_level slow)
    - [CLS] token pooling for 3-class classification
    
    Args:
        vocab_size: Vocabulary size
        d_model: Hidden dimension
        n_heads: Number of attention heads
        H_layers: Layers in H_level module
        L_layers: Layers in L_level module
        d_ff: Feedforward dimension
        dropout: Dropout rate
        H_cycles: Outer loop iterations
        L_cycles: Inner loop iterations
        max_seq_len: Maximum sequence length
        T: HRM period for pointer controller
        num_labels: Number of output classes (3 for NLI)
    """
    
    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 512,
        n_heads: int = 8,
        H_layers: int = 2,
        L_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.1,
        H_cycles: int = 2,
        L_cycles: int = 8,
        max_seq_len: int = 128,
        T: int = 4,
        num_labels: int = 3,
    ):
        # Initialize base class
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            H_layers=H_layers,
            L_layers=L_layers,
            d_ff=d_ff,
            seq_len=max_seq_len,
            H_cycles=H_cycles,
            L_cycles=L_cycles,
            dropout=dropout,
            T=T,
        )
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_labels = num_labels
        
        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        nn.init.normal_(self.token_emb.weight, mean=0, std=0.02)
        
        # Position embedding
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        nn.init.normal_(self.pos_emb.weight, mean=0, std=0.02)
        
        # Segment embedding (0 for premise, 1 for hypothesis)
        self.segment_emb = nn.Embedding(2, d_model)
        nn.init.normal_(self.segment_emb.weight, mean=0, std=0.02)
        
        # Embedding normalization and dropout
        self.emb_norm = nn.LayerNorm(d_model)
        self.emb_dropout = nn.Dropout(dropout)
        
        # Classification head
        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )
        self.classifier_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_labels)
        
        # Initialize classifier
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for NLI classification.
        
        Args:
            input_ids: [B, T] - token IDs
            attention_mask: [B, T] - 1 for valid tokens, 0 for padding
            token_type_ids: [B, T] - 0 for premise, 1 for hypothesis
            
        Returns:
            logits: [B, num_labels] - classification logits
        """
        B, T = input_ids.size()
        device = input_ids.device
        
        # Token embeddings
        x = self.token_emb(input_ids)
        
        # Position embeddings
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        x = x + self.pos_emb(positions)
        
        # Segment embeddings
        if token_type_ids is not None:
            x = x + self.segment_emb(token_type_ids)
        
        # Normalize and dropout
        x = self.emb_norm(x)
        x = self.emb_dropout(x)
        
        # Scale embeddings (HRM-style)
        input_emb = self.embed_scale * x
        
        # Pad/truncate to seq_len if needed
        if T < self.seq_len:
            padding = torch.zeros(B, self.seq_len - T, self.d_model, device=device)
            input_emb = torch.cat([input_emb, padding], dim=1)
        elif T > self.seq_len:
            input_emb = input_emb[:, :self.seq_len, :]
        
        # Run two-timescale reasoning loop
        hidden, q_halt, q_continue, steps = self.reasoning_loop(input_emb)
        
        # Use [CLS] token (position 0) for classification
        cls_output = hidden[:, 0, :]
        
        # Classification
        pooled = self.pooler(cls_output)
        pooled = self.classifier_dropout(pooled)
        logits = self.classifier(pooled)
        
        return logits
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

