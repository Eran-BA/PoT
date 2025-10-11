"""
Baseline dependency parser with vanilla multi-head attention.

Implements a standard transformer-based dependency parser for comparison
with the Pointer-over-Heads approach.

Classes:
    VanillaBlock: Standard transformer block
    BaselineParser: Baseline parser using vanilla MHA

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import ParserBase
from src.models.layers import BiaffinePointer, BiaffineLabeler
from src.utils.helpers import mean_pool_subwords, pad_words, make_targets


class VanillaBlock(nn.Module):
    """Standard transformer block with multi-head attention and FFN.

    Uses PyTorch's built-in MultiheadAttention for a fair baseline comparison.
    Implements pre-LayerNorm architecture for stable training.

    Args:
        d_model: Model hidden dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward network hidden dimension
        dropout: Dropout probability

    Example:
        >>> block = VanillaBlock(d_model=768, n_heads=8, d_ff=2048)
        >>> x = torch.randn(2, 10, 768)
        >>> y, aux = block(x)
        >>> print(y.shape)  # [2, 10, 768]
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Forward pass through transformer block.

        Args:
            x: Input representations [B, T, D]
            attn_mask: Optional attention mask

        Returns:
            Tuple of (output [B, T, D], auxiliary dict)
        """
        # Self-attention with residual
        h = self.ln1(x)
        y, _ = self.attn(h, h, h, need_weights=False, attn_mask=None)
        x = x + y

        # Feed-forward with residual
        y = self.ff(self.ln2(x))
        return x + y, {}


class BaselineParser(ParserBase):
    """Baseline dependency parser using vanilla multi-head attention.

    Serves as a control for comparing against the Pointer-over-Heads approach.
    Uses standard transformer architecture without dynamic head routing.

    Architecture:
        1. Pretrained encoder (e.g., DistilBERT)
        2. Subword-to-word pooling
        3. Standard transformer block
        4. Biaffine head prediction
        5. Optional biaffine label classification

    Args:
        enc_name: HuggingFace model name
        d_model: Model hidden dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward network dimension
        n_labels: Number of dependency labels
        use_labels: Whether to predict labels (LAS) or just heads (UAS)

    Attributes:
        block: Vanilla transformer block
        pointer: Biaffine pointer for head prediction
        labeler: Optional biaffine labeler for relation classification
        use_labels: Whether label prediction is enabled
        n_labels: Number of dependency relation labels

    Example:
        >>> parser = BaselineParser(
        ...     enc_name="distilbert-base-uncased",
        ...     d_model=768,
        ...     n_heads=8,
        ...     d_ff=2048,
        ...     n_labels=50,
        ...     use_labels=True
        ... )
        >>> # Training
        >>> loss, metrics = parser(subw_tokens, word_ids, heads_gold, labels_gold)
        >>> print(f"UAS: {metrics['uas']:.3f}, LAS: {metrics['las']:.3f}")
    """

    def __init__(
        self,
        enc_name: str = "distilbert-base-uncased",
        d_model: int = 768,
        n_heads: int = 8,
        d_ff: int = 2048,
        n_labels: int = 50,
        use_labels: bool = True,
    ):
        super().__init__(enc_name, d_model)

        # Vanilla transformer block
        self.block = VanillaBlock(d_model, n_heads, d_ff)

        # Head prediction
        self.pointer = BiaffinePointer(d_model)

        # Optional label prediction
        self.use_labels = use_labels
        self.n_labels = n_labels
        if use_labels:
            self.labeler = BiaffineLabeler(d_model, n_labels)

    def forward(
        self,
        subw: Dict[str, torch.Tensor],
        word_ids: List[List[Optional[int]]],
        heads_gold: List[List[int]],
        labels_gold: Optional[List[List[int]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Forward pass for dependency parsing.

        Args:
            subw: Subword tokens (dict with 'input_ids', 'attention_mask', etc.)
            word_ids: Word indices for each subword [B, S]
            heads_gold: Gold dependency heads [B, T] (0 = ROOT)
            labels_gold: Optional gold dependency labels [B, T]

        Returns:
            Tuple containing:
            - loss: Combined head + label loss (scalar tensor)
            - metrics: Dictionary with:
                - uas: Unlabeled Attachment Score
                - las: Labeled Attachment Score (if use_labels=True)
                - tokens: Number of tokens
                - (eval mode) pred_heads: Predicted heads
                - (eval mode) pred_labels: Predicted labels (if use_labels=True)

        Note:
            During training, uses gold heads for label prediction (teacher forcing).
            During evaluation, uses predicted heads for label prediction.
        """
        device = next(self.parameters()).device

        # Encode and pool to word level
        last_hidden = self.encoder(**subw).last_hidden_state
        words = mean_pool_subwords(last_hidden, word_ids)
        X, mask = pad_words(words)

        # Apply transformer block
        X, _ = self.block(X)

        # Head prediction
        head_logits = self.pointer(X, X, mask, mask)
        Y_heads, pad = make_targets(heads_gold, X.size(1), device)
        head_loss = F.cross_entropy(
            head_logits.view(-1, head_logits.size(-1))[pad.view(-1)], Y_heads.view(-1)[pad.view(-1)]
        )

        # Compute UAS
        with torch.no_grad():
            pred_heads = head_logits.argmax(-1)
            uas = (pred_heads[pad] == Y_heads[pad]).float().mean().item()

        # Label prediction (if enabled)
        total_loss = head_loss
        las = uas  # Default to UAS if no labels
        pred_labels = None

        if self.use_labels and labels_gold is not None:
            # Use gold heads for training, predicted heads for evaluation
            head_indices = Y_heads if self.training else pred_heads

            # Prepend ROOT representation
            root_repr = self.pointer.root.view(1, 1, -1).expand(X.size(0), 1, X.size(-1))
            X_with_root = torch.cat([root_repr, X], dim=1)

            # Compute label logits
            label_logits = self.labeler(X, X_with_root, head_indices, mask)
            Y_labels, label_pad = make_targets(labels_gold, X.size(1), device)
            label_loss = F.cross_entropy(
                label_logits.view(-1, label_logits.size(-1))[pad.view(-1)],
                Y_labels.view(-1)[pad.view(-1)],
            )
            total_loss = head_loss + label_loss

            # Compute LAS
            with torch.no_grad():
                pred_labels = label_logits.argmax(-1)
                las = (
                    ((pred_heads[pad] == Y_heads[pad]) & (pred_labels[pad] == Y_labels[pad]))
                    .float()
                    .mean()
                    .item()
                )

        # Prepare metrics
        metrics = {"uas": uas, "las": las, "tokens": pad.sum().item()}

        # Add predictions for CoNLL-U export (evaluation only)
        if not self.training:
            metrics["pred_heads"] = [
                pred_heads[b, : pad[b].sum()].cpu().tolist() for b in range(pred_heads.size(0))
            ]
            if pred_labels is not None:
                metrics["pred_labels"] = [
                    pred_labels[b, : pad[b].sum()].cpu().tolist()
                    for b in range(pred_labels.size(0))
                ]

        return total_loss, metrics
