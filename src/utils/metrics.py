"""
Evaluation metrics utilities with punctuation masking support

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import torch
from typing import List, Optional, Tuple


# UD punctuation relation tags to exclude from evaluation
PUNCT_TAGS = {"punct"}


def build_masks_for_metrics(
    heads: List[List[int]], deprels: Optional[List[List[str]]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build masks for metric computation with optional punctuation filtering.

    Args:
        heads: List of head indices per sentence (0 = ROOT)
        deprels: Optional list of dependency relation labels

    Returns:
        is_token: Boolean tensor [B, T] marking valid token positions
        is_eval: Boolean tensor [B, T] marking tokens to include in evaluation
                 (excludes punctuation if deprels provided)
    """
    B = len(heads)
    T = max(len(h) for h in heads) if B else 0

    is_token = torch.zeros(B, T, dtype=torch.bool)
    is_eval = torch.zeros(B, T, dtype=torch.bool)

    for b, h in enumerate(heads):
        L = len(h)
        is_token[b, :L] = True

        if deprels is None:
            # No punctuation info available, evaluate all tokens
            is_eval[b, :L] = True
        else:
            # Exclude punctuation from evaluation
            for i in range(L):
                is_eval[b, i] = deprels[b][i] not in PUNCT_TAGS

    return is_token, is_eval


def compute_uas_las(
    pred_heads: torch.Tensor,
    gold_heads: torch.Tensor,
    pred_labels: Optional[torch.Tensor],
    gold_labels: Optional[torch.Tensor],
    mask: torch.Tensor,
    deprels: Optional[List[List[str]]] = None,
    ignore_punct: bool = False,
) -> Tuple[float, float]:
    """
    Compute UAS and LAS with optional punctuation masking.

    Args:
        pred_heads: Predicted heads [B, T]
        gold_heads: Gold heads [B, T]
        pred_labels: Predicted labels [B, T] (optional)
        gold_labels: Gold labels [B, T] (optional)
        mask: Valid token mask [B, T]
        deprels: Dependency relation names (for punct masking)
        ignore_punct: Whether to exclude punctuation from evaluation

    Returns:
        uas: Unlabeled Attachment Score
        las: Labeled Attachment Score (or same as UAS if no labels)
    """
    # Base mask
    eval_mask = mask

    # Apply punctuation masking if requested
    if ignore_punct and deprels is not None:
        heads_list = gold_heads.cpu().tolist()
        _, is_eval = build_masks_for_metrics(heads_list, deprels)
        eval_mask = mask & is_eval.to(mask.device)

    # Compute UAS
    if eval_mask.any():
        correct_heads = pred_heads[eval_mask] == gold_heads[eval_mask]
        uas = correct_heads.float().mean().item()
    else:
        uas = 0.0

    # Compute LAS if labels available
    if pred_labels is not None and gold_labels is not None and eval_mask.any():
        correct_labels = pred_labels[eval_mask] == gold_labels[eval_mask]
        correct_both = correct_heads & correct_labels
        las = correct_both.float().mean().item()
    else:
        las = uas  # Default to UAS if no labels

    return uas, las
