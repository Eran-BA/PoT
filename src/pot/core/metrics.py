"""
Task-agnostic metrics for structured prediction.

Includes:
- Mask-aware Kendall-τ (ranking correlation)
- Accuracy metrics
- Structured prediction metrics

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import torch
import torch.nn.functional as F
from typing import Optional


def compute_mask_aware_kendall_tau(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    obs_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Kendall-τ rank correlation (mask-aware).

    Only considers pairs where both items are observable.

    Args:
        y_true: [B, N] ground truth scores
        y_pred: [B, N] predicted scores
        obs_mask: [B, N] boolean mask (True = observable)

    Returns:
        Scalar Kendall-τ correlation (averaged over batch)
    """
    B, N = y_true.shape
    
    # Pairwise differences
    true_diff = y_true.unsqueeze(2) - y_true.unsqueeze(1)  # [B, N, N]
    pred_diff = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)  # [B, N, N]
    
    # Pair mask: both i and j must be observable
    pair_mask = obs_mask.unsqueeze(2) & obs_mask.unsqueeze(1)  # [B, N, N]
    
    # Concordant: same sign
    # Discordant: opposite sign
    concordant = ((true_diff > 0) & (pred_diff > 0)) | ((true_diff < 0) & (pred_diff < 0))
    discordant = ((true_diff > 0) & (pred_diff < 0)) | ((true_diff < 0) & (pred_diff > 0))
    
    # Apply pair mask
    concordant = concordant & pair_mask
    discordant = discordant & pair_mask
    
    # Count per batch
    num_concordant = concordant.float().sum(dim=(-1, -2))  # [B]
    num_discordant = discordant.float().sum(dim=(-1, -2))  # [B]
    total_pairs = pair_mask.float().sum(dim=(-1, -2))  # [B]
    
    # Kendall-τ = (C - D) / (C + D)
    tau = (num_concordant - num_discordant) / total_pairs.clamp_min(1.0)
    
    return tau.mean()


def compute_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute accuracy (with optional masking).

    Args:
        pred: [B, ...] predictions
        target: [B, ...] targets
        mask: Optional [B, ...] mask

    Returns:
        Accuracy as float
    """
    correct = (pred == target)
    
    if mask is not None:
        correct = correct & mask
        total = mask.sum().item()
    else:
        total = pred.numel()
    
    return correct.sum().item() / max(1, total)


def compute_uas(
    pred_heads: torch.Tensor,
    true_heads: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """
    Compute Unlabeled Attachment Score (UAS) for dependency parsing.

    Args:
        pred_heads: [B, L] predicted head indices
        true_heads: [B, L] gold head indices
        mask: [B, L] token mask (True = valid token)

    Returns:
        UAS as float (% of correct attachments)
    """
    correct = (pred_heads == true_heads) & mask
    return correct.sum().item() / mask.sum().item()

