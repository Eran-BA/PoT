"""
Task-agnostic loss functions for structured prediction.

Includes:
- RankNet: Pairwise ranking loss
- Soft sorting losses (differentiable permutation)
- Mask-aware variants for partial observability

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def ranknet_loss(
    pred_diffs: torch.Tensor,
    true_labels: torch.Tensor,
    pair_mask: Optional[torch.Tensor] = None,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    RankNet pairwise ranking loss (mask-aware).

    Args:
        pred_diffs: [B, N, N] predicted score differences (pred[i] - pred[j])
        true_labels: [B, N, N] binary labels (1 if i > j, 0 otherwise)
        pair_mask: [B, N, N] mask for valid pairs (both items observable)
        margin: Margin for pairwise comparison

    Returns:
        Scalar loss
    """
    if pair_mask is not None:
        valid = pair_mask.float()
        loss = F.binary_cross_entropy_with_logits(
            pred_diffs[pair_mask],
            true_labels[pair_mask].float(),
            reduction='mean'
        )
    else:
        loss = F.binary_cross_entropy_with_logits(
            pred_diffs,
            true_labels.float(),
            reduction='mean'
        )
    
    return loss


def soft_sort_loss(
    pred_scores: torch.Tensor,
    true_scores: torch.Tensor,
    obs_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Differentiable soft sorting loss (simplified NeuralSort-style).

    Computes pairwise differences and applies soft ranking.

    Args:
        pred_scores: [B, N] predicted scores
        true_scores: [B, N] ground truth scores
        obs_mask: [B, N] observability mask
        temperature: Softmax temperature for soft ranking

    Returns:
        Scalar loss
    """
    B, N = pred_scores.shape
    
    # Compute pairwise differences
    pred_diff = pred_scores.unsqueeze(2) - pred_scores.unsqueeze(1)  # [B, N, N]
    true_diff = true_scores.unsqueeze(2) - true_scores.unsqueeze(1)  # [B, N, N]
    
    # Soft comparisons via sigmoid
    pred_soft = torch.sigmoid(pred_diff / temperature)  # [B, N, N]
    true_soft = (true_diff > 0).float()  # [B, N, N]
    
    # Mask-aware loss
    if obs_mask is not None:
        pair_mask = obs_mask.unsqueeze(2) & obs_mask.unsqueeze(1)  # [B, N, N]
        loss = F.binary_cross_entropy(
            pred_soft[pair_mask],
            true_soft[pair_mask],
            reduction='mean'
        )
    else:
        loss = F.binary_cross_entropy(pred_soft, true_soft, reduction='mean')
    
    return loss


def deep_supervision_loss(
    all_logits: list,  # List of [B, ...] tensors (one per iteration)
    targets: torch.Tensor,
    loss_fn: callable,
    weights: Optional[list] = None,
) -> torch.Tensor:
    """
    Deep supervision: average loss over all iterations.

    Args:
        all_logits: List of predictions at each iteration
        targets: Ground truth targets
        loss_fn: Loss function (e.g., F.cross_entropy)
        weights: Optional per-iteration weights (default: uniform)

    Returns:
        Weighted average loss
    """
    if weights is None:
        weights = [1.0 / len(all_logits)] * len(all_logits)
    
    total_loss = 0.0
    for logits, w in zip(all_logits, weights):
        total_loss += w * loss_fn(logits, targets)
    
    return total_loss

