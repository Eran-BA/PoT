"""
Mask-aware Ranking Utilities for Partial Observability Tasks

Implements:
1. Mask-aware Kendall-τ (only counts observable pairs)
2. Ranking-aware losses (RankNet pairwise + soft sorting)
3. Per-iteration diagnostics

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict


def compute_mask_aware_kendall_tau(
    pred: torch.Tensor,  # [B, L] predicted ranks/scores
    target: torch.Tensor,  # [B, L] ground-truth ranks/scores
    obs_mask: torch.Tensor,  # [B, L] observability mask (1=visible, 0=masked)
    return_components: bool = False,
) -> torch.Tensor:
    """
    Compute Kendall-τ only over pairs where BOTH elements are observable.
    
    This fixes the signal dilution problem when 50% of values are masked.
    
    Args:
        pred: Predicted ranks or scores [B, L]
        target: Ground-truth ranks or scores [B, L]
        obs_mask: Binary mask indicating observable positions [B, L]
        return_components: If True, return (tau, concordant, discordant, total_pairs)
    
    Returns:
        kendall_tau: Scalar correlation coefficient (mean over batch)
    """
    B, L = pred.shape
    device = pred.device
    
    # Pairwise observability mask: both i and j must be observable
    obs = obs_mask.bool()  # [B, L]
    pair_mask = obs.unsqueeze(-1) & obs.unsqueeze(-2)  # [B, L, L]
    
    # Exclude diagonal (i == j)
    diag_mask = ~torch.eye(L, dtype=torch.bool, device=device)
    pair_mask = pair_mask & diag_mask
    
    # Pairwise differences for ground-truth and predictions
    gt_diff = target.unsqueeze(-1) - target.unsqueeze(-2)  # [B, L, L]
    pred_diff = pred.unsqueeze(-1) - pred.unsqueeze(-2)  # [B, L, L]
    
    # Signs
    sgn_gt = torch.sign(gt_diff)  # -1, 0, +1
    sgn_pred = torch.sign(pred_diff)
    
    # Concordant: same sign and non-zero
    concordant = (sgn_gt == sgn_pred) & (sgn_gt != 0)
    
    # Discordant: opposite sign and both non-zero
    discordant = (sgn_gt != sgn_pred) & (sgn_gt != 0) & (sgn_pred != 0)
    
    # Apply observability mask
    concordant_masked = (concordant.float() * pair_mask.float()).sum(dim=(-1, -2))  # [B]
    discordant_masked = (discordant.float() * pair_mask.float()).sum(dim=(-1, -2))  # [B]
    total_pairs = pair_mask.float().sum(dim=(-1, -2)).clamp_min(1.0)  # [B]
    
    # Kendall-tau per sample
    tau = (concordant_masked - discordant_masked) / total_pairs  # [B]
    
    # Mean over batch
    tau_mean = tau.mean()
    
    if return_components:
        return tau_mean, concordant_masked.mean(), discordant_masked.mean(), total_pairs.mean()
    
    return tau_mean


def ranknet_pairwise_loss(
    pred_scores: torch.Tensor,  # [B, L] predicted scores
    target_ranks: torch.Tensor,  # [B, L] ground-truth ranks (lower=better)
    obs_mask: torch.Tensor,  # [B, L] observability mask
    margin: float = 0.0,  # Optional margin for contrastive learning
) -> torch.Tensor:
    """
    Mask-aware pairwise ranking loss (RankNet-style).
    
    Learns to predict relative order by minimizing:
        BCE(σ(s_i - s_j), label_ij)
    where label_ij = 1 if target_i < target_j (i should rank higher), else 0.
    
    Only pairs where both items are observable contribute to the loss.
    
    Args:
        pred_scores: Predicted scores (higher=better ranking) [B, L]
        target_ranks: Ground-truth ranks (lower number=better) [B, L]
        obs_mask: Binary observability mask [B, L]
        margin: Margin for ranking loss (default 0)
    
    Returns:
        loss: Scalar pairwise ranking loss
    """
    B, L = pred_scores.shape
    device = pred_scores.device
    
    # Pairwise observability mask
    obs = obs_mask.bool()
    pair_mask = obs.unsqueeze(-1) & obs.unsqueeze(-2)  # [B, L, L]
    diag_mask = ~torch.eye(L, dtype=torch.bool, device=device)
    pair_mask = pair_mask & diag_mask
    
    # Pairwise score differences
    score_diff = pred_scores.unsqueeze(-1) - pred_scores.unsqueeze(-2)  # [B, L, L]
    
    # Pairwise labels: 1 if i ranks higher than j (target_i < target_j)
    rank_diff = target_ranks.unsqueeze(-1) - target_ranks.unsqueeze(-2)  # [B, L, L]
    pair_labels = (rank_diff < 0).float()  # 1 if i should rank higher
    
    # Only consider pairs with strict ordering (ignore ties)
    non_tie_mask = (rank_diff != 0) & pair_mask
    
    if not non_tie_mask.any():
        return torch.tensor(0.0, device=device)
    
    # Binary cross-entropy loss (with margin)
    if margin > 0:
        # Hinge-style: max(0, margin - (s_i - s_j) * y_ij)
        signs = 2 * pair_labels - 1  # Convert to {-1, +1}
        loss = F.relu(margin - score_diff * signs)
        loss = (loss * non_tie_mask.float()).sum() / non_tie_mask.float().sum().clamp_min(1.0)
    else:
        # Standard sigmoid + BCE
        loss = F.binary_cross_entropy_with_logits(
            score_diff[non_tie_mask],
            pair_labels[non_tie_mask],
            reduction='mean'
        )
    
    return loss


def listnet_cross_entropy_loss(
    pred_scores: torch.Tensor,  # [B, L] predicted scores
    target_ranks: torch.Tensor,  # [B, L] ground-truth ranks
    obs_mask: torch.Tensor,  # [B, L] observability mask
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    ListNet-style cross-entropy loss over permutation distributions.
    
    Interprets scores as defining a probability distribution over permutations
    via top-one probabilities, and minimizes KL divergence to the target distribution.
    
    Args:
        pred_scores: Predicted scores [B, L]
        target_ranks: Ground-truth ranks (lower=better) [B, L]
        obs_mask: Binary observability mask [B, L]
        temperature: Softmax temperature
    
    Returns:
        loss: Scalar ListNet loss
    """
    B, L = pred_scores.shape
    
    # Convert ranks to scores (invert: lower rank → higher score)
    target_scores = -target_ranks.float()  # Negate so lower rank = higher score
    
    # Mask out non-observable positions with very negative scores
    pred_scores_masked = pred_scores.clone()
    target_scores_masked = target_scores.clone()
    pred_scores_masked[~obs_mask.bool()] = -1e9
    target_scores_masked[~obs_mask.bool()] = -1e9
    
    # Top-one probabilities (probability that each item is ranked first)
    pred_probs = F.softmax(pred_scores_masked / temperature, dim=-1)  # [B, L]
    target_probs = F.softmax(target_scores_masked / temperature, dim=-1)  # [B, L]
    
    # Cross-entropy loss (KL divergence)
    # Only compute over observable positions
    loss = -(target_probs * (pred_probs + 1e-10).log() * obs_mask.float()).sum() / obs_mask.float().sum().clamp_min(1.0)
    
    return loss


def soft_sorting_loss(
    pred_scores: torch.Tensor,  # [B, L] predicted scores
    target_ranks: torch.Tensor,  # [B, L] ground-truth ranks
    obs_mask: torch.Tensor,  # [B, L] observability mask
    tau: float = 0.1,  # Temperature for soft sorting
) -> torch.Tensor:
    """
    Differentiable sorting loss using soft ranking operators.
    
    Uses continuous relaxation of argsort to enable gradient flow through sorting.
    Based on "Fast Differentiable Sorting and Ranking" (Blondel et al., 2020).
    
    Args:
        pred_scores: Predicted scores [B, L]
        target_ranks: Ground-truth ranks [B, L]
        obs_mask: Binary observability mask [B, L]
        tau: Temperature (lower=sharper, higher=softer)
    
    Returns:
        loss: Scalar soft sorting loss
    """
    B, L = pred_scores.shape
    device = pred_scores.device
    
    # Mask non-observable positions
    pred_scores_masked = pred_scores.clone()
    pred_scores_masked[~obs_mask.bool()] = -1e9
    
    # Pairwise comparisons (differentiable via sigmoid)
    diff = pred_scores_masked.unsqueeze(-1) - pred_scores_masked.unsqueeze(-2)  # [B, L, L]
    soft_comp = torch.sigmoid(diff / tau)  # [B, L, L]
    
    # Soft ranks: sum of pairwise comparisons (how many items are ranked lower)
    soft_ranks = soft_comp.sum(dim=-1)  # [B, L]
    
    # Only compute loss over observable positions
    mse = F.mse_loss(
        soft_ranks[obs_mask.bool()],
        target_ranks[obs_mask.bool()].float(),
        reduction='mean'
    )
    
    return mse


def combined_ranking_loss(
    pred_scores: torch.Tensor,  # [B, L] predicted scores
    target_ranks: torch.Tensor,  # [B, L] ground-truth ranks
    obs_mask: torch.Tensor,  # [B, L] observability mask
    loss_type: str = "ranknet",  # "ranknet", "listnet", "soft_sort", or "combined"
    ranknet_weight: float = 1.0,
    listnet_weight: float = 0.5,
    soft_sort_weight: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined ranking loss with multiple objectives.
    
    Args:
        pred_scores: Predicted scores [B, L]
        target_ranks: Ground-truth ranks [B, L]
        obs_mask: Binary observability mask [B, L]
        loss_type: Which loss(es) to use
        *_weight: Weights for each loss component
    
    Returns:
        total_loss: Weighted sum of losses
        components: Dict with individual loss values
    """
    components = {}
    
    if loss_type == "ranknet" or loss_type == "combined":
        ranknet = ranknet_pairwise_loss(pred_scores, target_ranks, obs_mask)
        components['ranknet'] = ranknet.item()
        if loss_type == "ranknet":
            return ranknet, components
    
    if loss_type == "listnet" or loss_type == "combined":
        listnet = listnet_cross_entropy_loss(pred_scores, target_ranks, obs_mask)
        components['listnet'] = listnet.item()
        if loss_type == "listnet":
            return listnet, components
    
    if loss_type == "soft_sort" or loss_type == "combined":
        soft_sort = soft_sorting_loss(pred_scores, target_ranks, obs_mask)
        components['soft_sort'] = soft_sort.item()
        if loss_type == "soft_sort":
            return soft_sort, components
    
    # Combined
    total_loss = (
        ranknet_weight * components.get('ranknet', 0) * (1 if 'ranknet' in components else 0) +
        listnet_weight * components.get('listnet', 0) * (1 if 'listnet' in components else 0) +
        soft_sort_weight * components.get('soft_sort', 0) * (1 if 'soft_sort' in components else 0)
    )
    
    # Re-compute as tensor if combined
    if loss_type == "combined":
        total_loss = ranknet_weight * ranknet + listnet_weight * listnet + soft_sort_weight * soft_sort
        components['total'] = total_loss.item()
    
    return total_loss, components


# ===============================================================
# Numpy versions for final evaluation (no gradients)
# ===============================================================

def compute_kendall_tau_numpy(pred, target, obs_mask=None):
    """
    Numpy version of Kendall-τ with optional masking.
    
    Args:
        pred: numpy array [L] predicted ranks/scores
        target: numpy array [L] ground-truth ranks/scores
        obs_mask: optional numpy array [L] (1=observable, 0=masked)
    
    Returns:
        tau: Scalar correlation coefficient
    """
    n = len(pred)
    concordant = 0
    discordant = 0
    total_pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            # Skip if either position is masked
            if obs_mask is not None and (not obs_mask[i] or not obs_mask[j]):
                continue
            
            pred_order = np.sign(pred[i] - pred[j])
            target_order = np.sign(target[i] - target[j])
            
            if pred_order == target_order and pred_order != 0:
                concordant += 1
            elif pred_order == -target_order and pred_order != 0:
                discordant += 1
            
            total_pairs += 1
    
    if total_pairs == 0:
        return 0.0
    
    return (concordant - discordant) / total_pairs


def batch_kendall_tau_numpy(preds, targets, obs_masks=None):
    """
    Compute Kendall-τ for a batch (numpy).
    
    Args:
        preds: numpy array [B, L]
        targets: numpy array [B, L]
        obs_masks: optional numpy array [B, L]
    
    Returns:
        mean_tau: Average τ over batch
    """
    taus = []
    for i in range(len(preds)):
        mask_i = obs_masks[i] if obs_masks is not None else None
        tau = compute_kendall_tau_numpy(preds[i], targets[i], mask_i)
        taus.append(tau)
    
    return np.mean(taus)

