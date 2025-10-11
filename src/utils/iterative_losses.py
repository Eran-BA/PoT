"""
Iterative refinement loss functions for deep supervision and differentiable halting.

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import torch
import torch.nn.functional as F
from typing import Callable, Optional


def deep_supervision_loss(
    routed_seq: torch.Tensor,  # [B, iters, T, D]
    pointer_fn: Callable,       # Function: (X, X, mask, mask) -> logits [B, T, T+1]
    targets: torch.Tensor,      # [B, T]
    mask: torch.Tensor,         # [B, T] bool mask
    weight_schedule: str = "linear",  # 'linear', 'exp', or 'uniform'
) -> torch.Tensor:
    """
    Compute weighted sum of losses across all iterations.
    
    Args:
        routed_seq: Per-iteration hidden states [B, iters, T, D]
        pointer_fn: Function to compute pointer logits from hidden state
        targets: Gold head indices [B, T]
        mask: Valid token mask [B, T]
        weight_schedule: How to weight iterations ('linear', 'exp', 'uniform')
    
    Returns:
        Weighted loss scalar
    """
    B, IT, T, D = routed_seq.shape
    device = routed_seq.device
    
    # Define iteration weights
    if weight_schedule == "linear":
        # Ramp from 0.3 to 1.0
        weights = torch.linspace(0.3, 1.0, steps=IT, device=device)
    elif weight_schedule == "exp":
        # Exponential: 2^0, 2^1, 2^2, ... (normalized)
        weights = torch.pow(2.0, torch.arange(IT, dtype=torch.float32, device=device))
    elif weight_schedule == "uniform":
        weights = torch.ones(IT, device=device)
    else:
        raise ValueError(f"Unknown weight_schedule: {weight_schedule}")
    
    # Normalize weights to sum to 1
    weights = weights / weights.sum()
    
    total_loss = 0.0
    for t in range(IT):
        routed_t = routed_seq[:, t]  # [B, T, D]
        logits_t = pointer_fn(routed_t, routed_t, mask, mask)  # [B, T, T+1]
        
        # Cross-entropy loss (only on valid tokens)
        loss_t = F.cross_entropy(
            logits_t.view(-1, logits_t.size(-1))[mask.view(-1)],
            targets.view(-1)[mask.view(-1)]
        )
        
        total_loss = total_loss + weights[t] * loss_t
    
    return total_loss


def act_expected_loss(
    routed_seq: torch.Tensor,      # [B, iters, T, D]
    halt_logits_seq: torch.Tensor, # [B, iters, T, 1] or [B, iters, 1]
    pointer_fn: Callable,           # Function: (X, X, mask, mask) -> logits [B, T, T+1]
    targets: torch.Tensor,          # [B, T]
    mask: torch.Tensor,             # [B, T] bool mask
    ponder_coef: float = 1e-3,      # Weight for ponder cost
    per_token: bool = False,        # If True, per-token halting; else per-sequence
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ACT-style expected loss with differentiable halting.
    
    This computes E[loss] = sum_t p_t * loss_t where:
      p_t = c_t * σ(h_t)  (probability of halting at step t)
      c_t = product_{s<t} (1 - σ(h_s))  (probability of continuing to step t)
    
    Args:
        routed_seq: Per-iteration hidden states [B, iters, T, D]
        halt_logits_seq: Per-iteration halting logits (before sigmoid)
        pointer_fn: Function to compute pointer logits from hidden state
        targets: Gold head indices [B, T]
        mask: Valid token mask [B, T]
        ponder_coef: Weight for ponder regularization
        per_token: If True, halt per token; else per sequence (simpler)
    
    Returns:
        (total_loss, ponder_cost): Scalars
    """
    B, IT, T, D = routed_seq.shape
    device = routed_seq.device
    
    # Compute halt probabilities
    halt_probs = torch.sigmoid(halt_logits_seq)  # [B, iters, T, 1] or [B, iters, 1]
    
    if not per_token:
        # Per-sequence halting: reduce across tokens
        halt_probs = halt_probs.squeeze(-1)  # [B, iters]
        if halt_probs.dim() == 3:  # Was [B, iters, T]
            halt_probs = halt_probs.mean(dim=-1)  # [B, iters]
    
    # Initialize continue probability
    if per_token:
        c = torch.ones(B, T, 1, device=device)  # [B, T, 1]
    else:
        c = torch.ones(B, 1, device=device)     # [B, 1]
    
    total_loss = 0.0
    ponder = 0.0
    
    for t in range(IT):
        routed_t = routed_seq[:, t]  # [B, T, D]
        
        # Compute loss at this iteration
        logits_t = pointer_fn(routed_t, routed_t, mask, mask)  # [B, T, T+1]
        ce_t = F.cross_entropy(
            logits_t.view(-1, logits_t.size(-1))[mask.view(-1)],
            targets.view(-1)[mask.view(-1)],
            reduction='mean'
        )
        
        # Probability of halting at this step
        if per_token:
            p_t = c * halt_probs[:, t:t+1, :, :]  # [B, T, 1]
            p_t_mean = p_t.mean()
        else:
            p_t = c * halt_probs[:, t:t+1]  # [B, 1]
            p_t_mean = p_t.mean()
        
        # Add weighted loss
        total_loss = total_loss + p_t_mean * ce_t
        ponder = ponder + p_t_mean
        
        # Update continue probability for next iteration
        if per_token:
            c = c * (1.0 - halt_probs[:, t:t+1, :, :])
        else:
            c = c * (1.0 - halt_probs[:, t:t+1])
    
    # Assign remaining probability to the last iteration
    if per_token:
        c_mean = c.mean()
    else:
        c_mean = c.mean()
    
    # Final CE (using last routed state for any leftover probability)
    total_loss = total_loss + c_mean * ce_t
    
    # Add ponder cost
    total_loss = total_loss + ponder_coef * ponder
    
    return total_loss, ponder


def act_deep_supervision_loss(
    routed_seq: torch.Tensor,   # [B, I, T, D]
    halt_logits: torch.Tensor,  # [B, I] per-sequence or [B, I, T] per-token
    pointer_fn: Callable,        # callable: (dep, head, mask_dep, mask_head) -> logits [B, T, T+1]
    targets: torch.Tensor,       # [B, T] gold heads (0..T)
    mask: torch.Tensor,          # [B, T] valid tokens
    ponder_coef: float = 1e-3,
    ramp_strength: float = 1.0,  # 0 = no deep supervision; 1 = full ramp
) -> tuple[torch.Tensor, dict]:
    """
    Combined ACT-style halting + deep supervision loss.
    
    This is the RECOMMENDED approach when you want both:
      - Adaptive computation (ACT halting)
      - Progressive refinement (deep supervision)
    
    Args:
        routed_seq: Per-iteration hidden states [B, I, T, D]
        halt_logits: Halting logits (before sigmoid) [B, I] or [B, I, T]
        pointer_fn: Function to compute pointer logits
        targets: Gold head indices [B, T]
        mask: Valid token mask [B, T]
        ponder_coef: Weight for ponder regularization
        ramp_strength: 0=pure ACT, 1=full deep supervision ramp
    
    Returns:
        (loss, diagnostics): Loss scalar and dict with ACT stats
    """
    B, I, T, D = routed_seq.shape
    device = routed_seq.device
    
    # Per-sequence halting (simpler, common case)
    per_token = (halt_logits.dim() == 3)
    
    if per_token:
        # Per-token halting: halt_logits is [B, I, T]
        h = torch.sigmoid(halt_logits)  # [B, I, T]
        c = torch.ones(B, T, 1, device=device)  # [B, T, 1]
        p_list = []
        for t in range(I):
            p_t = c * h[:, t:t+1, :].transpose(1, 2)  # [B, T, 1]
            c = c * (1 - h[:, t:t+1, :].transpose(1, 2))
            p_list.append(p_t.squeeze(-1))  # [B, T]
        p = torch.stack(p_list, dim=1)  # [B, I, T]
        c_rem = c.squeeze(-1)  # [B, T]
    else:
        # Per-sequence halting: halt_logits is [B, I]
        h = torch.sigmoid(halt_logits)  # [B, I]
        c = torch.ones(B, 1, device=device)  # [B, 1]
        p_list = []
        for t in range(I):
            p_t = c * h[:, t:t+1]  # [B, 1]
            c = c * (1 - h[:, t:t+1])
            p_list.append(p_t)
        p = torch.cat(p_list, dim=1)  # [B, I]
        c_rem = c.squeeze(-1)  # [B]
    
    # Deep supervision ramp: linearly increase from 0.3 to 1.0
    w_lin = torch.linspace(0.3, 1.0, steps=I, device=device)  # [I]
    w_lin = w_lin / w_lin.sum()
    w = (1 - ramp_strength) * torch.ones_like(w_lin) / I + ramp_strength * w_lin  # [I]
    
    # Combine ACT weights and ramp: alpha_t = normalize(p_t * w_t)
    if per_token:
        pw = p * w.view(1, I, 1)  # [B, I, T]
        Z = pw.sum(dim=1, keepdim=True).clamp_min(1e-8)  # [B, 1, T]
        alpha = pw / Z  # [B, I, T]
    else:
        pw = p * w.view(1, I)  # [B, I]
        Z = pw.sum(dim=1, keepdim=True).clamp_min(1e-8)  # [B, 1]
        alpha = pw / Z  # [B, I]
    
    # Per-iteration CE
    loss = 0.0
    last_ce = None
    for t in range(I):
        routed_t = routed_seq[:, t]  # [B, T, D]
        logits_t = pointer_fn(routed_t, routed_t, mask, mask)  # [B, T, T+1]
        ce_flat = F.cross_entropy(
            logits_t.view(-1, logits_t.size(-1))[mask.view(-1)],
            targets.view(-1)[mask.view(-1)],
            reduction="none"
        )
        ce_t = ce_flat.mean()
        last_ce = ce_t
        
        # Expected contribution for step t
        if per_token:
            loss = loss + (alpha[:, t, :].mean() * ce_t)
        else:
            loss = loss + (alpha[:, t].mean() * ce_t)
    
    # Add leftover mass to the last step's CE
    if last_ce is not None:
        loss = loss + c_rem.mean() * last_ce
    
    # Ponder (compute) penalty
    ponder = p.mean()
    loss = loss + ponder_coef * ponder
    
    return loss, {
        "act_mean_halt_prob": p.mean().item(),
        "act_leftover_mass": c_rem.mean().item(),
        "ramp_strength": float(ramp_strength),
        "ponder": ponder.item()
    }


def compute_per_iter_metrics(
    routed_seq: torch.Tensor,  # [B, iters, T, D]
    pointer_fn: Callable,       # Function: (X, X, mask, mask) -> logits [B, T, T+1]
    targets: torch.Tensor,      # [B, T]
    mask: torch.Tensor,         # [B, T] bool mask
) -> dict:
    """
    Compute UAS for each iteration (for logging/analysis).
    
    Returns:
        dict with keys:
          - 'uas_per_iter': list of UAS scores [iter1_uas, iter2_uas, ...]
          - 'ce_per_iter': list of CE losses
          - 'refinement': list of |uas_{t+1} - uas_t|
    """
    B, IT, T, D = routed_seq.shape
    uas_list = []
    ce_list = []
    
    with torch.no_grad():
        for t in range(IT):
            routed_t = routed_seq[:, t]
            logits_t = pointer_fn(routed_t, routed_t, mask, mask)
            
            # UAS
            pred_t = logits_t.argmax(-1)
            uas_t = (pred_t[mask] == targets[mask]).float().mean().item()
            uas_list.append(uas_t)
            
            # CE
            ce_t = F.cross_entropy(
                logits_t.view(-1, logits_t.size(-1))[mask.view(-1)],
                targets.view(-1)[mask.view(-1)]
            ).item()
            ce_list.append(ce_t)
    
    # Compute refinement (delta UAS between consecutive iters)
    refinement = [uas_list[i+1] - uas_list[i] for i in range(len(uas_list)-1)]
    
    return {
        'uas_per_iter': uas_list,
        'ce_per_iter': ce_list,
        'refinement': refinement
    }

