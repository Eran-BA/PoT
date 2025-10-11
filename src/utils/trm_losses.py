# utils/trm_losses.py
"""
TRM-style (Tiny Recursive Model) deep supervision across answer refreshes.

Based on "Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871)
"""

import torch
import torch.nn.functional as F
from typing import List, Callable


def trm_supervised_loss(
    model_pointer: Callable,  # pointer(dep, head, mask_dep, mask_head) -> [B,T,T+1]
    Zs: List[torch.Tensor],   # List of z_t: each [B,T,D] from each supervision step
    Y: torch.Tensor,           # [B,T] gold heads
    pad: torch.Tensor,         # [B,T] padding mask
    ramp_strength: float = 1.0
):
    """
    Deep supervision across answer refreshes: CE at each z_t with ramped weights.
    
    Args:
        model_pointer: Callable that takes (dep, head, mask_dep, mask_head) and returns logits
        Zs: List of latent states from each TRM supervision step
        Y: Gold head indices
        pad: Valid token mask
        ramp_strength: Weight ramp (0=flat, 1=linear 0.3..1.0)
    
    Returns:
        loss: Weighted CE loss
        logits_last: Final pointer logits (for UAS computation)
    """
    S = len(Zs)
    device = Zs[0].device
    
    # Ramp weights: linear from 0.3 to 1.0
    w_lin = torch.linspace(0.3, 1.0, steps=S, device=device)
    w_lin = w_lin / w_lin.sum()
    
    # Interpolate between uniform and ramped
    w = (1 - ramp_strength) * (torch.ones_like(w_lin) / S) + ramp_strength * w_lin
    
    loss = 0.0
    logits_last = None
    
    for s, z in enumerate(Zs):
        logits = model_pointer(z, z, pad, pad)  # [B,T,T+1]
        logits_last = logits
        
        ce = F.cross_entropy(
            logits.view(-1, logits.size(-1))[pad.view(-1)],
            Y.view(-1)[pad.view(-1)]
        )
        loss = loss + w[s] * ce
    
    return loss, logits_last

