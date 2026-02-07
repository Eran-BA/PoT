"""
Stability Probes for PoT/HRM Solvers.

Diagnostic probes that measure whether the model behaves like an iterative
solver (converging to a fixed point) or a one-shot predictor. These run
during evaluation only with no impact on training.

Three probes:
- E_fp (fixed-point residual): does one more step change the state?
- Per-step delta: are hidden states converging across ACT steps?
- E_noise (noise sensitivity): is the final state a stable attractor?

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from tqdm import tqdm
from dataclasses import dataclass


def _is_main_process() -> bool:
    """Check if this is the main process (rank 0 or non-distributed)."""
    rank = int(os.environ.get('RANK', 0))
    return rank == 0


def compute_stability_probes(
    model: nn.Module,
    dataloader,
    device: torch.device,
    max_batches: int = 10,
    noise_std: float = 0.01,
) -> Dict[str, float]:
    """
    Compute stability probes on a trained model during evaluation.
    
    Measures three aspects of solver behavior:
    1. Fixed-point residual: does one more step change the output?
    2. Per-step delta: are hidden states converging across ACT steps?
    3. Noise sensitivity: is the final state robust to perturbations?
    
    All computations are done in torch.no_grad().
    
    Args:
        model: Trained HybridPoHHRMSolver (or DDP-wrapped)
        dataloader: Evaluation data loader
        device: Device
        max_batches: Maximum number of batches to probe (for speed)
        noise_std: Standard deviation of Gaussian noise for E_noise probe
        
    Returns:
        Dict of probe metrics (keys prefixed with "probe/")
    """
    model.eval()
    base_model = model.module if hasattr(model, 'module') else model
    
    # Check if model supports probes (needs ACT with halt_max_steps > 1)
    if not hasattr(base_model, 'forward_with_probes'):
        return {}
    if not hasattr(base_model, 'halt_max_steps') or base_model.halt_max_steps <= 1:
        return {}
    
    # Accumulators
    fp_H_sum = 0.0
    fp_L_sum = 0.0
    noise_sum = 0.0
    delta_sums = {}  # step_idx -> sum of deltas
    count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            
            inp = batch['input'].to(device)
            puzzle_ids = batch['puzzle_id'].to(device)
            B = inp.size(0)
            
            # Forward with probes
            probe_out = base_model.forward_with_probes(inp, puzzle_ids)
            
            final_carry = probe_out['final_carry']
            intermediate_hiddens = probe_out['intermediate_hiddens']
            input_emb = probe_out['input_emb']
            
            if final_carry is None:
                # No ACT, skip probes
                continue
            
            # ---- Probe 1: Fixed-point residual (E_fp) ----
            # Run one more ACT step from the final carry
            next_carry, next_hidden, _, _ = base_model._single_act_step(
                input_emb, final_carry, use_grad=False
            )
            
            # ||z_H_new - z_H_old|| averaged over batch
            fp_H = (next_carry.z_H - final_carry.z_H).norm(dim=-1).mean().item()
            fp_L = (next_carry.z_L - final_carry.z_L).norm(dim=-1).mean().item()
            fp_H_sum += fp_H * B
            fp_L_sum += fp_L * B
            
            # ---- Probe 2: Per-step delta ----
            for i in range(len(intermediate_hiddens) - 1):
                delta = (intermediate_hiddens[i + 1] - intermediate_hiddens[i]).norm(dim=-1).mean().item()
                step_key = i + 1  # delta between step i and i+1
                delta_sums[step_key] = delta_sums.get(step_key, 0.0) + delta * B
            
            # ---- Probe 3: Noise sensitivity (E_noise) ----
            # Create noisy carry
            from src.pot.models.hybrid_hrm import ACTCarry
            noisy_carry = ACTCarry(
                z_H=final_carry.z_H + torch.randn_like(final_carry.z_H) * noise_std,
                z_L=final_carry.z_L + torch.randn_like(final_carry.z_L) * noise_std,
                L_ptr_state=final_carry.L_ptr_state,
                H_ptr_state=final_carry.H_ptr_state,
                L_inj_mem=final_carry.L_inj_mem,
                H_inj_mem=final_carry.H_inj_mem,
            )
            
            # Run one step from clean and noisy carry
            _, clean_hidden, _, _ = base_model._single_act_step(
                input_emb, final_carry, use_grad=False
            )
            _, noisy_hidden, _, _ = base_model._single_act_step(
                input_emb, noisy_carry, use_grad=False
            )
            
            noise_diff = (noisy_hidden - clean_hidden).norm(dim=-1).mean().item()
            noise_sum += noise_diff * B
            
            count += B
    
    if count == 0:
        return {}
    
    # Build results
    result = {
        "probe/E_fp_H": fp_H_sum / count,
        "probe/E_fp_L": fp_L_sum / count,
        "probe/E_noise": noise_sum / count,
    }
    
    # Per-step deltas
    for step_key in sorted(delta_sums.keys()):
        result[f"probe/delta_step_{step_key}"] = delta_sums[step_key] / count
    
    # Delta ratio: last / first (< 1 means converging)
    if delta_sums:
        sorted_keys = sorted(delta_sums.keys())
        first_delta = delta_sums[sorted_keys[0]] / count
        last_delta = delta_sums[sorted_keys[-1]] / count
        if first_delta > 1e-10:
            result["probe/delta_ratio"] = last_delta / first_delta
        else:
            result["probe/delta_ratio"] = 0.0
    
    return result
