"""
Improved Training Utilities for Sorting Task

Implements:
1. Two-optimizer setup (encoder vs controller)
2. Temperature scheduling
3. Entropy regularization with decay
4. Deep supervision
5. Controller warm-up
6. Comprehensive diagnostics

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from ranking_utils import (
    ranknet_pairwise_loss,
    compute_mask_aware_kendall_tau,
    batch_kendall_tau_numpy
)


class ImprovedTrainer:
    """
    Enhanced trainer with all stability improvements.
    """
    
    def __init__(
        self,
        model,
        device,
        # Optimizer settings
        lr_encoder: float = 3e-4,
        lr_controller: float = 1e-4,
        weight_decay: float = 0.01,
        # Gradient clipping
        clip_encoder: float = 1.0,
        clip_controller: float = 0.5,
        # Temperature schedule
        temperature_init: float = 2.0,
        temperature_min: float = 0.8,
        temperature_decay: float = 0.95,
        # Entropy regularization
        entropy_reg_init: float = 1e-3,
        entropy_decay_rate: float = 0.5,
        entropy_decay_every: int = 5,
        # Controller warm-up
        controller_warmup_epochs: int = 5,
        # Loss type
        ranking_loss_type: str = "ranknet",  # "ranknet", "listnet", "soft_sort"
    ):
        self.model = model
        self.device = device
        
        # Separate optimizers
        encoder_params = model.get_encoder_parameters()
        controller_params = model.get_controller_parameters()
        
        self.optimizer_encoder = torch.optim.AdamW(
            encoder_params,
            lr=lr_encoder,
            betas=(0.9, 0.98),
            weight_decay=weight_decay
        )
        
        if len(controller_params) > 0:
            self.optimizer_controller = torch.optim.AdamW(
                controller_params,
                lr=lr_controller,
                betas=(0.9, 0.98),
                weight_decay=weight_decay
            )
        else:
            self.optimizer_controller = None
        
        # Clipping
        self.clip_encoder = clip_encoder
        self.clip_controller = clip_controller
        
        # Temperature schedule
        self.temperature_init = temperature_init
        self.temperature_min = temperature_min
        self.temperature_decay = temperature_decay
        self.current_temperature = temperature_init
        
        # Entropy regularization
        self.entropy_reg_init = entropy_reg_init
        self.entropy_decay_rate = entropy_decay_rate
        self.entropy_decay_every = entropy_decay_every
        self.current_entropy_weight = entropy_reg_init
        
        # Warm-up
        self.controller_warmup_epochs = controller_warmup_epochs
        self.current_epoch = 0
        
        # Loss type
        self.ranking_loss_type = ranking_loss_type
    
    def update_schedules(self, epoch: int):
        """Update temperature and entropy weight for current epoch."""
        self.current_epoch = epoch
        
        # Temperature annealing
        self.current_temperature = max(
            self.temperature_min,
            self.temperature_init * (self.temperature_decay ** epoch)
        )
        self.model.set_temperature(self.current_temperature)
        
        # Entropy weight decay
        decay_steps = epoch // self.entropy_decay_every
        self.current_entropy_weight = self.entropy_reg_init * (
            self.entropy_decay_rate ** decay_steps
        )
        
        # Controller warm-up: freeze controller for first few epochs
        if self.optimizer_controller is not None:
            if epoch < self.controller_warmup_epochs:
                # Freeze controller
                for param in self.model.get_controller_parameters():
                    param.requires_grad = False
            else:
                # Unfreeze controller
                for param in self.model.get_controller_parameters():
                    param.requires_grad = True
    
    def compute_loss(
        self,
        logits: torch.Tensor,  # [B, N, N]
        targets: torch.Tensor,  # [B, N]
        obs_mask: torch.Tensor,  # [B, N]
        diagnostics: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute ranking-aware loss with mask awareness.
        
        Args:
            logits: Pointer logits [B, N, N]
            targets: Ground-truth permutation [B, N]
            obs_mask: Observability mask [B, N]
            diagnostics: Optional routing diagnostics from model
        
        Returns:
            loss: Total loss
            components: Dict with loss breakdown
        """
        B, N, _ = logits.shape
        
        # Convert logits to scores for ranking loss
        # We need to invert the pointer network output:
        # logits[b, t, i] = score for selecting position i at rank t
        # We want: score[b, i] = how good is position i (for ranking)
        
        # Simple approach: average logits over all ranks for each position
        position_scores = logits.mean(dim=1)  # [B, N]
        
        # Ground-truth ranks: inverse of target permutation
        # targets[b, t] = position selected at rank t
        # We need: ranks[b, i] = what rank is position i
        device = targets.device
        ranks = torch.zeros_like(targets)  # [B, N]
        for b in range(B):
            ranks[b, targets[b]] = torch.arange(N, device=device)
        
        # Ranking loss (mask-aware)
        if self.ranking_loss_type == "ranknet":
            ranking_loss = ranknet_pairwise_loss(position_scores, ranks, obs_mask)
        else:
            # Fallback to standard CE (for compatibility)
            ranking_loss = F.cross_entropy(
                logits.reshape(B * N, N),
                targets.reshape(B * N),
                reduction="mean"
            )
        
        components = {"ranking_loss": ranking_loss.item()}
        
        # Entropy regularization (if diagnostics available)
        entropy_loss = torch.tensor(0.0, device=device)
        if diagnostics is not None and 'entropies' in diagnostics:
            # Average entropy across iterations
            mean_entropy = np.mean(diagnostics['entropies'])
            # We want to maximize entropy (minimize negative entropy)
            entropy_loss = -self.current_entropy_weight * mean_entropy
            components["entropy_reg"] = entropy_loss.item()
            components["mean_entropy"] = mean_entropy
        
        # Total loss
        total_loss = ranking_loss + entropy_loss
        components["total_loss"] = total_loss.item()
        
        return total_loss, components
    
    def train_step(
        self,
        batch_arrays: torch.Tensor,  # [B, N, 1]
        batch_targets: torch.Tensor,  # [B, N]
        batch_obs_masks: torch.Tensor,  # [B, N]
    ) -> Dict:
        """Single training step."""
        self.model.train()
        
        # Forward pass
        logits, _, diagnostics = self.model(
            batch_arrays,
            targets=batch_targets,
            return_diagnostics=True
        )
        
        # Compute loss
        loss, components = self.compute_loss(
            logits,
            batch_targets,
            batch_obs_masks,
            diagnostics
        )
        
        # Backward
        self.optimizer_encoder.zero_grad()
        if self.optimizer_controller is not None:
            self.optimizer_controller.zero_grad()
        
        loss.backward()
        
        # Gradient clipping (separate for encoder and controller)
        encoder_params = self.model.get_encoder_parameters()
        torch.nn.utils.clip_grad_norm_(encoder_params, self.clip_encoder)
        
        if self.optimizer_controller is not None and self.current_epoch >= self.controller_warmup_epochs:
            controller_params = self.model.get_controller_parameters()
            torch.nn.utils.clip_grad_norm_(controller_params, self.clip_controller)
        
        # Optimizer steps
        self.optimizer_encoder.step()
        if self.optimizer_controller is not None and self.current_epoch >= self.controller_warmup_epochs:
            self.optimizer_controller.step()
        
        # Add schedule info to components
        components["temperature"] = self.current_temperature
        components["entropy_weight"] = self.current_entropy_weight
        components["controller_frozen"] = self.current_epoch < self.controller_warmup_epochs
        
        return components
    
    def eval_step(
        self,
        batch_arrays: torch.Tensor,
        batch_targets: torch.Tensor,
        batch_obs_masks: torch.Tensor,
    ) -> Dict:
        """Single evaluation step."""
        self.model.eval()
        
        with torch.no_grad():
            logits, _, diagnostics = self.model(
                batch_arrays,
                targets=None,  # Greedy inference
                return_diagnostics=True
            )
            
            # Get predictions
            preds = logits.argmax(dim=-1)  # [B, N]
            
            # Compute metrics
            # 1. Accuracy
            correct = (preds == batch_targets).sum().item()
            total = batch_targets.numel()
            accuracy = correct / total
            
            # 2. Perfect sorts
            perfect = (preds == batch_targets).all(dim=1).sum().item()
            perfect_rate = perfect / len(preds)
            
            # 3. Mask-aware Kendall-τ
            # Convert predictions and targets to ranks
            B, N = preds.shape
            device = preds.device
            
            # For Kendall-τ, we need the inverse: position → rank
            pred_ranks = torch.zeros_like(preds, dtype=torch.float32)
            target_ranks = torch.zeros_like(batch_targets, dtype=torch.float32)
            
            for b in range(B):
                pred_ranks[b, preds[b]] = torch.arange(N, device=device, dtype=torch.float32)
                target_ranks[b, batch_targets[b]] = torch.arange(N, device=device, dtype=torch.float32)
            
            # Compute mask-aware Kendall-τ
            kendall = compute_mask_aware_kendall_tau(
                pred_ranks,
                target_ranks,
                batch_obs_masks
            )
            
            metrics = {
                "accuracy": accuracy,
                "perfect_rate": perfect_rate,
                "kendall_tau": kendall.item(),
            }
            
            # Add diagnostics if available
            if diagnostics is not None:
                if 'entropies' in diagnostics:
                    metrics["mean_entropy"] = np.mean(diagnostics['entropies'])
                if 'alphas' in diagnostics:
                    # Herfindahl index (concentration)
                    alphas = diagnostics['alphas']  # [B, iters, T, H]
                    alpha_final = alphas[:, -1]  # [B, T, H]
                    herfindahl = (alpha_final ** 2).sum(dim=-1).mean().item()
                    metrics["herfindahl"] = herfindahl
                    
                    # Max alpha (how peaked is the routing)
                    max_alpha = alpha_final.max(dim=-1)[0].mean().item()
                    metrics["max_alpha"] = max_alpha
            
            return metrics


def train_epoch_improved(
    trainer: ImprovedTrainer,
    arrays: torch.Tensor,
    targets: torch.Tensor,
    obs_masks: torch.Tensor,
    batch_size: int,
) -> Dict:
    """Train for one epoch with improved trainer."""
    num_samples = len(arrays)
    indices = torch.randperm(num_samples)
    
    all_components = []
    
    for start_idx in range(0, num_samples, batch_size):
        batch_idx = indices[start_idx : start_idx + batch_size]
        batch_arrays = arrays[batch_idx].to(trainer.device)
        batch_targets = targets[batch_idx].to(trainer.device)
        batch_obs_masks = obs_masks[batch_idx].to(trainer.device)
        
        components = trainer.train_step(batch_arrays, batch_targets, batch_obs_masks)
        all_components.append(components)
    
    # Average metrics
    avg_metrics = {}
    keys = all_components[0].keys()
    for key in keys:
        values = [c[key] for c in all_components if not isinstance(c[key], bool)]
        if values:
            avg_metrics[key] = np.mean(values)
    
    # Add boolean flags
    avg_metrics["controller_frozen"] = all_components[0]["controller_frozen"]
    
    return avg_metrics


def eval_epoch_improved(
    trainer: ImprovedTrainer,
    arrays: torch.Tensor,
    targets: torch.Tensor,
    obs_masks: torch.Tensor,
    batch_size: int,
) -> Dict:
    """Evaluate for one epoch with improved trainer."""
    num_samples = len(arrays)
    
    all_metrics = []
    
    for start_idx in range(0, num_samples, batch_size):
        batch_arrays = arrays[start_idx : start_idx + batch_size].to(trainer.device)
        batch_targets = targets[start_idx : start_idx + batch_size].to(trainer.device)
        batch_obs_masks = obs_masks[start_idx : start_idx + batch_size].to(trainer.device)
        
        metrics = trainer.eval_step(batch_arrays, batch_targets, batch_obs_masks)
        all_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {}
    keys = all_metrics[0].keys()
    for key in keys:
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = np.mean(values)
    
    return avg_metrics

