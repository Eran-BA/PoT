"""
HRM-optimized Trainer with stable defaults.

Features:
- Separate optimizers for encoder and controller
- Differential gradient clipping
- Deep supervision (average over inner steps)
- Temperature scheduling
- Entropy regularization with decay
- Controller warm-up
- Comprehensive diagnostics

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

from typing import Dict, List, Optional, Any
import time

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from src.models.layers import HRMState


class HRMTrainer:
    """
    Trainer optimized for HRM-style controllers with best practices.
    
    Key features:
    - Two optimizers (encoder + controller) with different LRs
    - Separate gradient clipping per module
    - Deep supervision across inner iterations
    - Temperature annealing per epoch
    - Entropy regularization with decay
    - Controller warm-up (freeze first N epochs)
    - AMP support for speed
    - Comprehensive diagnostics
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        # Optimizer configs
        lr_encoder: float = 3e-4,
        lr_controller: float = 1e-4,
        lr_other: float = 3e-4,
        weight_decay: float = 0.01,
        betas: tuple = (0.9, 0.98),
        # Gradient clipping
        clip_encoder: float = 1.0,
        clip_controller: float = 0.5,
        clip_other: float = 1.0,
        # Controller warm-up
        controller_warmup_epochs: int = 5,
        # Temperature scheduling
        temperature_init: float = 2.0,
        temperature_min: float = 0.7,
        temperature_decay: float = 0.95,
        # Entropy regularization
        entropy_reg_init: float = 1e-3,
        entropy_decay_epochs: int = 5,
        entropy_decay_rate: float = 0.5,
        # Training
        use_amp: bool = False,
        deep_supervision: bool = True,
        # Diagnostics
        log_diagnostics: bool = True
    ):
        self.model = model
        self.device = device
        
        # Split parameters into groups
        encoder_params = []
        controller_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if 'encoder' in name or 'embed' in name:
                encoder_params.append(param)
            elif 'controller' in name or 'hrm' in name.lower():
                controller_params.append(param)
            else:
                other_params.append(param)
        
        # Create optimizers
        self.optimizer_encoder = torch.optim.AdamW(
            encoder_params,
            lr=lr_encoder,
            betas=betas,
            weight_decay=weight_decay
        )
        
        self.optimizer_controller = torch.optim.AdamW(
            controller_params,
            lr=lr_controller,
            betas=betas,
            weight_decay=weight_decay
        )
        
        self.optimizer_other = torch.optim.AdamW(
            other_params,
            lr=lr_other,
            betas=betas,
            weight_decay=weight_decay
        )
        
        # Store parameter groups for clipping
        self.param_groups = {
            'encoder': encoder_params,
            'controller': controller_params,
            'other': other_params
        }
        
        self.clip_norms = {
            'encoder': clip_encoder,
            'controller': clip_controller,
            'other': clip_other
        }
        
        # Controller warm-up
        self.controller_warmup_epochs = controller_warmup_epochs
        
        # Temperature scheduling
        self.temperature_init = temperature_init
        self.temperature_min = temperature_min
        self.temperature_decay = temperature_decay
        
        # Entropy regularization
        self.entropy_reg_init = entropy_reg_init
        self.entropy_decay_epochs = entropy_decay_epochs
        self.entropy_decay_rate = entropy_decay_rate
        
        # Training config
        self.use_amp = use_amp
        self.deep_supervision = deep_supervision
        self.log_diagnostics = log_diagnostics
        
        # AMP scaler
        self.scaler = GradScaler(enabled=use_amp) if use_amp else None
        
        # Diagnostics
        self.diagnostics = {}
    
    def set_epoch(self, epoch: int):
        """Update epoch-dependent settings."""
        # Temperature annealing
        temperature = max(
            self.temperature_min,
            self.temperature_init * (self.temperature_decay ** epoch)
        )
        
        # Set temperature on controller (if it has the method)
        if hasattr(self.model, 'controller'):
            if hasattr(self.model.controller, 'set_temperature'):
                self.model.controller.set_temperature(temperature)
        
        # Controller warm-up: freeze controller for first N epochs
        if epoch < self.controller_warmup_epochs:
            for p in self.param_groups['controller']:
                p.requires_grad = False
        else:
            for p in self.param_groups['controller']:
                p.requires_grad = True
        
        # Entropy reg decay
        self.current_entropy_weight = self.entropy_reg_init * (
            self.entropy_decay_rate ** (epoch // self.entropy_decay_epochs)
        )
        
        self.current_temperature = temperature
        self.current_epoch = epoch
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        **kwargs
    ) -> Dict[str, float]:
        """
        Single training step with HRM optimizations.
        
        Returns:
            metrics: Dict with loss, entropy, diagnostics
        """
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        
        # Forward pass with AMP
        if self.use_amp:
            with autocast():
                outputs = self.model(**batch, **kwargs)
        else:
            outputs = self.model(**batch, **kwargs)
        
        # Extract loss and aux
        if isinstance(outputs, tuple):
            loss, aux = outputs
        else:
            loss = outputs
            aux = {}
        
        # Add entropy regularization
        if 'entropy' in aux and self.current_entropy_weight > 0:
            entropy_loss = self.current_entropy_weight * aux['entropy']
            total_loss = loss + entropy_loss
        else:
            total_loss = loss
            entropy_loss = torch.tensor(0.0)
        
        # Backward with AMP
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        # Gradient clipping (per group)
        for group_name, params in self.param_groups.items():
            if len(params) > 0:
                if self.use_amp:
                    self.scaler.unscale_(getattr(self, f'optimizer_{group_name}'))
                torch.nn.utils.clip_grad_norm_(
                    params,
                    self.clip_norms[group_name]
                )
        
        # Optimizer steps
        if self.use_amp:
            self.scaler.step(self.optimizer_encoder)
            self.scaler.step(self.optimizer_controller)
            self.scaler.step(self.optimizer_other)
            self.scaler.update()
        else:
            self.optimizer_encoder.step()
            self.optimizer_controller.step()
            self.optimizer_other.step()
        
        # Zero gradients
        self.optimizer_encoder.zero_grad()
        self.optimizer_controller.zero_grad()
        self.optimizer_other.zero_grad()
        
        # Collect metrics
        metrics = {
            'loss': loss.item(),
            'total_loss': total_loss.item(),
            'entropy_loss': entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else entropy_loss,
        }
        
        # Add aux diagnostics
        if self.log_diagnostics and aux:
            for k, v in aux.items():
                if isinstance(v, torch.Tensor):
                    metrics[f'aux_{k}'] = v.item() if v.ndim == 0 else v.mean().item()
        
        return metrics
    
    def train_epoch(
        self,
        data_loader,
        epoch: int,
        **kwargs
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.set_epoch(epoch)
        
        total_loss = 0.0
        total_samples = 0
        all_metrics = {}
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(data_loader):
            metrics = self.train_step(batch, **kwargs)
            
            batch_size = len(batch.get('targets', batch.get('input_ids', [1])))
            total_loss += metrics['loss'] * batch_size
            total_samples += batch_size
            
            # Accumulate metrics
            for k, v in metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = 0.0
                all_metrics[k] += v * batch_size
        
        elapsed = time.time() - start_time
        
        # Average metrics
        epoch_metrics = {
            k: v / total_samples for k, v in all_metrics.items()
        }
        
        # Add epoch-level diagnostics
        epoch_metrics.update({
            'epoch': epoch,
            'temperature': self.current_temperature,
            'entropy_weight': self.current_entropy_weight,
            'controller_frozen': epoch < self.controller_warmup_epochs,
            'epoch_time_s': elapsed,
            'samples_per_sec': total_samples / elapsed if elapsed > 0 else 0
        })
        
        return epoch_metrics
    
    @torch.no_grad()
    def eval_epoch(
        self,
        data_loader,
        **kwargs
    ) -> Dict[str, float]:
        """Evaluate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        all_metrics = {}
        
        for batch in data_loader:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch, **kwargs)
            
            if isinstance(outputs, tuple):
                loss, aux = outputs
            else:
                loss = outputs
                aux = {}
            
            batch_size = len(batch.get('targets', batch.get('input_ids', [1])))
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Accumulate aux
            for k, v in aux.items():
                if isinstance(v, torch.Tensor):
                    val = v.item() if v.ndim == 0 else v.mean().item()
                    if k not in all_metrics:
                        all_metrics[k] = 0.0
                    all_metrics[k] += val * batch_size
        
        # Average
        eval_metrics = {
            'loss': total_loss / total_samples
        }
        eval_metrics.update({
            k: v / total_samples for k, v in all_metrics.items()
        })
        
        return eval_metrics

