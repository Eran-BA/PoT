"""
Training manager for dependency parsing models.

Handles training loops, optimizer setup (including differentiated learning rates
for PoH), metric tracking, and CoNLL-U export.

Classes:
    Trainer: Main training manager

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import time
from math import ceil
from typing import List, Dict, Optional, Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from src.data.collate import collate_batch


class Trainer:
    """Training manager for dependency parsing models.
    
    Manages training loops, evaluation, optimizer setup with differentiated
    learning rates for PoH models, metric aggregation, and optional CoNLL-U export.
    
    Key Features:
        - Differentiated learning rates for PoH (controller gets 20x higher LR)
        - Gradient clipping for stability
        - Automatic metric aggregation (UAS, LAS, routing entropy, etc.)
        - CoNLL-U export support for official evaluation
        - Punctuation masking support
        - TRM-style forward pass support
        
    Args:
        model: Parser model (baseline or PoH)
        tokenizer: HuggingFace tokenizer
        device: Training device
        label_vocab: Optional label vocabulary for LAS
        
    Example:
        >>> from src.models import PoHParser
        >>> from transformers import AutoTokenizer
        >>> 
        >>> model = PoHParser(d_model=768, n_heads=8, d_ff=2048)
        >>> tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        >>> trainer = Trainer(model, tokenizer, device='cuda')
        >>> 
        >>> # Training
        >>> metrics = trainer.train_epoch(
        ...     train_data,
        ...     batch_size=32,
        ...     lr=3e-5,
        ...     weight_decay=0.01,
        ...     scheduler=scheduler
        ... )
        >>> print(f"Train UAS: {metrics['uas']:.3f}")
        >>> 
        >>> # Evaluation
        >>> eval_metrics = trainer.eval_epoch(dev_data, batch_size=32)
        >>> print(f"Dev UAS: {eval_metrics['uas']:.3f}, LAS: {eval_metrics['las']:.3f}")
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        device: torch.device,
        label_vocab: Optional[Dict[str, int]] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.label_vocab = label_vocab
        self.optimizers = {}  # Cache optimizers per model
        
    def _get_optimizer(
        self,
        lr: float,
        weight_decay: float
    ) -> torch.optim.Optimizer:
        """Get or create optimizer for current model.
        
        Uses differentiated learning rates for PoH models:
        - Encoder: base LR
        - Controller: 20x base LR (handles gradient imbalance)
        - Other components: 2x base LR
        
        Args:
            lr: Base learning rate
            weight_decay: Weight decay coefficient
            
        Returns:
            AdamW optimizer with appropriate parameter groups
        """
        model_id = id(self.model)
        
        if model_id not in self.optimizers:
            # Check if PoH model (has controller)
            if hasattr(self.model, 'block') and hasattr(self.model.block, 'controller'):
                # PoH model: use differentiated learning rates
                encoder_params = list(self.model.encoder.parameters())
                controller_params = list(self.model.block.controller.parameters())
                other_params = [
                    p for n, p in self.model.named_parameters()
                    if 'encoder' not in n and 'controller' not in n
                ]
                
                self.optimizers[model_id] = torch.optim.AdamW([
                    {'params': encoder_params, 'lr': lr},
                    {'params': controller_params, 'lr': lr * 20},  # 20x for controller
                    {'params': other_params, 'lr': lr * 2}  # 2x for others
                ], weight_decay=weight_decay)
            else:
                # Baseline model: uniform LR
                self.optimizers[model_id] = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay
                )
        
        return self.optimizers[model_id]
    
    def train_epoch(
        self,
        data: List[Dict],
        batch_size: int = 32,
        lr: float = 5e-5,
        weight_decay: float = 0.01,
        scheduler: Optional[Any] = None,
        args: Optional[Any] = None,
        *,
        use_amp: bool = False,
        grad_accum_steps: int = 1
    ) -> Dict[str, float]:
        """Run one training epoch.
        
        Args:
            data: List of training examples
            batch_size: Batch size
            lr: Learning rate
            weight_decay: Weight decay coefficient
            scheduler: Optional LR scheduler
            args: Optional arguments (for TRM mode, etc.)
            
        Returns:
            Dictionary with aggregated metrics
        """
        self.model.train()
        optimizer = self._get_optimizer(lr, weight_decay)
        steps = 0
        optimizer.zero_grad(set_to_none=True)

        # AMP scaler (CUDA only)
        has_cuda = torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and has_cuda))
        
        total_loss = 0.0
        total_tokens = 0
        total_iters = 0.0
        start_time = time.time()
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            subw, word_ids, heads, labels = collate_batch(
                batch, self.tokenizer, self.device, self.label_vocab
            )
            
            # TRM mode support
            if args and hasattr(args, 'trm_mode') and args.trm_mode and hasattr(self.model, 'forward_trm'):
                if scaler.is_enabled():
                    with torch.cuda.amp.autocast():
                        loss, metrics = self.model.forward_trm(subw, word_ids, heads, labels, args=args)
                else:
                    loss, metrics = self.model.forward_trm(subw, word_ids, heads, labels, args=args)
            else:
                if scaler.is_enabled():
                    with torch.cuda.amp.autocast():
                        loss, metrics = self.model(subw, word_ids, heads, labels)
                else:
                    loss, metrics = self.model(subw, word_ids, heads, labels)

            # Gradient scaling / accumulation
            loss_to_backprop = loss / max(1, grad_accum_steps)
            if scaler.is_enabled():
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            # Step when reaching accumulation boundary
            if (steps + 1) % max(1, grad_accum_steps) == 0:
                # Clip gradients
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Optimizer step
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if scheduler is not None:
                    scheduler.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            steps += 1
            total_tokens += metrics["tokens"]
            
            if "inner_iters_used" in metrics:
                total_iters += metrics["inner_iters_used"]
        
        # Compute epoch metrics
        elapsed = time.time() - start_time
        return {
            "loss": total_loss / steps if steps > 0 else 0.0,
            "uas": 0.0,  # UAS tracked per-batch in training
            "las": 0.0,
            "mean_inner_iters": total_iters / steps if steps > 0 else 0.0,
            "time": elapsed,
            "tokens": total_tokens
        }
    
    def eval_epoch(
        self,
        data: List[Dict],
        batch_size: int = 32,
        emit_conllu: bool = False,
        conllu_path: Optional[str] = None,
        ignore_punct: bool = False,
        args: Optional[Any] = None
    ) -> Dict[str, float]:
        """Run evaluation epoch.
        
        Args:
            data: List of evaluation examples
            batch_size: Batch size
            emit_conllu: Whether to write CoNLL-U predictions
            conllu_path: Path for CoNLL-U output
            ignore_punct: Whether to ignore punctuation in metrics
            args: Optional arguments (for TRM mode, etc.)
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        total_correct_uas = 0
        total_correct_las = 0
        total_iters = 0.0
        steps = 0
        start_time = time.time()
        
        # For CoNLL-U export
        all_tokens = []
        all_heads_gold = []
        all_deprels_gold = []
        all_heads_pred = []
        all_deprels_pred = []
        
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                
                if ignore_punct:
                    subw, word_ids, heads, labels, deprels_str = collate_batch(
                        batch, self.tokenizer, self.device,
                        self.label_vocab, return_deprels=True
                    )
                else:
                    subw, word_ids, heads, labels = collate_batch(
                        batch, self.tokenizer, self.device, self.label_vocab
                    )
                
                # Forward pass
                if args and hasattr(args, 'trm_mode') and args.trm_mode and hasattr(self.model, 'forward_trm'):
                    loss, metrics = self.model.forward_trm(subw, word_ids, heads, labels, args=args)
                else:
                    loss, metrics = self.model(subw, word_ids, heads, labels)
                
                # Accumulate metrics
                total_loss += loss.item()
                steps += 1
                total_tokens += metrics["tokens"]
                total_correct_uas += metrics["uas"] * metrics["tokens"]
                total_correct_las += metrics["las"] * metrics["tokens"]
                
                if "inner_iters_used" in metrics:
                    total_iters += metrics["inner_iters_used"]
                
                # Collect for CoNLL-U export
                if emit_conllu and "pred_heads" in metrics:
                    all_tokens.extend([ex["tokens"] for ex in batch])
                    all_heads_gold.extend(heads)
                    all_heads_pred.extend(metrics["pred_heads"])
                    
                    if "pred_labels" in metrics:
                        all_deprels_gold.extend([ex.get("deprel", []) for ex in batch])
                        all_deprels_pred.extend(metrics["pred_labels"])
        
        # Write CoNLL-U if requested
        if emit_conllu and conllu_path and all_tokens:
            from src.utils.conllu_writer import write_conllu
            write_conllu(
                conllu_path,
                all_tokens,
                all_heads_gold,
                all_deprels_gold if all_deprels_gold else None,
                all_heads_pred,
                all_deprels_pred if all_deprels_pred else None
            )
        
        # Compute epoch metrics
        elapsed = time.time() - start_time
        return {
            "loss": total_loss / steps if steps > 0 else 0.0,
            "uas": total_correct_uas / total_tokens if total_tokens > 0 else 0.0,
            "las": total_correct_las / total_tokens if total_tokens > 0 else 0.0,
            "mean_inner_iters": total_iters / steps if steps > 0 else 0.0,
            "time": elapsed,
            "tokens": total_tokens
        }

