"""
Sokoban Supervised Training.

IDENTICAL to Sudoku training:
- Cross-entropy loss on action prediction
- Q-halt loss for PoT models
- Same training loop structure

This is NOT RL - pure supervised learning like Sudoku.

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm


# =============================================================================
# Training Functions (Identical to Sudoku)
# =============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_pot: bool = True,
    scheduler: Optional[Any] = None,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """
    Train for one epoch - IDENTICAL to Sudoku's train_epoch.
    
    Loss structure (same as Sudoku):
    1. Cross-entropy loss on action prediction
    2. Q-halt loss (binary cross-entropy) if PoT model
    3. Q-continue loss (MSE) if ACT enabled
    
    Args:
        model: PoT or baseline Sokoban solver
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch number
        use_pot: If True, use PoT-specific losses (q_halt, etc.)
        scheduler: Optional LR scheduler
        grad_clip: Gradient clipping max norm
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    total_loss = 0.0
    correct_actions = 0
    total_actions = 0
    total_steps = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        # Get batch data
        inp = batch['input'].to(device)  # [B, H, W, 7] one-hot
        label = batch['label'].to(device)  # [B] action labels
        
        optimizer.zero_grad()
        
        # Forward pass
        # Model returns: (action_logits, value, q_halt, q_continue, aux)
        model_out = model(inp, return_aux=True)
        
        if len(model_out) == 5:
            logits, value, q_halt, q_continue, aux = model_out
        elif len(model_out) == 3:
            # Baseline model: (logits, value, aux)
            logits, value, aux = model_out
            q_halt = None
            q_continue = None
        else:
            raise ValueError(f"Unexpected model output length: {len(model_out)}")
        
        # =========================================
        # LOSS 1: Cross-entropy on action (like Sudoku on cells)
        # =========================================
        ce_loss = F.cross_entropy(logits, label)
        
        # =========================================
        # LOSS 2: Q-halt loss (like Sudoku)
        # =========================================
        if use_pot and q_halt is not None:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                is_correct = (preds == label).float()
            
            q_halt_loss = F.binary_cross_entropy_with_logits(q_halt, is_correct)
            loss = ce_loss + 0.5 * q_halt_loss
            
            # =========================================
            # LOSS 3: Q-continue loss (ACT Q-learning, like Sudoku)
            # =========================================
            if q_continue is not None:
                # Use model's target_q_continue if available (ACT look-ahead)
                # Otherwise fall back to is_correct (simpler supervised target)
                target_q_continue = None
                if aux is not None:
                    target_q_continue = aux.get('target_q_continue')
                if target_q_continue is None:
                    target_q_continue = is_correct
                q_continue_loss = F.mse_loss(torch.sigmoid(q_continue), target_q_continue)
                loss = loss + 0.5 * q_continue_loss
        else:
            loss = ce_loss
        
        # Backward pass
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Step scheduler
        if scheduler:
            scheduler.step()
        
        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct_actions += (preds == label).sum().item()
        total_actions += label.numel()
        total_steps += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct_actions / total_actions:.2%}',
        })
    
    return {
        'loss': total_loss / total_steps,
        'accuracy': correct_actions / total_actions,
        'correct': correct_actions,
        'total': total_actions,
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model - IDENTICAL structure to Sudoku's evaluate.
    
    Args:
        model: Trained model
        dataloader: Validation/test data loader
        device: Device to use
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    total_loss = 0.0
    correct_actions = 0
    total_actions = 0
    total_steps = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inp = batch['input'].to(device)
            label = batch['label'].to(device)
            
            # Forward pass
            model_out = model(inp, return_aux=False)
            
            if len(model_out) == 5:
                logits, value, q_halt, q_continue, _ = model_out
            elif len(model_out) == 3:
                logits, value, _ = model_out
            else:
                raise ValueError(f"Unexpected model output length: {len(model_out)}")
            
            # Loss
            ce_loss = F.cross_entropy(logits, label)
            total_loss += ce_loss.item()
            
            # Accuracy
            preds = logits.argmax(dim=-1)
            correct_actions += (preds == label).sum().item()
            total_actions += label.numel()
            total_steps += 1
    
    return {
        'loss': total_loss / total_steps,
        'accuracy': correct_actions / total_actions,
        'correct': correct_actions,
        'total': total_actions,
    }


# =============================================================================
# Full Training Loop
# =============================================================================

def train_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    warmup_steps: int = 100,
    lr_min_ratio: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
    use_pot: bool = True,
    save_dir: Optional[str] = None,
    wandb_log: bool = False,
) -> Dict[str, Any]:
    """
    Full supervised training loop for Sokoban.
    
    IDENTICAL to Sudoku training approach.
    
    Args:
        model: PoT or baseline Sokoban solver
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to use
        epochs: Number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        grad_clip: Gradient clipping
        warmup_steps: LR warmup steps
        lr_min_ratio: Minimum LR as fraction of peak (for cosine decay)
        betas: AdamW betas (same as Sudoku: 0.9, 0.95)
        use_pot: If True, use PoT-specific losses
        save_dir: Directory to save checkpoints
        wandb_log: If True, log to W&B
    
    Returns:
        Dictionary with training history and best model
    """
    from pathlib import Path
    
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    
    # Optimizer (same as Sudoku: AdamW with betas=(0.9, 0.95))
    import math
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
    )
    
    # Scheduler: cosine decay with warmup (IDENTICAL to Sudoku)
    total_steps = epochs * len(train_loader)
    
    def lr_lambda(step):
        # Linear warmup
        if step < warmup_steps:
            return step / warmup_steps
        # Cosine decay to lr_min_ratio
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr_min_ratio + (1 - lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    
    best_val_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'=' * 60}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch,
            use_pot=use_pot, scheduler=scheduler, grad_clip=grad_clip,
        )
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        
        # Log
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        print(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2%}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2%}")
        
        # W&B logging
        if wandb_log:
            import wandb
            wandb.log({
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'epoch': epoch,
            })
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            if save_dir:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': best_val_acc,
                }, save_path / 'best_model.pt')
                print(f"  → Saved best model (acc: {best_val_acc:.2%})")
    
    # Save final model
    if save_dir:
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
        }, save_path / 'final_model.pt')
    
    return {
        'history': history,
        'best_val_acc': best_val_acc,
        'model': model,
    }

