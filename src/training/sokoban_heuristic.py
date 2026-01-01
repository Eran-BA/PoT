"""
Sokoban Heuristic Pretraining.

Trains a Sokoban policy using heuristic pseudo-labels:
- For each state, enumerate legal actions
- Score successors with heuristic (boxes-on-target, deadlock penalty, Manhattan distance)
- Use argmax action as pseudo-label
- Train with cross-entropy loss

This is the Sudoku-style "constraint-guided" training approach.

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

from src.data.sokoban import (
    SokobanStateDataset,
    create_heuristic_dataset,
    board_to_onehot,
)
from src.data.sokoban_rules import (
    legal_actions,
    step,
    is_solved,
    is_deadlock,
    get_legal_action_list,
    compute_heuristic_score,
)
from src.pot.models.sokoban_solver import PoTSokobanSolver, BaselineSokobanSolver


# =============================================================================
# Config
# =============================================================================

@dataclass
class HeuristicTrainingConfig:
    """Configuration for heuristic pretraining."""
    # Data
    data_dir: str = "data"
    difficulty: str = "medium"
    steps_per_level: int = 500
    augment: bool = True
    cache_path: Optional[str] = None
    
    # Model
    model_type: str = "pot"  # "pot" or "baseline"
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.1
    R: int = 4  # PoT refinement steps
    
    # Training
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip: float = 1.0
    
    # Evaluation
    eval_interval: int = 5
    save_dir: str = "experiments/results/sokoban_heuristic"


# =============================================================================
# Training Functions
# =============================================================================

def create_model(config: HeuristicTrainingConfig, device: torch.device) -> nn.Module:
    """Create Sokoban solver model."""
    if config.model_type == "pot":
        model = PoTSokobanSolver(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            R=config.R,
        )
    else:
        model = BaselineSokobanSolver(
            n_filters=64,
            n_layers=4,
            d_hidden=config.d_model,
            dropout=config.dropout,
        )
    
    return model.to(device)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch of samples."""
    boards = torch.stack([item['board'] for item in batch])
    actions = torch.stack([item['action'] for item in batch])
    
    # Compute legal masks
    board_indices = torch.stack([item['board_indices'] for item in batch])
    legal_masks = []
    for i in range(len(batch)):
        board = board_indices[i].numpy()
        mask = legal_actions(board)
        legal_masks.append(torch.tensor(mask))
    legal_masks = torch.stack(legal_masks)
    
    return {
        'board': boards,
        'action': actions,
        'legal_mask': legal_masks,
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: HeuristicTrainingConfig,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        board = batch['board'].to(device)
        action = batch['action'].to(device)
        legal_mask = batch['legal_mask'].to(device)
        
        # Forward
        action_logits, _, _ = model(board, return_aux=False)
        
        # Mask illegal actions
        action_logits = action_logits.masked_fill(~legal_mask.bool(), float('-inf'))
        
        # Cross-entropy loss
        loss = F.cross_entropy(action_logits, action)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        optimizer.step()
        
        # Stats
        total_loss += loss.item() * board.size(0)
        pred = action_logits.argmax(dim=-1)
        total_correct += (pred == action).sum().item()
        total_samples += board.size(0)
    
    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate on validation set."""
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_legal_correct = 0  # Correct within legal actions
    total_samples = 0
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        board = batch['board'].to(device)
        action = batch['action'].to(device)
        legal_mask = batch['legal_mask'].to(device)
        
        # Forward
        action_logits, _, _ = model(board, return_aux=False)
        
        # Masked logits
        masked_logits = action_logits.masked_fill(~legal_mask.bool(), float('-inf'))
        
        # Loss
        loss = F.cross_entropy(masked_logits, action)
        
        # Stats
        total_loss += loss.item() * board.size(0)
        pred = masked_logits.argmax(dim=-1)
        total_correct += (pred == action).sum().item()
        total_samples += board.size(0)
    
    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
    }


# =============================================================================
# Rollout Evaluation (Solve Rate)
# =============================================================================

@torch.no_grad()
def evaluate_solve_rate(
    model: nn.Module,
    levels: List[np.ndarray],
    device: torch.device,
    max_steps: int = 100,
    temperature: float = 0.0,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Evaluate solve rate on levels.
    
    Args:
        model: Trained model
        levels: List of initial board states
        device: Torch device
        max_steps: Maximum steps per level
        temperature: Sampling temperature (0 = greedy)
        verbose: If True, show progress
    
    Returns:
        Dictionary with solve metrics
    """
    model.eval()
    
    solved = 0
    deadlocked = 0
    total_steps = []
    
    iterator = tqdm(levels, desc="Solve rate") if verbose else levels
    
    for level in iterator:
        board = level.copy()
        steps = 0
        
        for _ in range(max_steps):
            if is_solved(board):
                solved += 1
                total_steps.append(steps)
                break
            
            if is_deadlock(board):
                deadlocked += 1
                break
            
            # Get legal actions
            legal = get_legal_action_list(board)
            if not legal:
                break
            
            # Get model prediction
            board_onehot = board_to_onehot(board)
            board_tensor = torch.tensor(board_onehot, dtype=torch.float32).unsqueeze(0).to(device)
            
            action_logits, _, _ = model(board_tensor, return_aux=False)
            action_logits = action_logits.squeeze(0)
            
            # Mask illegal
            legal_mask = np.zeros(4, dtype=np.float32)
            for a in legal:
                legal_mask[a] = 1.0
            legal_mask_tensor = torch.tensor(legal_mask).to(device)
            action_logits = action_logits.masked_fill(~legal_mask_tensor.bool(), float('-inf'))
            
            # Select action
            if temperature > 0:
                probs = F.softmax(action_logits / temperature, dim=-1)
                action = torch.multinomial(probs, 1).item()
            else:
                action = action_logits.argmax().item()
            
            # Step
            board, _ = step(board, action)
            steps += 1
        else:
            # Hit max steps without solving
            pass
    
    n_levels = len(levels)
    
    return {
        'solve_rate': solved / n_levels if n_levels > 0 else 0.0,
        'deadlock_rate': deadlocked / n_levels if n_levels > 0 else 0.0,
        'median_steps': np.median(total_steps) if total_steps else 0.0,
        'mean_steps': np.mean(total_steps) if total_steps else 0.0,
        'solved': solved,
        'deadlocked': deadlocked,
        'total': n_levels,
    }


# =============================================================================
# Main Training Loop
# =============================================================================

def train(config: HeuristicTrainingConfig, device: torch.device) -> Dict[str, Any]:
    """
    Main training loop for heuristic pretraining.
    
    Args:
        config: Training configuration
        device: Torch device
    
    Returns:
        Dictionary with training results
    """
    # Create save directory
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    print("Creating training dataset...")
    train_dataset = create_heuristic_dataset(
        data_dir=config.data_dir,
        difficulty=config.difficulty,
        split='train',
        steps_per_level=config.steps_per_level,
        augment=config.augment,
        cache_path=os.path.join(config.data_dir, 'sokoban_train_states.npz') if config.cache_path is None else config.cache_path,
    )
    
    print("Creating validation dataset...")
    val_dataset = create_heuristic_dataset(
        data_dir=config.data_dir,
        difficulty=config.difficulty,
        split='valid',
        steps_per_level=config.steps_per_level // 2,
        augment=False,
        cache_path=os.path.join(config.data_dir, 'sokoban_val_states.npz'),
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    model = create_model(config, device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # Training loop
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    
    for epoch in range(config.epochs):
        start_time = time.time()
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, config)
        scheduler.step()
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        
        # Evaluate
        if (epoch + 1) % config.eval_interval == 0 or epoch == 0:
            val_metrics = evaluate(model, val_loader, device)
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': best_val_acc,
                    'config': config,
                }, save_dir / 'best_model.pt')
            
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{config.epochs} ({elapsed:.1f}s) - "
                  f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2%} | "
                  f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2%}")
        else:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{config.epochs} ({elapsed:.1f}s) - "
                  f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2%}")
    
    # Save final model
    torch.save({
        'epoch': config.epochs,
        'model_state_dict': model.state_dict(),
        'history': history,
        'config': config,
    }, save_dir / 'final_model.pt')
    
    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.2%}")
    
    return {
        'best_val_acc': best_val_acc,
        'history': history,
        'model': model,
    }

