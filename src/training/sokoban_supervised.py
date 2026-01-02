"""
Sokoban Supervised Training.

IDENTICAL to Sudoku training:
- Cross-entropy loss on action prediction
- Q-halt loss for PoT models
- Same training loop structure

Metrics (identical structure to Sudoku):
- action_acc: % of single actions correctly predicted (like cell_acc)
- solve_rate: % of puzzles fully solved by following policy (like grid_acc)

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
import numpy as np

from src.data.sokoban_rules import step as sokoban_step, is_solved, is_deadlock, NUM_CELL_TYPES


# =============================================================================
# Rollout for solve_rate (like grid_acc in Sudoku)
# =============================================================================

def board_to_onehot(board: np.ndarray) -> torch.Tensor:
    """Convert integer board to one-hot encoding."""
    H, W = board.shape
    onehot = np.zeros((H, W, NUM_CELL_TYPES), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            onehot[i, j, board[i, j]] = 1.0
    return torch.from_numpy(onehot)


def rollout_episode(
    model: nn.Module,
    board: np.ndarray,
    device: torch.device,
    max_steps: int = 100,
) -> bool:
    """
    Run model on a puzzle until solved or max steps.
    
    This is the Sokoban equivalent of checking if all 81 cells are correct in Sudoku.
    
    Args:
        model: Trained model
        board: Initial board state (H, W) integer array
        device: Device to run on
        max_steps: Maximum steps before giving up
        
    Returns:
        True if puzzle was solved, False otherwise
    """
    model.eval()
    current_board = board.copy()
    
    for _ in range(max_steps):
        # Check if already solved
        if is_solved(current_board):
            return True
        
        # Check if deadlock
        if is_deadlock(current_board):
            return False
        
        # Convert to tensor
        board_tensor = board_to_onehot(current_board).unsqueeze(0).to(device)  # [1, H, W, 7]
        
        # Get model prediction
        with torch.no_grad():
            model_out = model(board_tensor, return_aux=False)
            logits = model_out[0]  # [1, 4]
            action = logits.argmax(dim=-1).item()  # 0-indexed action
        
        # Apply action (actions are 0-indexed: 0=up, 1=down, 2=left, 3=right)
        # sokoban_step expects 1-indexed, so add 1
        next_board, moved = sokoban_step(current_board, action + 1)
        
        if not moved:
            # Invalid move, puzzle failed
            return False
        
        current_board = next_board
    
    # Max steps exceeded
    return False


def compute_solve_rate(
    model: nn.Module,
    dataset,
    device: torch.device,
    n_samples: int = 100,
    max_steps: int = 100,
) -> float:
    """
    Compute solve rate (like grid_acc in Sudoku).
    
    Args:
        model: Trained model
        dataset: Dataset with board_indices
        device: Device
        n_samples: Number of puzzles to test
        max_steps: Max steps per puzzle
        
    Returns:
        Fraction of puzzles successfully solved
    """
    model.eval()
    
    n_solved = 0
    n_tested = min(n_samples, len(dataset))
    
    # Sample random puzzles
    indices = np.random.choice(len(dataset), n_tested, replace=False)
    
    for idx in tqdm(indices, desc="Computing solve_rate"):
        sample = dataset[idx]
        
        # Get integer board
        if 'board_indices' in sample:
            board = sample['board_indices'].numpy()
        else:
            # Fallback: convert one-hot back to indices
            board = sample['input'].argmax(dim=-1).numpy()
        
        if rollout_episode(model, board, device, max_steps):
            n_solved += 1
    
    return n_solved / n_tested


# =============================================================================
# Helper Classes (Identical to Sudoku)
# =============================================================================

class InfiniteDataLoader:
    """
    Infinite data loader for async batching.
    
    Wraps a DataLoader to provide infinite iteration and batch slicing.
    Used for async batching where we need to replace halted samples.
    """
    
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.current_batch = None
        self.batch_idx = 0
    
    def _get_next_batch(self):
        """Get next batch, restarting iterator if needed."""
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch
    
    def get_next(self, n: int) -> dict:
        """
        Get next n samples, possibly spanning multiple batches.
        
        Args:
            n: Number of samples to return
            
        Returns:
            Dict with 'input', 'label' tensors of size n
        """
        if n == 0:
            # Return empty batch with correct keys
            return {
                'input': torch.empty(0),
                'label': torch.empty(0, dtype=torch.long),
            }
        
        samples = []
        remaining = n
        
        while remaining > 0:
            if self.current_batch is None or self.batch_idx >= self.current_batch['input'].size(0):
                self.current_batch = self._get_next_batch()
                self.batch_idx = 0
            
            available = self.current_batch['input'].size(0) - self.batch_idx
            take = min(available, remaining)
            
            samples.append({
                'input': self.current_batch['input'][self.batch_idx:self.batch_idx + take],
                'label': self.current_batch['label'][self.batch_idx:self.batch_idx + take],
            })
            
            self.batch_idx += take
            remaining -= take
        
        # Concatenate all samples
        return {
            'input': torch.cat([s['input'] for s in samples], dim=0),
            'label': torch.cat([s['label'] for s in samples], dim=0),
        }


# =============================================================================
# Training Functions (Identical to Sudoku)
# =============================================================================

def train_epoch_async(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_pot: bool = True,
    scheduler: Optional[Any] = None,
    samples_per_epoch: Optional[int] = None,
    track_halt_histogram: bool = False,
    grad_clip: float = 1.0,
) -> Dict[str, Any]:
    """
    Train for one epoch using HRM-style async batching - IDENTICAL to Sudoku.
    
    Each forward call runs one ACT step. Samples that halt are immediately
    replaced with new data, maximizing GPU utilization.
    
    Args:
        model: HybridPoTSokobanSolver with async support
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        use_pot: Whether to use PoT-specific losses
        scheduler: LR scheduler
        samples_per_epoch: Total samples to process (default: len(dataloader) * batch_size)
        track_halt_histogram: If True, track which halt step each sample finished at
        grad_clip: Maximum gradient norm
        
    Returns:
        Dictionary with loss, accuracy, and optionally halt_histogram
    """
    model.train()
    
    # Get underlying model (handles DDP wrapping)
    base_model = model.module if hasattr(model, 'module') else model
    
    # Determine batch size from first batch
    first_batch = next(iter(dataloader))
    batch_size = first_batch['input'].size(0)
    
    # Default samples per epoch = one pass through data
    if samples_per_epoch is None:
        samples_per_epoch = len(dataloader) * batch_size
    
    # Initialize infinite data loader
    inf_loader = InfiniteDataLoader(dataloader)
    
    # Initialize carry state (all halted, so first batch loads data)
    carry = base_model.create_async_carry(batch_size, device)
    
    # Metrics tracking
    total_loss = 0.0
    correct_actions = 0
    total_actions = 0
    completed_samples = 0
    forward_calls = 0
    
    # Halt histogram tracking: {step: {'halted': count, 'solved': count}}
    halt_histogram = {} if track_halt_histogram else None
    
    pbar = tqdm(total=samples_per_epoch, desc=f"Epoch {epoch} (async)")
    
    while completed_samples < samples_per_epoch:
        # Get new batch for replacing halted samples
        new_batch = inf_loader.get_next(carry.halted.sum().item())
        
        optimizer.zero_grad()
        
        # Record which samples were halted before this step
        was_halted = carry.halted.clone()
        
        # Forward: one ACT step
        new_carry, outputs = base_model.async_forward(carry, new_batch)
        
        logits = outputs['logits']  # [B, 4] action logits
        labels = outputs['labels']  # [B] action labels
        q_halt = outputs['q_halt_logits']
        q_continue = outputs['q_continue_logits']
        target_q_continue = outputs.get('target_q_continue')
        
        # Compute loss (IDENTICAL to Sudoku)
        ce_loss = F.cross_entropy(logits, labels)
        
        if use_pot and q_halt is not None:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                is_correct = (preds == labels).float()
            
            q_halt_loss = F.binary_cross_entropy_with_logits(q_halt, is_correct)
            loss = ce_loss + 0.5 * q_halt_loss
            
            # ACT Q-learning loss (if target available)
            if target_q_continue is not None:
                q_continue_loss = F.mse_loss(torch.sigmoid(q_continue), target_q_continue)
                loss = loss + 0.5 * q_continue_loss
        else:
            loss = ce_loss
        
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        # Detach carry to prevent gradient accumulation across steps
        new_carry = new_carry.detach()
        
        # Step scheduler
        if scheduler:
            scheduler.step()
        
        # Count newly halted samples
        newly_halted = new_carry.halted & ~was_halted
        num_completed = new_carry.halted.sum().item()
        
        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        action_correct = (preds == labels)
        correct_actions += action_correct.sum().item()
        total_actions += labels.size(0)
        forward_calls += 1
        
        # Track halt histogram
        if track_halt_histogram and newly_halted.any():
            steps_tensor = new_carry.steps
            for i in range(newly_halted.size(0)):
                if newly_halted[i]:
                    step = steps_tensor[i].item()
                    solved = action_correct[i].item()
                    if step not in halt_histogram:
                        halt_histogram[step] = {'halted': 0, 'solved': 0}
                    halt_histogram[step]['halted'] += 1
                    if solved:
                        halt_histogram[step]['solved'] += 1
        
        completed_samples += num_completed
        pbar.update(num_completed)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'act_acc': f'{correct_actions / max(1, total_actions):.2%}',
        })
        
        carry = new_carry
    
    pbar.close()
    
    result = {
        'loss': total_loss / max(1, forward_calls),
        'action_acc': correct_actions / max(1, total_actions),  # Like cell_acc in Sudoku
        'forward_calls': forward_calls,
    }
    
    if track_halt_histogram and halt_histogram:
        result['halt_histogram'] = halt_histogram
        print("\n📊 Train Halt Step Histogram:")
        print(f"{'Step':<6} {'Halted':<10} {'Correct':<10} {'Action Acc':<12}")
        for step in sorted(halt_histogram.keys()):
            data = halt_histogram[step]
            rate = 100 * data['solved'] / data['halted'] if data['halted'] > 0 else 0
            print(f"{step:<6} {data['halted']:<10} {data['solved']:<10} {rate:<10.2f}%")
        print()
    
    return result


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
            # LOSS 3: Q-continue loss (ACT Q-learning, IDENTICAL to Sudoku)
            # Only added when target_q_continue is available (ACT enabled)
            # =========================================
            target_q_continue = None
            if aux is not None:
                target_q_continue = aux.get('target_q_continue')
            if target_q_continue is not None:
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
            'act_acc': f'{correct_actions / total_actions:.2%}',
        })
    
    return {
        'loss': total_loss / total_steps,
        'action_acc': correct_actions / total_actions,  # Like cell_acc in Sudoku
        'correct': correct_actions,
        'total': total_actions,
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    compute_solve: bool = False,
    solve_samples: int = 100,
    solve_max_steps: int = 100,
) -> Dict[str, float]:
    """
    Evaluate model - IDENTICAL structure to Sudoku's evaluate.
    
    Metrics (like Sudoku):
    - action_acc: % of single actions correct (like cell_acc)
    - solve_rate: % of puzzles fully solved (like grid_acc) - optional
    
    Args:
        model: Trained model
        dataloader: Validation/test data loader
        device: Device to use
        compute_solve: If True, compute solve_rate (slower)
        solve_samples: Number of puzzles to test for solve_rate
        solve_max_steps: Max steps per puzzle for solve_rate
    
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
            
            # Action accuracy (like cell_acc)
            preds = logits.argmax(dim=-1)
            correct_actions += (preds == label).sum().item()
            total_actions += label.numel()
            total_steps += 1
    
    result = {
        'loss': total_loss / total_steps,
        'action_acc': correct_actions / total_actions,  # Like cell_acc in Sudoku
        'correct': correct_actions,
        'total': total_actions,
    }
    
    # Compute solve_rate (like grid_acc) if requested
    if compute_solve:
        dataset = dataloader.dataset
        solve_rate = compute_solve_rate(
            model, dataset, device, 
            n_samples=solve_samples, 
            max_steps=solve_max_steps
        )
        result['solve_rate'] = solve_rate  # Like grid_acc in Sudoku
    
    return result


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
    
    # Training history (like Sudoku: cell_acc → action_acc, grid_acc → solve_rate)
    history = {
        'train_loss': [],
        'train_action_acc': [],  # Like train_cell_acc in Sudoku
        'val_loss': [],
        'val_action_acc': [],    # Like val_cell_acc in Sudoku
        'val_solve_rate': [],    # Like val_grid_acc in Sudoku
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
        
        # Evaluate with solve_rate every 10 epochs (expensive)
        compute_solve = (epoch % 10 == 0) or (epoch == epochs)
        val_metrics = evaluate(
            model, val_loader, device,
            compute_solve=compute_solve,
            solve_samples=100,
        )
        
        # Log
        history['train_loss'].append(train_metrics['loss'])
        history['train_action_acc'].append(train_metrics['action_acc'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_action_acc'].append(val_metrics['action_acc'])
        if 'solve_rate' in val_metrics:
            history['val_solve_rate'].append(val_metrics['solve_rate'])
        
        # Print (like Sudoku: cell_acc + grid_acc)
        solve_str = f", Solve: {val_metrics['solve_rate']:.2%}" if 'solve_rate' in val_metrics else ""
        print(f"Train Loss: {train_metrics['loss']:.4f}, Action Acc: {train_metrics['action_acc']:.2%}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Action Acc: {val_metrics['action_acc']:.2%}{solve_str}")
        
        # W&B logging
        if wandb_log:
            import wandb
            log_dict = {
                'train/loss': train_metrics['loss'],
                'train/action_acc': train_metrics['action_acc'],  # Like cell_acc
                'val/loss': val_metrics['loss'],
                'val/action_acc': val_metrics['action_acc'],      # Like cell_acc
                'epoch': epoch,
            }
            if 'solve_rate' in val_metrics:
                log_dict['val/solve_rate'] = val_metrics['solve_rate']  # Like grid_acc
            wandb.log(log_dict)
        
        # Save best model (based on action_acc like Sudoku's cell_acc)
        if val_metrics['action_acc'] > best_val_acc:
            best_val_acc = val_metrics['action_acc']
            if save_dir:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_action_acc': best_val_acc,
                }, save_path / 'best_model.pt')
                print(f"  → Saved best model (action_acc: {best_val_acc:.2%})")
    
    # Save final model
    if save_dir:
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
        }, save_path / 'final_model.pt')
    
    # Get final solve_rate if computed
    final_solve_rate = history['val_solve_rate'][-1] if history['val_solve_rate'] else None
    
    return {
        'history': history,
        'best_val_acc': best_val_acc,           # Action accuracy (like cell_acc)
        'best_val_action_acc': best_val_acc,    # Explicit name
        'final_solve_rate': final_solve_rate,   # Like grid_acc
        'model': model,
    }

