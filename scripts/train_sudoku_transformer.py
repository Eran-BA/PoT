#!/usr/bin/env python3
"""
Sudoku Extreme Training Script - Transformer Controller

Train HybridPoHHRMSolver with Causal Depth Transformer controller on
Sudoku-Extreme dataset. This uses attention over the depth axis instead
of RNN-based controllers.

Features:
- Causal Depth Transformer controller (explicit attention over depth history)
- Token-conditioned routing (per-token α)
- Parallel training (causal mask allows batched forward)
- On-the-fly Sudoku augmentation (~1.9M variations per puzzle)
  - Digit permutation, transpose, row/column band shuffling
- W&B logging
- Checkpoint saving
- Cosine LR schedule with warmup

Usage:
    # Quick test (CPU)
    python scripts/train_sudoku_transformer.py --epochs 10 --batch-size 64 --device cpu
    
    # Full training (GPU)
    python scripts/train_sudoku_transformer.py --epochs 1000 --batch-size 768 --device cuda
    
    # With W&B logging
    python scripts/train_sudoku_transformer.py --epochs 1000 --wandb --project sudoku-transformer
    
    # Disable augmentation (for debugging)
    python scripts/train_sudoku_transformer.py --no-augment

Author: Eran Ben Artzy
Year: 2025
"""

import argparse
import os
import time
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pot.models.sudoku_solver import HybridPoHHRMSolver


# =============================================================================
# Sudoku Augmentation (from HRM paper - validity preserving transforms)
# =============================================================================

def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    """
    Apply validity-preserving augmentation to a Sudoku puzzle.
    
    Transforms include:
    - Digit permutation (1-9 -> random permutation)
    - Transpose (50% chance)
    - Row band shuffling (shuffle the 3 bands of 3 rows each)
    - Row shuffling within bands
    - Column stack shuffling (shuffle the 3 stacks of 3 columns each)  
    - Column shuffling within stacks
    
    This effectively multiplies dataset by ~1.9 million variations per puzzle.
    
    Args:
        board: Input puzzle [81] with 0=blank, 1-9=digits
        solution: Solution [81] with 1-9
        
    Returns:
        Augmented (board, solution) tuple
    """
    # Create a random digit mapping: a permutation of 1..9, with zero (blank) unchanged
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))
    
    # Randomly decide whether to transpose
    transpose_flag = np.random.rand() < 0.5

    # Generate a valid row permutation:
    # - Shuffle the 3 bands (each band = 3 rows) and for each band, shuffle its 3 rows.
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    # Similarly for columns (stacks).
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    # Build an 81->81 mapping
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        # Reshape to 9x9 for transpose
        x_2d = x.reshape(9, 9)
        if transpose_flag:
            x_2d = x_2d.T
        x_flat = x_2d.flatten()
        # Apply row/col permutation
        new_board = x_flat[mapping]
        # Apply digit mapping
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)


class SudokuDataset(Dataset):
    """Sudoku dataset with on-the-fly augmentation."""
    
    def __init__(self, inputs, solutions, puzzle_ids=None, augment=True):
        self.inputs = inputs
        self.solutions = solutions
        self.puzzle_ids = puzzle_ids if puzzle_ids is not None else torch.zeros(len(inputs), dtype=torch.long)
        self.augment = augment
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        inp = self.inputs[idx].numpy() if isinstance(self.inputs[idx], torch.Tensor) else self.inputs[idx]
        sol = self.solutions[idx].numpy() if isinstance(self.solutions[idx], torch.Tensor) else self.solutions[idx]
        pid = self.puzzle_ids[idx]
        
        if self.augment:
            inp, sol = shuffle_sudoku(inp, sol)
        
        return torch.tensor(inp, dtype=torch.long), torch.tensor(sol, dtype=torch.long), pid


def load_sudoku_data(data_dir: str, device: str = "cpu"):
    """
    Load Sudoku-Extreme dataset.
    
    Expected format: .pt files with 'inputs' and 'solutions' tensors.
    """
    data_path = Path(data_dir)
    
    train_data = {}
    val_data = {}
    
    # Try to load preprocessed .pt files
    train_pt = data_path / "train.pt"
    val_pt = data_path / "val.pt"
    
    if train_pt.exists() and val_pt.exists():
        print(f"Loading preprocessed data from {data_dir}...")
        train_data = torch.load(train_pt, map_location=device)
        val_data = torch.load(val_pt, map_location=device)
    else:
        # Create synthetic data for testing
        print("No preprocessed data found. Creating synthetic data for testing...")
        n_train, n_val = 1000, 200
        train_data = {
            'inputs': torch.randint(0, 10, (n_train, 81)),
            'solutions': torch.randint(1, 10, (n_train, 81)),
            'puzzle_ids': torch.zeros(n_train, dtype=torch.long),
        }
        val_data = {
            'inputs': torch.randint(0, 10, (n_val, 81)),
            'solutions': torch.randint(1, 10, (n_val, 81)),
            'puzzle_ids': torch.zeros(n_val, dtype=torch.long),
        }
    
    return train_data, val_data


def create_dataloaders(train_data, val_data, batch_size: int, device: str, augment: bool = True):
    """Create DataLoaders from data dicts with on-the-fly augmentation."""
    
    # Get or create puzzle IDs
    train_ids = train_data.get('puzzle_ids', torch.zeros(len(train_data['inputs']), dtype=torch.long))
    val_ids = val_data.get('puzzle_ids', torch.zeros(len(val_data['inputs']), dtype=torch.long))
    
    # Use SudokuDataset with augmentation for training
    train_dataset = SudokuDataset(
        train_data['inputs'],
        train_data['solutions'],
        train_ids,
        augment=augment,  # On-the-fly augmentation
    )
    
    # No augmentation for validation
    val_dataset = SudokuDataset(
        val_data['inputs'],
        val_data['solutions'],
        val_ids,
        augment=False,
    )
    
    # Note: data stays on CPU, moved to device in training loop
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_cells = 0
    total_grids_correct = 0
    total_grids = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for inputs, targets, puzzle_ids in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        puzzle_ids = puzzle_ids.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(inputs, puzzle_ids)
        logits = outputs[0]  # [B, 81, 10]
        
        # Loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, 10),
            targets.view(-1),
        )
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Metrics
        preds = logits.argmax(dim=-1)
        correct = (preds == targets).sum().item()
        total_correct += correct
        total_cells += targets.numel()
        
        # Grid accuracy
        grid_correct = (preds == targets).all(dim=1).sum().item()
        total_grids_correct += grid_correct
        total_grids += targets.size(0)
        
        total_loss += loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cell_acc': f'{100*correct/targets.numel():.1f}%',
            'grid_acc': f'{100*grid_correct/targets.size(0):.1f}%',
        })
    
    return {
        'loss': total_loss / len(dataloader),
        'cell_acc': total_correct / total_cells,
        'grid_acc': total_grids_correct / total_grids,
    }


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_cells = 0
    total_grids_correct = 0
    total_grids = 0
    
    for inputs, targets, puzzle_ids in tqdm(dataloader, desc="Evaluating"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        puzzle_ids = puzzle_ids.to(device)
        
        outputs = model(inputs, puzzle_ids)
        logits = outputs[0]
        
        loss = nn.functional.cross_entropy(
            logits.view(-1, 10),
            targets.view(-1),
        )
        
        preds = logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_cells += targets.numel()
        total_grids_correct += (preds == targets).all(dim=1).sum().item()
        total_grids += targets.size(0)
        total_loss += loss.item()
    
    return {
        'loss': total_loss / len(dataloader),
        'cell_acc': total_correct / total_cells,
        'grid_acc': total_grids_correct / total_grids,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Sudoku solver with Transformer controller")
    
    # Data
    parser.add_argument("--data-dir", type=str, default="data/sudoku-extreme", help="Data directory")
    
    # Model
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=8, help="Attention heads")
    parser.add_argument("--h-layers", type=int, default=2, help="H-level layers")
    parser.add_argument("--l-layers", type=int, default=2, help="L-level layers")
    parser.add_argument("--h-cycles", type=int, default=2, help="H cycles")
    parser.add_argument("--l-cycles", type=int, default=8, help="L cycles")
    parser.add_argument("--halt-max-steps", type=int, default=4, help="Max ACT steps")
    
    # Transformer controller specific
    parser.add_argument("--d-ctrl", type=int, default=256, help="Controller dimension")
    parser.add_argument("--n-ctrl-layers", type=int, default=2, help="Controller transformer layers")
    parser.add_argument("--n-ctrl-heads", type=int, default=4, help="Controller attention heads")
    parser.add_argument("--max-depth", type=int, default=32, help="Maximum depth steps for controller")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--no-augment", action="store_true", help="Disable on-the-fly augmentation")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Warmup steps")
    
    # Device
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    
    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--project", type=str, default="sudoku-transformer", help="W&B project")
    parser.add_argument("--run-name", type=str, default=None, help="W&B run name")
    
    # Checkpoints
    parser.add_argument("--save-dir", type=str, default="checkpoints/transformer", help="Checkpoint directory")
    parser.add_argument("--save-every", type=int, default=50, help="Save every N epochs")
    
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Data
    train_data, val_data = load_sudoku_data(args.data_dir, device)
    train_loader, val_loader = create_dataloaders(
        train_data, val_data, args.batch_size, device,
        augment=not args.no_augment
    )
    print(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")
    
    # Model with Transformer controller
    model = HybridPoHHRMSolver(
        d_model=args.d_model,
        n_heads=args.n_heads,
        H_layers=args.h_layers,
        L_layers=args.l_layers,
        H_cycles=args.h_cycles,
        L_cycles=args.l_cycles,
        halt_max_steps=args.halt_max_steps,
        controller_type="transformer",  # <-- Transformer controller!
        controller_kwargs={
            "d_ctrl": args.d_ctrl,
            "n_ctrl_layers": args.n_ctrl_layers,
            "n_ctrl_heads": args.n_ctrl_heads,
            "max_depth": args.max_depth,
            "token_conditioned": True,
        },
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Controller: Causal Depth Transformer")
    print(f"  - d_ctrl: {args.d_ctrl}")
    print(f"  - n_ctrl_layers: {args.n_ctrl_layers}")
    print(f"  - n_ctrl_heads: {args.n_ctrl_heads}")
    print(f"  - max_depth: {args.max_depth}")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    
    # Scheduler (cosine with warmup)
    total_steps = args.epochs * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # W&B
    if args.wandb:
        import wandb
        run_name = args.run_name or f"transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.project,
            name=run_name,
            config=vars(args),
        )
        wandb.watch(model, log="gradients", log_freq=100)
    
    # Checkpoint dir
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_grid_acc = 0
    print(f"\n{'='*60}")
    print("Starting Transformer Controller Training")
    print(f"{'='*60}\n")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        
        # Update scheduler
        scheduler.step()
        
        # Print
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train: loss={train_metrics['loss']:.4f}, "
              f"cell_acc={100*train_metrics['cell_acc']:.2f}%, "
              f"grid_acc={100*train_metrics['grid_acc']:.2f}%")
        print(f"  Val:   loss={val_metrics['loss']:.4f}, "
              f"cell_acc={100*val_metrics['cell_acc']:.2f}%, "
              f"grid_acc={100*val_metrics['grid_acc']:.2f}%")
        
        # W&B logging
        if args.wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_metrics['loss'],
                "train/cell_acc": train_metrics['cell_acc'],
                "train/grid_acc": train_metrics['grid_acc'],
                "val/loss": val_metrics['loss'],
                "val/cell_acc": val_metrics['cell_acc'],
                "val/grid_acc": val_metrics['grid_acc'],
                "lr": scheduler.get_last_lr()[0],
            })
        
        # Save best
        if val_metrics['grid_acc'] > best_grid_acc:
            best_grid_acc = val_metrics['grid_acc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_grid_acc': best_grid_acc,
                'config': vars(args),
            }, save_dir / "best_model.pt")
            print(f"  ✓ New best! Grid acc: {100*best_grid_acc:.2f}%")
        
        # Periodic save
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'grid_acc': val_metrics['grid_acc'],
                'config': vars(args),
            }, save_dir / f"checkpoint_epoch{epoch}.pt")
    
    print(f"\n{'='*60}")
    print(f"Training complete! Best grid accuracy: {100*best_grid_acc:.2f}%")
    print(f"{'='*60}")
    
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
