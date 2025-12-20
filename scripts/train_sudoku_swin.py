#!/usr/bin/env python3
"""
Sudoku Extreme Training Script - Swin Depth Controller

Train HybridPoHHRMSolver with Swin-style Hierarchical Depth Controller on
Sudoku-Extreme dataset.

Features:
- Swin Depth Controller (local window attention + shifted windows)
- Hierarchical spatial-temporal depth tracking (preserves spatial info!)
- Depth skip connections for better gradient flow
- On-the-fly Sudoku augmentation (~1.9M variations per puzzle)
- W&B logging
- Checkpoint saving
- Cosine LR schedule with warmup

Usage:
    # Quick test (CPU)
    python scripts/train_sudoku_swin.py --epochs 10 --batch-size 64 --device cpu
    
    # Full training (GPU)
    python scripts/train_sudoku_swin.py --epochs 1000 --batch-size 768 --device cuda
    
    # With W&B logging
    python scripts/train_sudoku_swin.py --epochs 1000 --wandb --project sudoku-swin

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
from src.data import download_sudoku_dataset, SudokuDataset as SudokuDatasetV2


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
    """
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))
    transpose_flag = np.random.rand() < 0.5
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        x_2d = x.reshape(9, 9)
        if transpose_flag:
            x_2d = x_2d.T
        x_flat = x_2d.flatten()
        new_board = x_flat[mapping]
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)


class SudokuDataset(Dataset):
    """Sudoku dataset with on-the-fly augmentation."""
    
    def __init__(self, inputs, solutions, puzzle_ids=None, augment=True):
        self.inputs = inputs
        self.solutions = solutions
        self.puzzle_ids = puzzle_ids if puzzle_ids is not None else np.zeros(len(inputs), dtype=np.int64)
        self.augment = augment
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        inp = self.inputs[idx]
        sol = self.solutions[idx]
        pid = self.puzzle_ids[idx]
        
        if self.augment:
            inp, sol = shuffle_sudoku(inp, sol)
        
        return torch.tensor(inp, dtype=torch.long), torch.tensor(sol, dtype=torch.long), torch.tensor(pid, dtype=torch.long)


def load_sudoku_data_npy(data_dir: str):
    """
    Load Sudoku-Extreme dataset from .npy files.
    
    Expected structure:
        data_dir/
            train/
                all__inputs.npy
                all__labels.npy
            val/
                all__inputs.npy
                all__labels.npy
            test/
                all__inputs.npy
                all__labels.npy
    """
    data_path = Path(data_dir)
    
    train_inputs = np.load(data_path / "train" / "all__inputs.npy")
    train_labels = np.load(data_path / "train" / "all__labels.npy")
    
    val_inputs = np.load(data_path / "val" / "all__inputs.npy")
    val_labels = np.load(data_path / "val" / "all__labels.npy")
    
    print(f"Loaded data from {data_dir}")
    print(f"  Train: {len(train_inputs)} samples")
    print(f"  Val: {len(val_inputs)} samples")
    
    return {
        'train_inputs': train_inputs,
        'train_labels': train_labels,
        'val_inputs': val_inputs,
        'val_labels': val_labels,
    }


def create_dataloaders(data, batch_size: int, augment: bool = True, num_workers: int = 4):
    """Create DataLoaders from data dict."""
    
    train_dataset = SudokuDataset(
        data['train_inputs'],
        data['train_labels'],
        augment=augment,
    )
    
    val_dataset = SudokuDataset(
        data['val_inputs'],
        data['val_labels'],
        augment=False,
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True, 
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
    """Cosine schedule with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, grad_clip: float = 1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_cells = 0
    total_grids_correct = 0
    total_grids = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for inputs, targets, puzzle_ids in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        puzzle_ids = puzzle_ids.to(device, non_blocking=True)
        
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        
        # Metrics
        preds = logits.argmax(dim=-1)
        correct = (preds == targets).sum().item()
        total_correct += correct
        total_cells += targets.numel()
        
        grid_correct = (preds == targets).all(dim=1).sum().item()
        total_grids_correct += grid_correct
        total_grids += targets.size(0)
        
        total_loss += loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cell': f'{100*correct/targets.numel():.1f}%',
            'grid': f'{100*grid_correct/targets.size(0):.1f}%',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}',
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
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        puzzle_ids = puzzle_ids.to(device, non_blocking=True)
        
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
    parser = argparse.ArgumentParser(description="Train Sudoku solver with Swin controller")
    
    # Data
    parser.add_argument("--data-dir", type=str, default="data/sudoku-extreme-10k-aug-100", help="Data directory")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--download", action="store_true", help="Download Sudoku-Extreme dataset from HuggingFace")
    
    # Model architecture
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument("--d-ff", type=int, default=2048, help="FFN dimension")
    parser.add_argument("--n-heads", type=int, default=8, help="Attention heads")
    parser.add_argument("--h-layers", type=int, default=2, help="H-level layers")
    parser.add_argument("--l-layers", type=int, default=2, help="L-level layers")
    parser.add_argument("--h-cycles", type=int, default=2, help="H cycles")
    parser.add_argument("--l-cycles", type=int, default=6, help="L cycles")
    parser.add_argument("--halt-max-steps", type=int, default=2, help="Max ACT steps")
    parser.add_argument("--dropout", type=float, default=0.039, help="Dropout")
    
    # Swin controller specific
    parser.add_argument("--d-ctrl", type=int, default=256, help="Controller dimension")
    parser.add_argument("--window-size", type=int, default=3, help="Swin window size (3 for 9x9 Sudoku)")
    parser.add_argument("--n-stages", type=int, default=2, help="Number of Swin stages")
    parser.add_argument("--n-ctrl-layers", type=int, default=2, help="Controller transformer layers")
    parser.add_argument("--n-ctrl-heads", type=int, default=4, help="Controller attention heads")
    parser.add_argument("--max-depth", type=int, default=32, help="Maximum depth steps")
    parser.add_argument("--no-depth-skip", action="store_true", help="Disable depth skip connection")
    
    # Training
    parser.add_argument("--epochs", type=int, default=1000, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=768, help="Batch size")
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation")
    parser.add_argument("--hrm-grad-style", action="store_true", help="Only last L+H get gradients (HRM-style)")
    parser.add_argument("--lr", type=float, default=3.7e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.108, help="Weight decay")
    parser.add_argument("--beta2", type=float, default=0.968, help="AdamW beta2")
    parser.add_argument("--warmup-steps", type=int, default=2000, help="Warmup steps")
    parser.add_argument("--lr-min-ratio", type=float, default=0.1, help="Min LR ratio")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--T", type=int, default=4, help="HRM period")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--project", type=str, default="sudoku-swin", help="W&B project")
    parser.add_argument("--run-name", type=str, default=None, help="W&B run name")
    
    # Checkpoints
    parser.add_argument("--save-dir", type=str, default="checkpoints/swin", help="Checkpoint directory")
    parser.add_argument("--save-every", type=int, default=50, help="Save every N epochs")
    parser.add_argument("--eval-interval", type=int, default=10, help="Eval every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Download dataset if requested
    if args.download:
        print("\nDownloading Sudoku-Extreme dataset from HuggingFace...")
        download_sudoku_dataset(args.data_dir, subsample_size=10000)
        print(f"✓ Dataset saved to {args.data_dir}\n")
    
    # Data
    data = load_sudoku_data_npy(args.data_dir)
    train_loader, val_loader = create_dataloaders(
        data, args.batch_size, 
        augment=not args.no_augment,
        num_workers=args.num_workers,
    )
    
    # Swin controller kwargs
    swin_kwargs = {
        "d_ctrl": args.d_ctrl,
        "window_size": args.window_size,
        "n_stages": args.n_stages,
        "max_depth": args.max_depth,
        "token_conditioned": True,
        "depth_skip": not args.no_depth_skip,
    }
    
    # Model
    model = HybridPoHHRMSolver(
        d_model=args.d_model,
        n_heads=args.n_heads,
        H_layers=args.h_layers,
        L_layers=args.l_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        H_cycles=args.h_cycles,
        L_cycles=args.l_cycles,
        T=args.T,
        halt_max_steps=args.halt_max_steps,
        hrm_grad_style=args.hrm_grad_style,
        controller_type="swin",  # <-- Swin controller!
        controller_kwargs=swin_kwargs,
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {param_count:,}")
    print(f"Controller: Swin Depth Controller")
    print(f"  - d_ctrl: {args.d_ctrl}")
    print(f"  - window_size: {args.window_size}")
    print(f"  - n_stages: {args.n_stages}")
    print(f"  - depth_skip: {not args.no_depth_skip}")
    print(f"  - hrm_grad_style: {args.hrm_grad_style}")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, args.beta2),
    )
    
    # Scheduler
    total_steps = args.epochs * len(train_loader)
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps, total_steps, args.lr_min_ratio)
    
    # W&B
    if args.wandb:
        import wandb
        run_name = args.run_name or f"swin_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.project,
            name=run_name,
            config=vars(args),
        )
        wandb.watch(model, log="gradients", log_freq=100)
    
    # Checkpoint dir
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_grid_acc = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"\nLoading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_grid_acc = checkpoint.get('best_grid_acc', 0)
            print(f"✓ Resumed from epoch {start_epoch - 1}, best_grid_acc={100*best_grid_acc:.2f}%")
        else:
            print(f"Warning: Checkpoint {args.resume} not found, starting from scratch")
    
    # Training loop
    print(f"\n{'='*60}")
    print("Starting Swin Controller Training")
    if args.resume:
        print(f"Resuming from epoch {start_epoch}")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, args.grad_clip
        )
        
        # Evaluate
        if epoch % args.eval_interval == 0 or epoch == 1:
            val_metrics = evaluate(model, val_loader, device)
            
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"  Train: loss={train_metrics['loss']:.4f}, "
                  f"cell={100*train_metrics['cell_acc']:.2f}%, "
                  f"grid={100*train_metrics['grid_acc']:.2f}%")
            print(f"  Val:   loss={val_metrics['loss']:.4f}, "
                  f"cell={100*val_metrics['cell_acc']:.2f}%, "
                  f"grid={100*val_metrics['grid_acc']:.2f}%")
            
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
                best_model_path = save_dir / "best_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_grid_acc': best_grid_acc,
                    'config': vars(args),
                }, best_model_path)
                print(f"  ✓ New best! Grid acc: {100*best_grid_acc:.2f}%")
                
                # Upload best model to W&B as artifact
                if args.wandb:
                    artifact = wandb.Artifact(
                        name="sudoku-swin-best",
                        type="model",
                        metadata={
                            "epoch": epoch,
                            "grid_acc": val_metrics['grid_acc'],
                            "cell_acc": val_metrics['cell_acc'],
                            "val_loss": val_metrics['loss'],
                        }
                    )
                    artifact.add_file(str(best_model_path))
                    wandb.log_artifact(artifact, aliases=["best", f"epoch-{epoch}"])
        else:
            # Just log train metrics
            if args.wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_metrics['loss'],
                    "train/cell_acc": train_metrics['cell_acc'],
                    "train/grid_acc": train_metrics['grid_acc'],
                    "lr": scheduler.get_last_lr()[0],
                })
        
        # Periodic save
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': vars(args),
            }, save_dir / f"checkpoint_epoch{epoch}.pt")
    
    print(f"\n{'='*60}")
    print(f"Training complete! Best grid accuracy: {100*best_grid_acc:.2f}%")
    print(f"{'='*60}")
    
    if args.wandb:
        # Log final best accuracy as summary metric
        wandb.run.summary["best_grid_acc"] = best_grid_acc
        wandb.finish()


if __name__ == "__main__":
    main()

