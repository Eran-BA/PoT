#!/usr/bin/env python3
"""
Diffusion HRM Sudoku Training Script
=====================================

Trains a fully diffusion-based Sudoku solver where:
- H,L cycles use diffusion denoising (not GRU)
- Timing for H-updates is learned via Gumbel-softmax (optional)
- adaLN conditioning (DiT-style)

Key Features:
- Configurable lastgrad (HRM-style gradient flow)
- Real-time accuracy tracking in progress bar
- W&B integration for experiment tracking
- Checkpoint saving with best model tracking

Usage:
    # Basic training
    python scripts/train_diffusion_sudoku.py --download
    
    # With custom settings
    python scripts/train_diffusion_sudoku.py \
        --max-steps 16 --T 4 --halt-max-steps 2 \
        --lr 3e-5 --batch-size 256 \
        --lastgrad 4 \
        --wandb-project diffusion-sudoku

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json

# Try to import wandb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Install with: pip install wandb")

from src.data import SudokuDataset, download_sudoku_dataset
from src.pot.models.diffusion_hrm_solver import DiffusionSudokuSolver


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    """Cosine learning rate schedule with warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model, loader, device):
    """Evaluate model on dataset."""
    model.eval()
    correct_cells = 0
    total_cells = 0
    correct_grids = 0
    total_grids = 0
    total_steps = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            inputs = batch['input'].to(device)
            targets = batch['label'].to(device)
            puzzle_ids = batch.get('puzzle_id', torch.zeros(inputs.size(0), dtype=torch.long)).to(device)
            
            logits, _, _, steps = model(inputs, puzzle_ids)
            preds = logits.argmax(dim=-1)
            
            mask = (inputs == 0)
            correct_cells += ((preds == targets) & mask).sum().item()
            total_cells += mask.sum().item()
            
            grid_correct = ((preds == targets) | ~mask).all(dim=1)
            correct_grids += grid_correct.sum().item()
            total_grids += inputs.size(0)
            
            total_steps += steps
            num_batches += 1
    
    return {
        'cell_acc': 100 * correct_cells / max(1, total_cells),
        'grid_acc': 100 * correct_grids / max(1, total_grids),
        'avg_steps': total_steps / max(1, num_batches),
    }


def train_epoch(model, loader, optimizer, scheduler, device, epoch, log_interval=50, use_wandb=True):
    """Train for one epoch with accuracy tracking."""
    model.train()
    total_loss = 0
    num_batches = 0
    global_step = (epoch - 1) * len(loader)
    
    # Track training accuracy
    correct_cells = 0
    total_cells = 0
    correct_grids = 0
    total_grids = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        inputs = batch['input'].to(device)
        targets = batch['label'].to(device)
        puzzle_ids = batch.get('puzzle_id', torch.zeros(inputs.size(0), dtype=torch.long)).to(device)
        
        logits, q_halt, q_continue, steps = model(inputs, puzzle_ids)
        
        # Clamp logits for numerical stability
        logits = torch.clamp(logits, -50, 50)
        
        mask = (inputs == 0)
        if not mask.any():
            continue  # Skip if no blank cells
        
        # Compute batch accuracy (no grad for efficiency)
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct_cells += ((preds == targets) & mask).sum().item()
            total_cells += mask.sum().item()
            grid_correct = ((preds == targets) | ~mask).all(dim=1)
            correct_grids += grid_correct.sum().item()
            total_grids += inputs.size(0)
            
        loss = F.cross_entropy(logits[mask], targets[mask])
        
        # Skip NaN/Inf losses
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss at batch {batch_idx}, skipping")
            continue
        
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        global_step += 1
        
        # Compute running accuracy
        cell_acc = 100 * correct_cells / max(1, total_cells)
        grid_acc = 100 * correct_grids / max(1, total_grids)
        
        if use_wandb and HAS_WANDB and batch_idx % log_interval == 0:
            wandb.log({
                'train/loss': loss.item(),
                'train/cell_acc': cell_acc,
                'train/grid_acc': grid_acc,
                'train/lr': scheduler.get_last_lr()[0],
                'train/grad_norm': grad_norm.item(),
                'train/act_steps': steps,
                'train/q_halt_mean': q_halt.mean().item(),
                'train/q_continue_mean': q_continue.mean().item(),
                'global_step': global_step,
            })
        
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'cell': f'{cell_acc:.1f}%',
            'grid': f'{grid_acc:.1f}%',
            'lr': f'{scheduler.get_last_lr()[0]:.1e}'
        })
    
    return total_loss / max(1, num_batches), cell_acc, grid_acc


def main():
    parser = argparse.ArgumentParser(
        description='Train Diffusion HRM Sudoku Solver',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data args
    parser.add_argument('--download', action='store_true', help='Download dataset if not present')
    parser.add_argument('--data-dir', type=str, default='data/sudoku-extreme-10k-aug-100',
                        help='Dataset directory')
    parser.add_argument('--subsample', type=int, default=10000, help='Number of puzzles to use')
    parser.add_argument('--num-aug', type=int, default=100, help='Augmentations per puzzle')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    
    # Model args
    parser.add_argument('--d-model', type=int, default=512, help='Model dimension')
    parser.add_argument('--n-heads', type=int, default=8, help='Attention heads')
    parser.add_argument('--max-steps', type=int, default=16, help='Diffusion steps per ACT iteration')
    parser.add_argument('--T', type=int, default=4, help='H/L timescale ratio')
    parser.add_argument('--noise-schedule', type=str, default='cosine',
                        choices=['linear', 'cosine', 'sqrt'], help='Noise schedule type')
    parser.add_argument('--learned-timing', action='store_true', help='Learn H-update timing')
    parser.add_argument('--no-learned-timing', action='store_false', dest='learned_timing')
    parser.set_defaults(learned_timing=False)  # More stable default
    parser.add_argument('--halt-max-steps', type=int, default=2, help='ACT outer iterations')
    parser.add_argument('--num-puzzles', type=int, default=10000, help='Puzzle embedding count')
    parser.add_argument('--init-noise-scale', type=float, default=0.1,
                        help='Initial noise scale (lower=more stable)')
    parser.add_argument('--lastgrad', type=int, default=2,
                        help='Number of final diffusion steps with gradient flow (HRM-style)')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=10000, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--warmup-steps', type=int, default=500, help='LR warmup steps')
    parser.add_argument('--lr-min-ratio', type=float, default=0.1, help='Min LR ratio for cosine decay')
    parser.add_argument('--grad-clip', type=float, default=0.5, help='Gradient clipping norm')
    
    # Eval args
    parser.add_argument('--eval-interval', type=int, default=100, help='Epochs between evaluations')
    parser.add_argument('--save-dir', type=str, default='experiments/results/diffusion_sudoku',
                        help='Checkpoint save directory')
    
    # W&B args
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging')
    parser.add_argument('--wandb-project', type=str, default='diffusion-sudoku', help='W&B project')
    parser.add_argument('--wandb-entity', type=str, default=None, help='W&B entity')
    parser.add_argument('--wandb-name', type=str, default=None, help='W&B run name')
    parser.add_argument('--wandb-tags', type=str, nargs='+', default=['diffusion', 'sudoku'],
                        help='W&B tags')
    
    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Download dataset if needed
    if args.download or not os.path.exists(args.data_dir):
        print(f"\nDownloading dataset to {args.data_dir}...")
        download_sudoku_dataset(
            output_dir=args.data_dir,
            subsample_size=args.subsample,
            num_aug=args.num_aug,
        )
    
    # Initialize W&B
    use_wandb = args.wandb and HAS_WANDB
    if use_wandb:
        run_name = args.wandb_name or f"diffusion-sudoku-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            tags=args.wandb_tags,
            config=vars(args),
        )
        print(f"W&B run: {wandb.run.url}")
    
    # Load data
    print("\nLoading datasets...")
    train_dataset = SudokuDataset(args.data_dir, 'train')
    val_dataset = SudokuDataset(args.data_dir, 'val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    # Create model
    print("\nCreating Diffusion HRM Sudoku Solver...")
    model = DiffusionSudokuSolver(
        d_model=args.d_model,
        n_heads=args.n_heads,
        max_steps=args.max_steps,
        T=args.T,
        noise_schedule=args.noise_schedule,
        num_puzzles=args.num_puzzles,
        halt_max_steps=args.halt_max_steps,
        dropout=args.dropout,
        learned_timing=args.learned_timing,
        init_noise_scale=args.init_noise_scale,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,}")
    
    if use_wandb:
        wandb.config.update({'num_params': num_params, 'trainable_params': trainable_params})
        wandb.watch(model, log='gradients', log_freq=500)
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        args.warmup_steps,
        total_steps,
        args.lr_min_ratio,
    )
    
    # Training loop
    best_grid_acc = 0.0
    best_cell_acc = 0.0
    history = []
    
    print(f"\n{'='*60}")
    print("Starting Training - Diffusion HRM Sudoku Solver")
    print(f"{'='*60}")
    print(f"  Diffusion steps: {args.max_steps}, T: {args.T}, Noise: {args.noise_schedule}")
    print(f"  Learned timing: {args.learned_timing}, ACT steps: {args.halt_max_steps}")
    print(f"  Lastgrad: {args.lastgrad} (gradient flow for last N diffusion steps)")
    print(f"  Init noise scale: {args.init_noise_scale}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"{'='*60}\n")
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_cell, train_grid = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch,
            use_wandb=use_wandb
        )
        
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/epoch_loss': train_loss,
                'train/epoch_cell_acc': train_cell,
                'train/epoch_grid_acc': train_grid,
            })
        
        if epoch % args.eval_interval == 0 or epoch == 1:
            val_metrics = evaluate(model, val_loader, device)
            
            print(f"\nEpoch {epoch}: loss={train_loss:.4f}")
            print(f"  Train: cell={train_cell:.2f}%, grid={train_grid:.2f}%")
            print(f"  Val:   cell={val_metrics['cell_acc']:.2f}%, grid={val_metrics['grid_acc']:.2f}%")
            
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_cell_acc': train_cell,
                'train_grid_acc': train_grid,
                'val_cell_acc': val_metrics['cell_acc'],
                'val_grid_acc': val_metrics['grid_acc'],
                'val_avg_steps': val_metrics['avg_steps'],
            })
            
            if use_wandb:
                wandb.log({
                    'val/cell_acc': val_metrics['cell_acc'],
                    'val/grid_acc': val_metrics['grid_acc'],
                    'val/avg_steps': val_metrics['avg_steps'],
                    'val/best_grid_acc': max(best_grid_acc, val_metrics['grid_acc']),
                    'epoch': epoch,
                })
            
            if val_metrics['grid_acc'] > best_grid_acc:
                best_grid_acc = val_metrics['grid_acc']
                best_cell_acc = val_metrics['cell_acc']
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_cell_acc': val_metrics['cell_acc'],
                    'val_grid_acc': val_metrics['grid_acc'],
                    'args': vars(args),
                }
                
                save_path = os.path.join(args.save_dir, 'diffusion_best.pt')
                torch.save(checkpoint, save_path)
                print(f"  ✓ New best model saved! Grid accuracy: {best_grid_acc:.2f}%")
                
                if use_wandb:
                    wandb.save(save_path)
                    wandb.run.summary['best_grid_acc'] = best_grid_acc
                    wandb.run.summary['best_cell_acc'] = best_cell_acc
                    wandb.run.summary['best_epoch'] = epoch
    
    # Save final results
    results = {
        'best_grid_acc': best_grid_acc,
        'best_cell_acc': best_cell_acc,
        'num_params': num_params,
        'args': vars(args),
        'history': history,
    }
    
    results_path = os.path.join(args.save_dir, 'diffusion_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"  Best cell accuracy: {best_cell_acc:.2f}%")
    print(f"  Best grid accuracy: {best_grid_acc:.2f}%")
    print(f"  Results saved to: {results_path}")
    print(f"{'='*60}")
    
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()

