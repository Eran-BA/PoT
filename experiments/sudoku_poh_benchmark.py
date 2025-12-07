#!/usr/bin/env python3
"""
PoT Sudoku Benchmark - Master-Level Sudoku Solver
==================================================

Replicates the HRM paper's Sudoku demo using PoT architecture.
Train a master-level Sudoku solver with 10000 extreme puzzles.

Based on: https://github.com/sapientinc/HRM

Task: Given a 9x9 Sudoku puzzle with blanks (0), output the complete solution.
Input:  81 tokens (flattened 9x9), vocab 0-9 (0=blank)
Output: 81 tokens (complete solution), vocab 1-9

Usage:
    # Download dataset and train hybrid model
    python experiments/sudoku_poh_benchmark.py --download --model hybrid
    
    # Train with custom settings
    python experiments/sudoku_poh_benchmark.py --model hybrid --H-cycles 3 --L-cycles 12
    
    # Evaluate checkpoint
    python experiments/sudoku_poh_benchmark.py --eval-only --checkpoint path/to/model.pt

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.optimizer import Optimizer


# ========== Distributed Training Utilities ==========

def is_distributed() -> bool:
    """Check if running in distributed mode (launched with torchrun)."""
    return 'RANK' in os.environ and 'WORLD_SIZE' in os.environ

def get_rank() -> int:
    """Get global rank (0 if not distributed)."""
    if is_distributed():
        return int(os.environ['RANK'])
    return 0

def get_local_rank() -> int:
    """Get local rank for device assignment (0 if not distributed)."""
    if is_distributed():
        return int(os.environ['LOCAL_RANK'])
    return 0

def get_world_size() -> int:
    """Get world size (1 if not distributed)."""
    if is_distributed():
        return int(os.environ['WORLD_SIZE'])
    return 1

def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0

def setup_distributed():
    """Initialize distributed training."""
    if not is_distributed():
        return
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(get_local_rank())

def cleanup_distributed():
    """Clean up distributed training."""
    if is_distributed():
        dist.destroy_process_group()

def print_rank0(*args, **kwargs):
    """Print only from rank 0."""
    if is_main_process():
        print(*args, **kwargs)

# Import from refactored modules
from src.data import SudokuDataset, download_sudoku_dataset
from src.pot.models import (
    PoHSudokuSolver,
    HybridPoHHRMSolver,
    BaselineSudokuSolver,
)
from src.training import train_epoch, train_epoch_async, evaluate

# Try to import adam_atan2 (HRM's optimizer)
try:
    from adam_atan2 import AdamATan2
    HAS_ADAM_ATAN2 = True
except ImportError:
    HAS_ADAM_ATAN2 = False
    AdamATan2 = None


class SignSGD(Optimizer):
    """
    SignSGD optimizer with decoupled weight decay (HRM-style for puzzle embeddings).
    
    Updates weights using only the sign of gradients, which can be more stable
    for sparse updates like puzzle embeddings.
    
    Args:
        params: Parameters to optimize
        lr: Learning rate
        weight_decay: Decoupled weight decay coefficient
    """
    def __init__(self, params, lr=1e-3, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Decoupled weight decay (apply before gradient step)
                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)
                
                # SignSGD update: p = p - lr * sign(grad)
                p.add_(torch.sign(p.grad), alpha=-lr)
        
        return loss


def main():
    parser = argparse.ArgumentParser(description='PoT Sudoku Benchmark')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='data/sudoku-extreme-10k-aug-100',
                       help='Path to Sudoku dataset')
    parser.add_argument('--download', action='store_true',
                       help='Download and build dataset from HuggingFace')
    parser.add_argument('--subsample', type=int, default=10000,
                       help='Number of puzzles to use (for download)')
    parser.add_argument('--num-aug', type=int, default=100,
                       help='Augmentations per puzzle (for download)')
    
    # Model
    parser.add_argument('--model', choices=['poh', 'baseline', 'hybrid'], default='hybrid')
    parser.add_argument('--d-model', type=int, default=512)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--n-layers', type=int, default=2, help='Layers for PoH (baseline uses 6)')
    parser.add_argument('--d-ff', type=int, default=2048)
    parser.add_argument('--R', type=int, default=8, help='Refinement iterations')
    parser.add_argument('--T', type=int, default=4, help='HRM outer period')
    parser.add_argument('--max-halt', type=int, default=16, help='Max halting steps')
    
    # Hybrid model args (balanced for A100 memory)
    parser.add_argument('--H-cycles', type=int, default=2, help='Hybrid H_level outer cycles (fixed per ACT step)')
    parser.add_argument('--L-cycles', type=int, default=8, help='Hybrid L_level inner cycles (fixed per H_cycle)')
    parser.add_argument('--H-layers', type=int, default=2, help='Layers in H_level module')
    parser.add_argument('--L-layers', type=int, default=2, help='Layers in L_level module')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for regularization (HRM default: 0.0)')
    parser.add_argument('--hrm-grad-style', action='store_true',
                       help='Use HRM-style gradients (only last L+H call). Default: all calls in last H_cycle.')
    
    # ACT (Adaptive Computation Time) - like HRM's adaptive outer steps
    parser.add_argument('--halt-max-steps', type=int, default=1,
                       help='Max ACT outer steps (1=no ACT, >1=ACT enabled like HRM). HRM uses 8 or 16.')
    parser.add_argument('--halt-exploration-prob', type=float, default=0.1,
                       help='Exploration probability for Q-learning halting')
    parser.add_argument('--async-batch', action='store_true',
                       help='Use HRM-style async batching: halted samples are immediately replaced with new puzzles')
    
    # Training (adjusted from HRM defaults for better convergence)
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'adam_atan2'],
                       help='Main optimizer. HRM uses adam_atan2 (pip install adam-atan2)')
    parser.add_argument('--puzzle-optimizer', type=str, default='adamw', choices=['adamw', 'signsgd'],
                       help='Puzzle embedding optimizer. HRM uses signsgd')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--puzzle-lr-multiplier', type=float, default=1.0,
                       help='Puzzle emb LR = lr * multiplier. HRM uses 100x (set to 100.0)')
    parser.add_argument('--weight-decay', type=float, default=0.01, 
                       help='Weight decay. HRM uses 0.1-1.0')
    parser.add_argument('--puzzle-weight-decay', type=float, default=0.01, 
                       help='Puzzle embedding weight decay. HRM uses 0.1-1.0')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999, 
                       help='Adam beta2. HRM uses 0.95 (Llama-style)')
    parser.add_argument('--warmup-steps', type=int, default=500, 
                       help='LR warmup steps. HRM uses 2000')
    parser.add_argument('--lr-min-ratio', type=float, default=0.0,
                       help='Min LR ratio for cosine schedule. HRM uses 0.1 or 1.0')
    parser.add_argument('--eval-interval', type=int, default=100)
    parser.add_argument('--constraint-weight', type=float, default=0.5, 
                       help='Weight for Sudoku constraint loss (not in HRM)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--debug-interval', type=int, default=10, help='Debug every N epochs')
    
    # Output
    parser.add_argument('--output', type=str, default='experiments/results/sudoku_poh')
    parser.add_argument('--seed', type=int, default=42)
    
    # Evaluation mode
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate, skip training')
    parser.add_argument('--checkpoint', type=str, default=None, 
                       help='Path to checkpoint for evaluation')
    
    args = parser.parse_args()
    
    # Setup distributed training (if launched with torchrun)
    setup_distributed()
    distributed = is_distributed()
    local_rank = get_local_rank()
    world_size = get_world_size()
    
    # Setup device
    if distributed:
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set seed (different per rank for proper shuffling, but reproducible)
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print_rank0(f"Device: {device}")
    if distributed:
        print_rank0(f"Distributed training: {world_size} GPUs")
    if torch.cuda.is_available():
        print_rank0(f"GPU: {torch.cuda.get_device_name(local_rank)}")
    
    # Download dataset if needed (only rank 0, then sync)
    if args.download:
        if is_main_process():
            download_sudoku_dataset(args.data_dir, args.subsample, args.num_aug)
        if distributed:
            dist.barrier()  # Wait for rank 0 to finish downloading
    
    # Load data
    train_dataset = SudokuDataset(args.data_dir, 'train')
    
    # Try to load val split (held-out puzzles from training distribution)
    # Fall back to test if val doesn't exist
    try:
        val_dataset = SudokuDataset(args.data_dir, 'val')
        print_rank0("Using VAL split (held-out training puzzles) for evaluation")
    except FileNotFoundError:
        val_dataset = SudokuDataset(args.data_dir, 'test')
        print_rank0("Using TEST split (422k new puzzles) for evaluation")
    
    # For async batching, drop_last=True ensures consistent batch sizes
    # (required because carry state has fixed batch dimension)
    drop_last = args.async_batch and args.halt_max_steps > 1 and args.model == 'hybrid'
    
    # Use DistributedSampler for multi-GPU training
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=(train_sampler is None),  # Only shuffle if not using sampler
        sampler=train_sampler,
        num_workers=4,
        drop_last=drop_last,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
    )
    
    # Single shared puzzle embedding like HRM
    num_puzzles = 1
    
    # Build model
    use_poh = args.model in ('poh', 'hybrid')
    
    if args.model == 'poh':
        model = PoHSudokuSolver(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            R=args.R,
            T=args.T,
            num_puzzles=num_puzzles,
            max_halting_steps=args.max_halt,
        ).to(device)
    elif args.model == 'hybrid':
        model = HybridPoHHRMSolver(
            d_model=args.d_model,
            n_heads=args.n_heads,
            H_layers=args.H_layers,
            L_layers=args.L_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            H_cycles=args.H_cycles,
            L_cycles=args.L_cycles,
            T=args.T,
            num_puzzles=num_puzzles,
            hrm_grad_style=args.hrm_grad_style,
            halt_max_steps=args.halt_max_steps,
            halt_exploration_prob=args.halt_exploration_prob,
        ).to(device)
        print_rank0(f"Hybrid model: H_cycles={args.H_cycles}, L_cycles={args.L_cycles}")
        print_rank0(f"H_layers={args.H_layers}, L_layers={args.L_layers}, dropout={args.dropout}")
        print_rank0(f"Gradient style: {'HRM (last L+H only)' if args.hrm_grad_style else 'Full (last H_cycle)'}")
        if args.halt_max_steps > 1:
            print_rank0(f"ACT enabled: halt_max_steps={args.halt_max_steps}, exploration={args.halt_exploration_prob}")
            if args.async_batch:
                print_rank0(f"Async batching enabled (HRM-style)")
        else:
            print_rank0(f"ACT disabled (halt_max_steps=1)")
            if args.async_batch:
                print_rank0(f"WARNING: --async-batch requires --halt-max-steps > 1. Disabling async batching.")
    else:
        model = BaselineSudokuSolver(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=6,
            d_ff=args.d_ff,
            dropout=args.dropout,
        ).to(device)
    
    # Wrap model in DDP for distributed training
    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    param_count = sum(p.numel() for p in model.parameters())
    print_rank0(f"\nModel: {args.model.upper()}")
    print_rank0(f"Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
    
    # Load checkpoint if specified
    if args.checkpoint:
        print_rank0(f"\nLoading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # Handle DDP wrapped model
        state_dict = checkpoint['model_state_dict']
        if distributed:
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        print_rank0(f"  Loaded from epoch {checkpoint.get('epoch', '?')}")
        print_rank0(f"  Previous accuracy: {checkpoint.get('test_grid_acc', '?')}%")
    
    # Eval-only mode
    if args.eval_only:
        print_rank0(f"\n{'='*60}")
        print_rank0("EVALUATION MODE")
        print_rank0(f"{'='*60}")
        
        val_metrics = evaluate(model, val_loader, device, use_poh=use_poh)
        
        print_rank0(f"\nTest Results:")
        print_rank0(f"  Loss: {val_metrics['loss']:.4f}")
        print_rank0(f"  Cell Accuracy: {val_metrics['cell_acc']:.2f}%")
        print_rank0(f"  Grid Accuracy: {val_metrics['grid_acc']:.2f}%")
        print_rank0(f"\n{'='*60}")
        cleanup_distributed()
        return
    
    # Optimizers
    puzzle_lr = args.lr * args.puzzle_lr_multiplier
    betas = (args.beta1, args.beta2)
    
    # Validate optimizer choices
    if args.optimizer == 'adam_atan2' and not HAS_ADAM_ATAN2:
        print_rank0("⚠️  adam_atan2 not installed. Install with: pip install adam-atan2")
        print_rank0("    Falling back to AdamW")
        args.optimizer = 'adamw'
    
    def create_optimizer(params, lr, wd, opt_type, is_puzzle=False):
        """Create optimizer based on type."""
        if opt_type == 'adam_atan2':
            return AdamATan2(params, lr=lr, weight_decay=wd, betas=betas)
        elif opt_type == 'signsgd':
            return SignSGD(params, lr=lr, weight_decay=wd)
        else:  # adamw
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)
    
    # Get underlying model for parameter access (handles DDP wrapping)
    base_model = model.module if distributed else model
    
    if use_poh:
        puzzle_params = list(base_model.puzzle_emb.parameters())
        model_params = [p for p in model.parameters() if p not in set(puzzle_params)]
        
        optimizer = create_optimizer(
            model_params, args.lr, args.weight_decay, args.optimizer
        )
        puzzle_optimizer = create_optimizer(
            puzzle_params, puzzle_lr, args.puzzle_weight_decay, args.puzzle_optimizer, is_puzzle=True
        )
        
        opt_name = args.optimizer.upper()
        puzzle_opt_name = args.puzzle_optimizer.upper()
        print_rank0(f"\nOptimizer: {opt_name} (betas={betas})")
        print_rank0(f"  Model params: {sum(p.numel() for p in model_params):,}, "
              f"lr={args.lr}, wd={args.weight_decay}")
        print_rank0(f"Puzzle Optimizer: {puzzle_opt_name}")
        print_rank0(f"  Puzzle params: {sum(p.numel() for p in puzzle_params):,}, "
              f"lr={puzzle_lr} ({args.puzzle_lr_multiplier}x), wd={args.puzzle_weight_decay}")
    else:
        optimizer = create_optimizer(
            model.parameters(), args.lr, args.weight_decay, args.optimizer
        )
        puzzle_optimizer = None
    
    # Learning rate scheduler with warmup (HRM-style cosine with min ratio)
    # In DDP, we want same total steps as single GPU training
    if distributed:
        # Use full dataset size / batch_size, not divided by world_size
        steps_per_epoch = (len(train_dataset) + args.batch_size - 1) // args.batch_size
    else:
        steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    min_ratio = args.lr_min_ratio
    
    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
        # Cosine decay from 1.0 to min_ratio (HRM uses 0.1 or 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    puzzle_scheduler = (
        torch.optim.lr_scheduler.LambdaLR(puzzle_optimizer, lr_lambda) 
        if puzzle_optimizer else None
    )
    
    print_rank0(f"\n{'='*60}")
    print_rank0(f"Training {args.model.upper()} Sudoku Solver")
    print_rank0(f"{'='*60}")
    print_rank0(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    if distributed:
        print_rank0(f"Distributed: {world_size} GPUs, effective batch size: {args.batch_size * world_size}")
    print_rank0(f"Optimizer: {args.optimizer}, Puzzle optimizer: {args.puzzle_optimizer}")
    print_rank0(f"LR: {args.lr}, Weight decay: {args.weight_decay}, Betas: {betas}")
    print_rank0(f"Warmup: {args.warmup_steps} steps, LR min ratio: {args.lr_min_ratio}")
    print_rank0(f"Total steps: {total_steps}")
    if use_poh:
        print_rank0(f"R={args.R}, T={args.T}, max_halt={args.max_halt}")
    
    best_grid_acc = 0
    results = []
    
    if is_main_process():
        os.makedirs(args.output, exist_ok=True)
    
    # Check if async batching is valid
    use_async = args.async_batch and args.halt_max_steps > 1 and args.model == 'hybrid'
    if args.async_batch and not use_async:
        if args.halt_max_steps <= 1:
            print_rank0("Note: Async batching disabled because halt_max_steps <= 1")
        if args.model != 'hybrid':
            print_rank0("Note: Async batching only supported for hybrid model")
    
    # For async batching, use FULL dataset size (not divided by world_size)
    # This ensures each GPU processes the same effective samples as single GPU
    if distributed:
        full_dataset_size = len(train_dataset)
    else:
        full_dataset_size = len(train_loader) * args.batch_size
    
    for epoch in range(1, args.epochs + 1):
        # Set epoch for distributed sampler (ensures proper shuffling)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        do_debug = args.debug and (epoch == 1 or epoch % args.debug_interval == 0)
        
        if use_async:
            # HRM-style async batching: halted samples replaced immediately
            # Use full dataset size so DDP trains same amount as single GPU
            train_metrics = train_epoch_async(
                model, train_loader, optimizer, puzzle_optimizer, 
                device, epoch, use_poh=use_poh, debug=do_debug,
                scheduler=scheduler, puzzle_scheduler=puzzle_scheduler,
                constraint_weight=args.constraint_weight,
                samples_per_epoch=full_dataset_size,
            )
        else:
            # Standard mini-batch training
            train_metrics = train_epoch(
                model, train_loader, optimizer, puzzle_optimizer, 
                device, epoch, use_poh=use_poh, debug=do_debug,
                scheduler=scheduler, puzzle_scheduler=puzzle_scheduler,
                constraint_weight=args.constraint_weight
            )
        
        # Resample augmentations for next epoch
        train_dataset.on_epoch_end()
        
        # Evaluate periodically
        if epoch % args.eval_interval == 0 or epoch == 1:
            val_metrics = evaluate(model, val_loader, device, use_poh=use_poh)
            
            print_rank0(f"\nEpoch {epoch}/{args.epochs}")
            print_rank0(f"  Train: Loss={train_metrics['loss']:.4f}, "
                  f"Cell={train_metrics['cell_acc']:.2f}%, Grid={train_metrics['grid_acc']:.2f}%")
            print_rank0(f"  Test:  Loss={val_metrics['loss']:.4f}, "
                  f"Cell={val_metrics['cell_acc']:.2f}%, Grid={val_metrics['grid_acc']:.2f}%")
            
            results.append({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_cell_acc': train_metrics['cell_acc'],
                'train_grid_acc': train_metrics['grid_acc'],
                'test_loss': val_metrics['loss'],
                'test_cell_acc': val_metrics['cell_acc'],
                'test_grid_acc': val_metrics['grid_acc'],
            })
            
            # Save best model (only rank 0)
            if val_metrics['grid_acc'] > best_grid_acc:
                best_grid_acc = val_metrics['grid_acc']
                if is_main_process():
                    # Get state dict from underlying model (handles DDP)
                    state_dict = base_model.state_dict()
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': state_dict,
                        'test_grid_acc': best_grid_acc,
                    }, os.path.join(args.output, f'{args.model}_best.pt'))
                print_rank0(f"  ✓ New best: {best_grid_acc:.2f}%")
            
            # Check for near-perfect accuracy
            if train_metrics['grid_acc'] >= 99.5:
                print_rank0(f"\n🎉 Reached 99.5% training accuracy!")
    
    # Final results
    print_rank0(f"\n{'='*60}")
    print_rank0("FINAL RESULTS")
    print_rank0(f"{'='*60}")
    print_rank0(f"Best Test Grid Accuracy: {best_grid_acc:.2f}%")
    print_rank0(f"HRM Paper Target: 55%")
    
    # Save results (only rank 0)
    if is_main_process():
        with open(os.path.join(args.output, f'{args.model}_results.json'), 'w') as f:
            json.dump({
                'model': args.model,
                'parameters': param_count,
                'best_grid_acc': best_grid_acc,
                'config': vars(args),
                'history': results,
            }, f, indent=2)
    
    print_rank0(f"\n💾 Results saved to: {args.output}")
    
    # Clean up distributed training
    cleanup_distributed()


if __name__ == '__main__':
    main()
