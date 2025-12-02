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
from torch.utils.data import DataLoader

# Import from refactored modules
from src.data import SudokuDataset, download_sudoku_dataset
from src.pot.models import (
    PoHSudokuSolver,
    HybridPoHHRMSolver,
    BaselineSudokuSolver,
)
from src.training import train_epoch, evaluate


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
    parser.add_argument('--H-cycles', type=int, default=2, help='Hybrid H_level outer cycles')
    parser.add_argument('--L-cycles', type=int, default=8, help='Hybrid L_level inner cycles')
    parser.add_argument('--H-layers', type=int, default=2, help='Layers in H_level module')
    parser.add_argument('--L-layers', type=int, default=2, help='Layers in L_level module')
    parser.add_argument('--hrm-grad-style', action='store_true',
                       help='Use HRM-style gradients (only last L+H call). Default: all calls in last H_cycle.')
    
    # Training (adjusted from HRM defaults for better convergence)
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4, help='Higher LR for faster learning')
    parser.add_argument('--puzzle-lr', type=float, default=3e-4, help='Same as main LR')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Much lower WD')
    parser.add_argument('--puzzle-weight-decay', type=float, default=0.01, help='Much lower WD')
    parser.add_argument('--warmup-steps', type=int, default=500, help='LR warmup steps')
    parser.add_argument('--eval-interval', type=int, default=100)
    parser.add_argument('--constraint-weight', type=float, default=0.5, 
                       help='Weight for Sudoku constraint loss')
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
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Download dataset if needed
    if args.download:
        download_sudoku_dataset(args.data_dir, args.subsample, args.num_aug)
    
    # Load data
    train_dataset = SudokuDataset(args.data_dir, 'train')
    
    # Try to load val split (held-out puzzles from training distribution)
    # Fall back to test if val doesn't exist
    try:
        val_dataset = SudokuDataset(args.data_dir, 'val')
        print("Using VAL split (held-out training puzzles) for evaluation")
    except FileNotFoundError:
        val_dataset = SudokuDataset(args.data_dir, 'test')
        print("Using TEST split (422k new puzzles) for evaluation")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
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
            H_cycles=args.H_cycles,
            L_cycles=args.L_cycles,
            T=args.T,
            num_puzzles=num_puzzles,
            hrm_grad_style=args.hrm_grad_style,
        ).to(device)
        print(f"Hybrid model: H_cycles={args.H_cycles}, L_cycles={args.L_cycles}")
        print(f"H_layers={args.H_layers}, L_layers={args.L_layers}")
        print(f"Gradient style: {'HRM (last L+H only)' if args.hrm_grad_style else 'Full (last H_cycle)'}")
    else:
        model = BaselineSudokuSolver(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=6,
            d_ff=args.d_ff,
        ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {args.model.upper()}")
    print(f"Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
    
    # Load checkpoint if specified
    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from epoch {checkpoint.get('epoch', '?')}")
        print(f"  Previous accuracy: {checkpoint.get('test_grid_acc', '?')}%")
    
    # Eval-only mode
    if args.eval_only:
        print(f"\n{'='*60}")
        print("EVALUATION MODE")
        print(f"{'='*60}")
        
        val_metrics = evaluate(model, val_loader, device, use_poh=use_poh)
        
        print(f"\nTest Results:")
        print(f"  Loss: {val_metrics['loss']:.4f}")
        print(f"  Cell Accuracy: {val_metrics['cell_acc']:.2f}%")
        print(f"  Grid Accuracy: {val_metrics['grid_acc']:.2f}%")
        print(f"\n{'='*60}")
        return
    
    # Optimizers
    if use_poh:
        puzzle_params = list(model.puzzle_emb.parameters())
        model_params = [p for p in model.parameters() if p not in set(puzzle_params)]
        
        optimizer = torch.optim.AdamW(
            model_params, lr=args.lr, weight_decay=args.weight_decay
        )
        puzzle_optimizer = torch.optim.AdamW(
            puzzle_params, lr=args.puzzle_lr, weight_decay=args.puzzle_weight_decay
        )
        
        print(f"\nOptimizer: AdamW")
        print(f"  Model params: {sum(p.numel() for p in model_params):,}, "
              f"lr={args.lr}, wd={args.weight_decay}")
        print(f"  Puzzle params: {sum(p.numel() for p in puzzle_params):,}, "
              f"lr={args.puzzle_lr}, wd={args.puzzle_weight_decay}")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        puzzle_optimizer = None
    
    # Learning rate scheduler with warmup
    total_steps = args.epochs * len(train_loader)
    
    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    puzzle_scheduler = (
        torch.optim.lr_scheduler.LambdaLR(puzzle_optimizer, lr_lambda) 
        if puzzle_optimizer else None
    )
    
    print(f"\n{'='*60}")
    print(f"Training {args.model.upper()} Sudoku Solver")
    print(f"{'='*60}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"LR: {args.lr}, Weight decay: {args.weight_decay}")
    print(f"Warmup: {args.warmup_steps} steps, Total: {total_steps} steps")
    if use_poh:
        print(f"R={args.R}, T={args.T}, max_halt={args.max_halt}")
    
    best_grid_acc = 0
    results = []
    
    os.makedirs(args.output, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        do_debug = args.debug and (epoch == 1 or epoch % args.debug_interval == 0)
        
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
            
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"  Train: Loss={train_metrics['loss']:.4f}, "
                  f"Cell={train_metrics['cell_acc']:.2f}%, Grid={train_metrics['grid_acc']:.2f}%")
            print(f"  Test:  Loss={val_metrics['loss']:.4f}, "
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
            
            # Save best model
            if val_metrics['grid_acc'] > best_grid_acc:
                best_grid_acc = val_metrics['grid_acc']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'test_grid_acc': best_grid_acc,
                }, os.path.join(args.output, f'{args.model}_best.pt'))
                print(f"  ✓ New best: {best_grid_acc:.2f}%")
            
            # Check for near-perfect accuracy
            if train_metrics['grid_acc'] >= 99.5:
                print(f"\n🎉 Reached 99.5% training accuracy!")
    
    # Final results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Best Test Grid Accuracy: {best_grid_acc:.2f}%")
    print(f"HRM Paper Target: 55%")
    
    # Save results
    with open(os.path.join(args.output, f'{args.model}_results.json'), 'w') as f:
        json.dump({
            'model': args.model,
            'parameters': param_count,
            'best_grid_acc': best_grid_acc,
            'config': vars(args),
            'history': results,
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {args.output}")


if __name__ == '__main__':
    main()
