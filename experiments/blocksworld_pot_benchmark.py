#!/usr/bin/env python3
"""
PoT Blocksworld Benchmark - Symbolic Planning with Depth Transformers
======================================================================

Train and evaluate PoT models on the Blocksworld planning domain from PlanBench.
Uses CausalDepthTransformerRouter for iterative state refinement.

Task: Given a state configuration, predict the next state (transition model).
Can also be used for goal-conditioned planning with rollout evaluation.

State representation:
- For N blocks: state is [pos_block_0, pos_block_1, ..., pos_block_{N-1}]
- Each pos_block_i in {0=table, 1..N = on_block_j, N+1 = holding}

Usage:
    # Download dataset and train PoT model
    python experiments/blocksworld_pot_benchmark.py --download --model pot
    
    # Train baseline comparison
    python experiments/blocksworld_pot_benchmark.py --model baseline
    
    # Evaluate checkpoint
    python experiments/blocksworld_pot_benchmark.py --eval-only --checkpoint path/to/model.pt

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import sys
import os
import math
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.blocksworld import (
    BlocksworldDataset,
    BlocksworldTrajectoryDataset,
    download_blocksworld_dataset,
)
from src.pot.models.blocksworld_solver import (
    HybridPoTBlocksworldSolver,
    BaselineBlocksworldSolver,
    SimplePoTBlocksworldSolver,
)


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(
    model,
    dataloader,
    device: torch.device,
    max_rollout_steps: int = 10,
) -> dict:
    """
    Compute evaluation metrics for Blocksworld transition model.
    
    Metrics:
    - slot_acc: Percentage of block positions predicted correctly
    - exact_match: Percentage of states fully correct (all blocks right)
    - loss: Average cross-entropy loss
    
    For goal-conditioned evaluation (if goals available):
    - plan_success_rate: Percentage of rollouts reaching goal state
    """
    model.eval()
    
    total_loss = 0
    total_correct_slots = 0
    total_slots = 0
    total_exact_match = 0
    total_samples = 0
    
    # Collect metrics by plan length (for length robustness analysis)
    length_metrics = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            if 'state' in batch:
                # Transition mode: (state, next_state) pairs
                state = batch['state'].to(device)
                target = batch['next_state'].to(device)
                num_blocks = batch['num_blocks'].to(device)
                goal = None
            else:
                # Goal mode: (init_state, goal_state) pairs
                state = batch['init_state'].to(device)
                target = None
                goal = batch['goal_state'].to(device)
                num_blocks = batch['num_blocks'].to(device)
                plan_lengths = batch.get('plan_length', None)
            
            B = state.shape[0]
            
            # Forward pass
            if hasattr(model, 'forward'):
                output = model(state, goal=goal if hasattr(model, 'goal_conditioned') and model.goal_conditioned else None)
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
            
            # Determine target for evaluation
            # In goal mode, we evaluate prediction against goal_state
            eval_target = target if target is not None else goal
            
            if eval_target is not None:
                # Compute loss
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    eval_target.view(-1),
                    reduction='mean',
                )
                total_loss += loss.item() * B
                
                # Slot accuracy
                preds = logits.argmax(dim=-1)  # [B, N]
                correct = (preds == eval_target).float()
                
                # Mask out padding positions (where num_blocks < N)
                N = state.shape[1]
                mask = torch.arange(N, device=device).unsqueeze(0) < num_blocks.unsqueeze(1)
                
                total_correct_slots += (correct * mask).sum().item()
                total_slots += mask.sum().item()
                
                # Exact match (all slots correct within valid range)
                all_correct = ((correct * mask).sum(dim=1) == mask.sum(dim=1)).float()
                total_exact_match += all_correct.sum().item()
            
            total_samples += B
    
    metrics = {
        'loss': total_loss / max(total_samples, 1),
        'slot_acc': 100 * total_correct_slots / max(total_slots, 1),
        'exact_match': 100 * total_exact_match / max(total_samples, 1),
        'num_samples': total_samples,
    }
    
    return metrics


def evaluate_rollouts(
    model,
    dataloader,
    device: torch.device,
    max_steps: int = 15,
) -> dict:
    """
    Evaluate rollout quality for goal-conditioned planning.
    
    For each (init, goal) pair, roll out the model and check if it reaches goal.
    
    Returns:
        Dict with plan_success_rate, avg_steps_to_goal, etc.
    """
    model.eval()
    
    total_success = 0
    total_samples = 0
    steps_to_goal = []
    
    # Metrics by plan length
    success_by_length = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Rollout eval", leave=False):
            if 'init_state' not in batch:
                continue
            
            init_state = batch['init_state'].to(device)
            goal_state = batch['goal_state'].to(device)
            plan_lengths = batch.get('plan_length', torch.zeros(init_state.size(0)))
            num_blocks = batch['num_blocks'].to(device)
            
            B, N = init_state.shape
            
            # Rollout
            state = init_state.clone()
            reached_goal = torch.zeros(B, dtype=torch.bool, device=device)
            first_success_step = torch.full((B,), max_steps + 1, dtype=torch.long, device=device)
            
            for step in range(max_steps):
                if reached_goal.all():
                    break
                
                # Predict next state
                if hasattr(model, 'predict_next_state'):
                    next_state = model.predict_next_state(
                        state, 
                        goal=goal_state if hasattr(model, 'goal_conditioned') and model.goal_conditioned else None
                    )
                else:
                    output = model(state)
                    if isinstance(output, tuple):
                        logits = output[0]
                    else:
                        logits = output
                    next_state = logits.argmax(dim=-1)
                
                state = next_state
                
                # Check if goal reached (within valid blocks)
                mask = torch.arange(N, device=device).unsqueeze(0) < num_blocks.unsqueeze(1)
                matches = ((state == goal_state) | ~mask)
                newly_reached = matches.all(dim=1) & ~reached_goal
                
                # Record step for newly successful samples
                first_success_step[newly_reached] = step + 1
                reached_goal = reached_goal | newly_reached
            
            # Aggregate
            total_success += reached_goal.sum().item()
            total_samples += B
            
            for i in range(B):
                if reached_goal[i]:
                    steps_to_goal.append(first_success_step[i].item())
                
                # Track by plan length
                pl = int(plan_lengths[i].item())
                if pl not in success_by_length:
                    success_by_length[pl] = {'success': 0, 'total': 0}
                success_by_length[pl]['total'] += 1
                if reached_goal[i]:
                    success_by_length[pl]['success'] += 1
    
    metrics = {
        'plan_success_rate': 100 * total_success / max(total_samples, 1),
        'avg_steps_to_goal': np.mean(steps_to_goal) if steps_to_goal else float('nan'),
        'total_samples': total_samples,
        'success_by_length': {
            k: 100 * v['success'] / max(v['total'], 1)
            for k, v in sorted(success_by_length.items())
        },
    }
    
    return metrics


# =============================================================================
# Training
# =============================================================================

def train_epoch(
    model,
    dataloader,
    optimizer,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
    goal_conditioned: bool = False,
) -> dict:
    """
    Train for one epoch.
    
    Handles both data formats:
    - Transition mode: batch has 'state', 'next_state'
    - Goal mode (sub-trajectory augmentation): batch has 'init_state', 'goal_state'
    """
    model.train()
    
    total_loss = 0
    total_correct = 0
    total_slots = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        # Handle both data formats
        if 'init_state' in batch:
            # Goal mode: predict goal_state from init_state
            state = batch['init_state'].to(device)
            target = batch['goal_state'].to(device)
        else:
            # Transition mode: predict next_state from state
            state = batch['state'].to(device)
            target = batch['next_state'].to(device)
        
        num_blocks = batch['num_blocks'].to(device)
        
        B, N = state.shape
        
        optimizer.zero_grad()
        
        # Forward (with goal conditioning if enabled)
        if goal_conditioned and 'goal_state' in batch:
            goal = batch['goal_state'].to(device)
            output = model(state, goal=goal)
        else:
            output = model(state)
        
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        
        # Mask for valid positions
        mask = torch.arange(N, device=device).unsqueeze(0) < num_blocks.unsqueeze(1)
        
        # Loss (only on valid positions)
        logits_flat = logits.view(-1, logits.size(-1))
        target_flat = target.view(-1)
        mask_flat = mask.view(-1)
        
        loss = F.cross_entropy(logits_flat, target_flat, reduction='none')
        loss = (loss * mask_flat.float()).sum() / mask_flat.sum()
        
        # Backward
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct = ((preds == target) * mask).sum().item()
        total_correct += correct
        total_slots += mask.sum().item()
        num_batches += 1
        
        # Update progress bar
        slot_acc = 100 * total_correct / max(total_slots, 1)
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'slot_acc': f'{slot_acc:.1f}%',
        })
    
    return {
        'loss': total_loss / max(num_batches, 1),
        'slot_acc': 100 * total_correct / max(total_slots, 1),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='PoT Blocksworld Benchmark')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='data/blocksworld',
                       help='Path to Blocksworld dataset')
    parser.add_argument('--download', action='store_true',
                       help='Download and build dataset from HuggingFace/synthetic')
    parser.add_argument('--max-blocks', type=int, default=6,
                       help='Maximum number of blocks to use')
    parser.add_argument('--max-plan-length', type=int, default=None,
                       help='Filter to plans with at most this many steps')
    
    # Planning mode
    parser.add_argument('--planning-mode', choices=['inner', 'none', 'external'], default='inner',
                       help='Planning mode: inner=PoT iterations, none=baseline R=1, external=FD augmentation')
    parser.add_argument('--generate-trajectories', action='store_true',
                       help='Use FastDownward to generate trajectories (required for external mode)')
    parser.add_argument('--fd-path', type=str, default='fast-downward.py',
                       help='Path to FastDownward executable')
    parser.add_argument('--fd-timeout', type=int, default=30,
                       help='Per-problem timeout for FastDownward (seconds)')
    
    # Model
    parser.add_argument('--model', choices=['pot', 'hybrid', 'baseline', 'simple'], default='simple',
                       help='Model type: pot/hybrid (full PoT), simple (explicit cycles), baseline')
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--d-ff', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--R', type=int, default=8, help='Refinement iterations')
    parser.add_argument('--goal-conditioned', action='store_true',
                       help='Use goal state as additional input')
    
    # Hybrid model args
    parser.add_argument('--H-cycles', type=int, default=2, help='H_level outer cycles')
    parser.add_argument('--L-cycles', type=int, default=8, help='L_level inner cycles')
    parser.add_argument('--H-layers', type=int, default=2, help='Layers in H_level')
    parser.add_argument('--L-layers', type=int, default=2, help='Layers in L_level')
    parser.add_argument('--T', type=int, default=4, help='HRM period')
    
    # Controller
    parser.add_argument('--controller', type=str, default='transformer',
                       choices=['gru', 'lstm', 'xlstm', 'mingru', 'transformer', 'pot_transformer'],
                       help='Controller type for depth routing')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--eval-interval', type=int, default=10)
    
    # Rollout evaluation
    parser.add_argument('--eval-rollouts', action='store_true',
                       help='Evaluate rollout success rate (slower)')
    parser.add_argument('--max-rollout-steps', type=int, default=15,
                       help='Maximum rollout steps for evaluation')
    
    # Output
    parser.add_argument('--output', type=str, default='experiments/results/blocksworld_pot')
    parser.add_argument('--seed', type=int, default=42)
    
    # Checkpoints
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for evaluation or resume')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only evaluate, skip training')
    
    # W&B
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging')
    parser.add_argument('--project', type=str, default='blocksworld-pot')
    parser.add_argument('--run-name', type=str, default=None)
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Download dataset if needed
    if args.download:
        # For external mode, we need to generate trajectories using FastDownward
        generate_trajs = args.generate_trajectories or args.planning_mode == 'external'
        download_blocksworld_dataset(
            args.data_dir,
            max_blocks=args.max_blocks,
            generate_trajectories=generate_trajs,
            fd_path=args.fd_path,
            fd_timeout=args.fd_timeout,
        )
    
    # Configure based on planning mode
    # - inner: PoT with R iterations, no augmentation (natural inner planning)
    # - none: Baseline with R=1, no augmentation (direct prediction)
    # - external: PoT with R iterations, C(n+1,2) sub-trajectory augmentation from FD
    print(f"\nPlanning mode: {args.planning_mode}")
    
    # Override R for baseline mode
    if args.planning_mode == 'none':
        print("  Mode 'none': Forcing R=1 (no refinement iterations)")
        args.R = 1
        use_augment = False
    elif args.planning_mode == 'inner':
        print(f"  Mode 'inner': PoT with R={args.R} refinement iterations (no augmentation)")
        use_augment = False
    else:  # external
        print(f"  Mode 'external': PoT with R={args.R} + C(n+1,2) sub-trajectory augmentation")
        use_augment = True
    
    # Load data
    # Training: sub-trajectory augmentation only in 'external' mode
    # Val/Test: never use augmentation (no data leakage)
    print("\nLoading dataset...")
    try:
        train_dataset = BlocksworldDataset(
            args.data_dir,
            split='train',
            max_blocks=args.max_blocks,
            mode='goal',  # (init, goal) pairs
            max_plan_length=args.max_plan_length,
            augment=use_augment,  # Only augment in external mode
        )
        val_dataset = BlocksworldDataset(
            args.data_dir,
            split='val',
            max_blocks=args.max_blocks,
            mode='goal',
            max_plan_length=args.max_plan_length,
            augment=False,  # Never augment validation
        )
        test_dataset = BlocksworldDataset(
            args.data_dir,
            split='test',
            max_blocks=args.max_blocks,
            mode='goal',
            max_plan_length=args.max_plan_length,
            augment=False,  # Never augment test
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run with --download to generate the dataset.")
        if args.planning_mode == 'external':
            print("For external mode, also add --generate-trajectories to use FastDownward.")
        return
    
    # For rollout evaluation, use the same test dataset (already goal mode)
    if args.eval_rollouts:
        goal_test_dataset = test_dataset
    else:
        goal_test_dataset = None
    
    # Note: shuffle=False because dataset handles shuffling internally
    # via on_epoch_end() (DQN-style experience replay)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Dataset shuffles internally
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    if args.eval_rollouts:
        goal_loader = DataLoader(
            goal_test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
        )
    else:
        goal_loader = None
    
    # Build model
    print(f"\nBuilding {args.model.upper()} model...")
    
    if args.model == 'simple':
        model = SimplePoTBlocksworldSolver(
            num_blocks=args.max_blocks,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            R=args.R,
            controller_type=args.controller,
            goal_conditioned=args.goal_conditioned,
        ).to(device)
    elif args.model in ('pot', 'hybrid'):
        model = HybridPoTBlocksworldSolver(
            num_blocks=args.max_blocks,
            d_model=args.d_model,
            n_heads=args.n_heads,
            H_layers=args.H_layers,
            L_layers=args.L_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            H_cycles=args.H_cycles,
            L_cycles=args.L_cycles,
            T=args.T,
            controller_type=args.controller,
            goal_conditioned=args.goal_conditioned,
        ).to(device)
    else:  # baseline
        model = BaselineBlocksworldSolver(
            num_blocks=args.max_blocks,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers * 2,  # More layers to match param count
            d_ff=args.d_ff,
            dropout=args.dropout,
            goal_conditioned=args.goal_conditioned,
        ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model.upper()}")
    print(f"Planning Mode: {args.planning_mode}")
    print(f"Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
    print(f"Max blocks: {args.max_blocks}, Vocab size: {model.vocab_size}")
    
    if args.model == 'simple':
        print(f"R={args.R}, Controller: {args.controller}")
    elif args.model in ('pot', 'hybrid'):
        print(f"H_cycles={args.H_cycles}, L_cycles={args.L_cycles}, Controller: {args.controller}")
    
    # Load checkpoint if specified
    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from epoch {checkpoint.get('epoch', '?')}")
    
    # Eval-only mode
    if args.eval_only:
        print(f"\n{'='*60}")
        print("EVALUATION MODE")
        print(f"{'='*60}")
        
        test_metrics = compute_metrics(model, test_loader, device)
        
        print(f"\nTest Results:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  Slot Accuracy: {test_metrics['slot_acc']:.2f}%")
        print(f"  Exact Match: {test_metrics['exact_match']:.2f}%")
        
        if args.eval_rollouts and goal_loader:
            rollout_metrics = evaluate_rollouts(
                model, goal_loader, device,
                max_steps=args.max_rollout_steps,
            )
            print(f"\nRollout Results:")
            print(f"  Plan Success Rate: {rollout_metrics['plan_success_rate']:.2f}%")
            print(f"  Avg Steps to Goal: {rollout_metrics['avg_steps_to_goal']:.2f}")
            if rollout_metrics['success_by_length']:
                print(f"  Success by Plan Length:")
                for length, rate in rollout_metrics['success_by_length'].items():
                    print(f"    Length {length}: {rate:.1f}%")
        
        print(f"\n{'='*60}")
        return
    
    # W&B logging
    if args.wandb:
        import wandb
        run_name = args.run_name or f"{args.model}_{args.controller}_{datetime.now():%Y%m%d_%H%M%S}"
        wandb.init(project=args.project, name=run_name, config=vars(args))
        wandb.watch(model, log="gradients", log_freq=100)
        print(f"W&B logging enabled: {args.project}/{run_name}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # LR scheduler with warmup
    total_steps = args.epochs * len(train_loader)
    
    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Training {args.model.upper()} Blocksworld Solver")
    print(f"{'='*60}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"LR: {args.lr}, Weight decay: {args.weight_decay}")
    
    best_val_slot_acc = -1  # Track best validation slot accuracy
    results = []
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch,
            grad_clip=args.grad_clip,
            goal_conditioned=args.goal_conditioned,
        )
        scheduler.step()
        
        # Shuffle training data for next epoch (DQN-style experience replay)
        if hasattr(train_dataset, 'on_epoch_end'):
            train_dataset.on_epoch_end()
        
        # Evaluate
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            val_metrics = compute_metrics(model, val_loader, device)
            
            print(f"\nEpoch {epoch}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Slot Acc: {train_metrics['slot_acc']:.2f}%")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Slot Acc: {val_metrics['slot_acc']:.2f}%, "
                  f"Exact: {val_metrics['exact_match']:.2f}%")
            
            # Save best model (using slot_acc since exact_match is often 0%)
            if val_metrics['slot_acc'] > best_val_slot_acc:
                best_val_slot_acc = val_metrics['slot_acc']
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_exact_match': val_metrics['exact_match'],
                    'val_slot_acc': val_metrics['slot_acc'],
                    'args': vars(args),
                }
                torch.save(checkpoint, os.path.join(args.output, 'best_model.pt'))
                print(f"  [NEW BEST] Saved checkpoint (slot_acc={best_val_slot_acc:.2f}%)")
            
            # Log to W&B
            if args.wandb:
                import wandb
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_metrics['loss'],
                    'train/slot_acc': train_metrics['slot_acc'],
                    'val/loss': val_metrics['loss'],
                    'val/slot_acc': val_metrics['slot_acc'],
                    'val/exact_match': val_metrics['exact_match'],
                    'lr': scheduler.get_last_lr()[0],
                })
            
            results.append({
                'epoch': epoch,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()},
            })
    
    # Final test evaluation
    print(f"\n{'='*60}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*60}")
    
    # Load best model
    best_ckpt = torch.load(os.path.join(args.output, 'best_model.pt'), map_location=device)
    model.load_state_dict(best_ckpt['model_state_dict'])
    
    test_metrics = compute_metrics(model, test_loader, device)
    
    print(f"\nTest Results (Best Model from Epoch {best_ckpt['epoch']}):")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Slot Accuracy: {test_metrics['slot_acc']:.2f}%")
    print(f"  Exact Match: {test_metrics['exact_match']:.2f}%")
    
    # Rollout evaluation
    if args.eval_rollouts and goal_loader:
        rollout_metrics = evaluate_rollouts(
            model, goal_loader, device,
            max_steps=args.max_rollout_steps,
        )
        print(f"\nRollout Results:")
        print(f"  Plan Success Rate: {rollout_metrics['plan_success_rate']:.2f}%")
        print(f"  Avg Steps to Goal: {rollout_metrics['avg_steps_to_goal']:.2f}")
        if rollout_metrics['success_by_length']:
            print(f"  Success by Plan Length:")
            for length, rate in list(rollout_metrics['success_by_length'].items())[:10]:
                print(f"    Length {length}: {rate:.1f}%")
        
        test_metrics.update(rollout_metrics)
    
    # Save final results
    final_results = {
        'test': test_metrics,
        'best_val_slot_acc': best_val_slot_acc,
        'training_history': results,
        'args': vars(args),
    }
    
    with open(os.path.join(args.output, 'results.json'), 'w') as f:
        json.dump(final_results, f, indent=2, default=float)
    
    print(f"\nResults saved to {args.output}")
    print(f"{'='*60}")
    
    if args.wandb:
        import wandb
        wandb.log({
            'test/loss': test_metrics['loss'],
            'test/slot_acc': test_metrics['slot_acc'],
            'test/exact_match': test_metrics['exact_match'],
        })
        wandb.finish()


if __name__ == '__main__':
    main()

