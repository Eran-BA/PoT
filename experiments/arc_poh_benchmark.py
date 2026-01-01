#!/usr/bin/env python3
"""
PoT ARC-2 Benchmark - Abstract Reasoning Solver
=================================================

Train an abstract reasoning solver on ARC-2 dataset using PoT architecture.

Based on: https://github.com/sapientinc/HRM

Task: Given a 30x30 grid with ARC puzzle input, output the solution.
Input:  900 tokens (flattened 30x30), vocab 0-11 (PAD, EOS, digits 0-9)
Output: 900 tokens (complete solution)

Usage:
    # Download dataset and train hybrid model
    python experiments/arc_poh_benchmark.py --download --model hybrid
    
    # Train with custom settings
    python experiments/arc_poh_benchmark.py --model hybrid --H-cycles 3 --L-cycles 12
    
    # Evaluate checkpoint
    python experiments/arc_poh_benchmark.py --eval-only --checkpoint path/to/model.pt

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
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm


# ========== Distributed Training Utilities ==========

def is_distributed() -> bool:
    return 'RANK' in os.environ and 'WORLD_SIZE' in os.environ

def get_rank() -> int:
    return int(os.environ.get('RANK', 0))

def get_local_rank() -> int:
    return int(os.environ.get('LOCAL_RANK', 0))

def get_world_size() -> int:
    return int(os.environ.get('WORLD_SIZE', 1))

def is_main_process() -> bool:
    return get_rank() == 0

def setup_distributed():
    if is_distributed():
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(get_local_rank())

def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()

def print_rank0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


# Import from refactored modules
from src.data.arc import ARCDataset, download_arc_dataset, ARC_VOCAB_SIZE, ARC_SEQ_LEN
from src.pot.models.arc_solver import HybridPoHARCSolver, BaselineARCSolver
from torch.optim.optimizer import Optimizer

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
                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)
                p.add_(torch.sign(p.grad), alpha=-lr)
        
        return loss


def train_epoch(model, dataloader, optimizer, puzzle_optimizer, device, epoch, 
                use_poh=True, scheduler=None, puzzle_scheduler=None, grad_clip=1.0):
    """Train for one epoch."""
    model.train()
    base_model = model.module if hasattr(model, 'module') else model
    
    total_loss = 0
    correct_cells = 0
    total_cells = 0
    correct_grids = 0
    total_grids = 0
    total_steps = 0
    
    # Ignore PAD token (0) in loss calculation
    pad_id = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not is_main_process())
    for batch in pbar:
        inp = batch['input'].to(device)
        label = batch['label'].to(device)
        puzzle_ids = batch['puzzle_id'].to(device)
        
        optimizer.zero_grad()
        if puzzle_optimizer:
            puzzle_optimizer.zero_grad()
        
        model_out = model(inp, puzzle_ids)
        if len(model_out) == 5:
            logits, q_halt, q_continue, steps, target_q_continue = model_out
        else:
            logits, q_halt, q_continue, steps = model_out
            target_q_continue = None
        
        # Cross entropy loss (ignore PAD tokens)
        lm_loss = F.cross_entropy(
            logits.view(-1, base_model.vocab_size),
            label.view(-1),
            ignore_index=pad_id,
        )
        
        # Q-halt loss
        if use_poh and q_halt is not None:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                # Mask out PAD positions for accuracy
                mask = label != pad_id
                is_correct = ((preds == label) | ~mask).all(dim=1).float()
            
            q_halt_loss = F.binary_cross_entropy_with_logits(q_halt, is_correct)
            loss = lm_loss + 0.5 * q_halt_loss
            
            if target_q_continue is not None:
                q_continue_loss = F.mse_loss(torch.sigmoid(q_continue), target_q_continue)
                loss = loss + 0.5 * q_continue_loss
        else:
            loss = lm_loss
        
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if puzzle_optimizer:
            puzzle_optimizer.step()
        
        if scheduler:
            scheduler.step()
        if puzzle_scheduler:
            puzzle_scheduler.step()
        
        # Metrics (only on non-PAD cells)
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        mask = label != pad_id
        correct_cells += ((preds == label) & mask).sum().item()
        total_cells += mask.sum().item()
        
        # Grid accuracy: all non-PAD cells correct
        grid_correct = ((preds == label) | ~mask).all(dim=1)
        correct_grids += grid_correct.sum().item()
        total_grids += inp.size(0)
        
        total_steps += steps if isinstance(steps, int) else steps.float().mean().item()
        
        pbar.set_postfix({
            'loss': loss.item(),
            'cell': f"{100*correct_cells/max(1,total_cells):.1f}%",
            'grid': f"{100*correct_grids/max(1,total_grids):.1f}%",
        })
    
    return {
        'loss': total_loss / len(dataloader),
        'cell_acc': 100 * correct_cells / max(1, total_cells),
        'grid_acc': 100 * correct_grids / max(1, total_grids),
        'avg_steps': total_steps / len(dataloader),
    }


@torch.no_grad()
def evaluate(model, dataloader, device, use_poh=True):
    """Evaluate model."""
    model.eval()
    base_model = model.module if hasattr(model, 'module') else model
    
    total_loss = 0
    correct_cells = 0
    total_cells = 0
    correct_grids = 0
    total_grids = 0
    
    pad_id = 0
    
    for batch in tqdm(dataloader, desc="Evaluating", disable=not is_main_process()):
        inp = batch['input'].to(device)
        label = batch['label'].to(device)
        puzzle_ids = batch['puzzle_id'].to(device)
        
        model_out = model(inp, puzzle_ids)
        logits = model_out[0]
        
        loss = F.cross_entropy(
            logits.view(-1, base_model.vocab_size),
            label.view(-1),
            ignore_index=pad_id,
        )
        
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        mask = label != pad_id
        correct_cells += ((preds == label) & mask).sum().item()
        total_cells += mask.sum().item()
        
        grid_correct = ((preds == label) | ~mask).all(dim=1)
        correct_grids += grid_correct.sum().item()
        total_grids += inp.size(0)
    
    return {
        'loss': total_loss / len(dataloader),
        'cell_acc': 100 * correct_cells / max(1, total_cells),
        'grid_acc': 100 * correct_grids / max(1, total_grids),
    }


def main():
    parser = argparse.ArgumentParser(description='PoT ARC-2 Benchmark')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='data/arc-2',
                        help='Path to ARC dataset')
    parser.add_argument('--download', action='store_true',
                        help='Download and build dataset')
    parser.add_argument('--version', type=str, default='arc-2',
                        choices=['arc-1', 'arc-2'], help='ARC version')
    
    # Model
    parser.add_argument('--model', choices=['baseline', 'hybrid'], default='hybrid')
    parser.add_argument('--d-model', type=int, default=512)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--n-layers', type=int, default=2, help='Layers for baseline model')
    parser.add_argument('--d-ff', type=int, default=2048)
    parser.add_argument('--R', type=int, default=8, help='Refinement iterations (baseline)')
    parser.add_argument('--T', type=int, default=4, help='HRM outer period')
    parser.add_argument('--max-halt', type=int, default=16, help='Max halting steps')
    parser.add_argument('--dropout', type=float, default=0.0)
    
    # Hybrid model args
    parser.add_argument('--H-cycles', type=int, default=2,
                        help='Hybrid H_level outer cycles (fixed per ACT step)')
    parser.add_argument('--L-cycles', type=int, default=8,
                        help='Hybrid L_level inner cycles (fixed per H_cycle)')
    parser.add_argument('--H-layers', type=int, default=2, help='Layers in H_level module')
    parser.add_argument('--L-layers', type=int, default=2, help='Layers in L_level module')
    parser.add_argument('--controller', type=str, default='gru',
                        choices=['gru', 'lstm', 'xlstm', 'mingru', 'transformer', 
                                 'pot_transformer', 'swin', 'mamba', 'diffusion'],
                        help='Controller type for depth routing')
    parser.add_argument('--d-ctrl', type=int, default=None,
                        help='Controller hidden dimension (default: d_model // 4)')
    parser.add_argument('--max-depth', type=int, default=32,
                        help='Maximum depth steps for controller (used by diffusion, swin)')
    parser.add_argument('--optimize-mamba', action='store_true',
                        help='Enable optimized Mamba inference with torch.compile')
    parser.add_argument('--hrm-grad-style', action='store_true',
                        help='Use HRM-style gradients (only last L+H call)')
    
    # Position embeddings and Flash Attention
    parser.add_argument('--use-rope', action='store_true', default=True,
                        help='Use RoPE (Rotary Position Embeddings) like HRM (default: True)')
    parser.add_argument('--no-rope', action='store_true',
                        help='Disable RoPE, use sinusoidal position embeddings instead')
    parser.add_argument('--use-flash-attn', action='store_true', default=True,
                        help='Use Flash Attention when available for 3-5x speedup (default: True)')
    parser.add_argument('--no-flash-attn', action='store_true',
                        help='Disable Flash Attention, use standard PyTorch attention')
    
    # Feature injection modes
    parser.add_argument('--injection-mode', type=str, default='none',
                        choices=['none', 'broadcast', 'film', 'depth_token', 'cross_attn', 'alpha_gated'],
                        help='Feature injection mode for controller knowledge into tokens')
    parser.add_argument('--injection-memory-size', type=int, default=16,
                        help='Memory bank size for cross_attn injection mode')
    parser.add_argument('--injection-n-heads', type=int, default=4,
                        help='Number of attention heads for cross_attn injection mode')
    parser.add_argument('--alpha-aggregation', type=str, default='mean',
                        choices=['mean', 'max', 'entropy'],
                        help='Alpha aggregation mode for alpha_gated injection')
    parser.add_argument('--no-learned-gate', action='store_true',
                        help='Disable learned gate in alpha_gated injection')
    
    # ACT (Adaptive Computation Time) - like HRM's adaptive outer steps
    parser.add_argument('--halt-max-steps', type=int, default=1,
                        help='Max ACT outer steps (1=no ACT, >1=ACT enabled). HRM uses 8 or 16.')
    parser.add_argument('--halt-exploration-prob', type=float, default=0.1,
                        help='Exploration probability for Q-learning halting')
    parser.add_argument('--async-batch', action='store_true',
                        help='Use HRM-style async batching: halted samples are immediately replaced')
    
    # Training (matching Sudoku defaults)
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (smaller than Sudoku due to seq_len=900)')
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
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping max norm (0 to disable)')
    parser.add_argument('--eval-interval', type=int, default=100)
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--debug-interval', type=int, default=10, help='Debug every N epochs')
    
    # W&B
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--project', type=str, default='arc-poh')
    parser.add_argument('--run-name', type=str, default=None)
    
    # Output
    parser.add_argument('--output', type=str, default='experiments/results/arc_poh')
    parser.add_argument('--seed', type=int, default=42)
    
    # Eval mode
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    # Setup
    setup_distributed()
    distributed = is_distributed()
    local_rank = get_local_rank()
    world_size = get_world_size()
    
    if distributed:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    torch.manual_seed(args.seed + get_rank())
    np.random.seed(args.seed + get_rank())
    
    print_rank0(f"Device: {device}")
    print_rank0(f"ARC Version: {args.version.upper()}")
    if distributed:
        print_rank0(f"Distributed training: {world_size} GPUs")
    
    # Download dataset if needed
    if args.download:
        if is_main_process():
            download_arc_dataset(args.data_dir, version=args.version)
        if distributed:
            dist.barrier()
    
    # Load data
    train_dataset = ARCDataset(args.data_dir, 'train')
    test_dataset = ARCDataset(args.data_dir, 'test')
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if distributed else None
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, sampler=test_sampler,
        num_workers=4, pin_memory=True,
    )
    
    # Build model
    use_poh = args.model == 'hybrid'
    
    if args.model == 'hybrid':
        # Build controller kwargs
        controller_kwargs = {}
        if args.d_ctrl is not None:
            controller_kwargs['d_ctrl'] = args.d_ctrl
        if args.controller in ('diffusion', 'swin'):
            controller_kwargs['max_depth'] = args.max_depth
        if args.controller == 'mamba' and args.optimize_mamba:
            controller_kwargs['use_fast_path'] = True
        
        # Build injection kwargs
        injection_kwargs = None
        if args.injection_mode == 'cross_attn':
            injection_kwargs = {
                'memory_size': args.injection_memory_size,
                'n_heads': args.injection_n_heads,
            }
        elif args.injection_mode == 'alpha_gated':
            injection_kwargs = {
                'alpha_aggregation': args.alpha_aggregation,
                'use_learned_gate': not args.no_learned_gate,
            }
        
        # Resolve --no-rope and --no-flash-attn flags
        use_rope = args.use_rope and not args.no_rope
        use_flash_attn = args.use_flash_attn and not args.no_flash_attn
        
        model = HybridPoHARCSolver(
            d_model=args.d_model,
            n_heads=args.n_heads,
            H_layers=args.H_layers,
            L_layers=args.L_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            H_cycles=args.H_cycles,
            L_cycles=args.L_cycles,
            controller_type=args.controller,
            controller_kwargs=controller_kwargs if controller_kwargs else None,
            halt_max_steps=args.halt_max_steps,
            halt_exploration_prob=args.halt_exploration_prob,
            hrm_grad_style=args.hrm_grad_style,
            injection_mode=args.injection_mode,
            injection_kwargs=injection_kwargs,
            use_rope=use_rope,
            use_flash_attn=use_flash_attn,
        ).to(device)
        print_rank0(f"Hybrid model: H_cycles={args.H_cycles}, L_cycles={args.L_cycles}")
        print_rank0(f"H_layers={args.H_layers}, L_layers={args.L_layers}, dropout={args.dropout}")
        print_rank0(f"Controller: {args.controller}")
        print_rank0(f"Injection mode: {args.injection_mode}")
        if injection_kwargs:
            print_rank0(f"  Injection kwargs: {injection_kwargs}")
        if args.controller == 'mamba' and args.optimize_mamba:
            print_rank0(f"  Mamba optimization: ENABLED (torch.compile)")
        print_rank0(f"Position encoding: {'RoPE' if use_rope else 'Sinusoidal'}")
        print_rank0(f"Flash Attention: {'ENABLED' if model.L_level.attn_layers[0].use_flash_attn else 'DISABLED'}")
        print_rank0(f"Gradient style: {'HRM (last L+H only)' if args.hrm_grad_style else 'Full (last H_cycle)'}")
        if args.halt_max_steps > 1:
            print_rank0(f"ACT enabled: halt_max_steps={args.halt_max_steps}, exploration={args.halt_exploration_prob}")
            if args.async_batch:
                print_rank0(f"Async batching enabled (HRM-style)")
        else:
            print_rank0(f"ACT disabled (halt_max_steps=1)")
    else:
        model = BaselineARCSolver(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=6,
            d_ff=args.d_ff,
            dropout=args.dropout,
        ).to(device)
    
    if distributed:
        model = DDP(model, device_ids=[local_rank])
    
    param_count = sum(p.numel() for p in model.parameters())
    print_rank0(f"\nModel: {args.model.upper()}")
    print_rank0(f"Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
    
    # Load checkpoint if specified
    if args.checkpoint:
        print_rank0(f"\nLoading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint['model_state_dict']
        if distributed:
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
    
    # Eval-only mode
    if args.eval_only:
        print_rank0(f"\n{'='*60}\nEVALUATION MODE\n{'='*60}")
        test_metrics = evaluate(model, test_loader, device, use_poh=use_poh)
        print_rank0(f"\nTest Results:")
        print_rank0(f"  Loss: {test_metrics['loss']:.4f}")
        print_rank0(f"  Cell Accuracy: {test_metrics['cell_acc']:.2f}%")
        print_rank0(f"  Grid Accuracy: {test_metrics['grid_acc']:.2f}%")
        cleanup_distributed()
        return
    
    # Optimizers
    base_model = model.module if distributed else model
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
    
    # LR scheduler
    if distributed:
        steps_per_epoch = (len(train_dataset) + args.batch_size - 1) // args.batch_size
    else:
        steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    min_ratio = args.lr_min_ratio
    
    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step) / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    puzzle_scheduler = torch.optim.lr_scheduler.LambdaLR(puzzle_optimizer, lr_lambda) if puzzle_optimizer else None
    
    # Resume training
    start_epoch = 1
    best_grid_acc = 0
    results = []
    
    if args.resume:
        print_rank0(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if distributed:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if puzzle_optimizer and 'puzzle_optimizer_state_dict' in checkpoint:
            puzzle_optimizer.load_state_dict(checkpoint['puzzle_optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_grid_acc = checkpoint.get('test_grid_acc', 0)
        print_rank0(f"  Resumed from epoch {start_epoch - 1}, best_grid_acc={best_grid_acc:.2f}%")
    
    # W&B logging
    if args.wandb and is_main_process():
        import wandb
        run_name = args.run_name or f"{args.model}_{args.controller}_{datetime.now():%Y%m%d_%H%M%S}"
        wandb.init(project=args.project, name=run_name, config=vars(args))
        wandb.watch(model, log="gradients", log_freq=100)
        print_rank0(f"W&B logging enabled: {args.project}/{run_name}")
    
    print_rank0(f"\n{'='*60}")
    print_rank0(f"Training {args.model.upper()} ARC Solver")
    print_rank0(f"{'='*60}")
    print_rank0(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    if distributed:
        print_rank0(f"Distributed: {world_size} GPUs, effective batch size: {args.batch_size * world_size}")
    print_rank0(f"LR: {args.lr}, Weight decay: {args.weight_decay}")
    print_rank0(f"Warmup: {args.warmup_steps} steps, LR min ratio: {args.lr_min_ratio}")
    print_rank0(f"Total steps: {total_steps}")
    
    if is_main_process():
        os.makedirs(args.output, exist_ok=True)
    
    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        train_metrics = train_epoch(
            model, train_loader, optimizer, puzzle_optimizer, device, epoch,
            use_poh=use_poh, scheduler=scheduler, puzzle_scheduler=puzzle_scheduler,
            grad_clip=args.grad_clip,
        )
        
        train_dataset.on_epoch_end()
        
        if epoch % args.eval_interval == 0 or epoch == 1:
            test_metrics = evaluate(model, test_loader, device, use_poh=use_poh)
            
            print_rank0(f"\nEpoch {epoch}/{args.epochs}")
            print_rank0(f"  Train: Loss={train_metrics['loss']:.4f}, "
                        f"Cell={train_metrics['cell_acc']:.2f}%, Grid={train_metrics['grid_acc']:.2f}%")
            print_rank0(f"  Test:  Loss={test_metrics['loss']:.4f}, "
                        f"Cell={test_metrics['cell_acc']:.2f}%, Grid={test_metrics['grid_acc']:.2f}%")
            
            results.append({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_cell_acc': train_metrics['cell_acc'],
                'train_grid_acc': train_metrics['grid_acc'],
                'test_loss': test_metrics['loss'],
                'test_cell_acc': test_metrics['cell_acc'],
                'test_grid_acc': test_metrics['grid_acc'],
            })
            
            if args.wandb and is_main_process():
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_metrics['loss'],
                    "train/cell_acc": train_metrics['cell_acc'],
                    "train/grid_acc": train_metrics['grid_acc'],
                    "test/loss": test_metrics['loss'],
                    "test/cell_acc": test_metrics['cell_acc'],
                    "test/grid_acc": test_metrics['grid_acc'],
                    "lr": scheduler.get_last_lr()[0],
                })
            
            if test_metrics['grid_acc'] > best_grid_acc:
                best_grid_acc = test_metrics['grid_acc']
                if is_main_process():
                    state_dict = base_model.state_dict()
                    checkpoint_path = os.path.join(args.output, f'{args.model}_best.pt')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'puzzle_optimizer_state_dict': puzzle_optimizer.state_dict() if puzzle_optimizer else None,
                        'test_grid_acc': best_grid_acc,
                        'config': vars(args),
                    }, checkpoint_path)
                    
                    if args.wandb:
                        import wandb
                        artifact = wandb.Artifact(f"{args.model}-{args.controller}-best", type="model")
                        artifact.add_file(checkpoint_path)
                        wandb.log_artifact(artifact)
                print_rank0(f"  ✓ New best: {best_grid_acc:.2f}%")
    
    print_rank0(f"\n{'='*60}")
    print_rank0("FINAL RESULTS")
    print_rank0(f"{'='*60}")
    print_rank0(f"Best Test Grid Accuracy: {best_grid_acc:.2f}%")
    
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
    
    if args.wandb and is_main_process():
        import wandb
        wandb.finish()
    
    cleanup_distributed()


if __name__ == '__main__':
    main()

