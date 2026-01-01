#!/usr/bin/env python3
"""
Sokoban PoT Benchmark.

Full benchmark for Sokoban puzzle solving with PoT iterative refinement.

Training: Pure PPO (reinforcement learning with reward shaping)

Ablations:
- Refinement steps R: {1, 2, 4, 8}
- Augmentations: on/off
- Model: PoT vs Baseline vs HybridPoT

Metrics:
- Solve Rate @N (N=50, 100, 200)
- Deadlock Rate
- Median Steps-to-Solve
- Legal Action Rate

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.sokoban import (
    SokobanDataset,
    SokobanStateDataset,
    load_boxoban_levels,
    download_boxoban_dataset,
    board_to_onehot,
)
from src.data.sokoban_rules import (
    legal_actions,
    step,
    is_solved,
    is_deadlock,
    get_legal_action_list,
)
from src.pot.models.sokoban_solver import (
    PoTSokobanSolver,
    BaselineSokobanSolver,
    SokobanActorCritic,
)
from src.training.sokoban_ppo import (
    PPOConfig,
    train_ppo,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Full benchmark configuration."""
    # Data
    data_dir: str = "data"
    difficulty: str = "medium"
    download: bool = True
    
    # Model type
    model_type: str = "pot"  # "pot", "hybrid", or "baseline"
    
    # Architecture
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.1
    conv_layers: int = 3
    conv_filters: int = 64
    
    # SimplePoT (model_type="pot")
    R: int = 4  # PoT refinement steps
    
    # HybridPoT (model_type="hybrid")
    H_cycles: int = 2
    L_cycles: int = 6
    H_layers: int = 2
    L_layers: int = 2
    T: int = 4
    halt_max_steps: int = 2
    
    # Controller
    controller_type: str = "transformer"
    d_ctrl: Optional[int] = None
    max_depth: int = 32
    hrm_grad_style: bool = False
    halt_exploration_prob: float = 0.1
    allow_early_halt_eval: bool = False
    
    # Feature injection
    injection_mode: str = "none"
    injection_memory_size: int = 16
    injection_n_heads: int = 4
    alpha_aggregation: str = "mean"
    
    # Training mode
    mode: str = "ppo"  # Pure PPO training (no domain heuristics)
    
    # PPO training
    ppo_timesteps: int = 500_000
    ppo_n_envs: int = 8
    ppo_n_steps: int = 128
    
    # Common
    batch_size: int = 64
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    lr_min_ratio: float = 0.0
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    eval_interval: int = 5
    
    # Evaluation
    max_eval_steps: int = 200
    eval_episodes: int = 100
    
    # Logging
    wandb: bool = False
    project: str = "sokoban-pot"
    run_name: Optional[str] = None
    
    # Output
    output_dir: str = "experiments/results/sokoban_benchmark"
    seed: int = 42
    
    # Device
    device: str = "auto"


def parse_args() -> BenchmarkConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Sokoban PoT Benchmark')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--difficulty', type=str, default='medium',
                        choices=['medium', 'hard', 'unfiltered'])
    parser.add_argument('--download', action='store_true',
                        help='Download Boxoban dataset if not present')
    
    # Model type
    parser.add_argument('--model-type', type=str, default='pot',
                        choices=['pot', 'hybrid', 'baseline'],
                        help='pot=SimplePoT, hybrid=HybridPoT with H/L cycles, baseline=CNN')
    
    # Architecture
    parser.add_argument('--d-model', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--n-heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--n-layers', type=int, default=2,
                        help='Number of transformer layers (for simple model)')
    parser.add_argument('--d-ff', type=int, default=512,
                        help='FFN dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--conv-layers', type=int, default=3,
                        help='Convolutional layers for feature extraction')
    parser.add_argument('--conv-filters', type=int, default=64,
                        help='Convolutional filters')
    
    # SimplePoT (model_type="pot")
    parser.add_argument('--R', type=int, default=4,
                        help='Number of PoT refinement steps (for simple model)')
    
    # HybridPoT (model_type="hybrid")
    parser.add_argument('--H-cycles', type=int, default=2,
                        help='H_level (slow) cycles per ACT step')
    parser.add_argument('--L-cycles', type=int, default=6,
                        help='L_level (fast) cycles per H_cycle')
    parser.add_argument('--H-layers', type=int, default=2,
                        help='Layers in H_level module')
    parser.add_argument('--L-layers', type=int, default=2,
                        help='Layers in L_level module')
    parser.add_argument('--T', type=int, default=4,
                        help='HRM period for pointer controller')
    parser.add_argument('--halt-max-steps', type=int, default=2,
                        help='Max halting steps for ACT')
    
    # Controller
    parser.add_argument('--controller-type', type=str, default='transformer',
                        choices=['transformer', 'pot_transformer', 'swin', 'diffusion', 'gru', 'lstm'],
                        help='Controller type for PoT')
    parser.add_argument('--d-ctrl', type=int, default=None,
                        help='Controller hidden dimension (default: d_model // 4)')
    parser.add_argument('--max-depth', type=int, default=32,
                        help='Maximum refinement depth for controller')
    parser.add_argument('--hrm-grad-style', action='store_true',
                        help='Use HRM-style gradients (only last L+H call)')
    parser.add_argument('--halt-exploration-prob', type=float, default=0.1,
                        help='Exploration probability for Q-learning halting')
    parser.add_argument('--allow-early-halt-eval', action='store_true',
                        help='Enable Q-learning based early halting during eval')
    
    # Feature injection
    parser.add_argument('--injection-mode', type=str, default='none',
                        choices=['none', 'broadcast', 'film', 'depth_token', 'cross_attn', 'alpha_gated'],
                        help='Feature injection mode for controller knowledge into tokens')
    parser.add_argument('--injection-memory-size', type=int, default=16,
                        help='Memory bank size for cross_attn injection mode')
    parser.add_argument('--injection-n-heads', type=int, default=4,
                        help='Number of attention heads for cross_attn injection mode')
    parser.add_argument('--alpha-aggregation', type=str, default='mean',
                        choices=['mean', 'max', 'last'],
                        help='How to aggregate alpha weights across layers')
    
    # Training mode (PPO only - pure RL without domain heuristics)
    parser.add_argument('--mode', type=str, default='ppo',
                        choices=['ppo'],
                        help='Training mode (only PPO supported)')
    
    # PPO training
    parser.add_argument('--ppo-timesteps', type=int, default=500_000)
    parser.add_argument('--ppo-n-envs', type=int, default=8)
    parser.add_argument('--ppo-n-steps', type=int, default=128)
    
    # Common training
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--warmup-steps', type=int, default=100,
                        help='LR warmup steps (HRM uses 2000)')
    parser.add_argument('--lr-min-ratio', type=float, default=0.0,
                        help='Minimum LR ratio for cosine annealing')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping max norm')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--eval-interval', type=int, default=5)
    
    # Evaluation
    parser.add_argument('--max-eval-steps', type=int, default=200)
    parser.add_argument('--eval-episodes', type=int, default=100)
    
    # Logging
    parser.add_argument('--wandb', action='store_true',
                        help='Enable W&B logging')
    parser.add_argument('--project', type=str, default='sokoban-pot',
                        help='W&B project name')
    parser.add_argument('--run-name', type=str, default=None,
                        help='W&B run name (auto-generated if not set)')
    
    # Output
    parser.add_argument('--output-dir', type=str,
                        default='experiments/results/sokoban_benchmark')
    parser.add_argument('--seed', type=int, default=42)
    
    # Ablation
    parser.add_argument('--ablate-R', action='store_true',
                        help='Run ablation over R={1,2,4,8}')
    
    args = parser.parse_args()
    
    return BenchmarkConfig(
        data_dir=args.data_dir,
        difficulty=args.difficulty,
        download=args.download,
        model_type=args.model_type,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        R=args.R,
        H_cycles=args.H_cycles,
        L_cycles=args.L_cycles,
        H_layers=args.H_layers,
        L_layers=args.L_layers,
        T=args.T,
        halt_max_steps=args.halt_max_steps,
        controller_type=args.controller_type,
        d_ctrl=args.d_ctrl,
        max_depth=args.max_depth,
        hrm_grad_style=args.hrm_grad_style,
        halt_exploration_prob=args.halt_exploration_prob,
        allow_early_halt_eval=args.allow_early_halt_eval,
        injection_mode=args.injection_mode,
        injection_memory_size=args.injection_memory_size,
        injection_n_heads=args.injection_n_heads,
        alpha_aggregation=args.alpha_aggregation,
        conv_layers=args.conv_layers,
        conv_filters=args.conv_filters,
        mode=args.mode,
        ppo_timesteps=args.ppo_timesteps,
        ppo_n_envs=args.ppo_n_envs,
        ppo_n_steps=args.ppo_n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_min_ratio=args.lr_min_ratio,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        eval_interval=args.eval_interval,
        max_eval_steps=args.max_eval_steps,
        eval_episodes=args.eval_episodes,
        wandb=args.wandb,
        project=args.project,
        run_name=args.run_name,
        output_dir=args.output_dir,
        seed=args.seed,
    )


# =============================================================================
# Model Creation
# =============================================================================

def create_model(config: BenchmarkConfig, device: torch.device) -> nn.Module:
    """Create Sokoban solver model based on config."""
    
    # Build controller kwargs
    controller_kwargs = {}
    if config.d_ctrl is not None:
        controller_kwargs['d_ctrl'] = config.d_ctrl
    controller_kwargs['max_depth'] = config.max_depth
    
    # Build injection kwargs
    injection_kwargs = {}
    if config.injection_mode != 'none':
        injection_kwargs = {
            'injection_mode': config.injection_mode,
            'memory_size': config.injection_memory_size,
            'n_heads': config.injection_n_heads,
            'alpha_aggregation': config.alpha_aggregation,
        }
    
    if config.model_type == "pot":
        # Simple PoT with R refinement steps
        model = PoTSokobanSolver(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            R=config.R,
            controller_type=config.controller_type,
            max_depth=config.max_depth,
            conv_layers=config.conv_layers,
            conv_filters=config.conv_filters,
        )
        print(f"\nModel: PoTSokobanSolver (Simple)")
        print(f"  R={config.R}, n_layers={config.n_layers}")
        
    elif config.model_type == "hybrid":
        # Full HybridPoTSokobanSolver (aligned with Sudoku's HybridPoHHRMSolver)
        from src.pot.models.sokoban_solver import HybridPoTSokobanSolver
        
        model = HybridPoTSokobanSolver(
            d_model=config.d_model,
            n_heads=config.n_heads,
            H_layers=config.H_layers,
            L_layers=config.L_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            H_cycles=config.H_cycles,
            L_cycles=config.L_cycles,
            T=config.T,
            conv_layers=config.conv_layers,
            conv_filters=config.conv_filters,
            controller_type=config.controller_type,
            controller_kwargs={'d_ctrl': config.d_ctrl, 'max_depth': config.max_depth},
            hrm_grad_style=config.hrm_grad_style,
            halt_max_steps=config.halt_max_steps,
            halt_exploration_prob=config.halt_exploration_prob,
            allow_early_halt_eval=config.allow_early_halt_eval,
            injection_mode=config.injection_mode,
            injection_kwargs={
                'memory_size': config.injection_memory_size,
                'n_heads': config.injection_n_heads,
                'alpha_aggregation': config.alpha_aggregation,
            } if config.injection_mode != 'none' else None,
        )
        print(f"\nModel: HybridPoTSokobanSolver")
        print(f"  H_cycles={config.H_cycles}, L_cycles={config.L_cycles}")
        print(f"  H_layers={config.H_layers}, L_layers={config.L_layers}")
        print(f"  T={config.T}, halt_max_steps={config.halt_max_steps}")
        print(f"  hrm_grad_style={config.hrm_grad_style}, injection_mode={config.injection_mode}")
        
    else:
        # Baseline CNN
        model = BaselineSokobanSolver(
            n_filters=config.conv_filters,
            n_layers=config.conv_layers + 1,
            d_hidden=config.d_model,
            dropout=config.dropout,
        )
        print(f"\nModel: BaselineSokobanSolver (CNN)")
    
    # Common info
    print(f"  d_model={config.d_model}, n_heads={config.n_heads}")
    print(f"  controller={config.controller_type}, max_depth={config.max_depth}")
    if config.injection_mode != 'none':
        print(f"  injection_mode={config.injection_mode}")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {num_params:,} ({num_params/1e6:.2f}M)")
    
    return model.to(device)


# =============================================================================
# Evaluation
# =============================================================================

def full_evaluation(
    model: nn.Module,
    test_levels: List[np.ndarray],
    device: torch.device,
    config: BenchmarkConfig,
) -> Dict[str, Any]:
    """
    Run full evaluation with multiple metrics.
    
    Returns:
        Dictionary with all evaluation metrics
    """
    results = {}
    
    # Solve rate at different step limits
    for max_steps in [50, 100, 200]:
        eval_stats = evaluate_solve_rate(
            model,
            test_levels[:config.eval_episodes],
            device,
            max_steps=max_steps,
            temperature=0.0,
            verbose=True,
        )
        
        results[f'solve_rate@{max_steps}'] = eval_stats['solve_rate']
        results[f'deadlock_rate@{max_steps}'] = eval_stats['deadlock_rate']
        
        if max_steps == 200:
            results['median_steps'] = eval_stats['median_steps']
            results['mean_steps'] = eval_stats['mean_steps']
            results['solved'] = eval_stats['solved']
            results['deadlocked'] = eval_stats['deadlocked']
            results['total'] = eval_stats['total']
    
    return results


# =============================================================================
# Main Benchmark
# =============================================================================

def run_benchmark(config: BenchmarkConfig) -> Dict[str, Any]:
    """
    Run full benchmark.
    
    Args:
        config: Benchmark configuration
    
    Returns:
        Dictionary with all results
    """
    # Setup
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    
    print(f"Using device: {device}")
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download data if needed
    if config.download:
        download_boxoban_dataset(config.data_dir)
    
    # Load levels
    print("Loading levels...")
    train_levels = load_boxoban_levels(config.data_dir, config.difficulty, 'train')
    val_levels = load_boxoban_levels(config.data_dir, config.difficulty, 'valid')
    
    # Try to load test split, fall back to using part of valid
    try:
        test_levels = load_boxoban_levels(config.data_dir, config.difficulty, 'test')
    except FileNotFoundError:
        print("No test split found, using validation levels for testing")
        test_levels = val_levels
    
    print(f"Train: {len(train_levels)}, Val: {len(val_levels)}, Test: {len(test_levels)}")
    
    # Create model
    model = create_model(config, device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    results = {
        'config': asdict(config),
        'num_params': num_params,
        'device': str(device),
    }
    
    start_time = time.time()
    
    # PPO Training
    print("\n" + "=" * 60)
    print("PPO Training")
    print("=" * 60)
    
    ppo_config = PPOConfig(
        max_episode_steps=config.max_eval_steps,
        n_envs=config.ppo_n_envs,
        n_steps=config.ppo_n_steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        total_timesteps=config.ppo_timesteps,
        eval_interval=config.ppo_timesteps // 10,
        eval_episodes=config.eval_episodes,
    )
    
    ppo_results = train_ppo(
        model,
        train_levels,
        val_levels,
        ppo_config,
        device,
        save_dir=str(output_dir / 'ppo'),
    )
    
    results['ppo'] = {
        'best_reward': ppo_results['best_reward'],
        'final_reward': ppo_results['history']['mean_reward'][-1] if ppo_results['history']['mean_reward'] else 0,
    }
    
    model = ppo_results['model']
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    eval_results = full_evaluation(model, test_levels, device, config)
    results['evaluation'] = eval_results
    
    # Timing
    total_time = time.time() - start_time
    results['total_time_seconds'] = total_time
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    if config.model_type == 'hybrid':
        print(f"Model: {config.model_type}, H_cycles={config.H_cycles}, L_cycles={config.L_cycles}")
    else:
        print(f"Model: {config.model_type}, R={config.R}")
    print(f"Mode: {config.mode}")
    print(f"Controller: {config.controller_type}, max_depth={config.max_depth}")
    print(f"Solve Rate @50:  {eval_results['solve_rate@50']:.2%}")
    print(f"Solve Rate @100: {eval_results['solve_rate@100']:.2%}")
    print(f"Solve Rate @200: {eval_results['solve_rate@200']:.2%}")
    print(f"Deadlock Rate:   {eval_results['deadlock_rate@200']:.2%}")
    print(f"Median Steps:    {eval_results['median_steps']:.1f}")
    print(f"Total Time:      {total_time / 60:.1f} min")
    
    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir / 'results.json'}")
    
    return results


def run_ablation(config: BenchmarkConfig) -> Dict[str, Any]:
    """
    Run ablation study over refinement steps R.
    """
    R_values = [1, 2, 4, 8]
    all_results = {}
    
    base_output_dir = Path(config.output_dir)
    
    for R in R_values:
        print(f"\n{'#' * 60}")
        print(f"ABLATION: R = {R}")
        print(f"{'#' * 60}")
        
        config.R = R
        config.output_dir = str(base_output_dir / f'R_{R}')
        
        results = run_benchmark(config)
        all_results[f'R={R}'] = results
    
    # Summary table
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)
    print(f"{'R':<5} {'Solve@50':<12} {'Solve@100':<12} {'Solve@200':<12} {'Median Steps':<12}")
    print("-" * 55)
    
    for R in R_values:
        r = all_results[f'R={R}']['evaluation']
        print(f"{R:<5} {r['solve_rate@50']:<12.2%} {r['solve_rate@100']:<12.2%} "
              f"{r['solve_rate@200']:<12.2%} {r['median_steps']:<12.1f}")
    
    # Save summary
    with open(base_output_dir / 'ablation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    return all_results


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Main entry point."""
    # Check for ablation flag
    if '--ablate-R' in sys.argv:
        sys.argv.remove('--ablate-R')
        config = parse_args()
        run_ablation(config)
    else:
        config = parse_args()
        run_benchmark(config)


if __name__ == '__main__':
    main()

