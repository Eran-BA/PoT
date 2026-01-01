#!/usr/bin/env python3
"""
Sokoban PoT Benchmark.

Full benchmark for Sokoban puzzle solving with PoT iterative refinement.

Training modes:
1. Heuristic pretraining: Cross-entropy on heuristic-guided pseudo-labels
2. PPO fine-tuning: Reinforcement learning with reward shaping
3. Combined: Pretrain with heuristic, then fine-tune with PPO

Ablations:
- Refinement steps R: {1, 2, 4, 8}
- Augmentations: on/off
- Model: PoT vs Baseline

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
from src.pot.models.sokoban_solver import (
    PoTSokobanSolver,
    BaselineSokobanSolver,
    SokobanActorCritic,
)
from src.training.sokoban_heuristic import (
    HeuristicTrainingConfig,
    train as train_heuristic,
    evaluate_solve_rate,
    collate_fn,
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
    
    # Model
    model_type: str = "pot"  # "pot" or "baseline"
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.1
    R: int = 4  # PoT refinement steps
    conv_layers: int = 3
    conv_filters: int = 64
    
    # Training mode
    mode: str = "heuristic"  # "heuristic", "ppo", or "combined"
    
    # Heuristic training
    heuristic_epochs: int = 30
    steps_per_level: int = 500
    augment: bool = True
    
    # PPO training
    ppo_timesteps: int = 500_000
    ppo_n_envs: int = 8
    ppo_n_steps: int = 128
    
    # Common
    batch_size: int = 64
    learning_rate: float = 1e-4
    eval_interval: int = 5
    
    # Evaluation
    max_eval_steps: int = 200
    eval_episodes: int = 100
    
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
    
    # Model
    parser.add_argument('--model-type', type=str, default='pot',
                        choices=['pot', 'baseline'])
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--d-ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--R', type=int, default=4,
                        help='Number of PoT refinement steps')
    parser.add_argument('--conv-layers', type=int, default=3)
    parser.add_argument('--conv-filters', type=int, default=64)
    
    # Training mode
    parser.add_argument('--mode', type=str, default='heuristic',
                        choices=['heuristic', 'ppo', 'combined'])
    
    # Heuristic training
    parser.add_argument('--heuristic-epochs', type=int, default=30)
    parser.add_argument('--steps-per-level', type=int, default=500)
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable geometric augmentations')
    
    # PPO training
    parser.add_argument('--ppo-timesteps', type=int, default=500_000)
    parser.add_argument('--ppo-n-envs', type=int, default=8)
    parser.add_argument('--ppo-n-steps', type=int, default=128)
    
    # Common
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--eval-interval', type=int, default=5)
    
    # Evaluation
    parser.add_argument('--max-eval-steps', type=int, default=200)
    parser.add_argument('--eval-episodes', type=int, default=100)
    
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
        conv_layers=args.conv_layers,
        conv_filters=args.conv_filters,
        mode=args.mode,
        heuristic_epochs=args.heuristic_epochs,
        steps_per_level=args.steps_per_level,
        augment=not args.no_augment,
        ppo_timesteps=args.ppo_timesteps,
        ppo_n_envs=args.ppo_n_envs,
        ppo_n_steps=args.ppo_n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_interval=args.eval_interval,
        max_eval_steps=args.max_eval_steps,
        eval_episodes=args.eval_episodes,
        output_dir=args.output_dir,
        seed=args.seed,
    )


# =============================================================================
# Model Creation
# =============================================================================

def create_model(config: BenchmarkConfig, device: torch.device) -> nn.Module:
    """Create Sokoban solver model based on config."""
    if config.model_type == "pot":
        model = PoTSokobanSolver(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            R=config.R,
            conv_layers=config.conv_layers,
            conv_filters=config.conv_filters,
        )
    else:
        model = BaselineSokobanSolver(
            n_filters=config.conv_filters,
            n_layers=config.conv_layers + 1,
            d_hidden=config.d_model,
            dropout=config.dropout,
        )
    
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
    
    # Training
    if config.mode in ['heuristic', 'combined']:
        print("\n" + "=" * 60)
        print("Phase 1: Heuristic Pretraining")
        print("=" * 60)
        
        heuristic_config = HeuristicTrainingConfig(
            data_dir=config.data_dir,
            difficulty=config.difficulty,
            steps_per_level=config.steps_per_level,
            augment=config.augment,
            model_type=config.model_type,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            R=config.R,
            epochs=config.heuristic_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            eval_interval=config.eval_interval,
            save_dir=str(output_dir / 'heuristic'),
        )
        
        heuristic_results = train_heuristic(heuristic_config, device)
        
        results['heuristic'] = {
            'best_val_acc': heuristic_results['best_val_acc'],
            'history': {
                'train_loss': heuristic_results['history']['train_loss'][-5:],
                'train_acc': heuristic_results['history']['train_acc'][-5:],
            }
        }
        
        # Use the trained model
        model = heuristic_results['model']
    
    if config.mode in ['ppo', 'combined']:
        print("\n" + "=" * 60)
        print("Phase 2: PPO Fine-tuning" if config.mode == 'combined' else "PPO Training")
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
    print(f"Model: {config.model_type}, R={config.R}")
    print(f"Mode: {config.mode}")
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

