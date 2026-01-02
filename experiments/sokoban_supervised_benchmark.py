#!/usr/bin/env python3
"""
Sokoban Supervised Benchmark.

IDENTICAL training approach to Sudoku:
- Cross-entropy loss on action prediction
- PoT refinement architecture
- On-the-fly augmentation

Dataset: Xiaofeng77/sokoban from HuggingFace

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
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.sokoban_hf import SokobanHFDataset, download_sokoban_hf_dataset
from src.data.sokoban_generator import SokobanGeneratedDataset
from src.training.sokoban_supervised import train_supervised, evaluate


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Benchmark configuration - same structure as Sudoku."""
    
    # Model type
    model_type: str = "pot"  # "pot", "hybrid", or "baseline"
    
    # Architecture (same defaults as Sudoku)
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.1
    
    # PoT refinement
    R: int = 4  # Refinement steps (like Sudoku)
    
    # HybridPoT (like Sudoku)
    H_cycles: int = 2
    L_cycles: int = 6
    H_layers: int = 2
    L_layers: int = 2
    T: int = 4
    
    # Conv encoder
    conv_layers: int = 3
    conv_filters: int = 64
    
    # Controller
    controller_type: str = "transformer"
    max_depth: int = 32
    
    # Training (same as Sudoku)
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 100
    
    # Data
    augment: bool = True
    
    # Logging
    wandb: bool = False
    project: str = "sokoban-supervised"
    run_name: Optional[str] = None
    
    # Output
    output_dir: str = "experiments/results/sokoban_supervised"
    seed: int = 42
    
    # Device
    device: str = "auto"
    
    # Multi-difficulty evaluation
    eval_difficulties: str = "simple,complex"  # comma-separated
    eval_samples_per_difficulty: int = 200


def parse_args() -> BenchmarkConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Sokoban Supervised Benchmark')
    
    # Model type
    parser.add_argument('--model-type', type=str, default='pot',
                        choices=['pot', 'hybrid', 'baseline'])
    
    # Architecture
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--d-ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # PoT
    parser.add_argument('--R', type=int, default=4)
    
    # HybridPoT
    parser.add_argument('--H-cycles', type=int, default=2)
    parser.add_argument('--L-cycles', type=int, default=6)
    parser.add_argument('--H-layers', type=int, default=2)
    parser.add_argument('--L-layers', type=int, default=2)
    parser.add_argument('--T', type=int, default=4)
    
    # Conv
    parser.add_argument('--conv-layers', type=int, default=3)
    parser.add_argument('--conv-filters', type=int, default=64)
    
    # Controller
    parser.add_argument('--controller-type', type=str, default='transformer')
    parser.add_argument('--max-depth', type=int, default=32)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--warmup-steps', type=int, default=100)
    
    # Data
    parser.add_argument('--no-augment', action='store_true')
    
    # Multi-difficulty evaluation
    parser.add_argument('--eval-difficulties', type=str, default='simple,complex',
                        help='Comma-separated difficulties: simple,larger,two_boxes,complex')
    parser.add_argument('--eval-samples', type=int, default=200,
                        help='Samples per difficulty for evaluation')
    
    # Logging
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--project', type=str, default='sokoban-supervised')
    parser.add_argument('--run-name', type=str, default=None)
    
    # Output
    parser.add_argument('--output-dir', type=str,
                        default='experiments/results/sokoban_supervised')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    return BenchmarkConfig(
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
        conv_layers=args.conv_layers,
        conv_filters=args.conv_filters,
        controller_type=args.controller_type,
        max_depth=args.max_depth,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        augment=not args.no_augment,
        eval_difficulties=args.eval_difficulties,
        eval_samples_per_difficulty=args.eval_samples,
        wandb=args.wandb,
        project=args.project,
        run_name=args.run_name,
        output_dir=args.output_dir,
        seed=args.seed,
    )


# =============================================================================
# Model Creation
# =============================================================================

def create_model(config: BenchmarkConfig, board_shape: tuple, device: torch.device) -> nn.Module:
    """Create Sokoban solver model for supervised learning."""
    
    from src.pot.models.sokoban_solver import (
        PoTSokobanSolver,
        HybridPoTSokobanSolver,
        BaselineSokobanSolver,
    )
    
    H, W = board_shape
    
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
            controller_type=config.controller_type,
            max_depth=config.max_depth,
        )
        # Adjust for smaller board if needed (HF dataset is 6x6)
        if H != 10 or W != 10:
            model.seq_len = H * W
            model.pos_embed = nn.Parameter(torch.randn(1, H * W, config.d_model) * 0.02)
        
        print(f"\nModel: PoTSokobanSolver")
        print(f"  R={config.R}, n_layers={config.n_layers}")
        
    elif config.model_type == "hybrid":
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
        )
        print(f"\nModel: HybridPoTSokobanSolver")
        print(f"  H_cycles={config.H_cycles}, L_cycles={config.L_cycles}")
        
    else:
        model = BaselineSokobanSolver(
            n_filters=config.conv_filters,
            n_layers=config.conv_layers + 1,
            d_hidden=config.d_model,
            dropout=config.dropout,
        )
        print(f"\nModel: BaselineSokobanSolver (CNN)")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  d_model={config.d_model}, n_heads={config.n_heads}")
    print(f"  Board: {H}x{W}")
    print(f"  Total params: {num_params:,} ({num_params/1e6:.2f}M)")
    
    return model.to(device)


# =============================================================================
# Main Benchmark
# =============================================================================

def run_benchmark(config: BenchmarkConfig) -> Dict[str, Any]:
    """Run supervised Sokoban benchmark."""
    
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
    
    # Load dataset
    print("\nLoading Sokoban dataset from HuggingFace...")
    train_ds = SokobanHFDataset(split='train', augment=config.augment)
    test_ds = SokobanHFDataset(split='test', augment=False)
    
    # Split train into train/val (90/10)
    n_val = len(train_ds) // 10
    n_train = len(train_ds) - n_val
    
    # Simple split by indexing
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, len(train_ds)))
    
    train_subset = torch.utils.data.Subset(train_ds, train_indices)
    val_subset = torch.utils.data.Subset(train_ds, val_indices)
    
    print(f"Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_ds)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # Create model
    model = create_model(config, train_ds.board_shape, device)
    
    # Initialize W&B
    if config.wandb:
        import wandb
        wandb.init(
            project=config.project,
            name=config.run_name or f"sokoban-{config.model_type}-supervised",
            config=asdict(config),
        )
    
    # Train
    print("\n" + "=" * 60)
    print("SUPERVISED TRAINING (Identical to Sudoku)")
    print("=" * 60)
    
    start_time = time.time()
    
    results = train_supervised(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        grad_clip=config.grad_clip,
        warmup_steps=config.warmup_steps,
        use_pot=(config.model_type != "baseline"),
        save_dir=str(output_dir),
        wandb_log=config.wandb,
    )
    
    train_time = time.time() - start_time
    
    # Final test evaluation on HuggingFace test set
    print("\n" + "=" * 60)
    print("TEST EVALUATION (HuggingFace test set)")
    print("=" * 60)
    
    test_metrics = evaluate(model, test_loader, device)
    print(f"HF Test Accuracy: {test_metrics['accuracy']:.2%}")
    
    # Multi-difficulty evaluation
    print("\n" + "=" * 60)
    print("MULTI-DIFFICULTY EVALUATION")
    print("=" * 60)
    
    difficulty_results = {}
    difficulties = [d.strip() for d in config.eval_difficulties.split(',')]
    
    for difficulty in difficulties:
        print(f"\nEvaluating {difficulty}...")
        try:
            eval_ds = SokobanGeneratedDataset(
                difficulty=difficulty,
                n_samples=config.eval_samples_per_difficulty,
                seed=config.seed + 1000,  # Different seed for eval
                augment=False,
            )
            
            eval_loader = DataLoader(
                eval_ds,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
            )
            
            # Need to adjust model for different board sizes
            board_h, board_w = eval_ds.board_shape
            orig_seq_len = model.seq_len if hasattr(model, 'seq_len') else 36
            orig_pos_embed = model.pos_embed.data.clone() if hasattr(model, 'pos_embed') else None
            
            new_seq_len = board_h * board_w
            if new_seq_len != orig_seq_len and hasattr(model, 'pos_embed'):
                # Interpolate position embeddings for different board size
                model.seq_len = new_seq_len
                model.pos_embed = torch.nn.Parameter(
                    torch.randn(1, new_seq_len, model.d_model, device=device) * 0.02
                )
            
            eval_metrics = evaluate(model, eval_loader, device)
            
            # Restore original
            if orig_pos_embed is not None and new_seq_len != orig_seq_len:
                model.seq_len = orig_seq_len
                model.pos_embed = torch.nn.Parameter(orig_pos_embed)
            
            difficulty_results[difficulty] = {
                'accuracy': eval_metrics['accuracy'],
                'loss': eval_metrics['loss'],
                'board_shape': eval_ds.board_shape,
            }
            
            print(f"  {difficulty}: {eval_metrics['accuracy']:.2%} accuracy")
            
        except Exception as e:
            print(f"  {difficulty}: FAILED - {e}")
            difficulty_results[difficulty] = {'error': str(e)}
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Model: {config.model_type}, R={config.R}")
    print(f"Best Val Accuracy: {results['best_val_acc']:.2%}")
    print(f"HF Test Accuracy: {test_metrics['accuracy']:.2%}")
    print(f"\nMulti-difficulty results:")
    for diff, metrics in difficulty_results.items():
        if 'accuracy' in metrics:
            print(f"  {diff}: {metrics['accuracy']:.2%}")
        else:
            print(f"  {diff}: ERROR")
    print(f"\nTraining Time: {train_time / 60:.1f} min")
    
    # Log to W&B
    if config.wandb:
        import wandb
        log_dict = {
            'test/accuracy': test_metrics['accuracy'],
            'test/loss': test_metrics['loss'],
            'best_val_accuracy': results['best_val_acc'],
            'training_time_min': train_time / 60,
        }
        for diff, metrics in difficulty_results.items():
            if 'accuracy' in metrics:
                log_dict[f'eval/{diff}_accuracy'] = metrics['accuracy']
        wandb.log(log_dict)
        wandb.finish()
    
    # Save results
    final_results = {
        'config': asdict(config),
        'best_val_acc': results['best_val_acc'],
        'test_accuracy': test_metrics['accuracy'],
        'test_loss': test_metrics['loss'],
        'difficulty_results': difficulty_results,
        'training_time_seconds': train_time,
        'history': results['history'],
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'results.json'}")
    
    # Comparison with known baselines
    print("\n" + "=" * 60)
    print("COMPARISON WITH KNOWN BASELINES")
    print("=" * 60)
    print("""
    Reference baselines from literature:

    | Method                    | Simple (6x6,1) | Complex (10x10,2) | Notes                    |
    |---------------------------|----------------|-------------------|--------------------------|
    | SFT (paper)               | ~50%           | ~15%              | Supervised fine-tuning   |
    | GPT-4 + LangGraph*        | ~varies        | ~varies           | Zero-shot with workflow  |
    | RL (PPO, sparse reward)   | ~20%           | <5%               | Very hard to train       |
    | Random                    | 25%            | 25%               | 4 actions = 25% chance   |
    | PoT (this benchmark)      | {:.1%}         | {:.1%}            | Pondering over Thoughts  |

    *See: https://blog.gopenai.com/using-llms-and-langgraph-to-tackle-sokoban-puzzles-5f50b43b9515
    """.format(
        difficulty_results.get('simple', {}).get('accuracy', 0),
        difficulty_results.get('complex', {}).get('accuracy', 0),
    ))
    
    return final_results


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Main entry point."""
    config = parse_args()
    run_benchmark(config)


if __name__ == '__main__':
    main()

