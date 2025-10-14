#!/usr/bin/env python3
"""
Hyperparameter search for maze solving: find optimal R, T, and n_heads.

This script performs a grid search over:
- R (refinement steps): [2, 4, 6, 8]
- T (HRM period): [2, 4, 8]
- n_heads (attention heads): [2, 4, 8]

Results are saved to CSV for analysis.
"""

import os
import sys
import csv
import itertools
from datetime import datetime
from pathlib import Path

# Setup paths
script_dir = Path(__file__).parent.absolute()
repo_root = script_dir.parent
if repo_root not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import the main test function
from maze_ab_proper_generation import run_ab_test


def save_results(results_file, config, results):
    """Save hyperparameter search results to CSV."""
    file_exists = os.path.exists(results_file)
    
    with open(results_file, 'a', newline='') as f:
        fieldnames = [
            'timestamp', 'maze_size', 'n_train', 'n_test', 'min_path_length',
            'R', 'T', 'n_heads', 'epochs', 'seed',
            'baseline_acc', 'baseline_opt', 
            'poh_acc', 'poh_opt',
            'poh_improvement_acc', 'poh_improvement_opt'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        # Calculate improvements
        baseline_acc = results['baseline']['acc']
        baseline_opt = results['baseline']['opt']
        poh_acc = results['poh']['acc']
        poh_opt = results['poh']['opt']
        
        improvement_acc = ((poh_acc - baseline_acc) / max(baseline_acc, 0.01)) * 100 if baseline_acc > 0 else 0
        improvement_opt = ((poh_opt - baseline_opt) / max(baseline_opt, 0.01)) * 100 if baseline_opt > 0 else 0
        
        row = {
            'timestamp': datetime.now().isoformat(),
            'maze_size': config['maze_size'],
            'n_train': config['n_train'],
            'n_test': config['n_test'],
            'min_path_length': config['min_path_length'],
            'R': config['R'],
            'T': config['T'],
            'n_heads': config['n_heads'],
            'epochs': config['epochs'],
            'seed': config['seed'],
            'baseline_acc': baseline_acc,
            'baseline_opt': baseline_opt,
            'poh_acc': poh_acc,
            'poh_opt': poh_opt,
            'poh_improvement_acc': improvement_acc,
            'poh_improvement_opt': improvement_opt,
        }
        writer.writerow(row)


def hyperparameter_search(
    maze_size=12,
    n_train=1000,
    n_test=100,
    min_path_length=40,
    epochs=50,
    seed=42,
    output_dir='experiments/results'
):
    """
    Run grid search over R, T, n_heads.
    
    Args:
        maze_size: Maze grid size
        n_train: Number of training samples
        n_test: Number of test samples
        min_path_length: Minimum solution path length
        epochs: Training epochs per configuration
        seed: Random seed
        output_dir: Directory for results CSV
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f'maze_hyperparam_search_{maze_size}x{maze_size}.csv')
    
    # Hyperparameter grid
    R_values = [2, 4, 6, 8]
    T_values = [2, 4, 8]
    heads_values = [2, 4, 8]
    
    total_configs = len(R_values) * len(T_values) * len(heads_values)
    
    print("="*80)
    print("MAZE HYPERPARAMETER SEARCH")
    print("="*80)
    print(f"Maze size: {maze_size}×{maze_size}")
    print(f"Training samples: {n_train}")
    print(f"Test samples: {n_test}")
    print(f"Min path length: {min_path_length}")
    print(f"Epochs per config: {epochs}")
    print(f"")
    print(f"Grid:")
    print(f"  R (refinement steps): {R_values}")
    print(f"  T (HRM period): {T_values}")
    print(f"  n_heads: {heads_values}")
    print(f"")
    print(f"Total configurations: {total_configs}")
    print(f"Results will be saved to: {results_file}")
    print("="*80)
    print("")
    
    # Run baseline once at the beginning with 100 epochs
    print(f"\n{'='*80}")
    print("RUNNING BASELINE TRANSFORMER FIRST (100 epochs)")
    print(f"{'='*80}\n")
    
    baseline_config = {
        'maze_size': maze_size,
        'n_train': n_train,
        'n_test': n_test,
        'min_path_length': min_path_length,
        'R': 1,  # Baseline doesn't use refinement
        'T': 1,  # Baseline doesn't use HRM
        'n_heads': 4,  # Standard baseline heads
        'epochs': 100,  # Always use 100 epochs for baseline
        'seed': seed,
        'skip_baseline': False,  # Run baseline
        'skip_poh': True         # Skip PoH for baseline-only run
    }
    
    try:
        baseline_results = run_ab_test(**baseline_config)
        print(f"\n✓ Baseline complete")
        print(f"  Baseline: Acc={baseline_results['baseline']['acc']:.2f}%, Opt={baseline_results['baseline']['opt']:.2f}%")
        
        # Append baseline to results file
        save_results(results_file, baseline_config, baseline_results)
    except Exception as e:
        print(f"\n✗ Baseline failed: {e}")
    
    # Run grid search (PoH configs only)
    config_num = 0
    for R, T, n_heads in itertools.product(R_values, T_values, heads_values):
        config_num += 1
        
        print(f"\n{'='*80}")
        print(f"Configuration {config_num}/{total_configs}")
        print(f"R={R}, T={T}, n_heads={n_heads}")
        print(f"{'='*80}\n")
        
        config = {
            'maze_size': maze_size,
            'n_train': n_train,
            'n_test': n_test,
            'min_path_length': min_path_length,
            'R': R,
            'T': T,
            'n_heads': n_heads,
            'epochs': epochs,
            'seed': seed
        }
        
        try:
            # Run A/B test with this configuration (skip baseline to save time)
            config['skip_baseline'] = True
            config['skip_poh'] = False
            results = run_ab_test(**config)
            
            # Save results
            save_results(results_file, config, results)
            
            print(f"\n✓ Configuration {config_num}/{total_configs} complete")
            print(f"  PoH-HRM:  Acc={results['poh']['acc']:.2f}%, Opt={results['poh']['opt']:.2f}%")
            
        except Exception as e:
            print(f"\n✗ Configuration {config_num}/{total_configs} failed: {e}")
            continue
    
    print(f"\n{'='*80}")
    print("HYPERPARAMETER SEARCH COMPLETE")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}\n")
    
    # Print summary
    print(f"\n{'='*80}")
    print("To analyze results, run:")
    print(f"  python experiments/analyze_maze_hyperparam_results.py {results_file}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Maze hyperparameter search')
    parser.add_argument('--maze-size', type=int, default=12, help='Maze size (default: 12)')
    parser.add_argument('--train', type=int, default=1000, help='Training samples (default: 1000)')
    parser.add_argument('--test', type=int, default=100, help='Test samples (default: 100)')
    parser.add_argument('--min-path-length', type=int, default=40, help='Min path length (default: 40)')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs per config (default: 50)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default='experiments/results', help='Output directory')
    
    # Quick mode for testing
    parser.add_argument('--quick', action='store_true', help='Quick test (10x10, 500 train, custom epochs via --epochs)')
    
    args = parser.parse_args()
    
    if args.quick:
        print("🚀 Quick mode enabled\n")
        hyperparameter_search(
            maze_size=10,
            n_train=500,
            n_test=50,
            min_path_length=30,
            epochs=args.epochs,  # Use --epochs argument instead of hardcoded 30
            seed=args.seed,
            output_dir=args.output_dir
        )
    else:
        hyperparameter_search(
            maze_size=args.maze_size,
            n_train=args.train,
            n_test=args.test,
            min_path_length=args.min_path_length,
            epochs=args.epochs,
            seed=args.seed,
            output_dir=args.output_dir
        )

