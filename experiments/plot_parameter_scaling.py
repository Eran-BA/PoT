#!/usr/bin/env python3
"""
Visualize parameter scaling results.
Creates plots showing:
1. Accuracy vs. Parameters
2. Optimality vs. Parameters
3. PoH Advantage vs. Parameters
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def plot_scaling_results(results_file, output_dir=None):
    """Plot parameter scaling results."""
    
    # Load results
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    config = data['config']
    
    # Extract data
    sizes = [r['size'] for r in results]
    baseline_params = [r['baseline_params'] / 1e6 for r in results]
    poh_params = [r['poh_params'] / 1e6 for r in results]
    
    baseline_acc = [r['baseline_acc'] for r in results]
    poh_acc = [r['poh_acc'] for r in results]
    
    baseline_opt = [r['baseline_opt'] for r in results]
    poh_opt = [r['poh_opt'] for r in results]
    
    poh_adv_acc = [r['poh_advantage_acc'] for r in results]
    poh_adv_opt = [r['poh_advantage_opt'] for r in results]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Accuracy vs Parameters
    ax = axes[0, 0]
    ax.plot(baseline_params, baseline_acc, 'o-', label='Baseline', linewidth=2, markersize=8)
    ax.plot(poh_params, poh_acc, 's-', label='PoH-HRM', linewidth=2, markersize=8)
    ax.set_xlabel('Parameters (M)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Accuracy vs. Model Size\n(Maze {config["maze_size"]}×{config["maze_size"]})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Plot 2: Optimality vs Parameters
    ax = axes[0, 1]
    ax.plot(baseline_params, baseline_opt, 'o-', label='Baseline', linewidth=2, markersize=8)
    ax.plot(poh_params, poh_opt, 's-', label='PoH-HRM', linewidth=2, markersize=8)
    ax.set_xlabel('Parameters (M)', fontsize=12)
    ax.set_ylabel('Optimality (%)', fontsize=12)
    ax.set_title(f'Optimality vs. Model Size\n(Maze {config["maze_size"]}×{config["maze_size"]})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Plot 3: PoH Advantage in Accuracy
    ax = axes[1, 0]
    colors = ['green' if x > 0 else 'red' for x in poh_adv_acc]
    ax.bar(sizes, poh_adv_acc, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Model Size', fontsize=12)
    ax.set_ylabel('PoH Advantage (%)', fontsize=12)
    ax.set_title('PoH-HRM Accuracy Advantage\n(PoH - Baseline)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: PoH Advantage in Optimality
    ax = axes[1, 1]
    colors = ['green' if x > 0 else 'red' for x in poh_adv_opt]
    ax.bar(sizes, poh_adv_opt, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Model Size', fontsize=12)
    ax.set_ylabel('PoH Advantage (%)', fontsize=12)
    ax.set_title('PoH-HRM Optimality Advantage\n(PoH - Baseline)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        output_dir = Path(results_file).parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f'scaling_plot_maze{config["maze_size"]}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_file}")
    
    # Also create a summary table
    summary_file = output_dir / f'scaling_summary_maze{config["maze_size"]}.txt'
    with open(summary_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write(f"PARAMETER SCALING RESULTS - Maze {config['maze_size']}×{config['maze_size']}\n")
        f.write("="*100 + "\n")
        f.write(f"Training: {config['n_train']} samples, {config['epochs']} epochs\n")
        f.write(f"Testing: {config['n_test']} samples\n")
        f.write(f"PoH Config: R={config['R']}, T={config['T']}\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"{'Size':<10} {'Params (M)':<20} {'Accuracy (%)':<25} {'Optimality (%)':<25}\n")
        f.write(f"{'':<10} {'Baseline / PoH':<20} {'Baseline / PoH / Δ':<25} {'Baseline / PoH / Δ':<25}\n")
        f.write("-"*100 + "\n")
        
        for r in results:
            f.write(f"{r['size']:<10} "
                   f"{r['baseline_params']/1e6:>5.1f} / {r['poh_params']/1e6:>5.1f}   "
                   f"{r['baseline_acc']:>5.1f} / {r['poh_acc']:>5.1f} / {r['poh_advantage_acc']:>+5.1f}   "
                   f"{r['baseline_opt']:>5.1f} / {r['poh_opt']:>5.1f} / {r['poh_advantage_opt']:>+5.1f}\n")
        
        f.write("="*100 + "\n")
        f.write("\nKey Findings:\n")
        f.write("-"*100 + "\n")
        
        # Calculate average advantage
        avg_adv_acc = np.mean([r['poh_advantage_acc'] for r in results])
        avg_adv_opt = np.mean([r['poh_advantage_opt'] for r in results])
        
        f.write(f"Average PoH Advantage (Accuracy): {avg_adv_acc:+.2f}%\n")
        f.write(f"Average PoH Advantage (Optimality): {avg_adv_opt:+.2f}%\n\n")
        
        # Find best size for PoH
        best_acc_idx = np.argmax([r['poh_acc'] for r in results])
        best_opt_idx = np.argmax([r['poh_opt'] for r in results])
        
        f.write(f"Best PoH Accuracy: {results[best_acc_idx]['size']} "
               f"({results[best_acc_idx]['poh_acc']:.1f}% @ {results[best_acc_idx]['poh_params']/1e6:.1f}M params)\n")
        f.write(f"Best PoH Optimality: {results[best_opt_idx]['size']} "
               f"({results[best_opt_idx]['poh_opt']:.1f}% @ {results[best_opt_idx]['poh_params']/1e6:.1f}M params)\n")
    
    print(f"✓ Summary saved to: {summary_file}")
    
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot parameter scaling results')
    parser.add_argument('results_file', type=str, help='Path to results JSON file')
    parser.add_argument('--output', type=str, default=None, 
                       help='Output directory (default: same as results file)')
    
    args = parser.parse_args()
    
    plot_scaling_results(args.results_file, args.output)

