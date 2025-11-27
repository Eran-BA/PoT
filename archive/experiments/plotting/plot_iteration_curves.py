"""
Plot Iteration Convergence Curves

Visualizes:
1. Kendall-τ vs iteration count
2. Improvement over baseline vs iterations
3. Plateau detection
4. Task-specific convergence patterns

Author: Eran Ben Artzy
Year: 2025
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_convergence_curve(csv_file, output_dir, task_name):
    """Plot convergence curve for a single task."""
    # Load data
    df = pd.read_csv(csv_file)
    
    # Separate baseline and PoH
    baseline_df = df[df['model'] == 'baseline']
    poh_df = df[df['model'] == 'pot']
    
    # Compute statistics per iteration count
    baseline_mean = baseline_df['test_kendall'].mean()
    baseline_std = baseline_df['test_kendall'].std()
    
    iter_stats = poh_df.groupby('max_inner_iters')['test_kendall'].agg(['mean', 'std', 'count'])
    iter_stats = iter_stats.reset_index()
    iter_stats = iter_stats.sort_values('max_inner_iters')
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Iteration Convergence Analysis: {task_name}', fontsize=16, fontweight='bold')
    
    # ----------------------------------------------------------------
    # Plot 1: Absolute Kendall-τ vs Iterations
    # ----------------------------------------------------------------
    ax = axes[0, 0]
    
    # Plot PoH curve
    ax.errorbar(
        iter_stats['max_inner_iters'],
        iter_stats['mean'],
        yerr=iter_stats['std'],
        marker='o',
        linewidth=2,
        capsize=5,
        label='PoH',
        color='#2E86AB'
    )
    
    # Plot baseline
    ax.axhline(
        baseline_mean,
        color='#A23B72',
        linestyle='--',
        linewidth=2,
        label=f'Baseline: {baseline_mean:.4f} ± {baseline_std:.4f}'
    )
    ax.fill_between(
        iter_stats['max_inner_iters'],
        baseline_mean - baseline_std,
        baseline_mean + baseline_std,
        color='#A23B72',
        alpha=0.2
    )
    
    ax.set_xlabel('Number of Iterations', fontsize=12, fontweight='bold')
    ax.set_ylabel('Kendall-τ', fontsize=12, fontweight='bold')
    ax.set_title('Absolute Performance', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # ----------------------------------------------------------------
    # Plot 2: Improvement over Baseline
    # ----------------------------------------------------------------
    ax = axes[0, 1]
    
    improvements = ((iter_stats['mean'] - baseline_mean) / baseline_mean) * 100
    improvements_abs = iter_stats['mean'] - baseline_mean
    
    bars = ax.bar(
        range(len(iter_stats)),
        improvements,
        color=['#06A77D' if imp > 0 else '#D62828' for imp in improvements],
        alpha=0.7,
        edgecolor='black',
        linewidth=1.5
    )
    
    # Add value labels on bars
    for i, (bar, imp_abs) in enumerate(zip(bars, improvements_abs)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + (1 if height > 0 else -1),
            f'{imp_abs:+.3f}',
            ha='center',
            va='bottom' if height > 0 else 'top',
            fontsize=8,
            fontweight='bold'
        )
    
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xticks(range(len(iter_stats)))
    ax.set_xticklabels(iter_stats['max_inner_iters'], fontsize=10)
    ax.set_xlabel('Number of Iterations', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement over Baseline (%)', fontsize=12, fontweight='bold')
    ax.set_title('Relative Improvement', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # ----------------------------------------------------------------
    # Plot 3: Marginal Gains (Diminishing Returns)
    # ----------------------------------------------------------------
    ax = axes[1, 0]
    
    marginal_gains = []
    iters_list = list(iter_stats['max_inner_iters'])
    means_list = list(iter_stats['mean'])
    
    for i in range(1, len(means_list)):
        gain = means_list[i] - means_list[i-1]
        marginal_gains.append(gain)
    
    if marginal_gains:
        ax.plot(
            iters_list[1:],
            marginal_gains,
            marker='s',
            linewidth=2,
            markersize=8,
            color='#F77F00',
            label='Marginal Gain'
        )
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.axhline(0.005, color='green', linestyle=':', alpha=0.5, label='Plateau threshold (0.005)')
        
        # Shade diminishing returns region
        plateau_idx = None
        for i, gain in enumerate(marginal_gains):
            if gain < 0.005:
                plateau_idx = i
                break
        
        if plateau_idx is not None:
            ax.axvspan(
                iters_list[plateau_idx + 1],
                iters_list[-1],
                color='red',
                alpha=0.1,
                label='Diminishing returns'
            )
    
    ax.set_xlabel('Number of Iterations', fontsize=12, fontweight='bold')
    ax.set_ylabel('Marginal Gain in τ', fontsize=12, fontweight='bold')
    ax.set_title('Diminishing Returns Analysis', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ----------------------------------------------------------------
    # Plot 4: Variance Analysis
    # ----------------------------------------------------------------
    ax = axes[1, 1]
    
    ax.errorbar(
        iter_stats['max_inner_iters'],
        iter_stats['mean'],
        yerr=iter_stats['std'],
        fmt='none',
        ecolor='gray',
        elinewidth=2,
        capsize=5,
        alpha=0.6
    )
    ax.scatter(
        iter_stats['max_inner_iters'],
        iter_stats['std'],
        s=100,
        c=iter_stats['std'],
        cmap='RdYlGn_r',
        edgecolor='black',
        linewidth=1.5,
        alpha=0.8
    )
    
    ax.set_xlabel('Number of Iterations', fontsize=12, fontweight='bold')
    ax.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
    ax.set_title('Variance Across Seeds', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(vmin=iter_stats['std'].min(), vmax=iter_stats['std'].max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Std Dev', rotation=270, labelpad=15, fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    output_file = Path(output_dir) / f'convergence_{task_name.lower().replace(" ", "_")}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    
    plt.close()
    
    return iter_stats, baseline_mean, baseline_std


def plot_combined_comparison(csv_files, task_names, output_dir):
    """Plot comparison across all tasks."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Iteration Convergence: Task Difficulty Comparison', fontsize=16, fontweight='bold')
    
    colors = ['#2E86AB', '#A23B72', '#06A77D']
    
    # ----------------------------------------------------------------
    # Plot 1: All tasks on same axes
    # ----------------------------------------------------------------
    ax = axes[0]
    
    for csv_file, task_name, color in zip(csv_files, task_names, colors):
        df = pd.read_csv(csv_file)
        baseline_df = df[df['model'] == 'baseline']
        poh_df = df[df['model'] == 'pot']
        
        baseline_mean = baseline_df['test_kendall'].mean()
        iter_stats = poh_df.groupby('max_inner_iters')['test_kendall'].agg(['mean', 'std']).reset_index()
        
        ax.plot(
            iter_stats['max_inner_iters'],
            iter_stats['mean'],
            marker='o',
            linewidth=2.5,
            markersize=7,
            label=task_name,
            color=color
        )
        ax.fill_between(
            iter_stats['max_inner_iters'],
            iter_stats['mean'] - iter_stats['std'],
            iter_stats['mean'] + iter_stats['std'],
            alpha=0.2,
            color=color
        )
    
    ax.set_xlabel('Number of Iterations', fontsize=13, fontweight='bold')
    ax.set_ylabel('Kendall-τ', fontsize=13, fontweight='bold')
    ax.set_title('Absolute Performance', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # ----------------------------------------------------------------
    # Plot 2: Improvement over baseline
    # ----------------------------------------------------------------
    ax = axes[1]
    
    for csv_file, task_name, color in zip(csv_files, task_names, colors):
        df = pd.read_csv(csv_file)
        baseline_df = df[df['model'] == 'baseline']
        poh_df = df[df['model'] == 'pot']
        
        baseline_mean = baseline_df['test_kendall'].mean()
        iter_stats = poh_df.groupby('max_inner_iters')['test_kendall'].agg(['mean']).reset_index()
        
        improvements = ((iter_stats['mean'] - baseline_mean) / baseline_mean) * 100
        
        ax.plot(
            iter_stats['max_inner_iters'],
            improvements,
            marker='s',
            linewidth=2.5,
            markersize=7,
            label=task_name,
            color=color
        )
    
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Number of Iterations', fontsize=13, fontweight='bold')
    ax.set_ylabel('Improvement over Baseline (%)', fontsize=13, fontweight='bold')
    ax.set_title('Relative Improvement', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = Path(output_dir) / 'combined_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Saved combined plot: {output_file}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot iteration convergence curves")
    parser.add_argument('--results_dir', type=str, default='experiments/results',
                        help='Directory containing CSV results')
    parser.add_argument('--output_dir', type=str, default='experiments/plots',
                        help='Output directory for plots')
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("PLOTTING ITERATION CONVERGENCE CURVES")
    print("=" * 80)
    print()
    
    # Process each task
    tasks = [
        ('iteration_sweep_len12.csv', 'Length 12 (Easy)'),
        ('iteration_sweep_len16.csv', 'Length 16 (Medium)'),
        ('iteration_sweep_len20.csv', 'Length 20 (Hard)'),
    ]
    
    csv_files = []
    task_names = []
    
    for csv_name, task_name in tasks:
        csv_file = Path(args.results_dir) / csv_name
        if not csv_file.exists():
            print(f"⚠ Skipping {task_name}: {csv_file} not found")
            continue
        
        print(f"\nProcessing: {task_name}")
        print("─" * 40)
        
        iter_stats, baseline_mean, baseline_std = plot_convergence_curve(
            csv_file, args.output_dir, task_name
        )
        
        # Print summary
        best_iter = iter_stats.loc[iter_stats['mean'].idxmax()]
        print(f"  Baseline: {baseline_mean:.4f} ± {baseline_std:.4f}")
        print(f"  Best iteration count: {int(best_iter['max_inner_iters'])}")
        print(f"  Best τ: {best_iter['mean']:.4f} ± {best_iter['std']:.4f}")
        print(f"  Improvement: {best_iter['mean'] - baseline_mean:+.4f} ({((best_iter['mean'] - baseline_mean)/baseline_mean)*100:+.1f}%)")
        
        csv_files.append(csv_file)
        task_names.append(task_name)
    
    # Create combined plot
    if len(csv_files) > 1:
        print("\n" + "=" * 80)
        print("CREATING COMBINED COMPARISON")
        print("=" * 80)
        plot_combined_comparison(csv_files, task_names, args.output_dir)
    
    print("\n" + "=" * 80)
    print("PLOTTING COMPLETE")
    print("=" * 80)
    print(f"\nPlots saved to: {args.output_dir}/")
    print()


if __name__ == '__main__':
    main()

