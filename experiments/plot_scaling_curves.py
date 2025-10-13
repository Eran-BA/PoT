"""
Plot Model Scaling Curves

Visualizes:
1. Performance vs model size (baseline vs PoH)
2. PoH improvement vs model size
3. Scaling efficiency (performance per parameter)
4. Optimal size-iteration tradeoff

Author: Eran Ben Artzy
Year: 2025
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_scaling_analysis(csv_file, output_dir):
    """Create comprehensive scaling analysis plots."""
    # Load data
    df = pd.read_csv(csv_file)
    
    # Separate baseline and PoH
    baseline_df = df[df['use_poh'] == False]
    poh_df = df[df['use_poh'] == True]
    
    # Compute statistics per model size
    baseline_stats = baseline_df.groupby(['d_model', 'n_params'])['test_kendall'].agg(['mean', 'std']).reset_index()
    poh_stats = poh_df.groupby(['d_model', 'n_params'])['test_kendall'].agg(['mean', 'std']).reset_index()
    
    # Merge
    stats = pd.merge(
        baseline_stats,
        poh_stats,
        on=['d_model', 'n_params'],
        suffixes=('_baseline', '_poh')
    )
    
    # Compute improvement
    stats['improvement_abs'] = stats['mean_poh'] - stats['mean_baseline']
    stats['improvement_pct'] = (stats['improvement_abs'] / stats['mean_baseline']) * 100
    
    # Sort by model size
    stats = stats.sort_values('d_model')
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # ----------------------------------------------------------------
    # Plot 1: Absolute Performance vs Model Size
    # ----------------------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    
    x_pos = np.arange(len(stats))
    width = 0.35
    
    # Baseline bars
    ax.bar(
        x_pos - width/2,
        stats['mean_baseline'],
        width,
        yerr=stats['std_baseline'],
        label='Baseline',
        color='#A23B72',
        alpha=0.7,
        capsize=5,
        edgecolor='black',
        linewidth=1.5
    )
    
    # PoH bars
    ax.bar(
        x_pos + width/2,
        stats['mean_poh'],
        width,
        yerr=stats['std_poh'],
        label='PoH (12 iters)',
        color='#2E86AB',
        alpha=0.7,
        capsize=5,
        edgecolor='black',
        linewidth=1.5
    )
    
    ax.set_xlabel('Model Size (d_model)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test Kendall-τ', fontsize=13, fontweight='bold')
    ax.set_title('Performance vs Model Size: Baseline vs PoH', fontsize=15, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{int(d)}-dim\n({p/1e6:.2f}M)" for d, p in zip(stats['d_model'], stats['n_params'])], fontsize=10)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # ----------------------------------------------------------------
    # Plot 2: PoH Improvement vs Model Size
    # ----------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    
    # Plot both absolute and percentage
    ax2 = ax.twinx()
    
    # Absolute improvement (left y-axis)
    line1 = ax.plot(
        stats['d_model'],
        stats['improvement_abs'],
        marker='o',
        linewidth=2.5,
        markersize=10,
        color='#06A77D',
        label='Absolute improvement'
    )
    ax.fill_between(
        stats['d_model'],
        0,
        stats['improvement_abs'],
        alpha=0.2,
        color='#06A77D'
    )
    
    # Percentage improvement (right y-axis)
    line2 = ax2.plot(
        stats['d_model'],
        stats['improvement_pct'],
        marker='s',
        linewidth=2.5,
        markersize=10,
        color='#F77F00',
        linestyle='--',
        label='Relative improvement'
    )
    
    ax.set_xlabel('Model Size (d_model)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Improvement (Δτ)', fontsize=12, fontweight='bold', color='#06A77D')
    ax2.set_ylabel('Relative Improvement (%)', fontsize=12, fontweight='bold', color='#F77F00')
    ax.set_title('PoH Improvement vs Model Size', fontsize=13, fontweight='bold')
    ax.tick_params(axis='y', labelcolor='#06A77D')
    ax2.tick_params(axis='y', labelcolor='#F77F00')
    ax.grid(True, alpha=0.3)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left', fontsize=10)
    
    # ----------------------------------------------------------------
    # Plot 3: Scaling Efficiency (Performance per Parameter)
    # ----------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    
    # Compute efficiency: tau per million parameters
    baseline_efficiency = stats['mean_baseline'] / (stats['n_params'] / 1e6)
    poh_efficiency = stats['mean_poh'] / (stats['n_params'] / 1e6)
    
    ax.plot(
        stats['d_model'],
        baseline_efficiency,
        marker='o',
        linewidth=2.5,
        markersize=8,
        label='Baseline',
        color='#A23B72'
    )
    ax.plot(
        stats['d_model'],
        poh_efficiency,
        marker='s',
        linewidth=2.5,
        markersize=8,
        label='PoH',
        color='#2E86AB'
    )
    
    ax.set_xlabel('Model Size (d_model)', fontsize=12, fontweight='bold')
    ax.set_ylabel('τ per Million Parameters', fontsize=12, fontweight='bold')
    ax.set_title('Scaling Efficiency', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # ----------------------------------------------------------------
    # Plot 4: Scaling Correlation
    # ----------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    
    # Scatter plot: model size vs improvement
    scatter = ax.scatter(
        stats['n_params'] / 1e6,
        stats['improvement_abs'],
        s=200,
        c=stats['improvement_pct'],
        cmap='RdYlGn',
        edgecolor='black',
        linewidth=2,
        alpha=0.8
    )
    
    # Fit line
    z = np.polyfit(stats['n_params'] / 1e6, stats['improvement_abs'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(stats['n_params'].min() / 1e6, stats['n_params'].max() / 1e6, 100)
    ax.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=2, label=f'Fit: y={z[0]:.4f}x+{z[1]:.4f}')
    
    # Compute correlation
    corr = np.corrcoef(stats['n_params'], stats['improvement_abs'])[0, 1]
    
    ax.set_xlabel('Model Parameters (Millions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('PoH Improvement (Δτ)', fontsize=12, fontweight='bold')
    ax.set_title(f'Scaling Correlation (r={corr:.3f})', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Improvement (%)', rotation=270, labelpad=20, fontsize=10)
    
    # ----------------------------------------------------------------
    # Plot 5: Cost-Benefit Analysis
    # ----------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 1])
    
    # Compute training cost (params × iterations)
    baseline_cost = stats['n_params']
    poh_cost = stats['n_params'] * 12  # 12 iterations
    
    # Plot performance vs cost
    ax.scatter(
        baseline_cost / 1e6,
        stats['mean_baseline'],
        s=150,
        marker='o',
        color='#A23B72',
        edgecolor='black',
        linewidth=2,
        label='Baseline',
        alpha=0.7
    )
    ax.scatter(
        poh_cost / 1e6,
        stats['mean_poh'],
        s=150,
        marker='s',
        color='#2E86AB',
        edgecolor='black',
        linewidth=2,
        label='PoH',
        alpha=0.7
    )
    
    # Draw arrows from baseline to PoH
    for i in range(len(stats)):
        ax.annotate(
            '',
            xy=(poh_cost.iloc[i] / 1e6, stats['mean_poh'].iloc[i]),
            xytext=(baseline_cost.iloc[i] / 1e6, stats['mean_baseline'].iloc[i]),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, alpha=0.5)
        )
    
    ax.set_xlabel('Computational Cost (M params × iters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance (τ)', fontsize=12, fontweight='bold')
    ax.set_title('Cost-Benefit Analysis', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Model Scaling Analysis: PoH vs Baseline', fontsize=17, fontweight='bold', y=0.995)
    
    # Save
    output_file = Path(output_dir) / 'scaling_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.close()
    
    return stats, corr


def print_summary(stats, corr):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS SUMMARY")
    print("=" * 80)
    print()
    
    print(f"{'Model':<15} {'Baseline':<12} {'PoH':<12} {'Improvement':<20} {'Status'}")
    print("─" * 80)
    
    for _, row in stats.iterrows():
        d_model = int(row['d_model'])
        n_params = row['n_params'] / 1e6
        baseline = row['mean_baseline']
        poh = row['mean_poh']
        imp_abs = row['improvement_abs']
        imp_pct = row['improvement_pct']
        
        status = "✅" if imp_abs > 0.01 else "⚠️" if imp_abs > 0 else "❌"
        
        print(f"{d_model}-dim ({n_params:.1f}M)  "
              f"{baseline:.4f}      "
              f"{poh:.4f}      "
              f"{imp_abs:+.4f} ({imp_pct:+.1f}%)      "
              f"{status}")
    
    print("─" * 80)
    
    print(f"\nScaling Correlation: r = {corr:.3f}")
    if corr > 0.7:
        print("  ✓ STRONG positive correlation: Larger models benefit MORE from PoH!")
    elif corr > 0.4:
        print("  ✓ Moderate positive correlation: Some scaling benefit")
    elif corr > 0.1:
        print("  ⚠ Weak correlation: Limited scaling benefit")
    else:
        print("  ✗ No clear scaling benefit")
    
    # Find best size
    best_idx = stats['mean_poh'].idxmax()
    best_row = stats.iloc[best_idx]
    
    print(f"\nBest Model Size:")
    print(f"  d_model: {int(best_row['d_model'])}")
    print(f"  Parameters: {best_row['n_params']/1e6:.2f}M")
    print(f"  PoH Performance: {best_row['mean_poh']:.4f}")
    print(f"  Improvement: {best_row['improvement_abs']:+.4f} ({best_row['improvement_pct']:+.1f}%)")
    
    # Efficiency analysis
    baseline_eff = stats['mean_baseline'] / (stats['n_params'] / 1e6)
    poh_eff = stats['mean_poh'] / (stats['n_params'] / 1e6)
    
    print(f"\nMost Efficient Model:")
    most_eff_idx = poh_eff.idxmax()
    most_eff_row = stats.iloc[most_eff_idx]
    print(f"  d_model: {int(most_eff_row['d_model'])}")
    print(f"  τ per M params: {poh_eff.iloc[most_eff_idx]:.4f}")
    print(f"  Best balance of performance and cost")


def main():
    parser = argparse.ArgumentParser(description="Plot model scaling curves")
    parser.add_argument('--csv_file', type=str,
                        default='experiments/results/scaling_full.csv',
                        help='CSV file with scaling results')
    parser.add_argument('--output_dir', type=str,
                        default='experiments/plots',
                        help='Output directory for plots')
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("PLOTTING MODEL SCALING ANALYSIS")
    print("=" * 80)
    print()
    
    if not Path(args.csv_file).exists():
        print(f"❌ Error: {args.csv_file} not found")
        print("Run experiments first: ./experiments/run_scaling_experiments.sh")
        return
    
    print(f"Loading data from: {args.csv_file}")
    stats, corr = plot_scaling_analysis(args.csv_file, args.output_dir)
    
    print_summary(stats, corr)
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nPlot saved to: {args.output_dir}/scaling_analysis.png")
    print()


if __name__ == '__main__':
    main()

