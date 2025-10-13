#!/usr/bin/env python3
"""
Generate a comprehensive results plot for the README.

Creates a publication-quality comparison of Baseline vs PoH across different
array lengths for the partial observability sorting task.

Author: Eran Ben Artzy
Year: 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

def load_results():
    """Load all experiment results."""
    results_dir = Path("examples/synthetic/results")
    
    data = []
    
    # Baseline results
    for file in results_dir.glob("fair_ab_baseline*.csv"):
        df = pd.read_csv(file)
        length = 12  # default
        if "len16" in file.name:
            length = 16
        elif "len20" in file.name:
            length = 20
        df['length'] = length
        data.append(df)
    
    # PoH results
    for file in results_dir.glob("fair_ab_pot*.csv"):
        if "bptt" in file.name:
            continue  # Skip BPTT variants for main comparison
        df = pd.read_csv(file)
        length = 12  # default
        if "len16" in file.name:
            length = 16
        elif "len20" in file.name:
            length = 20
        df['length'] = length
        data.append(df)
    
    return pd.concat(data, ignore_index=True)

def plot_results():
    """Create the main results plot."""
    df = load_results()
    
    # Aggregate by model and length
    summary = df.groupby(['model', 'length'])['test_kendall'].agg(['mean', 'std', 'count']).reset_index()
    summary['sem'] = summary['std'] / np.sqrt(summary['count'])
    summary['ci95'] = 1.96 * summary['sem']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Plot 1: Bar chart comparison
    x = np.arange(len(summary['length'].unique()))
    width = 0.35
    
    baseline_data = summary[summary['model'] == 'baseline'].sort_values('length')
    poh_data = summary[summary['model'] == 'pot'].sort_values('length')
    
    bars1 = ax1.bar(x - width/2, baseline_data['mean'], width, 
                    yerr=baseline_data['ci95'], label='Baseline',
                    color='#3498db', alpha=0.8, capsize=5)
    bars2 = ax1.bar(x + width/2, poh_data['mean'], width,
                    yerr=poh_data['ci95'], label='PoH',
                    color='#e74c3c', alpha=0.8, capsize=5)
    
    ax1.set_xlabel('Array Length')
    ax1.set_ylabel('Kendall-τ (higher is better)')
    ax1.set_title('Baseline vs PoH: Partial Observability Sorting')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'L={l}' for l in sorted(baseline_data['length'].unique())])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, max(summary['mean'] + summary['ci95']) * 1.15)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Per-seed scatter
    for model, color, marker in [('baseline', '#3498db', 'o'), ('pot', '#e74c3c', '^')]:
        model_data = df[df['model'] == model]
        for length in sorted(model_data['length'].unique()):
            length_data = model_data[model_data['length'] == length]
            x_vals = [length] * len(length_data)
            y_vals = length_data['test_kendall']
            
            label = model.upper() if length == sorted(model_data['length'].unique())[0] else None
            ax2.scatter(x_vals, y_vals, alpha=0.6, s=50, 
                       color=color, marker=marker, label=label)
    
    # Add means
    for model, color in [('baseline', '#3498db'), ('pot', '#e74c3c')]:
        model_summary = summary[summary['model'] == model].sort_values('length')
        ax2.plot(model_summary['length'], model_summary['mean'], 
                color=color, linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Array Length')
    ax2.set_ylabel('Kendall-τ')
    ax2.set_title('Individual Seeds (with means)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xticks(sorted(df['length'].unique()))
    
    plt.tight_layout()
    
    # Save
    output_dir = Path("figs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "baseline_vs_poh_sorting.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved plot to {output_path}")
    
    # Also save as PDF for publication
    pdf_path = output_dir / "baseline_vs_poh_sorting.pdf"
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✅ Saved PDF to {pdf_path}")
    
    return summary

def print_summary_table(summary):
    """Print a summary table."""
    print("\n" + "="*60)
    print("RESULTS SUMMARY: Baseline vs PoH")
    print("="*60)
    print("\nPartial Observability Sorting (50% masking)")
    print("-" * 60)
    
    for length in sorted(summary['length'].unique()):
        print(f"\n📊 Array Length = {length}")
        print("-" * 40)
        
        for model in ['baseline', 'pot']:
            row = summary[(summary['model'] == model) & (summary['length'] == length)].iloc[0]
            model_name = "Baseline" if model == 'baseline' else "PoH     "
            print(f"  {model_name}: {row['mean']:.4f} ± {row['ci95']:.4f} (n={int(row['count'])})")
        
        # Calculate improvement
        baseline_mean = summary[(summary['model'] == 'baseline') & (summary['length'] == length)]['mean'].iloc[0]
        poh_mean = summary[(summary['model'] == 'pot') & (summary['length'] == length)]['mean'].iloc[0]
        improvement = ((poh_mean - baseline_mean) / baseline_mean) * 100
        
        symbol = "✅" if improvement > 0 else "⚠️"
        print(f"  {symbol} PoH improvement: {improvement:+.2f}%")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    print("Generating results plot...")
    summary = plot_results()
    print_summary_table(summary)
    print("\n✅ Results plot generated successfully!")
    print("📊 Plot saved to: figs/baseline_vs_poh_sorting.png")

