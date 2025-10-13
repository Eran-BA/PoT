#!/usr/bin/env python3
"""
Auto-generate publication-quality plots from experiment CSVs.

Reads experiments/results/*.csv and writes figs/
- Baseline vs PoH comparison (bar charts with error bars)
- Iterations → UAS/LAS curves
- Multi-seed mean ± 95% CI
- Training time/epoch

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --output_dir figs/custom
    python scripts/plot_results.py --task parsing --metric uas

Author: Eran Ben Artzy
Year: 2025
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set publication style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


def load_results(registry_path: str = "experiments/registry.json") -> Dict:
    """Load experiment registry."""
    with open(registry_path, 'r') as f:
        return json.load(f)


def load_csv_safe(path: str) -> Optional[pd.DataFrame]:
    """Load CSV if exists."""
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


def plot_baseline_vs_poh(
    baseline_df: pd.DataFrame,
    poh_df: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path
):
    """Bar chart: Baseline vs PoH with error bars."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract scores
    baseline_scores = baseline_df[f'test_{metric}'].values if f'test_{metric}' in baseline_df.columns else baseline_df[metric].values
    poh_scores = poh_df[f'test_{metric}'].values if f'test_{metric}' in poh_df.columns else poh_df[metric].values
    
    # Compute stats
    baseline_mean = baseline_scores.mean()
    baseline_std = baseline_scores.std()
    baseline_ci = 1.96 * baseline_std / np.sqrt(len(baseline_scores))
    
    poh_mean = poh_scores.mean()
    poh_std = poh_scores.std()
    poh_ci = 1.96 * poh_std / np.sqrt(len(poh_scores))
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(poh_scores, baseline_scores)
    
    # Plot
    models = ['Baseline', 'PoH']
    means = [baseline_mean, poh_mean]
    cis = [baseline_ci, poh_ci]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax.bar(models, means, yerr=cis, capsize=10, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, mean, ci in zip(bars, means, cis):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + ci + 0.005,
                f'{mean:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Add improvement annotation
    improvement = poh_mean - baseline_mean
    pct_improvement = (improvement / baseline_mean) * 100
    
    ax.annotate(f'Δ = {improvement:+.4f}\n({pct_improvement:+.1f}%)\np = {p_value:.4f}',
                xy=(0.5, max(means) + max(cis) + 0.01),
                xytext=(0.5, max(means) + max(cis) + 0.05),
                ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_ylabel(metric.upper(), fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_ylim(0, max(means) + max(cis) + 0.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def plot_iterations_curve(
    results_dict: Dict[str, pd.DataFrame],
    metric: str,
    title: str,
    output_path: Path
):
    """Line plot: Iterations → Metric (with confidence intervals)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by iteration count
    iter_data = []
    for name, df in results_dict.items():
        # Extract iteration count from name (if present)
        if 'iter' in name.lower():
            iters = int(''.join(filter(str.isdigit, name.split('iter')[0].split('_')[-1])))
        else:
            iters = 1
        
        scores = df[f'test_{metric}'].values if f'test_{metric}' in df.columns else df[metric].values
        iter_data.append({
            'iters': iters,
            'mean': scores.mean(),
            'std': scores.std(),
            'ci': 1.96 * scores.std() / np.sqrt(len(scores)),
            'n': len(scores)
        })
    
    # Sort by iteration count
    iter_data.sort(key=lambda x: x['iters'])
    
    iters = [d['iters'] for d in iter_data]
    means = [d['mean'] for d in iter_data]
    cis = [d['ci'] for d in iter_data]
    
    # Plot with error band
    ax.plot(iters, means, marker='o', linewidth=2, markersize=8, color='#e74c3c', label='PoH')
    ax.fill_between(iters, 
                     [m - c for m, c in zip(means, cis)],
                     [m + c for m, c in zip(means, cis)],
                     alpha=0.3, color='#e74c3c')
    
    # Add value labels
    for i, m, c in zip(iters, means, cis):
        ax.text(i, m + c + 0.005, f'{m:.4f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Iterations', fontweight='bold')
    ax.set_ylabel(metric.upper(), fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xticks(iters)
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def plot_seeds_variance(
    df: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path
):
    """Box plot: Multi-seed variance."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    scores = df[f'test_{metric}'].values if f'test_{metric}' in df.columns else df[metric].values
    
    bp = ax.boxplot([scores], labels=['PoH'], patch_artist=True,
                     boxprops=dict(facecolor='#e74c3c', alpha=0.7),
                     medianprops=dict(color='black', linewidth=2),
                     whiskerprops=dict(color='black', linewidth=1.5),
                     capprops=dict(color='black', linewidth=1.5))
    
    # Add individual points
    ax.scatter([1] * len(scores), scores, alpha=0.5, s=100, color='black', zorder=3)
    
    # Stats annotation
    mean = scores.mean()
    std = scores.std()
    ax.text(1.3, mean, f'μ = {mean:.4f}\nσ = {std:.4f}\nn = {len(scores)}',
            fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_ylabel(metric.upper(), fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Auto-generate plots from experiment results')
    parser.add_argument('--registry', type=str, default='experiments/registry.json',
                        help='Path to registry.json')
    parser.add_argument('--output_dir', type=str, default='figs',
                        help='Output directory for figures')
    parser.add_argument('--task', type=str, default=None,
                        help='Specific task to plot (default: all)')
    parser.add_argument('--metric', type=str, default=None,
                        help='Metric to plot (default: from registry)')
    args = parser.parse_args()

    # Load registry
    registry = load_results(args.registry)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("GENERATING PLOTS FROM EXPERIMENT RESULTS")
    print("="*80)
    print()

    # Process each task
    tasks_to_plot = [args.task] if args.task else registry['tasks'].keys()
    
    for task_name in tasks_to_plot:
        if task_name not in registry['tasks']:
            print(f"⚠️  Task '{task_name}' not found in registry")
            continue
        
        task_configs = registry['tasks'][task_name]
        metric = args.metric or registry['metrics'].get(task_name, 'accuracy')
        
        print(f"\n📊 Task: {task_name.upper()} (metric: {metric})")
        print("-" * 80)
        
        for config_name, config_info in task_configs.items():
            print(f"\n  Config: {config_name}")
            
            # Load baseline
            baseline_path = config_info.get('baseline')
            baseline_df = load_csv_safe(baseline_path) if baseline_path else None
            
            # Load PoH variants
            poh_variants = {}
            for key, value in config_info.items():
                if key.startswith('hrm_poh') or key == 'poh':
                    poh_df = load_csv_safe(value) if value else None
                    if poh_df is not None:
                        poh_variants[key] = poh_df
            
            if not poh_variants:
                print("    ⚠️  No PoH results found")
                continue
            
            # Plot 1: Baseline vs PoH (use first PoH variant)
            if baseline_df is not None and poh_variants:
                first_poh_name, first_poh_df = list(poh_variants.items())[0]
                output_path = output_dir / f'{task_name}_{config_name}_baseline_vs_poh.png'
                plot_baseline_vs_poh(
                    baseline_df,
                    first_poh_df,
                    metric,
                    f'{task_name.capitalize()} - {config_name}: Baseline vs PoH',
                    output_path
                )
            
            # Plot 2: Iterations curve (if multiple PoH variants)
            if len(poh_variants) > 1:
                output_path = output_dir / f'{task_name}_{config_name}_iterations.png'
                plot_iterations_curve(
                    poh_variants,
                    metric,
                    f'{task_name.capitalize()} - {config_name}: Iterations → {metric.upper()}',
                    output_path
                )
            
            # Plot 3: Seed variance (use first PoH variant)
            if poh_variants:
                first_poh_name, first_poh_df = list(poh_variants.items())[0]
                output_path = output_dir / f'{task_name}_{config_name}_variance.png'
                plot_seeds_variance(
                    first_poh_df,
                    metric,
                    f'{task_name.capitalize()} - {config_name}: Multi-Seed Variance',
                    output_path
                )
    
    print("\n" + "="*80)
    print(f"✅ ALL PLOTS SAVED TO: {output_dir}/")
    print("="*80)
    print()


if __name__ == '__main__':
    main()

