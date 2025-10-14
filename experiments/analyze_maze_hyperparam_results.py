#!/usr/bin/env python3
"""
Analyze maze hyperparameter search results.

Reads CSV from maze_hyperparam_search.py and shows:
- Best configurations by accuracy
- Best configurations by optimality
- Heatmaps for each hyperparameter
"""

import sys
import csv
import pandas as pd
import numpy as np
from pathlib import Path


def load_results(csv_path):
    """Load results from CSV."""
    df = pd.read_csv(csv_path)
    return df


def print_top_configs(df, metric='poh_acc', top_n=10):
    """Print top N configurations by metric."""
    df_sorted = df.sort_values(metric, ascending=False)
    
    print(f"\n{'='*80}")
    print(f"Top {top_n} Configurations by {metric}")
    print(f"{'='*80}\n")
    
    for i, (idx, row) in enumerate(df_sorted.head(top_n).iterrows(), 1):
        print(f"{i}. R={int(row['R'])}, T={int(row['T'])}, heads={int(row['n_heads'])}")
        print(f"   PoH: Acc={row['poh_acc']:.2f}%, Opt={row['poh_opt']:.2f}%")
        print(f"   Baseline: Acc={row['baseline_acc']:.2f}%, Opt={row['baseline_opt']:.2f}%")
        print(f"   Improvement: Acc={row['poh_improvement_acc']:.1f}%, Opt={row['poh_improvement_opt']:.1f}%")
        print()


def analyze_by_param(df, param, metric='poh_acc'):
    """Analyze results grouped by a single parameter."""
    grouped = df.groupby(param)[metric].agg(['mean', 'std', 'count'])
    grouped = grouped.sort_values('mean', ascending=False)
    
    print(f"\n{'='*80}")
    print(f"Analysis by {param} (metric: {metric})")
    print(f"{'='*80}\n")
    print(grouped.to_string())
    print()


def print_summary_statistics(df):
    """Print summary statistics."""
    print(f"\n{'='*80}")
    print("Summary Statistics")
    print(f"{'='*80}\n")
    
    print(f"Total configurations tested: {len(df)}")
    print(f"\nPoH Accuracy:")
    print(f"  Mean:   {df['poh_acc'].mean():.2f}%")
    print(f"  Std:    {df['poh_acc'].std():.2f}%")
    print(f"  Min:    {df['poh_acc'].min():.2f}%")
    print(f"  Max:    {df['poh_acc'].max():.2f}%")
    
    print(f"\nPoH Optimality:")
    print(f"  Mean:   {df['poh_opt'].mean():.2f}%")
    print(f"  Std:    {df['poh_opt'].std():.2f}%")
    print(f"  Min:    {df['poh_opt'].min():.2f}%")
    print(f"  Max:    {df['poh_opt'].max():.2f}%")
    
    print(f"\nBaseline Accuracy:")
    print(f"  Mean:   {df['baseline_acc'].mean():.2f}%")
    print(f"  Std:    {df['baseline_acc'].std():.2f}%")
    
    print(f"\nBaseline Optimality:")
    print(f"  Mean:   {df['baseline_opt'].mean():.2f}%")
    print(f"  Std:    {df['baseline_opt'].std():.2f}%")
    print()


def find_best_overall(df):
    """Find best overall configuration."""
    # Weighted score: 70% accuracy + 30% optimality
    df['score'] = 0.7 * df['poh_acc'] + 0.3 * df['poh_opt']
    best_idx = df['score'].idxmax()
    best = df.loc[best_idx]
    
    print(f"\n{'='*80}")
    print("BEST OVERALL CONFIGURATION (weighted: 70% acc + 30% opt)")
    print(f"{'='*80}\n")
    print(f"R={int(best['R'])}, T={int(best['T'])}, n_heads={int(best['n_heads'])}")
    print(f"\nPoH-HRM Results:")
    print(f"  Accuracy:   {best['poh_acc']:.2f}%")
    print(f"  Optimality: {best['poh_opt']:.2f}%")
    print(f"  Score:      {best['score']:.2f}")
    print(f"\nBaseline Results:")
    print(f"  Accuracy:   {best['baseline_acc']:.2f}%")
    print(f"  Optimality: {best['baseline_opt']:.2f}%")
    print(f"\nImprovement:")
    print(f"  Accuracy:   {best['poh_improvement_acc']:.1f}%")
    print(f"  Optimality: {best['poh_improvement_opt']:.1f}%")
    print()


def main(csv_path):
    """Main analysis function."""
    if not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    print(f"\nLoading results from: {csv_path}")
    df = load_results(csv_path)
    
    if len(df) == 0:
        print("Error: No results found in CSV")
        sys.exit(1)
    
    # Print summary
    print_summary_statistics(df)
    
    # Find best overall
    find_best_overall(df)
    
    # Top configurations by accuracy
    print_top_configs(df, metric='poh_acc', top_n=5)
    
    # Top configurations by optimality
    print_top_configs(df, metric='poh_opt', top_n=5)
    
    # Analysis by parameter
    analyze_by_param(df, 'R', metric='poh_acc')
    analyze_by_param(df, 'T', metric='poh_acc')
    analyze_by_param(df, 'n_heads', metric='poh_acc')
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python analyze_maze_hyperparam_results.py <results.csv>")
        sys.exit(1)
    
    main(sys.argv[1])

