#!/usr/bin/env python3
"""
Generate Markdown tables from experiment CSVs for README.

Reads experiments/results/*.csv and outputs Markdown tables
that you can copy-paste into README.md

Usage:
    python scripts/make_readme_tables.py
    python scripts/make_readme_tables.py --task parsing
    python scripts/make_readme_tables.py --format github  # GitHub-flavored Markdown

Author: Eran Ben Artzy
Year: 2025
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np
from scipy import stats


def load_csv_safe(path: str) -> Optional[pd.DataFrame]:
    """Load CSV if exists."""
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


def cohens_d(x1, x2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(x1), len(x2)
    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(x1) - np.mean(x2)) / pooled_std if pooled_std > 0 else 0


def format_number(value: float, decimals: int = 4) -> str:
    """Format number with fixed decimals."""
    return f"{value:.{decimals}f}"


def format_improvement(delta: float, pct: float) -> str:
    """Format improvement string."""
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.4f} ({sign}{pct:.1f}%)"


def generate_comparison_table(
    task_name: str,
    config_name: str,
    baseline_df: pd.DataFrame,
    poh_variants: Dict[str, pd.DataFrame],
    metric: str,
    format_style: str = 'github'
) -> str:
    """Generate comparison table for a single configuration."""
    
    # Header
    table = f"\n### {task_name.capitalize()} - {config_name}\n\n"
    
    # Table header
    if format_style == 'github':
        table += "| Model | Iterations | Mean | Std | Seeds | Δ (vs Baseline) | p-value | Cohen's d | Status |\n"
        table += "|-------|-----------|------|-----|-------|----------------|---------|-----------|--------|\n"
    else:
        table += "| Model | Iters | Mean | Std | n | Improvement | p | d | Status |\n"
        table += "|---|---|---|---|---|---|---|---|---|\n"
    
    # Baseline row
    baseline_scores = baseline_df[f'test_{metric}'].values if f'test_{metric}' in baseline_df.columns else baseline_df[metric].values
    baseline_mean = baseline_scores.mean()
    baseline_std = baseline_scores.std()
    baseline_n = len(baseline_scores)
    
    table += f"| **Baseline** | 1 | {format_number(baseline_mean)} | {format_number(baseline_std)} | {baseline_n} | - | - | - | 🥇 |\n"
    
    # PoH variant rows
    for poh_name, poh_df in poh_variants.items():
        poh_scores = poh_df[f'test_{metric}'].values if f'test_{metric}' in poh_df.columns else poh_df[metric].values
        poh_mean = poh_scores.mean()
        poh_std = poh_scores.std()
        poh_n = len(poh_scores)
        
        # Extract iteration count
        if 'iter' in poh_name.lower():
            iters = ''.join(filter(str.isdigit, poh_name.split('iter')[0].split('_')[-1]))
        else:
            iters = '2'
        
        # Statistical tests
        t_stat, p_value = stats.ttest_ind(poh_scores, baseline_scores)
        effect = cohens_d(poh_scores, baseline_scores)
        
        delta = poh_mean - baseline_mean
        pct = (delta / baseline_mean) * 100
        
        # Status
        if p_value < 0.05 and delta > 0 and abs(effect) >= 0.5:
            status = "🏆 **WIN**"
        elif p_value < 0.05 and delta > 0:
            status = "✅ Win"
        elif delta > 0:
            status = "⚠️ Marginal"
        else:
            status = "❌ Worse"
        
        # Effect size interpretation
        if abs(effect) >= 0.8:
            effect_str = f"{effect:.3f} (large)"
        elif abs(effect) >= 0.5:
            effect_str = f"{effect:.3f} (med)"
        elif abs(effect) >= 0.2:
            effect_str = f"{effect:.3f} (small)"
        else:
            effect_str = f"{effect:.3f}"
        
        improvement_str = format_improvement(delta, pct)
        
        table += f"| PoH | {iters} | {format_number(poh_mean)} | {format_number(poh_std)} | {poh_n} | {improvement_str} | {p_value:.4f} | {effect_str} | {status} |\n"
    
    return table


def generate_reproduce_command(task_name: str, config_info: Dict) -> str:
    """Generate reproduce command."""
    config_path = config_info.get('config', 'experiments/configs/<task>/<config>.yaml')
    
    cmd = f"\n**Reproduce:**\n```bash\npython scripts/train.py --task {task_name} --config {config_path}\npython scripts/analyze.py --task {task_name}\n```\n"
    return cmd


def main():
    parser = argparse.ArgumentParser(description='Generate Markdown tables from results')
    parser.add_argument('--registry', type=str, default='experiments/registry.json',
                        help='Path to registry.json')
    parser.add_argument('--task', type=str, default=None,
                        help='Specific task (default: all)')
    parser.add_argument('--format', type=str, default='github', choices=['github', 'simple'],
                        help='Table format style')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file (default: stdout)')
    args = parser.parse_args()

    # Load registry
    with open(args.registry, 'r') as f:
        registry = json.load(f)
    
    # Build output
    output = []
    output.append("# Experiment Results\n")
    output.append("*Auto-generated from experiments/results/*.csv*\n")
    
    # Process each task
    tasks_to_process = [args.task] if args.task else registry['tasks'].keys()
    
    for task_name in tasks_to_process:
        if task_name not in registry['tasks']:
            continue
        
        task_configs = registry['tasks'][task_name]
        metric = registry['metrics'].get(task_name, 'accuracy')
        
        output.append(f"\n## {task_name.capitalize()}\n")
        output.append(f"**Primary Metric:** `{metric.upper()}`\n")
        
        for config_name, config_info in task_configs.items():
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
            
            if baseline_df is None or not poh_variants:
                continue
            
            # Generate table
            table = generate_comparison_table(
                task_name,
                config_name,
                baseline_df,
                poh_variants,
                metric,
                args.format
            )
            output.append(table)
            
            # Add reproduce command
            reproduce = generate_reproduce_command(task_name, config_info)
            output.append(reproduce)
    
    # Add footer
    output.append("\n---\n")
    output.append("\n**Legend:**\n")
    output.append("- 🏆 **WIN**: Statistically significant (p < 0.05), large effect size (|d| ≥ 0.5)\n")
    output.append("- ✅ Win: Statistically significant, positive improvement\n")
    output.append("- ⚠️ Marginal: Positive but not significant\n")
    output.append("- ❌ Worse: Negative improvement\n")
    output.append("\n**Statistical Tests:**\n")
    output.append("- p-value: Welch's t-test (two-tailed)\n")
    output.append("- Cohen's d: Effect size (small: 0.2, medium: 0.5, large: 0.8)\n")
    
    # Output
    result = '\n'.join(output)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(result)
        print(f"✅ Saved tables to: {args.output}")
    else:
        print(result)


if __name__ == '__main__':
    main()

