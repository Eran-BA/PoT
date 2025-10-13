"""
Unified analyzer for all PoT task results.

Reads experiments/registry.json and generates:
- Per-task result tables
- Leaderboard across tasks
- Statistical comparisons

Usage:
    python scripts/analyze.py
    python scripts/analyze.py --task sorting
    python scripts/analyze.py --output_dir experiments/reports

Author: Eran Ben Artzy
Year: 2025
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from scipy import stats


def load_registry(registry_path: str = "experiments/registry.json") -> dict:
    """Load experiment registry."""
    with open(registry_path, 'r') as f:
        return json.load(f)


def load_results_csv(csv_path: str) -> pd.DataFrame:
    """Load results CSV if it exists."""
    path = Path(csv_path)
    if path.exists():
        return pd.read_csv(path)
    return None


def cohens_d(x1, x2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(x1), len(x2)
    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(x1) - np.mean(x2)) / pooled_std if pooled_std > 0 else 0


def analyze_task(task_name: str, task_configs: dict, metric_name: str) -> dict:
    """Analyze all configurations for a single task."""
    print(f"\n{'='*100}")
    print(f"TASK: {task_name.upper()}")
    print(f"Primary Metric: {metric_name}")
    print(f"{'='*100}\n")

    results = []

    for config_name, config_info in task_configs.items():
        print(f"Configuration: {config_name}")
        print(f"  Description: {config_info.get('description', 'N/A')}")
        print(f"  Config file: {config_info.get('config', 'N/A')}")
        
        # Load baseline
        baseline_path = config_info.get('baseline')
        baseline_df = load_results_csv(baseline_path) if baseline_path else None
        
        # Load PoH variants
        poh_results = {}
        for key, value in config_info.items():
            if key.startswith('hrm_poh') or key == 'poh':
                poh_df = load_results_csv(value) if value else None
                if poh_df is not None:
                    poh_results[key] = poh_df

        # Compare
        if baseline_df is not None:
            baseline_scores = baseline_df[f'test_{metric_name}'].values if f'test_{metric_name}' in baseline_df.columns else baseline_df[metric_name].values
            baseline_mean = baseline_scores.mean()
            baseline_std = baseline_scores.std()
            
            print(f"\n  Baseline: {baseline_mean:.4f} ± {baseline_std:.4f} (n={len(baseline_scores)})")
            
            for poh_name, poh_df in poh_results.items():
                poh_scores = poh_df[f'test_{metric_name}'].values if f'test_{metric_name}' in poh_df.columns else poh_df[metric_name].values
                poh_mean = poh_scores.mean()
                poh_std = poh_scores.std()
                
                # Statistical tests
                t_stat, p_value = stats.ttest_ind(poh_scores, baseline_scores)
                effect = cohens_d(poh_scores, baseline_scores)
                
                delta = poh_mean - baseline_mean
                pct = (delta / baseline_mean) * 100
                
                # Interpret
                sig = "✅" if p_value < 0.05 and delta > 0 else "❌" if p_value < 0.05 else "⚠️"
                effect_str = f"d={effect:.3f}"
                if abs(effect) >= 0.8:
                    effect_str += " (LARGE)"
                elif abs(effect) >= 0.5:
                    effect_str += " (medium)"
                elif abs(effect) >= 0.2:
                    effect_str += " (small)"
                
                print(f"  {poh_name}: {poh_mean:.4f} ± {poh_std:.4f} (n={len(poh_scores)})")
                print(f"    Δ = {delta:+.4f} ({pct:+.1f}%), p={p_value:.4f}, {effect_str} {sig}")
                
                results.append({
                    'task': task_name,
                    'config': config_name,
                    'model': poh_name,
                    'metric': metric_name,
                    'baseline_mean': baseline_mean,
                    'baseline_std': baseline_std,
                    'poh_mean': poh_mean,
                    'poh_std': poh_std,
                    'delta': delta,
                    'pct_improvement': pct,
                    'p_value': p_value,
                    'cohens_d': effect,
                    'significant': p_value < 0.05 and delta > 0
                })
        else:
            print("  ⚠️ No baseline results found")
        
        print()

    return results


def generate_leaderboard(all_results: List[dict], output_path: str):
    """Generate cross-task leaderboard."""
    df = pd.DataFrame(all_results)
    
    print("\n" + "="*100)
    print("LEADERBOARD: Best PoH Configuration Per Task")
    print("="*100 + "\n")
    
    # Group by task and find best
    for task in df['task'].unique():
        task_df = df[df['task'] == task]
        best = task_df.loc[task_df['poh_mean'].idxmax()]
        
        print(f"{best['task']} ({best['config']}):")
        print(f"  Best Model: {best['model']}")
        print(f"  Performance: {best['poh_mean']:.4f} ± {best['poh_std']:.4f}")
        print(f"  vs Baseline: {best['delta']:+.4f} ({best['pct_improvement']:+.1f}%)")
        print(f"  Statistical: p={best['p_value']:.4f}, d={best['cohens_d']:.3f}")
        print(f"  Status: {'🏆 SIGNIFICANT WIN' if best['significant'] else '⚠️ Not significant'}")
        print()
    
    # Save CSV
    output_file = Path(output_path) / "leaderboard.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"✅ Saved leaderboard to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze PoT experiment results')
    parser.add_argument('--registry', type=str, default='experiments/registry.json',
                        help='Path to registry.json')
    parser.add_argument('--task', type=str, default=None,
                        help='Analyze specific task only')
    parser.add_argument('--output_dir', type=str, default='experiments/reports',
                        help='Output directory for reports')
    args = parser.parse_args()

    # Load registry
    registry = load_registry(args.registry)
    
    all_results = []
    
    # Analyze each task
    tasks_to_analyze = [args.task] if args.task else registry['tasks'].keys()
    
    for task_name in tasks_to_analyze:
        if task_name not in registry['tasks']:
            print(f"⚠️ Task '{task_name}' not found in registry")
            continue
        
        task_configs = registry['tasks'][task_name]
        metric_name = registry['metrics'].get(task_name, 'accuracy')
        
        task_results = analyze_task(task_name, task_configs, metric_name)
        all_results.extend(task_results)
    
    # Generate leaderboard
    if all_results:
        generate_leaderboard(all_results, args.output_dir)
    else:
        print("\n⚠️ No results found to analyze")


if __name__ == '__main__':
    main()

