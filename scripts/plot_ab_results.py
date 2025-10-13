#!/usr/bin/env python3
"""
Plot A/B comparison results for Baseline GPT vs PoH-GPT.

Reads experiments/results/lm/ab_results.csv and generates side-by-side
bar charts for perplexity and training time.

Author: Eran Ben Artzy
Year: 2025
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

def plot_ab_results():
    """Generate side-by-side comparison plots."""
    
    # Load results
    results_path = Path("experiments/results/lm/ab_results.csv")
    if not results_path.exists():
        print(f"❌ No results found at {results_path}")
        print("   Run: python experiments/fair_ab_lm.py first")
        return
    
    df = pd.read_csv(results_path)
    
    # Get latest run
    latest = df.groupby('model').last().reset_index()
    
    models = latest['model'].tolist()
    ppls = latest['perplexity'].astype(float).tolist()
    times = latest['time_min'].astype(float).tolist()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['#3498db', '#e74c3c']  # Blue for baseline, red for PoH
    
    # Plot 1: Perplexity
    bars1 = ax1.bar(models, ppls, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Perplexity (lower is better)', fontweight='bold')
    ax1.set_title('Validation Perplexity Comparison', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, max(ppls) * 1.2)
    
    # Add value labels
    for bar, ppl in zip(bars1, ppls):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{ppl:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Plot 2: Training time
    bars2 = ax2.bar(models, times, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Training Time (minutes)', fontweight='bold')
    ax2.set_title('Training Time Comparison', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, max(times) * 1.2)
    
    # Add value labels
    for bar, t in zip(bars2, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{t:.1f}m',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.suptitle('Baseline GPT vs PoH-GPT: Language Modeling', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    output_dir = Path("figs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "ab_lm_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    print(f"✅ Plot saved to {output_path}")
    
    # Print summary
    baseline_ppl = ppls[0]
    poh_ppl = ppls[1] if len(ppls) > 1 else baseline_ppl
    improvement = ((baseline_ppl - poh_ppl) / baseline_ppl) * 100
    
    print(f"\n📊 Latest Results:")
    print(f"   Baseline: {baseline_ppl:.2f} perplexity")
    print(f"   PoH:      {poh_ppl:.2f} perplexity")
    print(f"   Improvement: {improvement:+.2f}%")

if __name__ == "__main__":
    plot_ab_results()

