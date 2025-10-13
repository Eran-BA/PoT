#!/usr/bin/env python3
"""Simple results plot generator."""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Simple style
plt.style.use('default')

# Load results
results_dir = Path("examples/synthetic/results")

baseline = pd.read_csv(results_dir / "fair_ab_baseline.csv")
poh = pd.read_csv(results_dir / "fair_ab_pot.csv")

# Calculate means and std
baseline_mean = baseline['test_kendall'].mean()
baseline_std = baseline['test_kendall'].std()
poh_mean = poh['test_kendall'].mean()
poh_std = poh['test_kendall'].std()

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))

models = ['Baseline', 'PoH']
means = [baseline_mean, poh_mean]
stds = [baseline_std, poh_std]

bars = ax.bar(models, means, yerr=stds, capsize=10, 
              color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')

ax.set_ylabel('Kendall-τ (Test)')
ax.set_title('Baseline vs PoH: Partial Observability Sorting (L=12, 50% mask)')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(means) * 1.3)

# Add value labels
for bar, mean, std in zip(bars, means, stds):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.005,
            f'{mean:.3f}±{std:.3f}',
            ha='center', va='bottom', fontweight='bold')

# Save
output_dir = Path("figs")
output_dir.mkdir(exist_ok=True)
plt.tight_layout()
plt.savefig(output_dir / "baseline_vs_poh.png", dpi=150, bbox_inches='tight')
print(f"✅ Plot saved to figs/baseline_vs_poh.png")

# Print summary
print(f"\n📊 Results Summary:")
print(f"   Baseline: {baseline_mean:.4f} ± {baseline_std:.4f}")
print(f"   PoH:      {poh_mean:.4f} ± {poh_std:.4f}")
improvement = ((poh_mean - baseline_mean) / baseline_mean) * 100
print(f"   Improvement: {improvement:+.2f}%")

