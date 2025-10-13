#!/usr/bin/env python3
"""
Visualize inner-loop vs outer-loop dynamics.

Generates:
1. Inner-loop convergence per epoch (does loss drop across inner steps?)
2. Outer learning curve (loss at last inner step vs global step)
3. UAS probe curve (optional sanity check)
4. Attention entropy over time (optional)
5. Halting statistics (if ACT enabled)

Usage:
    python scripts/plot_inner_vs_outer.py --csv experiments/results/<run_id>/innerloop_log.csv
    python scripts/plot_inner_vs_outer.py --csv innerloop.csv --epochs 1,5,10,20,40

Author: Eran Ben Artzy
Year: 2025
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


def load_rows(path):
    """Load inner-loop log CSV."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for d in reader:
            rows.append({
                "epoch": int(d["epoch"]),
                "global_step": int(d["global_step"]),
                "inner_step": int(d["inner_step"]),
                "loss": float(d["loss"]),
                "grad_norm": float(d["grad_norm"]),
                "uas_probe": float(d["uas_probe"]) if d.get("uas_probe") and d["uas_probe"] != "None" else None,
                "attn_entropy_mean": float(d["attn_entropy_mean"]) if d.get("attn_entropy_mean") and d["attn_entropy_mean"] != "None" else None,
                "halted_frac": float(d["halted_frac"]) if d.get("halted_frac") and d["halted_frac"] != "None" else None,
                "ms_forward": float(d["ms_forward"]) if d.get("ms_forward") and d["ms_forward"] != "None" else None,
            })
    return rows


def plot_inner_convergence(rows, epochs_to_plot, output_dir):
    """Plot inner-step loss curves for selected epochs."""
    by_epoch = defaultdict(list)
    for r in rows:
        by_epoch[r["epoch"]].append(r)
    
    inner_max = max(r["inner_step"] for r in rows)
    xs = list(range(1, inner_max + 1))
    
    # Plot selected epochs
    for epoch in sorted(by_epoch.keys()):
        if epochs_to_plot and epoch not in epochs_to_plot:
            continue
        
        ep_rows = by_epoch[epoch]
        sums = defaultdict(float)
        counts = defaultdict(int)
        
        for r in ep_rows:
            sums[r["inner_step"]] += r["loss"]
            counts[r["inner_step"]] += 1
        
        ys = [sums[i] / max(1, counts[i]) for i in xs]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(xs, ys, marker='o', markersize=8, linewidth=2, color='#e74c3c')
        
        # Show marginal improvements
        for i in range(len(xs) - 1):
            delta = ys[i+1] - ys[i]
            pct = (delta / ys[i]) * 100 if ys[i] > 0 else 0
            ax.text(xs[i] + 0.5, (ys[i] + ys[i+1]) / 2, 
                   f'{delta:.3f}\n({pct:.1f}%)', 
                   ha='center', va='center', fontsize=9, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_title(f'Inner-Loop Convergence (Epoch {epoch})', fontweight='bold', fontsize=14)
        ax.set_xlabel('Inner Step', fontweight='bold')
        ax.set_ylabel('Loss (averaged over batches)', fontweight='bold')
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.3)
        
        # Highlight diminishing returns
        if len(ys) > 1:
            first_drop = ys[0] - ys[1]
            last_drop = ys[-2] - ys[-1] if len(ys) > 2 else 0
            efficiency = (last_drop / first_drop * 100) if first_drop > 0 else 0
            ax.text(0.05, 0.95, 
                   f'1st→2nd: {first_drop:.4f}\n{len(ys)-1}→{len(ys)}: {last_drop:.4f}\nEfficiency: {efficiency:.1f}%',
                   transform=ax.transAxes, va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        output_file = output_dir / f'inner_convergence_epoch{epoch}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_file}")


def plot_outer_curve(rows, output_dir):
    """Plot outer learning curve (loss at last inner step)."""
    inner_max = max(r["inner_step"] for r in rows)
    last_inner = [r for r in rows if r["inner_step"] == inner_max]
    
    steps = [r["global_step"] for r in last_inner]
    losses = [r["loss"] for r in last_inner]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, losses, linewidth=2, color='#3498db')
    
    ax.set_title('Outer Learning Curve (Loss @ Last Inner Step)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Global Step', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'outer_learning_curve.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_file}")


def plot_uas_probe(rows, output_dir):
    """Plot UAS probe over training."""
    inner_max = max(r["inner_step"] for r in rows)
    last_inner = [r for r in rows if r["inner_step"] == inner_max]
    
    uas_pts = [(r["global_step"], r["uas_probe"]) for r in last_inner if r["uas_probe"] is not None]
    
    if not uas_pts:
        print("⚠️  No UAS probe data found")
        return
    
    xs, ys = zip(*uas_pts)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, ys, linewidth=2, color='#2ecc71', marker='o', markersize=4)
    
    ax.set_title('UAS Probe Over Training (Tiny Batch)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Global Step', fontweight='bold')
    ax.set_ylabel('UAS', fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'uas_probe_curve.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_file}")


def plot_attention_entropy(rows, output_dir):
    """Plot attention entropy over time."""
    entropy_pts = [(r["global_step"], r["attn_entropy_mean"]) 
                   for r in rows if r["attn_entropy_mean"] is not None]
    
    if not entropy_pts:
        print("⚠️  No attention entropy data found")
        return
    
    xs, ys = zip(*entropy_pts)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, ys, linewidth=1, alpha=0.7, color='#9b59b6')
    
    # Add moving average
    window = min(50, len(xs) // 10)
    if window > 1:
        import numpy as np
        ys_smooth = np.convolve(ys, np.ones(window)/window, mode='valid')
        xs_smooth = xs[window-1:]
        ax.plot(xs_smooth, ys_smooth, linewidth=2, color='#8e44ad', label=f'MA({window})')
        ax.legend()
    
    ax.set_title('Attention Entropy Over Training', fontweight='bold', fontsize=14)
    ax.set_xlabel('Global Step', fontweight='bold')
    ax.set_ylabel('Mean Attention Entropy', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'attention_entropy.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_file}")


def plot_timing_analysis(rows, output_dir):
    """Plot forward pass timing per inner step."""
    by_inner = defaultdict(list)
    for r in rows:
        if r["ms_forward"] is not None:
            by_inner[r["inner_step"]].append(r["ms_forward"])
    
    if not by_inner:
        print("⚠️  No timing data found")
        return
    
    inner_steps = sorted(by_inner.keys())
    means = [sum(by_inner[i]) / len(by_inner[i]) for i in inner_steps]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(inner_steps, means, color='#e67e22', alpha=0.8, edgecolor='black')
    
    for i, (step, mean) in enumerate(zip(inner_steps, means)):
        ax.text(step, mean + max(means) * 0.02, f'{mean:.1f}ms', 
               ha='center', va='bottom', fontweight='bold')
    
    ax.set_title('Forward Pass Timing Per Inner Step', fontweight='bold', fontsize=14)
    ax.set_xlabel('Inner Step', fontweight='bold')
    ax.set_ylabel('Mean Forward Time (ms)', fontweight='bold')
    ax.set_xticks(inner_steps)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'timing_per_inner_step.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Plot inner-loop vs outer-loop dynamics')
    parser.add_argument('--csv', required=True, help='Path to innerloop_log.csv')
    parser.add_argument('--epochs', type=str, default=None, 
                       help='Comma-separated epochs to plot (e.g., "1,5,10,20,40")')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: same as CSV)')
    args = parser.parse_args()
    
    # Load data
    print(f"\n📊 Loading inner-loop log: {args.csv}")
    rows = load_rows(args.csv)
    print(f"   Loaded {len(rows)} rows")
    
    # Determine output directory
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse epochs to plot
    epochs_to_plot = None
    if args.epochs:
        epochs_to_plot = set(int(e.strip()) for e in args.epochs.split(','))
        print(f"   Plotting epochs: {sorted(epochs_to_plot)}")
    
    # Generate plots
    print("\n🎨 Generating plots...")
    print()
    
    plot_inner_convergence(rows, epochs_to_plot, output_dir)
    plot_outer_curve(rows, output_dir)
    plot_uas_probe(rows, output_dir)
    plot_attention_entropy(rows, output_dir)
    plot_timing_analysis(rows, output_dir)
    
    print()
    print(f"✅ All plots saved to: {output_dir}/")
    print()


if __name__ == "__main__":
    main()

