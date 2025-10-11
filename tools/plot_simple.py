#!/usr/bin/env python3
"""
Simple plotting script for quick UAS vs iterations visualization

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description="Quick plot from CSV logs")
    ap.add_argument("csv", type=str, help="results.csv from training runs")
    ap.add_argument("--out", type=str, default="plot.png", help="Output plot file")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    plt.figure(figsize=(10, 6))

    try:
        # UAS vs inner iters (PoH)
        if "poh_mean_inner_iters" in df.columns and "poh_dev_uas" in df.columns:
            # Convert to numeric, handling any non-numeric values
            iters = pd.to_numeric(df["poh_mean_inner_iters"], errors="coerce")
            uas = pd.to_numeric(df["poh_dev_uas"], errors="coerce")

            # Filter out NaN values
            valid = ~(iters.isna() | uas.isna())
            if valid.any():
                plt.scatter(iters[valid], uas[valid], label="PoH", s=50, alpha=0.7, color="#e74c3c")
    except Exception as e:
        print(f"Note: Could not plot PoH data: {e}")

    try:
        # Plot baseline as horizontal reference
        if "baseline_dev_uas" in df.columns:
            bl_uas = pd.to_numeric(df["baseline_dev_uas"], errors="coerce")
            bl_mean = bl_uas.mean()
            if not pd.isna(bl_mean):
                plt.axhline(
                    bl_mean,
                    linestyle="--",
                    label=f"Baseline mean UAS={bl_mean:.3f}",
                    color="#3498db",
                    linewidth=2,
                )
    except Exception as e:
        print(f"Note: Could not plot baseline: {e}")

    plt.xlabel("Mean Inner Iterations (PoH)", fontsize=12)
    plt.ylabel("Dev UAS", fontsize=12)
    plt.title("PoH: Accuracy vs Computational Depth", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"✓ Saved {args.out}")

    # Print summary statistics
    print(f"\n{'='*60}")
    print("Summary Statistics:")
    print(f"{'='*60}")
    try:
        if "baseline_dev_uas" in df.columns:
            bl = pd.to_numeric(df["baseline_dev_uas"], errors="coerce")
            print(f"Baseline UAS: {bl.mean():.4f} ± {bl.std():.4f}")
        if "poh_dev_uas" in df.columns:
            poh = pd.to_numeric(df["poh_dev_uas"], errors="coerce")
            print(f"PoH UAS:      {poh.mean():.4f} ± {poh.std():.4f}")
    except Exception as e:
        print(f"Could not compute statistics: {e}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
