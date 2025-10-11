"""
Plot sorting experiment results

Author: Eran Ben Artzy
License: Apache 2.0
"""

import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_sample_efficiency(csv_path, output_path):
    """Plot sample efficiency curves."""
    results = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_len = int(row["test_length"])
            if test_len not in results:
                results[test_len] = {
                    "samples": [],
                    "baseline_acc": [],
                    "poh_acc": [],
                    "baseline_perfect": [],
                    "poh_perfect": [],
                }
            results[test_len]["samples"].append(int(row["train_samples"]))
            results[test_len]["baseline_acc"].append(float(row["baseline_acc"]))
            results[test_len]["poh_acc"].append(float(row["poh_acc"]))
            results[test_len]["baseline_perfect"].append(
                float(row["baseline_perfect"])
            )
            results[test_len]["poh_perfect"].append(float(row["poh_perfect"]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for test_len, data in results.items():
        # Element accuracy
        axes[0].plot(
            data["samples"],
            data["baseline_acc"],
            "o-",
            label=f"Baseline (len={test_len})",
            alpha=0.7,
        )
        axes[0].plot(
            data["samples"],
            data["poh_acc"],
            "s-",
            label=f"PoH (len={test_len})",
            alpha=0.7,
        )

        # Perfect sort rate
        axes[1].plot(
            data["samples"],
            data["baseline_perfect"],
            "o-",
            label=f"Baseline (len={test_len})",
            alpha=0.7,
        )
        axes[1].plot(
            data["samples"],
            data["poh_perfect"],
            "s-",
            label=f"PoH (len={test_len})",
            alpha=0.7,
        )

    axes[0].set_xlabel("Training Samples")
    axes[0].set_ylabel("Element Accuracy")
    axes[0].set_title("Sample Efficiency: Element Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale("log")

    axes[1].set_xlabel("Training Samples")
    axes[1].set_ylabel("Perfect Sort Rate")
    axes[1].set_title("Sample Efficiency: Perfect Sorts")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale("log")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved sample efficiency plot to {output_path}")


def plot_length_generalization(csv_path, output_path):
    """Plot length generalization curves."""
    samples = []
    lengths = []
    baseline_acc = []
    poh_acc = []
    baseline_kendall = []
    poh_kendall = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            lengths.append(int(row["test_length"]))
            baseline_acc.append(float(row["baseline_acc"]))
            poh_acc.append(float(row["poh_acc"]))
            baseline_kendall.append(float(row["baseline_kendall"]))
            poh_kendall.append(float(row["poh_kendall"]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(lengths, baseline_acc, "o-", label="Baseline", linewidth=2)
    axes[0].plot(lengths, poh_acc, "s-", label="PoH", linewidth=2)
    axes[0].axvline(12, color="red", linestyle="--", alpha=0.5, label="Train length")
    axes[0].set_xlabel("Test Sequence Length")
    axes[0].set_ylabel("Element Accuracy")
    axes[0].set_title("Length Generalization: Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Kendall-τ
    axes[1].plot(lengths, baseline_kendall, "o-", label="Baseline", linewidth=2)
    axes[1].plot(lengths, poh_kendall, "s-", label="PoH", linewidth=2)
    axes[1].axvline(12, color="red", linestyle="--", alpha=0.5, label="Train length")
    axes[1].set_xlabel("Test Sequence Length")
    axes[1].set_ylabel("Kendall-τ")
    axes[1].set_title("Length Generalization: Kendall-τ")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved length generalization plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot sorting experiment results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="experiments/results",
        help="Results directory",
    )
    parser.add_argument(
        "--output_dir", type=str, default="experiments/plots", help="Output directory"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find latest result files
    sample_eff_files = glob.glob(
        os.path.join(args.results_dir, "sample_efficiency_*.csv")
    )
    length_gen_files = glob.glob(os.path.join(args.results_dir, "length_gen_*.csv"))

    if sample_eff_files:
        latest = max(sample_eff_files, key=os.path.getctime)
        output = os.path.join(args.output_dir, "sample_efficiency.png")
        plot_sample_efficiency(latest, output)

    if length_gen_files:
        latest = max(length_gen_files, key=os.path.getctime)
        output = os.path.join(args.output_dir, "length_generalization.png")
        plot_length_generalization(latest, output)

    print("\n✓ All plots generated")


if __name__ == "__main__":
    main()

