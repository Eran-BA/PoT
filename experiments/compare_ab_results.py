"""
Compare A/B results and generate publication table

Author: Eran Ben Artzy
License: Apache 2.0
"""

import argparse
import json


def load_summary(path):
    """Load summary JSON."""
    with open(path) as f:
        return json.load(f)


def format_metric(mean, ci):
    """Format metric with CI."""
    return f"{mean:.3f} ± {ci:.3f}"


def compute_delta(pot_mean, baseline_mean):
    """Compute improvement."""
    delta = pot_mean - baseline_mean
    rel = (delta / baseline_mean * 100) if baseline_mean != 0 else 0.0
    return delta, rel


def main():
    parser = argparse.ArgumentParser(description="Compare A/B results")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline summary JSON")
    parser.add_argument("--pot", type=str, required=True, help="PoT summary JSON")
    args = parser.parse_args()

    baseline = load_summary(args.baseline)
    pot = load_summary(args.pot)

    print("=" * 80)
    print("FAIR A/B COMPARISON RESULTS")
    print("=" * 80)
    print(f"\nTask: {pot['task']}")
    print(f"Array length: {pot['array_len']}, Mask rate: {pot['mask_rate']}")
    print(f"Seeds: {pot['n_seeds']}")
    print()

    # Kendall-τ
    delta_k, rel_k = compute_delta(pot["kendall_mean"], baseline["kendall_mean"])
    print(f"Kendall-τ:")
    print(f"  Baseline: {format_metric(baseline['kendall_mean'], baseline['kendall_ci'])}")
    print(f"  PoT:      {format_metric(pot['kendall_mean'], pot['kendall_ci'])}")
    print(f"  Δ:        {delta_k:+.3f} ({rel_k:+.1f}% relative)")
    print()

    # Perfect sort
    delta_p, rel_p = compute_delta(pot["perfect_mean"], baseline["perfect_mean"])
    print(f"Perfect Sort %:")
    print(f"  Baseline: {format_metric(baseline['perfect_mean']*100, baseline['perfect_ci']*100)}%")
    print(f"  PoT:      {format_metric(pot['perfect_mean']*100, pot['perfect_ci']*100)}%")
    print(f"  Δ:        {delta_p*100:+.1f}% points")
    print()

    # Hamming distance
    delta_h, rel_h = compute_delta(pot["hamming_mean"], baseline["hamming_mean"])
    print(f"Hamming Distance:")
    print(f"  Baseline: {format_metric(baseline['hamming_mean'], baseline['hamming_ci'])}")
    print(f"  PoT:      {format_metric(pot['hamming_mean'], pot['hamming_ci'])}")
    print(f"  Δ:        {delta_h:+.3f} ({'better' if delta_h < 0 else 'worse'})")
    print()

    # Publication table
    print("=" * 80)
    print("PUBLICATION TABLE")
    print("=" * 80)
    print()
    print("| Task             | Length | Mask | Metric              | Baseline          | PoT (iters=2,last) |      Δ |")
    print("| ---------------- | -----: | ---: | ------------------- | ----------------: | -----------------: | -----: |")
    print(f"| Partial-obs sort |     {pot['array_len']:2d} |  {int(pot['mask_rate']*100):2d}% "
          f"| Kendall-τ (mean±CI) | {format_metric(baseline['kendall_mean'], baseline['kendall_ci']):>17s} "
          f"| {format_metric(pot['kendall_mean'], pot['kendall_ci']):>18s} | {delta_k:+.3f} |")
    print(f"|                  |        |      | Perfect-sort %      | "
          f"{baseline['perfect_mean']*100:>5.1f} ± {baseline['perfect_ci']*100:>4.1f}     "
          f"| {pot['perfect_mean']*100:>6.1f} ± {pot['perfect_ci']*100:>4.1f}      | {delta_p*100:+.1f}% |")
    print(f"|                  |        |      | Hamming distance    | {format_metric(baseline['hamming_mean'], baseline['hamming_ci']):>17s} "
          f"| {format_metric(pot['hamming_mean'], pot['hamming_ci']):>18s} | {delta_h:+.3f} |")
    print()

    # Statistical significance check (rough)
    print("=" * 80)
    print("STATISTICAL NOTES")
    print("=" * 80)
    print()
    
    # Check if CIs overlap
    baseline_lower = baseline["kendall_mean"] - baseline["kendall_ci"]
    baseline_upper = baseline["kendall_mean"] + baseline["kendall_ci"]
    pot_lower = pot["kendall_mean"] - pot["kendall_ci"]
    pot_upper = pot["kendall_mean"] + pot["kendall_ci"]
    
    if pot_lower > baseline_upper:
        print("✅ PoT Kendall-τ CIs do NOT overlap → likely significant (p<0.05)")
    elif baseline_lower > pot_upper:
        print("⚠️  Baseline Kendall-τ CIs do NOT overlap → baseline better")
    else:
        print("⚖️  Kendall-τ CIs overlap → difference may not be significant")
    
    print(f"\nFor publication: report as mean ± 95% CI over n={pot['n_seeds']} seeds")
    print(f"Recommend: t-test or bootstrap for formal significance testing")


if __name__ == "__main__":
    main()

