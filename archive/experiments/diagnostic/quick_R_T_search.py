#!/usr/bin/env python3
"""
Quick targeted R & T search based on diagnostic findings.

Based on diagnostic results:
- Best LR: 1e-3 (already updated in main script)
- Best R from quick test: 8
- Now search around R=8 with different T values

This will run much faster than the full grid.
"""

import subprocess
import sys

def main():
    print("="*60)
    print("QUICK TARGETED R & T SEARCH")
    print("="*60)
    print("\nBased on diagnostic findings:")
    print("  ✅ LR = 1e-3 (optimal)")
    print("  ✅ R = 8 showed best results (51.3% in quick test)")
    print("\nNow testing R × T combinations around R=8:")
    print("  R values: [6, 8, 12]")
    print("  T values: [2, 4, 8]")
    print("  Seeds: [42, 43] (2 seeds for faster iteration)")
    print("  Total: 18 experiments")
    print("="*60)
    
    # Run the search
    cmd = [
        "python", "experiments/find_optimal_R_T_real_nli.py",
        "--R-values", "6", "8", "12",
        "--T-values", "2", "4", "8",
        "--seeds", "42", "43",
        "--max-steps", "500",  # Faster for initial search
        "--max-train-samples", "5000",
        "--max-val-samples", "1000",
        "--batch-size", "32"
    ]
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    
    # Set PYTHONPATH
    import os
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()
    
    result = subprocess.run(cmd, env=env)
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("✅ Search complete!")
        print("="*60)
        print("\nCheck results:")
        print("  experiments/results/R_T_search_real_nli/grid_search_results.csv")
        print("  experiments/results/R_T_search_real_nli/grid_search_results_summary.txt")
    else:
        print(f"\n❌ Search failed with exit code {result.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()

