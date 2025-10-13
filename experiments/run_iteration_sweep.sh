#!/bin/bash
#
# Comprehensive Iteration Sweep: Find the Plateau Point
#
# Tests: 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20 iterations
# Seeds: 3 runs each for statistical significance
# Goal: Find exact point where performance plateaus
#
# Author: Eran Ben Artzy
# Year: 2025

set -e

echo "================================================================================"
echo "COMPREHENSIVE ITERATION SWEEP"
echo "================================================================================"
echo ""
echo "Purpose: Find the exact iteration count where performance plateaus"
echo ""
echo "Configurations:"
echo "  - Iterations: 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20"
echo "  - Seeds: 3 runs each"
echo "  - Total: 1 baseline + 11 iteration counts × 3 seeds = 34 experiments"
echo ""
echo "Tasks to test:"
echo "  1. Length 12 (easy)"
echo "  2. Length 16 (medium)"
echo "  3. Length 20 (hard)"
echo ""
echo "================================================================================"
echo ""

# ================================================================
# Length 12 (Easy Task)
# ================================================================

echo "📊 TASK 1: Length 12 (Easy)"
echo "================================================================"
echo ""

PYTHONPATH=. python experiments/iteration_plateau_analysis.py \
  --array_len 12 \
  --mask_rate 0.5 \
  --train_samples 1000 \
  --dev_samples 200 \
  --test_samples 500 \
  --epochs 40 \
  --batch_size 64 \
  --lr 3e-4 \
  --seeds 1 2 3 \
  --iterations 1 2 3 4 5 6 8 10 12 16 20 \
  --output_csv experiments/results/iteration_sweep_len12.csv

echo ""
echo "✓ Length 12 complete"
echo ""

# ================================================================
# Length 16 (Medium Task)
# ================================================================

echo "📊 TASK 2: Length 16 (Medium)"
echo "================================================================"
echo ""

PYTHONPATH=. python experiments/iteration_plateau_analysis.py \
  --array_len 16 \
  --mask_rate 0.5 \
  --train_samples 1000 \
  --dev_samples 200 \
  --test_samples 500 \
  --epochs 50 \
  --batch_size 32 \
  --lr 3e-4 \
  --seeds 1 2 3 \
  --iterations 1 2 3 4 5 6 8 10 12 16 20 \
  --output_csv experiments/results/iteration_sweep_len16.csv

echo ""
echo "✓ Length 16 complete"
echo ""

# ================================================================
# Length 20 (Hard Task)
# ================================================================

echo "📊 TASK 3: Length 20 (Hard)"
echo "================================================================"
echo ""

PYTHONPATH=. python experiments/iteration_plateau_analysis.py \
  --array_len 20 \
  --mask_rate 0.5 \
  --train_samples 1000 \
  --dev_samples 200 \
  --test_samples 500 \
  --epochs 50 \
  --batch_size 32 \
  --lr 3e-4 \
  --seeds 1 2 3 \
  --iterations 1 2 3 4 5 6 8 10 12 16 20 \
  --output_csv experiments/results/iteration_sweep_len20.csv

echo ""
echo "✓ Length 20 complete"
echo ""

# ================================================================
# Summary
# ================================================================

echo "================================================================================"
echo "ALL ITERATION SWEEPS COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved:"
echo "  - experiments/results/iteration_sweep_len12.csv"
echo "  - experiments/results/iteration_sweep_len16.csv"
echo "  - experiments/results/iteration_sweep_len20.csv"
echo ""
echo "Total experiments run: 3 tasks × 34 configs = 102 experiments"
echo ""
echo "Next steps:"
echo "  1. Plot convergence curves: python experiments/plot_iteration_curves.py"
echo "  2. Find plateau points for each task difficulty"
echo "  3. Analyze task-specific patterns"
echo ""
echo "Expected findings:"
echo "  - Easy tasks (len=12): Plateau at 2-4 iterations"
echo "  - Medium tasks (len=16): Plateau at 6-8 iterations"
echo "  - Hard tasks (len=20): Plateau at 10-12 iterations"
echo "  - Diminishing returns after plateau"
echo ""

