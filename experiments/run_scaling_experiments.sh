#!/bin/bash
#
# Model Scaling Experiments: Test PoH across model sizes
#
# Hypothesis: Larger models benefit more from PoH
# - More heads = more specialization opportunities
# - Larger capacity = better routing patterns
#
# Author: Eran Ben Artzy
# Year: 2025

set -e

echo "================================================================================"
echo "MODEL SCALING EXPERIMENTS"
echo "================================================================================"
echo ""
echo "Testing PoH across 5 model sizes:"
echo "  - Tiny:   64-dim,  2 heads    (~15k params)"
echo "  - Small:  128-dim, 4 heads    (~100k params)"
echo "  - Medium: 256-dim, 8 heads    (~800k params)"
echo "  - Large:  512-dim, 8 heads    (~3M params)"
echo "  - XLarge: 768-dim, 12 heads   (~10M params)"
echo ""
echo "For each size:"
echo "  - Baseline (single-pass)"
echo "  - PoH (12 iterations)"
echo "  - 3 seeds each"
echo ""
echo "Total: 5 sizes × 2 models × 3 seeds = 30 experiments"
echo "================================================================================"
echo ""

# Create results directory
mkdir -p experiments/results

# ================================================================
# Quick Test (Small + Medium only, faster)
# ================================================================

echo "📊 QUICK TEST: Small & Medium Models"
echo "================================================================"
echo ""

PYTHONPATH=. python experiments/model_scaling_experiments.py \
  --array_len 20 \
  --mask_rate 0.5 \
  --train_samples 1000 \
  --dev_samples 200 \
  --test_samples 500 \
  --epochs 50 \
  --batch_size 32 \
  --lr 3e-4 \
  --sizes small medium \
  --max_inner_iters 12 \
  --seeds 1 2 3 \
  --output_csv experiments/results/scaling_quick.csv

echo ""
echo "✓ Quick test complete"
echo ""

# ================================================================
# Full Test (All 5 sizes)
# ================================================================

echo "📊 FULL TEST: All Model Sizes"
echo "================================================================"
echo ""

PYTHONPATH=. python experiments/model_scaling_experiments.py \
  --array_len 20 \
  --mask_rate 0.5 \
  --train_samples 1000 \
  --dev_samples 200 \
  --test_samples 500 \
  --epochs 50 \
  --batch_size 32 \
  --lr 3e-4 \
  --sizes tiny small medium large xlarge \
  --max_inner_iters 12 \
  --seeds 1 2 3 \
  --output_csv experiments/results/scaling_full.csv

echo ""
echo "✓ Full test complete"
echo ""

# ================================================================
# Scaling Test on Easy Task (Length 12)
# ================================================================

echo "📊 EASY TASK: Length 12"
echo "================================================================"
echo ""

PYTHONPATH=. python experiments/model_scaling_experiments.py \
  --array_len 12 \
  --mask_rate 0.5 \
  --train_samples 1000 \
  --epochs 40 \
  --batch_size 64 \
  --lr 3e-4 \
  --sizes small medium large \
  --max_inner_iters 12 \
  --seeds 1 2 3 \
  --output_csv experiments/results/scaling_len12.csv

echo ""
echo "✓ Easy task scaling complete"
echo ""

# ================================================================
# Summary
# ================================================================

echo "================================================================================"
echo "ALL SCALING EXPERIMENTS COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved:"
echo "  - experiments/results/scaling_quick.csv (quick test)"
echo "  - experiments/results/scaling_full.csv (all sizes)"
echo "  - experiments/results/scaling_len12.csv (easy task)"
echo ""
echo "Next steps:"
echo "  1. Plot scaling curves: python experiments/plot_scaling_curves.py"
echo "  2. Analyze correlation: Does PoH benefit increase with model size?"
echo "  3. Find optimal size-iteration tradeoff"
echo ""
echo "Expected findings:"
echo "  - Tiny models: Minimal PoH benefit (too small)"
echo "  - Small/Medium: Moderate PoH benefit (~10-20%)"
echo "  - Large/XLarge: Maximum PoH benefit (~20-30%)"
echo "  - Positive correlation: larger → more benefit"
echo ""

