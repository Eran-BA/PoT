#!/bin/bash
# Fair A/B Comparison: Baseline vs PoT on Partial Observability
# 
# Protocol:
# - Same data/task/split (controlled by seeds)
# - Parameter-matched models
# - Identical training (40 epochs, LR=3e-4, batch=64, clip=1.0)
# - PoT: max_inner_iters=2, grad_mode=last (HRM-style)
# - 5 seeds for statistical robustness
#
# Author: Eran Ben Artzy
# License: Apache 2.0

set -e

# Configuration
ARRAY_LEN=12
MASK_RATE=0.5
TRAIN_SAMPLES=1000
DEV_SAMPLES=300
TEST_SAMPLES=300
EPOCHS=40
BATCH_SIZE=64
LR=3e-4
SEEDS="1 2 3 4 5"
D_MODEL=128
N_HEADS=4
MAX_INNER_ITERS=2

# Output directory
mkdir -p experiments/results

echo "================================================================================"
echo "FAIR A/B COMPARISON: Baseline vs PoT"
echo "================================================================================"
echo "Task: Partial Observability Sorting"
echo "Array length: $ARRAY_LEN, Mask rate: $MASK_RATE"
echo "Training: $TRAIN_SAMPLES samples, $EPOCHS epochs, LR=$LR"
echo "Seeds: $SEEDS"
echo ""

# Run Baseline
echo "Running BASELINE..."
python experiments/fair_ab_comparison.py \
  --model baseline \
  --array_len $ARRAY_LEN \
  --mask_rate $MASK_RATE \
  --train_samples $TRAIN_SAMPLES \
  --dev_samples $DEV_SAMPLES \
  --test_samples $TEST_SAMPLES \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --d_model $D_MODEL \
  --n_heads $N_HEADS \
  --seeds $SEEDS \
  --output_csv experiments/results/fair_ab_baseline.csv

echo ""
echo "Running PoT..."
python experiments/fair_ab_comparison.py \
  --model pot \
  --array_len $ARRAY_LEN \
  --mask_rate $MASK_RATE \
  --train_samples $TRAIN_SAMPLES \
  --dev_samples $DEV_SAMPLES \
  --test_samples $TEST_SAMPLES \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --d_model $D_MODEL \
  --n_heads $N_HEADS \
  --max_inner_iters $MAX_INNER_ITERS \
  --seeds $SEEDS \
  --output_csv experiments/results/fair_ab_pot.csv

echo ""
echo "================================================================================"
echo "COMPARISON"
echo "================================================================================"
python experiments/compare_ab_results.py \
  --baseline experiments/results/fair_ab_baseline_summary.json \
  --pot experiments/results/fair_ab_pot_summary.json

echo ""
echo "✓ Fair A/B comparison complete!"
echo "Results saved to experiments/results/"

