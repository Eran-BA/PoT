#!/bin/bash
#
# Systematic Experiment Grid for Improved PoH Sorting
#
# Runs the exact experiments requested by the user:
# 1. Baseline vs HRM (ranking loss)
# 2. Iteration sweep
# 3. HRM period (T) sweep
# 4. Entropy schedule sweep
#
# Author: Eran Ben Artzy
# Year: 2025

set -e

# Configuration
MASK_RATE=0.5
ARRAY_LEN=20
TRAIN_SAMPLES=1000
EPOCHS=100
BATCH_SIZE=32
SEEDS="1 2 3 4 5"

RESULTS_DIR="experiments/results_improved"
mkdir -p $RESULTS_DIR

echo "================================================================================"
echo "IMPROVED POH EXPERIMENTS"
echo "================================================================================"
echo ""
echo "Task: Partial observability sorting"
echo "Mask rate: ${MASK_RATE} (50%)"
echo "Array length: ${ARRAY_LEN}"
echo "Training: ${TRAIN_SAMPLES} samples, ${EPOCHS} epochs"
echo "Seeds: ${SEEDS}"
echo ""
echo "Improvements:"
echo "  ✓ Mask-aware Kendall-τ (only observable pairs)"
echo "  ✓ RankNet pairwise ranking loss"
echo "  ✓ Deep supervision (all iterations)"
echo "  ✓ Temperature scheduling (2.0 → 0.8)"
echo "  ✓ Entropy regularization with decay"
echo "  ✓ Two optimizers (encoder 3e-4, controller 1e-4)"
echo "  ✓ Differential clipping (encoder 1.0, controller 0.5)"
echo "  ✓ Controller warm-up (5 epochs)"
echo ""
echo "================================================================================"
echo ""

# ================================================================
# EXPERIMENT 1: Baseline vs HRM (with ranking loss)
# ================================================================

echo "📊 EXPERIMENT 1: Baseline vs HRM (RankNet loss)"
echo "================================================================"
echo ""

echo "  Running Baseline (no PoH)..."
python experiments/improved_ab_comparison.py \
  --model baseline \
  --array_len $ARRAY_LEN \
  --mask_rate $MASK_RATE \
  --train_samples $TRAIN_SAMPLES \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --seeds $SEEDS \
  --output_csv $RESULTS_DIR/baseline_ranknet.csv

echo ""
echo "  Running HRM (12 iters, T=4, RankNet)..."
python experiments/improved_ab_comparison.py \
  --model pot \
  --max_inner_iters 12 \
  --hrm_period 4 \
  --array_len $ARRAY_LEN \
  --mask_rate $MASK_RATE \
  --train_samples $TRAIN_SAMPLES \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --seeds $SEEDS \
  --output_csv $RESULTS_DIR/hrm_12iters_T4_ranknet.csv

echo ""
echo "✓ Experiment 1 complete"
echo ""

# ================================================================
# EXPERIMENT 2: Iteration Sweep (HRM, T=4)
# ================================================================

echo "📊 EXPERIMENT 2: Iteration Sweep"
echo "================================================================"
echo ""

for ITERS in 6 8 12 16; do
  echo "  Running HRM with ${ITERS} iterations (T=4)..."
  python experiments/improved_ab_comparison.py \
    --model pot \
    --max_inner_iters $ITERS \
    --hrm_period 4 \
    --array_len $ARRAY_LEN \
    --mask_rate $MASK_RATE \
    --train_samples $TRAIN_SAMPLES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --seeds $SEEDS \
    --output_csv $RESULTS_DIR/hrm_${ITERS}iters_T4.csv
  echo ""
done

echo "✓ Experiment 2 complete"
echo ""

# ================================================================
# EXPERIMENT 3: HRM Period (T) Sweep (12 iters)
# ================================================================

echo "📊 EXPERIMENT 3: HRM Period (T) Sweep"
echo "================================================================"
echo ""

for T in 3 4 6; do
  echo "  Running HRM with T=${T} (12 iterations)..."
  python experiments/improved_ab_comparison.py \
    --model pot \
    --max_inner_iters 12 \
    --hrm_period $T \
    --array_len $ARRAY_LEN \
    --mask_rate $MASK_RATE \
    --train_samples $TRAIN_SAMPLES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --seeds $SEEDS \
    --output_csv $RESULTS_DIR/hrm_12iters_T${T}.csv
  echo ""
done

echo "✓ Experiment 3 complete"
echo ""

# ================================================================
# EXPERIMENT 4: Entropy Schedule Sweep (12 iters, T=4)
# ================================================================

echo "📊 EXPERIMENT 4: Entropy Regularization Sweep"
echo "================================================================"
echo ""

for ENT in 0.0005 0.001 0.002; do
  echo "  Running HRM with entropy_reg=${ENT}..."
  python experiments/improved_ab_comparison.py \
    --model pot \
    --max_inner_iters 12 \
    --hrm_period 4 \
    --entropy_reg $ENT \
    --array_len $ARRAY_LEN \
    --mask_rate $MASK_RATE \
    --train_samples $TRAIN_SAMPLES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --seeds $SEEDS \
    --output_csv $RESULTS_DIR/hrm_12iters_T4_ent${ENT}.csv
  echo ""
done

echo "✓ Experiment 4 complete"
echo ""

# ================================================================
# EXPERIMENT 5: Temperature Schedule Sweep (12 iters, T=4)
# ================================================================

echo "📊 EXPERIMENT 5: Temperature Initialization Sweep"
echo "================================================================"
echo ""

for TEMP in 1.5 2.0 2.5; do
  echo "  Running HRM with temperature_init=${TEMP}..."
  python experiments/improved_ab_comparison.py \
    --model pot \
    --max_inner_iters 12 \
    --hrm_period 4 \
    --temperature_init $TEMP \
    --array_len $ARRAY_LEN \
    --mask_rate $MASK_RATE \
    --train_samples $TRAIN_SAMPLES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --seeds $SEEDS \
    --output_csv $RESULTS_DIR/hrm_12iters_T4_temp${TEMP}.csv
  echo ""
done

echo "✓ Experiment 5 complete"
echo ""

# ================================================================
# Summary
# ================================================================

echo "================================================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to: $RESULTS_DIR/"
echo ""
echo "Summary:"
echo "  1. Baseline vs HRM (RankNet loss)"
echo "  2. Iteration sweep: 6, 8, 12, 16 iterations"
echo "  3. HRM period sweep: T ∈ {3, 4, 6}"
echo "  4. Entropy reg sweep: {5e-4, 1e-3, 2e-3}"
echo "  5. Temperature sweep: {1.5, 2.0, 2.5}"
echo ""
echo "Total configurations: 13"
echo "Total runs: 13 × 5 seeds = 65 experiments"
echo ""
echo "Next steps:"
echo "  1. Analyze results: python experiments/analyze_improved_results.py"
echo "  2. Plot τ vs time: python experiments/plot_improved_results.py"
echo "  3. Check diagnostics: python experiments/check_routing_diagnostics.py"
echo ""
echo "Expected improvements:"
echo "  - Baseline → HRM: +30-50% relative (from 0.09 to 0.12-0.13)"
echo "  - Lower variance across seeds"
echo "  - Routing specialization visible in diagnostics"
echo ""

