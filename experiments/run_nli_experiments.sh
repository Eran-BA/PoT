#!/bin/bash
#
# NLI Experiments: Baseline vs PoH on Stanford NLI
#
# Tests PoH on a challenging NLP encoding task requiring:
# - Compositional reasoning
# - Semantic understanding
# - Logical inference
#
# Author: Eran Ben Artzy
# Year: 2025

set -e

RESULTS_DIR="experiments/results_nli"
mkdir -p $RESULTS_DIR

echo "================================================================================"
echo "NLI EXPERIMENTS: Baseline vs PoH Transformer"
echo "================================================================================"
echo ""
echo "Task: Natural Language Inference (SNLI)"
echo "Challenge: 3-way classification (entailment, contradiction, neutral)"
echo "Dataset: 570k sentence pairs"
echo ""
echo "Why This Tests PoH:"
echo "  ✓ Requires compositional reasoning"
echo "  ✓ Different reasoning patterns (lexical, semantic, logical)"
echo "  ✓ Long-range dependencies"
echo "  ✓ Heads can specialize in different inference types"
echo ""
echo "================================================================================"
echo ""

# ================================================================
# Quick Sanity Test (10k samples, 3 epochs)
# ================================================================

echo "📊 QUICK SANITY TEST (10k samples, 3 epochs)"
echo "================================================================"
echo ""

echo "  Running Baseline (BERT + classification head)..."
PYTHONPATH=. python experiments/nli_poh_experiment.py \
  --encoder bert-base-uncased \
  --epochs 3 \
  --batch_size 32 \
  --lr 2e-5 \
  --num_train_samples 10000 \
  --num_val_samples 2000 \
  --seed 42 \
  --output_dir $RESULTS_DIR

echo ""
echo "  Running PoH (BERT + PoH layer + classification head)..."
PYTHONPATH=. python experiments/nli_poh_experiment.py \
  --encoder bert-base-uncased \
  --use_poh \
  --max_inner_iters 3 \
  --epochs 3 \
  --batch_size 32 \
  --lr 2e-5 \
  --num_train_samples 10000 \
  --num_val_samples 2000 \
  --seed 42 \
  --output_dir $RESULTS_DIR

echo ""
echo "✓ Quick sanity test complete"
echo ""

# ================================================================
# Full Experiment (50k samples, 5 epochs, 3 seeds)
# ================================================================

echo "📊 FULL EXPERIMENT (50k samples, 5 epochs, 3 seeds)"
echo "================================================================"
echo ""

for SEED in 42 43 44; do
  echo ""
  echo "  Seed ${SEED}"
  echo "  ----------------------------------------"
  
  echo "    Baseline..."
  PYTHONPATH=. python experiments/nli_poh_experiment.py \
    --encoder bert-base-uncased \
    --epochs 5 \
    --batch_size 32 \
    --lr 2e-5 \
    --num_train_samples 50000 \
    --num_val_samples 5000 \
    --seed $SEED \
    --output_dir $RESULTS_DIR
  
  echo "    PoH (3 iterations)..."
  PYTHONPATH=. python experiments/nli_poh_experiment.py \
    --encoder bert-base-uncased \
    --use_poh \
    --max_inner_iters 3 \
    --epochs 5 \
    --batch_size 32 \
    --lr 2e-5 \
    --num_train_samples 50000 \
    --num_val_samples 5000 \
    --seed $SEED \
    --output_dir $RESULTS_DIR
  
  echo "    PoH (5 iterations)..."
  PYTHONPATH=. python experiments/nli_poh_experiment.py \
    --encoder bert-base-uncased \
    --use_poh \
    --max_inner_iters 5 \
    --epochs 5 \
    --batch_size 32 \
    --lr 2e-5 \
    --num_train_samples 50000 \
    --num_val_samples 5000 \
    --seed $SEED \
    --output_dir $RESULTS_DIR
done

echo ""
echo "✓ Full experiment complete"
echo ""

# ================================================================
# Frozen Encoder Experiment (test PoH layer alone)
# ================================================================

echo "📊 FROZEN ENCODER EXPERIMENT (PoH layer only)"
echo "================================================================"
echo ""

echo "  Baseline (frozen BERT)..."
PYTHONPATH=. python experiments/nli_poh_experiment.py \
  --encoder bert-base-uncased \
  --freeze_encoder \
  --epochs 5 \
  --batch_size 32 \
  --lr 1e-3 \
  --num_train_samples 50000 \
  --num_val_samples 5000 \
  --seed 42 \
  --output_dir $RESULTS_DIR

echo ""
echo "  PoH (frozen BERT + trainable PoH)..."
PYTHONPATH=. python experiments/nli_poh_experiment.py \
  --encoder bert-base-uncased \
  --freeze_encoder \
  --use_poh \
  --max_inner_iters 3 \
  --epochs 5 \
  --batch_size 32 \
  --lr 1e-3 \
  --num_train_samples 50000 \
  --num_val_samples 5000 \
  --seed 42 \
  --output_dir $RESULTS_DIR

echo ""
echo "✓ Frozen encoder experiment complete"
echo ""

# ================================================================
# Summary
# ================================================================

echo "================================================================================"
echo "ALL NLI EXPERIMENTS COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to: $RESULTS_DIR/"
echo ""
echo "Summary:"
echo "  1. Quick sanity test (10k samples)"
echo "  2. Full experiment (50k samples, 3 seeds)"
echo "  3. Frozen encoder test (PoH layer only)"
echo ""
echo "Total configurations: 8"
echo "Total runs: 1 (sanity) + 9 (full) + 2 (frozen) = 12 experiments"
echo ""
echo "Next steps:"
echo "  1. Analyze results: python experiments/analyze_nli_results.py"
echo "  2. Check routing patterns: Inspect entropy and alpha diagnostics"
echo "  3. Compare to baseline: Look for accuracy improvements"
echo ""
echo "Expected findings:"
echo "  - PoH should show +1-3% absolute accuracy on test set"
echo "  - Different heads specialize in different reasoning patterns"
echo "  - Frozen encoder test shows PoH layer contribution"
echo ""

