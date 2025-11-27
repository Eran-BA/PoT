#!/bin/bash
# Quick experiments to investigate 16×16 optimality failure
# Tests different R, T, and n_heads values

set -e

MAZE_SIZE=16
TRAIN=200
TEST=50
EPOCHS=30
SEED=42

echo "🔬 Quick 16×16 Investigation Experiments"
echo "========================================="
echo ""
echo "Each test takes ~50 min on Mac"
echo "Total: ~4-5 hours for all tests"
echo ""

cd /Users/rnbnrzy/Desktop/PoT

# Test 1: Vary R (refinement iterations)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 1: Varying R (refinement iterations)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for R in 2 4 6 8; do
    echo ""
    echo "▶ Testing R=$R, T=4, heads=4"
    python3 experiments/maze_scaling_benchmark.py \
        --maze-sizes $MAZE_SIZE \
        --train $TRAIN --test $TEST \
        --R $R --T 4 --heads 4 \
        --epochs $EPOCHS --seed $SEED \
        --output "experiments/results/16x16_R${R}_T4_h4"
    
    # Extract key result
    RESULT=$(grep "PoH-HRM.*Optimality:" "experiments/results/16x16_R${R}_T4_h4.json" 2>/dev/null || echo "N/A")
    echo "  Result R=$R: $RESULT"
done

# Test 2: Vary T (HRM period)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 2: Varying T (HRM outer loop period)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for T in 2 4 8 16; do
    echo ""
    echo "▶ Testing R=4, T=$T, heads=4"
    python3 experiments/maze_scaling_benchmark.py \
        --maze-sizes $MAZE_SIZE \
        --train $TRAIN --test $TEST \
        --R 4 --T $T --heads 4 \
        --epochs $EPOCHS --seed $SEED \
        --output "experiments/results/16x16_R4_T${T}_h4"
    
    # Extract key result
    RESULT=$(grep "PoH-HRM.*Optimality:" "experiments/results/16x16_R4_T${T}_h4.json" 2>/dev/null || echo "N/A")
    echo "  Result T=$T: $RESULT"
done

# Test 3: Vary n_heads
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 3: Varying n_heads (attention heads)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for HEADS in 2 4 8; do
    echo ""
    echo "▶ Testing R=4, T=4, heads=$HEADS"
    python3 experiments/maze_scaling_benchmark.py \
        --maze-sizes $MAZE_SIZE \
        --train $TRAIN --test $TEST \
        --R 4 --T 4 --heads $HEADS \
        --epochs $EPOCHS --seed $SEED \
        --output "experiments/results/16x16_R4_T4_h${HEADS}"
    
    # Extract key result
    RESULT=$(grep "PoH-HRM.*Optimality:" "experiments/results/16x16_R4_T4_h${HEADS}.json" 2>/dev/null || echo "N/A")
    echo "  Result heads=$HEADS: $RESULT"
done

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 SUMMARY OF RESULTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "R variation (T=4, heads=4):"
for R in 2 4 6 8; do
    if [ -f "experiments/results/16x16_R${R}_T4_h4.json" ]; then
        OPT=$(python3 -c "import json; r=json.load(open('experiments/results/16x16_R${R}_T4_h4.json')); print(f\"{r['poh']['optimality'][-1]*100:.1f}%\")" 2>/dev/null || echo "N/A")
        echo "  R=$R: $OPT"
    fi
done

echo ""
echo "T variation (R=4, heads=4):"
for T in 2 4 8 16; do
    if [ -f "experiments/results/16x16_R4_T${T}_h4.json" ]; then
        OPT=$(python3 -c "import json; r=json.load(open('experiments/results/16x16_R4_T${T}_h4.json')); print(f\"{r['poh']['optimality'][-1]*100:.1f}%\")" 2>/dev/null || echo "N/A")
        echo "  T=$T: $OPT"
    fi
done

echo ""
echo "n_heads variation (R=4, T=4):"
for HEADS in 2 4 8; do
    if [ -f "experiments/results/16x16_R4_T4_h${HEADS}.json" ]; then
        OPT=$(python3 -c "import json; r=json.load(open('experiments/results/16x16_R4_T4_h${HEADS}.json')); print(f\"{r['poh']['optimality'][-1]*100:.1f}%\")" 2>/dev/null || echo "N/A")
        echo "  heads=$HEADS: $OPT"
    fi
done

echo ""
echo "✅ All tests complete!"
echo "Results saved to: experiments/results/16x16_*"

