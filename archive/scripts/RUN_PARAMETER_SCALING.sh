#!/bin/bash
# Quick runner for parameter scaling benchmark

cd /Users/rnbnrzy/Desktop/PoT

# Activate virtual environment if available
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Ensure maze-dataset is installed
echo "📦 Checking dependencies..."
pip install -q maze-dataset matplotlib

# Set matplotlib backend to non-interactive
export MPLBACKEND=Agg

echo ""
echo "=========================================="
echo "PARAMETER SCALING BENCHMARK"
echo "=========================================="
echo ""

# Run the benchmark
# Default: 16x16 mazes, 1000 train, 100 test, 50 epochs
# Tests all 5 model sizes (tiny, small, medium, large, xl)
python3 experiments/parameter_scaling_benchmark.py \
    --maze-size 16 \
    --train 1000 \
    --test 100 \
    --epochs 50 \
    --R 4 \
    --T 4 \
    --seed 42 \
    --output experiments/results/parameter_scaling \
    2>&1 | tee experiments/results/parameter_scaling/benchmark.log

echo ""
echo "=========================================="
echo "GENERATING PLOTS"
echo "=========================================="
echo ""

# Generate visualization
python3 experiments/plot_parameter_scaling.py \
    experiments/results/parameter_scaling/scaling_results_maze16.json

echo ""
echo "✅ Done! Check experiments/results/parameter_scaling/ for results"
echo ""

