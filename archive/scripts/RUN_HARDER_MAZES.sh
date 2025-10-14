#!/bin/bash
# Run harder mazes benchmark - execute this directly in your terminal

cd /Users/rnbnrzy/Desktop/PoT

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Set matplotlib backend to avoid GUI issues
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/matplotlib_temp

echo ""
echo "=========================================="
echo "RUNNING HARDER MAZES QUICK TEST"
echo "=========================================="
echo "Wall probability: 0.6 (hard mazes)"
echo "Maze size: 16×16"
echo "Training: 100 samples, Test: 20 samples"
echo "Epochs: 20"
echo "=========================================="
echo ""

python3 experiments/maze_scaling_benchmark.py \
    --maze-sizes 16 \
    --train 100 \
    --test 20 \
    --R 4 \
    --T 4 \
    --heads 4 \
    --epochs 20 \
    --seed 42 \
    --output experiments/results/maze_scaling_wall60_hard \
    --wall-prob 0.6

echo ""
echo "=========================================="
echo "✅ DONE!"
echo "Results saved to:"
echo "  - experiments/results/maze_scaling_wall60_hard.json"
echo "  - experiments/results/maze_scaling_wall60_hard.png"
echo "=========================================="

