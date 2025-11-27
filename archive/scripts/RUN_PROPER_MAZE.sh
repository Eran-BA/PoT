#!/bin/bash
# Run proper maze generation with maze-dataset library

cd /Users/rnbnrzy/Desktop/PoT

# Activate virtual environment
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Install maze-dataset if not installed
echo "📦 Checking maze-dataset installation..."
pip show maze-dataset > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Installing maze-dataset..."
    pip install maze-dataset
fi

# Set matplotlib backend
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/matplotlib_temp

echo ""
echo "=========================================="
echo "PROPER MAZE GENERATION A/B TEST"
echo "=========================================="
echo "Maze size: 20×20"
echo "Min path length: 80 (long planning horizon)"
echo "Training: 300 samples, Test: 50 samples"
echo "Epochs: 30"
echo "Using DFS algorithm from maze-dataset"
echo "=========================================="
echo ""

python3 experiments/maze_ab_proper_generation.py \
    --maze-size 20 \
    --train 300 \
    --test 50 \
    --min-path-length 80 \
    --epochs 30 \
    --R 4 --T 4 --heads 4 \
    --seed 42

echo ""
echo "=========================================="
echo "✅ DONE!"
echo "=========================================="

