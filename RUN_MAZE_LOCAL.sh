#!/bin/bash
# Run proper maze generation benchmark locally

cd /Users/rnbnrzy/Desktop/PoT

# Activate virtual environment
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Check if maze-dataset is installed, install if not
echo "📦 Checking maze-dataset installation..."
python3 -c "import maze_dataset" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing maze-dataset (this may take a few minutes)..."
    pip install maze-dataset --quiet
    if [ $? -ne 0 ]; then
        echo "❌ Installation failed. Trying without optional dependencies..."
        pip install maze-dataset --no-deps
        pip install muutils jaxtyping zanj numpy matplotlib tqdm
    fi
fi

# Set matplotlib backend
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/matplotlib_temp

echo ""
echo "=========================================="
echo "MAZE A/B TEST - PROPER GENERATION"
echo "=========================================="
echo "Configuration:"
echo "  Maze size: 20×20"
echo "  Min path length: 80"
echo "  Training: 300 samples"
echo "  Test: 50 samples"
echo "  Epochs: 30"
echo "  R=4, T=4, heads=4"
echo "=========================================="
echo ""
echo "Starting benchmark..."
echo ""

python3 experiments/maze_ab_proper_generation.py \
    --maze-size 20 \
    --train 300 \
    --test 50 \
    --min-path-length 80 \
    --epochs 30 \
    --R 4 --T 4 --heads 4 \
    --seed 42

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ BENCHMARK COMPLETE!"
else
    echo "❌ Benchmark failed (exit code: $EXIT_CODE)"
fi
echo "=========================================="

exit $EXIT_CODE

