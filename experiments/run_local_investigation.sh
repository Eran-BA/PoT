#!/bin/bash
# Run local 16x16, 20x20, 30x30 investigation
# Usage: ./run_local_investigation.sh

cd /Users/rnbnrzy/Desktop/PoT

echo "🚀 Starting Maze Scaling Investigation"
echo "======================================"
echo ""
echo "Maze sizes: 16×16, 20×20, 30×30"
echo "Training: 200 samples per size"
echo "Testing: 50 samples per size"
echo "Epochs: 30 per size"
echo "Config: R=4, T=4, heads=4"
echo ""
echo "Log file: experiments/results/maze_local_investigation.log"
echo ""
echo "Monitor with:"
echo "  ./experiments/check_status.sh"
echo "  ./experiments/monitor_progress.sh"
echo ""
echo "======================================"
echo ""

# Run the benchmark
python3 experiments/maze_scaling_benchmark.py \
    --maze-sizes 16 20 30 \
    --train 200 \
    --test 50 \
    --R 4 \
    --T 4 \
    --heads 4 \
    --epochs 30 \
    --seed 42 \
    --output experiments/results/maze_local_investigation \
    2>&1 | tee experiments/results/maze_local_investigation.log

