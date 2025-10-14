#!/bin/bash
# Simple script to run the maze investigation
# Run this in your Mac terminal (outside Cursor)

cd /Users/rnbnrzy/Desktop/PoT

# Activate venv
source venv/bin/activate

# Set matplotlib to non-interactive backend
export MPLBACKEND=Agg

# Run the investigation
echo "Starting maze scaling investigation..."
echo "This will take approximately 3-3.5 hours"
echo "Log file: experiments/results/maze_local_investigation.log"
echo ""

python3 experiments/maze_scaling_benchmark.py \
    --maze-sizes 16 20 30 \
    --train 200 \
    --test 50 \
    --R 4 \
    --T 4 \
    --heads 4 \
    --epochs 30 \
    --seed 42 \
    --output experiments/results/maze_local_investigation

echo ""
echo "✅ Investigation complete!"
echo "Results saved to: experiments/results/maze_local_investigation.json"

