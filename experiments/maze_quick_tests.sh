#!/bin/bash
# Quick maze tests with different configurations

echo "=================================="
echo "MAZE QUICK TESTS"
echo "=================================="

# Test 1: 10x10 maze (easier)
echo ""
echo "Test 1: 10×10 maze (easier baseline)"
echo "  - 500 train, 100 test"
echo "  - Min path: 30"
echo "  - 50 epochs"
python experiments/maze_ab_proper_generation.py \
  --maze-size 10 \
  --train 500 \
  --test 100 \
  --min-path-length 30 \
  --epochs 50 \
  --R 4 --T 4 --heads 4

# Test 2: 12x12 maze (medium)
echo ""
echo "Test 2: 12×12 maze (medium difficulty)"
echo "  - 1000 train, 100 test"
echo "  - Min path: 40"
echo "  - 50 epochs"
python experiments/maze_ab_proper_generation.py \
  --maze-size 12 \
  --train 1000 \
  --test 100 \
  --min-path-length 40 \
  --epochs 50 \
  --R 4 --T 4 --heads 4

# Test 3: 15x15 maze (harder)
echo ""
echo "Test 3: 15×15 maze (harder)"
echo "  - 1000 train, 100 test"
echo "  - Min path: 60"
echo "  - 60 epochs"
python experiments/maze_ab_proper_generation.py \
  --maze-size 15 \
  --train 1000 \
  --test 100 \
  --min-path-length 60 \
  --epochs 60 \
  --R 4 --T 4 --heads 4

echo ""
echo "=================================="
echo "All tests complete!"
echo "=================================="

