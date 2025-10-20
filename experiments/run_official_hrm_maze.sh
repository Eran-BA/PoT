#!/bin/bash
# Run official HRM on maze 30x30 hard dataset
# Based on: https://github.com/sapientinc/HRM

set -e

cd vendor/hrm

echo "=================================="
echo "Running Official HRM on Maze 30x30 Hard"
echo "=================================="

# Training configuration from official repo
OMP_NUM_THREADS=8 python pretrain.py \
  data_path=../../vendor/hrm/data/maze-30x30-hard-1k \
  epochs=20000 \
  eval_interval=2000 \
  lr=1e-4 \
  puzzle_emb_lr=1e-4 \
  weight_decay=1.0 \
  puzzle_emb_weight_decay=1.0 \
  project_name=hrm_maze_benchmark \
  run_name=official_hrm_maze30x30_hard

echo "=================================="
echo "HRM Training Complete!"
echo "=================================="

