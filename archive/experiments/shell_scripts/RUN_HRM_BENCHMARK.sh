#!/usr/bin/env bash
set -euo pipefail

# Benchmark original HRM (vendor/hrm) vs PoT-HRM on Maze 30x30 Hard
# - HRM side follows commands adapted from the official repo README
# - PoT side uses our maze runner with HRM-style normalization

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

echo "✓ PoT root: $ROOT_DIR"

if [ -d "venv" ]; then
  echo "🔧 Activating virtualenv"
  source venv/bin/activate
fi

HRM_DIR="vendor/hrm"
if [ ! -d "$HRM_DIR" ]; then
  echo "✗ vendor/hrm not found. Run: git submodule update --init --recursive"; exit 1
fi

echo "=== Environment Checks ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "🚀 NVIDIA GPU detected"
else
  if python -c 'import torch,sys;print(getattr(getattr(torch.backends,"mps",None),"is_available",lambda:False)())' 2>/dev/null | grep -q True; then
    echo "🚀 Apple MPS detected (HRM repo expects CUDA; recommend Linux + CUDA for speed)"
  else
    echo "⚠️  No GPU detected. HRM training will be extremely slow on CPU."
  fi
fi

echo "=== HRM Requirements (optional) ==="
echo "This may require CUDA-specific wheels. You can skip if already installed."
echo "If this fails on macOS, run HRM on a CUDA machine/Colab."
echo "(Skipping auto-install to avoid breaking local env)"

# Prepare HRM Maze dataset path
HRM_DATA_DIR="$ROOT_DIR/vendor/hrm/data/maze-30x30-hard-1k"
mkdir -p "$HRM_DATA_DIR" || true

echo "\n=== Build HRM Maze 30x30 Hard dataset (1000 examples) ==="
echo "(If this fails, please run on a CUDA Linux box or Colab as per HRM README)"
set +e
python "$HRM_DIR/dataset/build_maze_dataset.py" --output-dir "$HRM_DATA_DIR" --grid-n 30 --num-examples 1000
HRM_DATA_RC=$?
set -e

if [ $HRM_DATA_RC -ne 0 ]; then
  echo "⚠️  HRM dataset build failed here; proceed to run PoT side and use HRM on a CUDA box."
fi

echo "\n=== Run HRM pretrain (reduced settings for single GPU) ==="
echo "Tip: On multi-GPU CUDA: OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 vendor/hrm/pretrain.py data_path=$HRM_DATA_DIR epochs=20000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4"

HRM_OUT_DIR="$ROOT_DIR/experiments/results/hrm_original_maze30"
mkdir -p "$HRM_OUT_DIR"

set +e
python "$HRM_DIR/pretrain.py" data_path="$HRM_DATA_DIR" epochs=2000 eval_interval=200 lr=1e-4 puzzle_emb_lr=1e-4 \
  > "$HRM_OUT_DIR/train.log" 2>&1
HRM_TRAIN_RC=$?
set -e

if [ $HRM_TRAIN_RC -ne 0 ]; then
  echo "⚠️  HRM training did not complete here. See $HRM_OUT_DIR/train.log."
fi

echo "\n=== Run PoT-HRM baseline on 30x30 (reduced) ==="
POT_OUT_DIR="$ROOT_DIR/experiments/results/pot_maze30_reduced"
mkdir -p "$POT_OUT_DIR"

python -u experiments/maze_scaling_benchmark.py \
  --maze-sizes 30 \
  --train 300 --test 60 \
  --R 4 --T 4 --heads 8 \
  --epochs 60 --seed 42 \
  --output "$POT_OUT_DIR" \
  > "$POT_OUT_DIR/run.log" 2>&1

echo "\n✓ Benchmark orchestrated. Artifacts:"
echo "- HRM logs: $HRM_OUT_DIR/train.log (if succeeded)"
echo "- PoT logs: $POT_OUT_DIR/run.log"
echo "\nNext: For a fair HRM run, execute on CUDA per repo instructions:"
echo "  OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 vendor/hrm/pretrain.py data_path=$HRM_DATA_DIR epochs=20000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4"

