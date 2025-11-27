"""
Run PoH-HRM with ALL CRITICAL FIXES APPLIED:
- Q-continue loss always computed
- Exploration logic fixed (OR instead of AND)
- Inference uses learned policy
- PUZZLE EMBEDDINGS: weight_decay=0.0 (NOT 1.0!)

This version bypasses Colab caching issues.
"""

import subprocess
import sys
import os

print("="*80)
print("🔥 PoH-HRM with ALL CRITICAL FIXES")
print("="*80)

# Clone and setup (force fresh)
print("\n1. Cloning repository...")
subprocess.run("rm -rf /content/PoT", shell=True, check=False)
subprocess.run("git clone https://github.com/Eran-BA/PoT.git /content/PoT", shell=True, check=True)
subprocess.run("cd /content/PoT && git checkout scaling_parameter_size", shell=True, check=True)

print("\n2. Installing dependencies...")
subprocess.run("pip install -q tqdm huggingface_hub", shell=True, check=True)

# Download HRM dataset
print("\n3. Downloading HRM maze-30x30-hard dataset from HuggingFace...")

dataset_script = """
import os
import csv
import json
import numpy as np
from huggingface_hub import hf_hub_download
from tqdm import tqdm

output_dir = '/content/PoT/vendor/hrm/data/maze-30x30-hard-1k'
os.makedirs(output_dir, exist_ok=True)

CHARSET = "# SGo"
repo = "sapientinc/maze-30x30-hard-1k"

for split in ['train', 'test']:
    print(f"Downloading {split}...")
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    
    # Download CSV
    csv_file = hf_hub_download(repo, f"{split}.csv", repo_type="dataset")
    
    # Read and convert
    inputs, labels = [], []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for source, q, a, rating in tqdm(reader):
            grid_size = int(len(q) ** 0.5)
            inputs.append(np.frombuffer(q.encode(), dtype=np.uint8).reshape(grid_size, grid_size))
            labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(grid_size, grid_size))
    
    # Convert to ids
    char2id = np.zeros(256, np.uint8)
    char2id[np.array(list(map(ord, CHARSET)))] = np.arange(len(CHARSET)) + 1
    
    inputs_arr = np.vstack([char2id[inp.reshape(-1)] for inp in inputs])
    labels_arr = np.vstack([char2id[lab.reshape(-1)] for lab in labels])
    
    # Save
    np.save(os.path.join(split_dir, 'all__inputs.npy'), inputs_arr)
    np.save(os.path.join(split_dir, 'all__labels.npy'), labels_arr)
    
    # Save metadata
    with open(os.path.join(split_dir, 'dataset.json'), 'w') as f:
        json.dump({'seq_len': 900, 'vocab_size': 6}, f)
    
    print(f"{split}: {len(inputs)} mazes")

print("✓ Dataset downloaded!")
"""

with open('/content/hrm_download.py', 'w') as f:
    f.write(dataset_script)

subprocess.run("python /content/hrm_download.py", shell=True, check=True)

print("\n" + "="*80)
print("🚀 TRAINING POH-HRM (ALL BUGS FIXED)")
print("="*80)
print()
print("✅ CRITICAL FIXES APPLIED:")
print("  1. Q-continue loss: NOW ALWAYS COMPUTED (was broken)")
print("  2. Exploration logic: FIXED (OR instead of AND)")
print("  3. Inference halting: USES LEARNED POLICY (not forced max)")
print("  4. Puzzle emb weight decay: 0.0 (was 1.0 - killing learning!)")
print()
print("Configuration:")
print("  - Puzzle Embeddings: 1000 mazes × 256-dim (0.02 std init)")
print("  - Q-Halting: max 16 steps, adaptive")
print("  - Learning Rate: 1e-4 (main), 1e-4 (puzzle emb)")
print("  - Weight Decay: 1.0 (main model), 0.0 (puzzle emb) ⭐ KEY FIX!")
print("  - Early Stopping: patience=50, max_epochs=5000")
print()
print("Expected results:")
print("  - Q_cont > 0 (should see ~0.04-0.1)")
print("  - Avg Steps < 16 (should vary 8-14)")
print("  - Token Acc > 87.5% (should break plateau!)")
print("  - Grid Acc > 0% (first perfect grids!)")
print()
print("🎯 Starting training...")
sys.stdout.flush()
print()

# Run PoH-HRM with ALL FIXES
try:
    subprocess.run("""
cd /content/PoT
python -u experiments/maze_grid2grid_hrm.py \
  --data-dir vendor/hrm/data/maze-30x30-hard-1k \
  --model poh \
  --d-model 256 \
  --n-heads 8 \
  --n-layers 1 \
  --T 4 \
  --batch-size 32 \
  --max-epochs 5000 \
  --patience 50 \
  --lr 1e-4 \
  --puzzle-emb-lr 1e-4 \
  --weight-decay 1.0 \
  --num-puzzles 1000 \
  --puzzle-emb-dim 256 \
  --max-halting-steps 16 \
  --output experiments/results/grid2grid_poh_fixed \
  --seed 42
""", shell=True, check=True)
    
    print("\n" + "="*80)
    print("✅ PoH-HRM Training Complete!")
    print("="*80)
except subprocess.CalledProcessError as e:
    print("\n" + "="*80)
    print("❌ Training failed with error!")
    print("="*80)
    print(f"Error: {e}")
    raise

