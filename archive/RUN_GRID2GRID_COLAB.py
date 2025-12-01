"""
Paste this entire script into a Colab cell and run it.
It will train both Baseline and PoH-HRM on the grid-to-grid maze task.
"""

# Setup
import subprocess
import sys

# Clone and setup
subprocess.run("git clone https://github.com/Eran-BA/PoT.git /content/PoT 2>/dev/null || (cd /content/PoT && git pull)", shell=True)
subprocess.run("cd /content/PoT && git checkout scaling_parameter_size", shell=True)
subprocess.run("pip install -q tqdm", shell=True)

# Download HRM dataset using huggingface_hub directly
print("Downloading HRM maze-30x30-hard dataset from HuggingFace...")
subprocess.run("pip install -q huggingface_hub", shell=True)

# Create a simple dataset downloader
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

subprocess.run("python /content/hrm_download.py", shell=True)

print("\n" + "="*80)
print("TRAINING BASELINE TRANSFORMER")
print("="*80)

# Run Baseline
subprocess.run("""
cd /content/PoT
python -u experiments/maze_grid2grid_hrm.py \
  --data-dir vendor/hrm/data/maze-30x30-hard-1k \
  --model baseline \
  --d-model 256 \
  --n-heads 8 \
  --n-layers 4 \
  --batch-size 32 \
  --epochs 100 \
  --lr 1e-3 \
  --output experiments/results/grid2grid_baseline \
  --seed 42
""", shell=True)

print("\n" + "="*80)
print("TRAINING POH-HRM")
print("="*80)

# Run PoH-HRM
subprocess.run("""
cd /content/PoT
python -u experiments/maze_grid2grid_hrm.py \
  --data-dir vendor/hrm/data/maze-30x30-hard-1k \
  --model poh \
  --d-model 256 \
  --n-heads 8 \
  --n-layers 1 \
  --R 4 \
  --T 4 \
  --batch-size 32 \
  --epochs 100 \
  --lr 1e-3 \
  --output experiments/results/grid2grid_poh \
  --seed 42
""", shell=True)

# Show results
import json

baseline_results = json.load(open('/content/PoT/experiments/results/grid2grid_baseline/baseline_results.json'))
poh_results = json.load(open('/content/PoT/experiments/results/grid2grid_poh/poh_results.json'))

print("\n" + "="*80)
print("GRID-TO-GRID MAZE BENCHMARK RESULTS (HRM Task Format)")
print("="*80)
print(f"\nHRM Paper (30x30 Hard): ~74% grid accuracy\n")
print(f"Baseline Transformer:")
print(f"  Parameters: {baseline_results['parameters']:,}")
print(f"  Grid Accuracy: {baseline_results['best_grid_acc']:.2f}%")
print(f"  Token Accuracy: {baseline_results['final_token_acc']:.2f}%")
print(f"\nPoH-HRM (R=4, T=4):")
print(f"  Parameters: {poh_results['parameters']:,}")
print(f"  Grid Accuracy: {poh_results['best_grid_acc']:.2f}%")
print(f"  Token Accuracy: {poh_results['final_token_acc']:.2f}%")
print(f"\n" + "="*80)

