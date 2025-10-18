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

# Download HRM dataset
print("Setting up HRM dataset...")
subprocess.run("mkdir -p /content/PoT/vendor", shell=True, check=False)

# Clone HRM repo if needed
clone_result = subprocess.run(
    "git clone https://github.com/sapientinc/HRM /content/PoT/vendor/hrm",
    shell=True,
    capture_output=True,
    text=True
)
if clone_result.returncode != 0 and "already exists" not in clone_result.stderr.lower():
    print(f"HRM clone warning: {clone_result.stderr}")

# Install HRM requirements
subprocess.run("cd /content/PoT/vendor/hrm && pip install -q -r requirements.txt 2>/dev/null || pip install -q argdantic pydantic omegaconf hydra-core huggingface_hub", shell=True, check=False)

# Download dataset
dataset_result = subprocess.run(
    "cd /content/PoT/vendor/hrm && python dataset/build_maze_dataset.py --output-dir data/maze-30x30-hard-1k",
    shell=True,
    capture_output=True,
    text=True
)
if dataset_result.returncode != 0:
    print(f"Dataset download output: {dataset_result.stdout}")
    print(f"Dataset download error: {dataset_result.stderr}")
    print("Continuing anyway - dataset may already exist...")

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

