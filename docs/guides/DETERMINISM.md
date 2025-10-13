# Determinism & Reproducibility

This document describes how to achieve reproducible results with PoH.

---

## TL;DR

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic operations (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# At the start of your script
set_seed(42)
```

---

## Seed Management

### Single-Seed Runs

For a single experiment:

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

set_seed(args.seed)
```

### Multi-Seed Runs

For statistical validation:

```python
seeds = [1, 2, 3, 4, 5]  # At least 3-5 seeds recommended

for seed in seeds:
    set_seed(seed)
    
    # Run experiment
    model = build_model(cfg)
    results = train(model, ...)
    
    # Save with seed in filename
    results.to_csv(f"results/run_seed{seed}.csv")
```

---

## PyTorch Determinism

### CUDA Operations

Some CUDA operations are non-deterministic by default:

```python
# Force deterministic algorithms (may be slower)
torch.use_deterministic_algorithms(True)

# Or set environment variable before running
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py
```

**Note:** Some operations (e.g., scatter_add on CUDA) don't have deterministic implementations. If you hit errors, you may need to:
1. Run on CPU for exact reproducibility
2. Accept minor variance on GPU (~1e-6)

### cuDNN Backend

```python
# Disable cuDNN benchmarking (trades speed for determinism)
torch.backends.cudnn.benchmark = False

# Force deterministic cuDNN ops
torch.backends.cudnn.deterministic = True
```

---

## DataLoader Determinism

### Random Shuffling

```python
from torch.utils.data import DataLoader

# Set worker_init_fn for multi-worker determinism
def worker_init_fn(worker_id):
    np.random.seed(args.seed + worker_id)
    random.seed(args.seed + worker_id)

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    worker_init_fn=worker_init_fn,  # Important!
    generator=torch.Generator().manual_seed(args.seed)  # For shuffling
)
```

### Sampling

For tasks with random data generation (e.g., synthetic sorting):

```python
# Inside dataset __getitem__
rng = np.random.default_rng(seed=self.base_seed + index)
array = rng.permutation(length)
```

---

## Model Initialization

### Weight Initialization

```python
def init_weights(module, seed=42):
    """Initialize model weights deterministically."""
    torch.manual_seed(seed)
    
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)

model.apply(init_weights)
```

### Dropout

Dropout is deterministic when seed is fixed:

```python
set_seed(42)
model.eval()  # Dropout off
with torch.no_grad():
    out1 = model(x)
    out2 = model(x)
    assert torch.allclose(out1, out2)  # Identical
```

---

## Training Determinism

### Optimizer State

```python
# Optimizer state is deterministic if initialized after set_seed
set_seed(42)
model = build_model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Checkpoint includes optimizer state
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'seed': args.seed,
}, f"checkpoint_seed{args.seed}.pt")
```

### Gradient Accumulation

```python
# Deterministic with fixed seed
optimizer.zero_grad()
for i, batch in enumerate(batches):
    loss = model(batch)
    loss.backward()
    
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## Expected Variance

### Multi-Seed Statistics

Even with perfect determinism, **statistical variance across seeds is expected**:

```python
# Example: Kendall-τ on synthetic sorting (L=12, 50% mask, 3 seeds)
seeds = [1, 2, 3]
scores = [0.091, 0.094, 0.089]

mean = np.mean(scores)  # 0.091
std = np.std(scores)    # 0.002
ci_95 = 1.96 * std      # ±0.004

print(f"Kendall-τ: {mean:.3f} ± {ci_95:.3f}")  # 0.091 ± 0.004
```

**Recommendation:** Report mean ± 95% CI over 3-5 seeds.

### GPU vs CPU

Small numerical differences (~1e-6) are expected between GPU and CPU due to:
- Different floating-point precision
- Different operation ordering
- Non-deterministic CUDA kernels

**For publication:** Use the same hardware (GPU model) across all runs.

---

## Reproducibility Checklist

### ✅ Code

- [ ] Pin all random seeds (Python, NumPy, PyTorch, CUDA)
- [ ] Set `torch.backends.cudnn.deterministic = True`
- [ ] Set `torch.backends.cudnn.benchmark = False`
- [ ] Use `worker_init_fn` in DataLoader
- [ ] Use `generator` for DataLoader shuffling

### ✅ Environment

- [ ] Pin library versions (requirements.txt or environment.yml)
- [ ] Document hardware (CPU/GPU model, CUDA version)
- [ ] Document OS and Python version

### ✅ Experiments

- [ ] Run with 3-5 different seeds
- [ ] Report mean ± 95% CI
- [ ] Save random seed with each result
- [ ] Include "reproduce this row" command in README

### ✅ Documentation

- [ ] Add seed to result CSVs
- [ ] Add determinism notes to README
- [ ] Include full command in experiment logs

---

## Example: Reproducible Training Script

```python
#!/usr/bin/env python3
"""Reproducible training script."""

import argparse
import random
import numpy as np
import torch
from pathlib import Path

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()
    
    # Set seed BEFORE any randomness
    set_seed(args.seed)
    
    # Build model
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Train
    for epoch in range(args.epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
    
    # Save results with seed
    results = {"seed": args.seed, "final_loss": loss.item()}
    save_csv(results, args.output_csv)
    
    print(f"✅ Reproducible run with seed={args.seed} complete")

if __name__ == "__main__":
    main()
```

---

## Debugging Non-Determinism

If you get different results with the same seed:

### 1. Check seed is set BEFORE model creation

```python
# ✅ Good
set_seed(42)
model = build_model()

# ❌ Bad
model = build_model()
set_seed(42)  # Too late!
```

### 2. Check for unintended randomness

```python
# Search for random operations
git grep "random\." src/
git grep "np.random" src/
git grep "torch.rand" src/
```

### 3. Test determinism on CPU

```python
# Run on CPU to eliminate CUDA non-determinism
device = torch.device("cpu")
model = model.to(device)
```

### 4. Compare outputs bit-by-bit

```python
set_seed(42)
out1 = model(x)

set_seed(42)
out2 = model(x)

torch.testing.assert_close(out1, out2, rtol=0, atol=0)  # Exact match
```

---

## Hardware & Software Versions

For full reproducibility, document:

```yaml
# environment.yml
name: poh
dependencies:
  - python=3.10
  - pytorch=2.0.1
  - numpy=1.24.3
  - cuda=11.8  # If using GPU

# Or requirements.txt with exact versions
torch==2.0.1+cu118
numpy==1.24.3
scipy==1.10.1
```

**Hardware:**
```
CPU: Intel Core i9-9900K
GPU: NVIDIA RTX 3090 (24GB, CUDA 11.8, cuDNN 8.6.0)
RAM: 64GB DDR4
OS: Ubuntu 22.04 LTS
```

---

## References

- PyTorch Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
- cuDNN Determinism: https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html

---

**Bottom line:** 
- For development: Determinism helps debugging
- For publication: Multi-seed statistics (mean ± CI) are more important than perfect bit-level reproducibility

---

**Status:** Documented ✅

