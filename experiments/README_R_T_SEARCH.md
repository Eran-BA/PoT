# Finding Optimal R and T for PoH

This directory contains scripts to find the optimal hyperparameters for the PoH architecture:

- **R**: Number of refinement steps (how many times the model refines its representation)
- **T**: HRM outer loop period (how often the slow f_H module updates)

---

## 🎯 Quick Start

### Option 1: Real NLI (SNLI) - Recommended

```bash
# Activate environment
cd /Users/rnbnrzy/Desktop/PoT
source venv/bin/activate

# Install datasets if needed
pip install datasets pandas

# Quick test (3 experiments, ~30 min)
PYTHONPATH=$PWD python experiments/find_optimal_R_T_real_nli.py \
  --R-values 4 12 \
  --T-values 2 4 \
  --seeds 42 \
  --max-steps 500 \
  --max-train-samples 5000

# Full search (72 experiments, ~24 hours)
PYTHONPATH=$PWD python experiments/find_optimal_R_T_real_nli.py \
  --R-values 1 2 4 6 8 12 \
  --T-values 1 2 4 8 \
  --seeds 42 43 44 \
  --max-steps 2000 \
  --max-train-samples 10000
```

### Option 2: Synthetic Data (Fast Testing)

```bash
# Quick test on synthetic data (63 experiments, ~3 hours)
PYTHONPATH=$PWD python experiments/find_optimal_R_and_T.py \
  --R-values 1 2 3 4 6 8 12 \
  --T-values 1 2 4 \
  --seeds 42 43 44 \
  --max-steps 200
```

---

## 📊 What Gets Tested

### Default Search Space

**R values (refinement steps):**
- 1, 2, 4, 6, 8, 12

**T values (HRM outer loop period):**
- 1, 2, 4, 8

**Seeds:**
- 42, 43, 44 (3 seeds for robustness)

**Total experiments:** 6 × 4 × 3 = 72

---

## 🔬 Hyperparameters

### Real NLI (SNLI)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-steps` | 2000 | Training steps per experiment |
| `--max-train-samples` | 10000 | SNLI training samples |
| `--max-val-samples` | 2000 | SNLI validation samples |
| `--batch-size` | 32 | Batch size |
| `--d-model` | 256 | Model dimension |
| `--n-heads` | 8 | Number of attention heads |
| `--depth` | 4 | Number of PoH blocks |

### Synthetic Data

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-steps` | 200 | Training steps per experiment |
| `--batch-size` | 32 | Batch size |
| `--d-model` | 128 | Model dimension |
| `--n-heads` | 4 | Number of attention heads |
| `--depth` | 2 | Number of PoH blocks |

---

## 📈 Expected Results

The script will output:

1. **Progress**: Live training progress for each (R, T, seed) combination
2. **CSV**: `grid_search_results.csv` with all results
3. **Analysis**: Best accuracy and best efficiency configurations
4. **Summary**: `grid_search_results_summary.txt` with recommendations

### Example Output

```
Best Accuracy Configuration:
  R=12 (refinement steps)
  T=4 (HRM outer loop period)
  Accuracy: 0.6234 ± 0.0089
  Time: 45.23 min

Best Efficiency Configuration:
  R=6 (refinement steps)
  T=2 (HRM outer loop period)
  Accuracy: 0.6145 ± 0.0072
  Time: 22.15 min
  Efficiency: 0.000891
```

---

## 🎯 Recommended Configurations

### For Quick Testing:

```bash
# 2 experiments, ~10 min
--R-values 4 12 \
--T-values 4 \
--seeds 42 \
--max-steps 500
```

### For Publication-Quality Results:

```bash
# 108 experiments, ~48 hours
--R-values 1 2 3 4 6 8 12 16 20 \
--T-values 1 2 4 8 12 \
--seeds 42 43 44 \
--max-steps 5000 \
--max-train-samples 50000
```

---

## 🔍 Understanding the Parameters

### R (Refinement Steps)

- **What it does**: Number of times the model refines its representation
- **Trade-off**: Higher R = better quality but slower training
- **Range**: 1-20 (we test 1, 2, 4, 6, 8, 12)
- **Current default**: 12 (from previous analysis)

### T (HRM Outer Loop Period)

- **What it does**: How often the slow f_H module updates
- **Trade-off**: 
  - T=1: f_H updates every step (most expressive, slowest)
  - T=∞: f_H never updates (fastest, least expressive)
- **Range**: 1-12 (we test 1, 2, 4, 8)
- **Current default**: 4 (from HRM paper)

### Efficiency Metric

```
efficiency = accuracy / (time × √(R × T))
```

This balances accuracy against computational cost.

---

## 📂 Results Directory

```
experiments/results/R_T_search_real_nli/
├── grid_search_results.csv           # All results
└── grid_search_results_summary.txt   # Best configurations
```

---

## 🚀 Running on GPU

For faster training:

```bash
PYTHONPATH=$PWD python experiments/find_optimal_R_T_real_nli.py \
  --device cuda \
  --batch-size 64 \
  --max-steps 5000
```

---

## 📊 Analyzing Results

The script automatically analyzes results and prints:

1. **Best Accuracy**: Configuration that achieves highest validation accuracy
2. **Best Efficiency**: Configuration with best accuracy/time trade-off
3. **Full Table**: Complete results sorted by accuracy

You can also manually analyze the CSV:

```python
import pandas as pd

df = pd.read_csv('experiments/results/R_T_search_real_nli/grid_search_results.csv')

# Average over seeds
grouped = df.groupby(['R', 'T']).agg({
    'best_val_acc': ['mean', 'std'],
    'time_minutes': 'mean'
}).reset_index()

print(grouped.sort_values(('best_val_acc', 'mean'), ascending=False))
```

---

## 🎓 Interpretation Guide

### When R increases:
- ✅ Better quality (more refinement steps)
- ❌ Slower training (linear in R)
- 📊 Diminishing returns after R=12

### When T increases:
- ✅ Faster training (fewer f_H updates)
- ❌ Less strategic planning
- 📊 Sweet spot around T=4

### Optimal combinations (expected):
- **High accuracy**: R=12, T=4 (current defaults)
- **Fast training**: R=4, T=8
- **Best balance**: R=6-8, T=2-4

---

## ⏱️ Time Estimates

### Real NLI (10K samples, 2000 steps)

| R | T | Time/Experiment | Total (3 seeds) |
|---|---|----------------|-----------------|
| 1 | 1 | ~15 min | ~45 min |
| 4 | 4 | ~30 min | ~90 min |
| 12 | 4 | ~45 min | ~135 min |

**Full grid (6 R × 4 T × 3 seeds):** ~36-48 hours on CPU

### Synthetic (5K samples, 200 steps)

**Full grid (7 R × 3 T × 3 seeds):** ~3-4 hours on CPU

---

## 🐛 Troubleshooting

### Error: "No module named 'datasets'"
```bash
pip install datasets
```

### Out of memory
```bash
# Reduce batch size
--batch-size 16

# Reduce model size
--d-model 128 --n-heads 4

# Reduce data
--max-train-samples 5000
```

### Too slow
```bash
# Use GPU
--device cuda

# Reduce search space
--R-values 4 8 12
--T-values 2 4
--seeds 42

# Fewer steps
--max-steps 1000
```

---

## 📚 Related Documentation

- [POH_ITERATION_GUIDE.md](../docs/POH_ITERATION_GUIDE.md) - Why R=12 was chosen
- [HRM_VS_REFINEMENT_LOOPS.md](../docs/HRM_VS_REFINEMENT_LOOPS.md) - Understanding R vs T
- [TERMINOLOGY_GUIDE.md](../docs/TERMINOLOGY_GUIDE.md) - Official terminology

---

**Questions?** See main README or open an issue!

