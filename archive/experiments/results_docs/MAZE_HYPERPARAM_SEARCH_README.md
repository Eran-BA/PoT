# Maze Hyperparameter Search

Find optimal R (refinement steps), T (HRM period), and n_heads for maze solving.

## Quick Start

### 1. Quick Test (Recommended First)
Fast test on 10×10 mazes to verify everything works:
```bash
python experiments/maze_hyperparam_search.py --quick
```
- 10×10 mazes, min path 30
- 500 train, 50 test
- 30 epochs per config
- **Total: 36 configs × 30 epochs ≈ 30-60 minutes on GPU**

### 2. Full Search on 12×12 Mazes
More comprehensive search on medium-sized mazes:
```bash
python experiments/maze_hyperparam_search.py \
  --maze-size 12 \
  --train 1000 \
  --test 100 \
  --min-path-length 40 \
  --epochs 50
```
- **Total: 36 configs × 50 epochs ≈ 2-3 hours on GPU**

### 3. Analyze Results
Once the search completes, analyze the results:
```bash
python experiments/analyze_maze_hyperparam_results.py \
  experiments/results/maze_hyperparam_search_12x12.csv
```

## Search Grid

The search tests all combinations of:
- **R (refinement steps)**: [2, 4, 6, 8]
- **T (HRM period)**: [2, 4, 8]
- **n_heads**: [2, 4, 8]

**Total configurations**: 4 × 3 × 3 = **36 configs**

## Output

Results are saved to:
```
experiments/results/maze_hyperparam_search_{size}x{size}.csv
```

CSV columns:
- `timestamp`: When the test ran
- `R`, `T`, `n_heads`: Hyperparameters tested
- `baseline_acc`, `baseline_opt`: Baseline Transformer results
- `poh_acc`, `poh_opt`: PoH-HRM results
- `poh_improvement_acc`, `poh_improvement_opt`: % improvement

## Analysis Output

The analysis script provides:
1. **Summary statistics** (mean, std, min, max)
2. **Best overall configuration** (weighted: 70% acc + 30% opt)
3. **Top 5 configs by accuracy**
4. **Top 5 configs by optimality**
5. **Parameter analysis** (which R, T, n_heads work best)

## Example Output

```
================================================================================
BEST OVERALL CONFIGURATION (weighted: 70% acc + 30% opt)
================================================================================

R=4, T=4, n_heads=4

PoH-HRM Results:
  Accuracy:   45.23%
  Optimality: 12.50%
  Score:      35.41

Baseline Results:
  Accuracy:   32.10%
  Optimality: 5.20%

Improvement:
  Accuracy:   +41.0%
  Optimality: +140.4%
```

## Running on Colab

For faster GPU execution, run on Colab:

```python
# In Colab cell
!git clone https://github.com/Eran-BA/PoT.git
!cd PoT && pip install maze-dataset

# Quick test
!python PoT/experiments/maze_hyperparam_search.py --quick

# Analyze
!python PoT/experiments/analyze_maze_hyperparam_results.py \
  PoT/experiments/results/maze_hyperparam_search_10x10.csv
```

## Advanced Options

### Custom maze size
```bash
python experiments/maze_hyperparam_search.py --maze-size 15 --min-path-length 60
```

### More training data
```bash
python experiments/maze_hyperparam_search.py --train 2000 --test 200
```

### Longer training
```bash
python experiments/maze_hyperparam_search.py --epochs 100
```

## Tips

1. **Start small**: Run `--quick` first to verify everything works
2. **Use GPU**: This search is compute-intensive, A100 recommended
3. **Monitor progress**: Results are saved after each config, so you can analyze partial results
4. **Resume failed runs**: Edit the search script to skip completed configs

## Interpreting Results

**What to look for:**
- **Accuracy**: % of correct next-step predictions
- **Optimality**: % of paths matching optimal length

**Good performance:**
- Accuracy: >50% (task-dependent)
- Optimality: >10% (task-dependent)
- PoH improvement over baseline: >20%

**If results are poor (<10% accuracy):**
- Task may be too hard (try smaller mazes)
- Need more training data (increase --train)
- Need more epochs (increase --epochs)
- Try different min-path-length (easier mazes)

