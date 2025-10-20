# Parameter Scaling Benchmark

This directory contains experiments to test how **Baseline Transformer** and **PoH-HRM** performance scales with model size on maze solving tasks.

## Overview

We test 5 model size configurations:

| Size   | d_model | n_heads | d_ff  | depth | Target Params |
|--------|---------|---------|-------|-------|---------------|
| Tiny   | 128     | 4       | 512   | 2     | ~1M           |
| Small  | 256     | 4       | 1024  | 3     | ~3M           |
| Medium | 512     | 8       | 2048  | 4     | ~10M          |
| Large  | 768     | 12      | 3072  | 6     | ~30M          |
| XL     | 1024    | 16      | 4096  | 8     | ~100M         |

For each size, we compare:
- **Baseline Transformer**: Standard multi-head attention with stacked layers
- **PoH-HRM**: Pointer-over-Heads with Hierarchical Reasoning Module (R=4, T=4)

## Key Questions

1. **Does PoH-HRM maintain its advantage across different model sizes?**
2. **At what scale does PoH-HRM's dynamic routing provide the most benefit?**
3. **Does performance saturate at larger sizes, or continue improving?**
4. **Is the advantage more pronounced in accuracy or optimality?**

## Running the Benchmark

### Basic Usage

```bash
# Default: 16×16 mazes, 1000 train, 100 test, 50 epochs
python experiments/parameter_scaling_benchmark.py
```

### Custom Configuration

```bash
# Larger mazes with more training
python experiments/parameter_scaling_benchmark.py \
    --maze-size 20 \
    --train 2000 \
    --test 200 \
    --epochs 100 \
    --min-path 160
```

### Arguments

- `--maze-size`: Maze grid size (default: 16)
- `--train`: Number of training samples (default: 1000)
- `--test`: Number of test samples (default: 100)
- `--min-path`: Minimum solution path length filter (default: auto)
- `--epochs`: Training epochs per model (default: 50)
- `--R`: PoH refinement iterations (default: 4)
- `--T`: HRM outer loop period (default: 4)
- `--seed`: Random seed (default: 42)
- `--output`: Output directory (default: experiments/results/parameter_scaling)

## Visualizing Results

After running the benchmark, visualize the results:

```bash
python experiments/plot_parameter_scaling.py \
    experiments/results/parameter_scaling/scaling_results_maze16.json
```

This generates:
- **scaling_plot_maze16.png**: 4-panel plot showing accuracy, optimality, and PoH advantages
- **scaling_summary_maze16.txt**: Text summary with key findings

## Expected Outputs

### Results JSON Structure

```json
{
  "config": {
    "maze_size": 16,
    "n_train": 1000,
    "n_test": 100,
    "epochs": 50,
    "R": 4,
    "T": 4,
    "seed": 42
  },
  "results": [
    {
      "size": "tiny",
      "d_model": 128,
      "n_heads": 4,
      "d_ff": 512,
      "depth": 2,
      "baseline_params": 1234567,
      "baseline_acc": 45.0,
      "baseline_opt": 30.0,
      "poh_params": 1456789,
      "poh_acc": 52.0,
      "poh_opt": 38.0,
      "poh_advantage_acc": 7.0,
      "poh_advantage_opt": 8.0
    },
    ...
  ]
}
```

### Plots Generated

1. **Accuracy vs. Parameters**: Shows how both models improve with scale (log scale on x-axis)
2. **Optimality vs. Parameters**: Shows optimal path finding vs. scale
3. **PoH Accuracy Advantage**: Bar chart of PoH-Baseline gap at each size
4. **PoH Optimality Advantage**: Bar chart of optimality gap at each size

Green bars = PoH wins, Red bars = Baseline wins

## Hypothesis

We expect:
- **Small models (1-3M)**: PoH-HRM may struggle due to controller overhead
- **Medium models (10M)**: PoH-HRM advantage becomes clear
- **Large models (30-100M)**: PoH-HRM maintains or increases advantage due to better credit assignment

## Quick Test (Small Scale)

To quickly test the pipeline without running all sizes:

```bash
# Test only tiny and small models, fewer epochs
python experiments/parameter_scaling_benchmark.py \
    --maze-size 12 \
    --train 500 \
    --test 50 \
    --epochs 30
```

Then manually edit `MODEL_CONFIGS` in the script to only include `'tiny'` and `'small'`.

## Notes

- Each model size takes ~5-30 minutes to train (depending on size and GPU)
- Full benchmark (5 sizes × 2 models = 10 runs) may take 2-4 hours on GPU
- Mazes are generated once and reused for all models
- Results are saved incrementally as each model completes
- GPU memory requirements scale with model size (largest ~8GB VRAM for XL)

## Integration with Main Repository

After running experiments, results can be referenced in:
- `README.md` (add scaling results section)
- `docs/project/results.md` (detailed analysis)
- Paper/preprint (scaling analysis section)

## Related Files

- `maze_ab_proper_generation.py`: Core maze A/B testing (single configuration)
- `maze_hyperparam_search.py`: Hyperparameter search for R, T, n_heads
- `parameter_scaling_benchmark.py`: This scaling benchmark (multiple model sizes)
- `plot_parameter_scaling.py`: Visualization script

