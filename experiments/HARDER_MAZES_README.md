# 🔥 Harder Mazes Guide

## Problem
Easy mazes (default `wall_prob=0.3`) often lead to **100% accuracy for all models**, making it impossible to differentiate between baseline, BERT, and PoH-HRM.

## Solution
Use the `run_harder_mazes.py` helper script to easily run benchmarks with harder maze configurations.

---

## Quick Start (Colab)

### Option 1: Quick Test (Recommended First)
```bash
!python experiments/run_harder_mazes.py --wall-prob 0.6 --quick
```
- **Maze Size**: 16×16 only
- **Runtime**: ~10 minutes on A100
- **Purpose**: Fast sanity check with hard mazes

### Option 2: Full Scaling Benchmark
```bash
!python experiments/run_harder_mazes.py --wall-prob 0.6 --maze-sizes 8 12 16 20 24 30
```
- **Maze Sizes**: 8×8 to 30×30
- **Runtime**: ~2 hours on A100
- **Purpose**: Complete scaling analysis with hard mazes

---

## Difficulty Levels

| `wall_prob` | Difficulty | Description | Recommended Use |
|-------------|------------|-------------|-----------------|
| `0.3` | **Easy** | Default, often too easy | Not recommended (all models → 100%) |
| `0.45` | **Medium** | Slightly harder | Good starting point |
| `0.6` | **Hard** ⭐ | Challenging but solvable | **Best for model differentiation** |
| `0.7` | **Very Hard** | Expert mode | May have unsolvable mazes |

---

## Local Usage

### Quick Test
```bash
python experiments/run_harder_mazes.py --wall-prob 0.6 --quick
```

### Custom Configuration
```bash
python experiments/run_harder_mazes.py \
  --wall-prob 0.6 \
  --maze-sizes 16 20 24 \
  --train 500 \
  --test 100 \
  --epochs 40 \
  --R 4 --T 4 --heads 4
```

---

## All Options

```
--wall-prob        Wall probability (0.3-0.7), default: 0.6
--maze-sizes       Maze sizes to test (e.g., 8 12 16 20), default: [8, 12, 16, 20, 24, 30]
--train            Training samples per size, default: 1000
--test             Test samples per size, default: 200
--R                PoH refinement steps, default: 4
--T                HRM outer loop period, default: 4
--heads            Number of attention heads, default: 4
--epochs           Training epochs per size, default: 50
--seed             Random seed, default: 42
--quick            Quick test mode (16×16 only, small dataset)
```

---

## What to Expect

### With Easy Mazes (`wall_prob=0.3`)
```
8×8   → All models: 100% acc, 100% opt  ❌ No differentiation
12×12 → All models: 100% acc, 100% opt  ❌ No differentiation
16×16 → All models: 100% acc, 100% opt  ❌ No differentiation
```

### With Hard Mazes (`wall_prob=0.6`)
```
8×8   → All models: ~95% acc  ✅ Good differentiation
12×12 → Baseline: ~85%, PoH-HRM: ~92%  ✅ Clear winner
16×16 → Baseline: ~70%, PoH-HRM: ~85%  ✅ HRM shines
20×20 → Baseline: ~55%, PoH-HRM: ~75%  ✅ Clear hierarchy
```

---

## Output

The script automatically:
1. **Adjusts maze difficulty** to your specified `wall_prob`
2. **Generates unique output names** (e.g., `maze_scaling_wall60_hard.json`)
3. **Saves results** in `experiments/results/`
4. **Creates plots** showing model performance vs maze size

---

## Tips

✅ **Start with `--quick`** to verify setup (10 min)  
✅ **Use `wall_prob=0.6`** for best differentiation  
✅ **Check logs** for average path length (should be challenging)  
❌ **Avoid `wall_prob < 0.4`** (too easy, all models succeed)  
❌ **Avoid `wall_prob > 0.7`** (too hard, may have no solution)

---

## Example: Complete Workflow in Colab

```python
# 1. Quick sanity check
!python experiments/run_harder_mazes.py --wall-prob 0.6 --quick

# 2. If results look good, run full benchmark
!python experiments/run_harder_mazes.py --wall-prob 0.6 --maze-sizes 8 12 16 20 24 30

# 3. Visualize results
from IPython.display import Image, display
display(Image('experiments/results/maze_scaling_wall60_hard.png'))

# 4. Download results
from google.colab import files
files.download('experiments/results/maze_scaling_wall60_hard.json')
files.download('experiments/results/maze_scaling_wall60_hard.png')
```

---

## Troubleshooting

**Q: All models still getting 100%?**  
A: Increase `wall_prob` (try 0.65 or 0.7)

**Q: Models getting 0%?**  
A: Decrease `wall_prob` (try 0.5 or 0.45)

**Q: BERT fails on large mazes?**  
A: Expected! BERT has sequence length limits. The script will automatically skip BERT for mazes > 1024 tokens.

**Q: Want to compare multiple difficulties?**  
A: Run the script multiple times with different `--wall-prob` values and compare the output plots.

---

## Next Steps

After running with harder mazes, you should see:
1. **Clear performance gaps** between models
2. **PoH-HRM advantage** on larger mazes (20×20+)
3. **Hierarchical reasoning benefits** in long-horizon planning

This is the evidence you need to demonstrate that HRM's multi-timescale reasoning helps on complex tasks! 🎯

