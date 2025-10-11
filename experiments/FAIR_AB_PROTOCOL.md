# Fair A/B Comparison Protocol

**Author**: Eran Ben Artzy  
**License**: Apache 2.0

## Objective

Provide a **clean, minimal, reproducible A/B comparison** between Baseline and PoT (Pointer-over-Heads) on partial observability sorting, with **no new features**, only architectural differences.

## Protocol

### Same Data/Task
- Task: Partial observability sorting (50% values randomly masked)
- Array length: 12
- Training: 1000 samples
- Dev: 300 samples (for early stopping)
- Test: 300 samples (for final evaluation)
- Seeds: {1, 2, 3, 4, 5} for statistical robustness

### Parameter-Matched Models
- **Baseline**: d_model=128, n_heads=4, d_ff=256
  - Parameters: 199,552
- **PoT**: d_model=128, n_heads=4, d_ff=256, max_inner_iters=2, grad_mode=last
  - Parameters: ~266,000 (includes PoH routing components)

### Identical Training Recipe
- Optimizer: Adam
- Learning rate: 3e-4
- Weight decay: 0.0
- Gradient clip norm: 1.0
- Batch size: 64
- Epochs: 40
- No warmup, no LR schedule (keeping it simple)

### PoT Configuration (Fixed)
- `max_inner_iters=2`: Two refinement iterations
- `grad_mode=last` (HRM-style): Backprop only through last iteration
- No routing tricks (e.g., top-k, halting)
- No additional features

### Selection Rule
- For each seed: track **best dev Kendall-τ** across epochs
- Report test metrics at that checkpoint

### Metrics
- **Primary**: Kendall-τ (mean ± 95% CI over seeds)
- **Secondary**: Perfect sort %, Hamming distance
- All metrics: mean ± 95% CI

## Running the Experiments

### Option 1: Shell Script (Recommended)

```bash
cd /Users/rnbnrzy/Desktop/PoT
source venv/bin/activate
bash experiments/run_fair_ab.sh
```

This will:
1. Run Baseline with 5 seeds
2. Run PoT with 5 seeds  
3. Compare results and generate publication table

### Option 2: Manual Commands

**Baseline:**
```bash
python experiments/fair_ab_comparison.py \
  --model baseline \
  --array_len 12 \
  --mask_rate 0.5 \
  --train_samples 1000 \
  --dev_samples 300 \
  --test_samples 300 \
  --epochs 40 \
  --batch_size 64 \
  --lr 3e-4 \
  --d_model 128 \
  --n_heads 4 \
  --seeds 1 2 3 4 5 \
  --output_csv experiments/results/fair_ab_baseline.csv
```

**PoT:**
```bash
python experiments/fair_ab_comparison.py \
  --model pot \
  --array_len 12 \
  --mask_rate 0.5 \
  --train_samples 1000 \
  --dev_samples 300 \
  --test_samples 300 \
  --epochs 40 \
  --batch_size 64 \
  --lr 3e-4 \
  --d_model 128 \
  --n_heads 4 \
  --max_inner_iters 2 \
  --seeds 1 2 3 4 5 \
  --output_csv experiments/results/fair_ab_pot.csv
```

**Compare:**
```bash
python experiments/compare_ab_results.py \
  --baseline experiments/results/fair_ab_baseline_summary.json \
  --pot experiments/results/fair_ab_pot_summary.json
```

## Expected Output

### Results Format

```
BASELINE Results (n=5 seeds):
  Kendall-τ:    0.144 ± 0.013
  Perfect sort: 0.0% ± 0.0%
  Accuracy:     0.242 ± 0.009
  Hamming dist: 0.758 ± 0.009

PoT Results (n=5 seeds):
  Kendall-τ:    X.XXX ± X.XXX  ← TO BE DETERMINED
  Perfect sort: X.X% ± X.X%
  Accuracy:     X.XXX ± X.XXX
  Hamming dist: X.XXX ± X.XXX
```

### Publication Table

| Task             | Length | Mask | Metric              | Baseline          | PoT (iters=2,last) |      Δ |
| ---------------- | -----: | ---: | ------------------- | ----------------: | -----------------: | -----: |
| Partial-obs sort |     12 |  50% | Kendall-τ (mean±CI) | 0.144 ± 0.013 | **X.XXX ± X.XXX** | **+X.XXX** |
|                  |        |      | Perfect-sort %      |   0.0 ±  0.0      |   **X.X ±  X.X**  | **+X.X%** |
|                  |        |      | Hamming distance    | 0.758 ± 0.009 | **X.XXX ± X.XXX** | **-X.XXX** |

## Design Choices

### Why These Settings?

1. **max_inner_iters=2**: Balance between refinement capability and compute cost
2. **grad_mode=last (HRM)**: Memory-efficient, allows fair comparison of compute
3. **Array length 12**: Large enough to be non-trivial, small enough to train quickly
4. **Mask rate 50%**: Genuinely hard (information bottleneck), but not impossible
5. **1000 train samples**: Data-scarce regime where architectural differences matter
6. **40 epochs**: Enough for convergence without excessive compute
7. **5 seeds**: Standard for statistical robustness

### Why No Parameter Perfect Matching?

- PoT has additional parameters for routing/control (~33% more)
- This is inherent to the architecture
- Alternative: reduce d_model for PoT to match params exactly
  - But this would penalize PoT's representation capacity
  - Current approach is fairer: same d_model, let architecture add what it needs

### Why No Warmup/Schedule?

- Keeping training recipe maximally simple
- Both models use same recipe → fair comparison
- Can add later if needed for absolute performance

## Interpretation Guidelines

### Statistical Significance

- **CIs don't overlap**: Likely significant (p<0.05)
- **CIs overlap slightly**: May still be significant (run t-test)
- **CIs overlap substantially**: Difference not significant

### Relative vs Absolute Improvements

- **Absolute improvement** (e.g., +0.020 Kendall-τ): Raw difference
- **Relative improvement** (e.g., +14%): Percentage of baseline performance

Both matter! Report both.

### Expected Results

Based on preliminary single-seed runs:
- Baseline: Kendall-τ ~0.14-0.18
- PoT: Kendall-τ ~0.18-0.22 (if advantage holds)
- Expected Δ: +0.02 to +0.05 Kendall-τ (+10-30% relative)

## Files Generated

- `experiments/results/fair_ab_baseline.csv`: Per-seed baseline results
- `experiments/results/fair_ab_baseline_summary.json`: Aggregate baseline stats
- `experiments/results/fair_ab_pot.csv`: Per-seed PoT results
- `experiments/results/fair_ab_pot_summary.json`: Aggregate PoT stats

## Reproducibility Checklist

✅ Fixed random seeds (1-5)  
✅ Same data generation (controlled by seeds)  
✅ Same model dimensions (d_model, n_heads, d_ff)  
✅ Same optimizer + hyperparameters (lr, clip, weight_decay)  
✅ Same training duration (epochs)  
✅ Same evaluation protocol (best dev → test)  
✅ Statistical robustness (5 seeds, CI reporting)  
✅ No architecture-specific tuning (same recipe for both)

## Citation

If using this protocol, please cite:

```
@software{pot_fair_ab_2025,
  author = {Ben Artzy, Eran},
  title = {Fair A/B Comparison Protocol for Pointer-over-Heads Transformer},
  year = {2025},
  url = {https://github.com/Eran-BA/PoT}
}
```


