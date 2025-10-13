# Synthetic Task Examples

This directory contains experiments on **synthetic tasks** to demonstrate PoH's capabilities on controlled problems.

## Partial-Observability Sorting

**Task:** Sort an array when only a subset of elements are visible.

**Metric:** Mask-aware Kendall-τ (ranking correlation over observable pairs)

### Quick Start

```bash
cd /Users/rnbnrzy/Desktop/PoT

# Baseline (no inner iterations)
python examples/synthetic/fair_ab_comparison.py \
  --model baseline \
  --array_len 12 \
  --mask_rate 0.5 \
  --epochs 40 \
  --seeds 1 2 3

# PoH (with inner refinement)
python examples/synthetic/fair_ab_comparison.py \
  --model pot \
  --array_len 12 \
  --mask_rate 0.5 \
  --max_inner_iters 4 \
  --epochs 40 \
  --seeds 1 2 3
```

### Results

See `results/` for experiment CSVs.

**Key findings:**
- PoH shows gains on longer sequences (L=20) with high masking (50%)
- Inner iterations help when the task requires multi-step reasoning
- Diminishing returns after ~8-12 iterations

### Files

- `fair_ab_comparison.py` - Main experiment script
- `sort_pointer_fixed.py` - PointerDecoderSort model
- `ranking_utils.py` - Mask-aware Kendall-τ and RankNet loss
- `results/` - Experiment CSVs

### Analysis

```bash
# Plot results
python scripts/plot_results.py

# Generate tables
python scripts/make_readme_tables.py
```

### Notes

This is a **synthetic task** to demonstrate PoH's iterative refinement. For real-world NLP tasks (dependency parsing, etc.), see the main README.

**Position encoding:** Use `pos_encoding="none"` for sorting (permutation-invariant task).

---

**Back to main repo:** [../../README.md](../../README.md)

