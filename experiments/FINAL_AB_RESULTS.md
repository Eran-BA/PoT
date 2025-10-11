# Final A/B Comparison Results

**Author**: Eran Ben Artzy  
**License**: Apache 2.0

## Task

**Partial Observability Sorting**:
- Array length: 12
- Mask rate: 50% (randomly mask half the values)
- Training: 1000 samples, 40 epochs
- Model: d_model=128, n_heads=4
- Seeds: 5 for statistical robustness

## Results Summary

| Configuration | Iters | Grad Mode | Kendall-τ (mean±CI) | Δ vs Baseline | Parameters |
|---------------|-------|-----------|---------------------|---------------|------------|
| **Baseline** | N/A | N/A | **0.144 ± 0.013** | - | 199,552 |
| PoT | 2 | HRM (last) | 0.149 ± 0.021 | +0.005 (+3.6%) | 332,548 |
| PoT | 4 | HRM (last) | 0.146 ± 0.008 | +0.002 (+1.1%) | 332,548 |
| PoT | 4 | Full BPTT | **RUNNING...** | TBD | 332,548 |

### Statistical Significance

**All differences so far are NOT statistically significant** - confidence intervals overlap substantially.

## Key Findings

### 1. HRM Gradients May Be Limiting

With HRM-style (last-iterate) gradients:
- 2 iterations: +0.005 improvement
- 4 iterations: +0.002 improvement (worse!)

**Hypothesis**: HRM only backprops through last iteration, which may not provide enough signal for learning.

**Test**: Full BPTT with 4 iterations (currently running)

### 2. Task is Extremely Hard

Both models achieve only **~14-15% Kendall-τ** (random is 0%, perfect is 1.0). This indicates:
- Strong information bottleneck with 50% masking
- Current model scale (d_model=128) insufficient
- Task may require fundamentally different approaches

### 3. More Iterations Don't Help (with HRM)

4 iterations performed **worse** than 2 iterations under HRM:
- 2 iters: 0.149 ± 0.021
- 4 iters: 0.146 ± 0.008

**Possible reasons**:
- HRM gradient truncation gets worse with more iterations
- Vanishing gradients through long chains
- Optimization difficulty

## Honest Assessment

**Current results do NOT support a strong PoT advantage on this task.**

Possible conclusions:
1. **Task mismatch**: PoT designed for dependency parsing, not missing data imputation
2. **Scale issue**: Need larger models (d_model > 256) for complex reasoning
3. **Gradient flow**: HRM may not be appropriate for this task
4. **Information bottleneck**: 50% masking too severe for any architecture

## Next Experiments

### Currently Running ✅

**Full BPTT with 4 iterations**: Testing if gradient flow is the issue

### If Full BPTT Doesn't Help

1. **Easier task** (30% masking instead of 50%)
2. **Larger models** (d_model=256-512)
3. **Different task** (dependency parsing - original PoT domain)
4. **Accept results** and report honestly

## Publication Strategy

### If Full BPTT Shows Improvement

> "On partial observability sorting with full gradient flow, PoT with 4 iterations achieves X.XXX Kendall-τ vs baseline's 0.144±0.013, demonstrating that iterative refinement requires proper gradient propagation."

### If No Significant Improvement

**Option A: Honest negative result (recommended)**
> "We conducted rigorous A/B testing on partial observability sorting (5 seeds, identical training). PoT with various configurations (2-4 iterations, HRM vs full BPTT) shows no significant advantage over baseline (Kendall-τ: ~0.14-0.15, CIs overlap). This suggests the task's information bottleneck may be too severe for architectural differences to matter at this scale."

**Option B: Focus on other tasks**
> Move to dependency parsing (original PoT domain) where advantages are more likely

**Option C: Frame as exploratory**
> "We establish baseline performance on challenging partial observability tasks (~15% Kendall-τ) and identify that gradient flow and model scale are critical factors for future work."

## Lessons Learned

1. **Fair comparison is hard**: Parameter matching, gradient flow, initialization all matter
2. **Task selection matters**: Too easy → no difference, too hard → no difference  
3. **HRM may not be universal**: Last-iterate gradients work for some tasks, not others
4. **Absolute scores matter**: Low absolute performance limits architectural differentiation
5. **Statistical rigor is essential**: Single-seed results can be misleading

## Data Availability

All results saved in `experiments/results/`:
- `fair_ab_baseline.csv` + `_summary.json`
- `fair_ab_pot_full_bptt.csv` + `_summary.json` (2 iters, HRM)
- `fair_ab_pot_bptt_4iters.csv` + `_summary.json` (4 iters, full BPTT - running)

## Reproducibility

Full protocol documented in `FAIR_AB_PROTOCOL.md`.

To reproduce:
```bash
bash experiments/run_fair_ab.sh
```

Or manually:
```bash
# Baseline
python experiments/fair_ab_comparison.py --model baseline --seeds 1 2 3 4 5 ...

# PoT (HRM)
python experiments/fair_ab_comparison.py --model pot --max_inner_iters 4 --seeds 1 2 3 4 5 ...

# PoT (Full BPTT)
python experiments/fair_ab_comparison.py --model pot --max_inner_iters 4 --use_full_bptt --seeds 1 2 3 4 5 ...
```

---

**Status**: Awaiting full BPTT results to make final determination.

**Updated**: 2025-01-11

