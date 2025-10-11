# Length Scaling Results: Partial Observability Sorting

**Author**: Eran Ben Artzy  
**License**: Apache 2.0

## Hypothesis

**Longer sequences should show clearer PoH advantages** because:
1. More positions to reason about
2. Longer-range dependencies
3. More opportunity for iterative refinement to help
4. Baseline may struggle more with longer sequences

## Experimental Setup

**Task**: Partial Observability Sorting (50% masked)

**Configuration**:
- Training: 1000 samples, 50 epochs
- Batch size: 48
- Learning rate: 3e-4
- Model: d_model=128, n_heads=4
- PoT: max_inner_iters=4, grad_mode=last (HRM)
- Seeds: 3 for faster iteration

## Results

### Baseline Performance vs Length

| Array Length | Baseline Kendall-τ (mean±CI) | Perfect Sort % | Task Difficulty |
|--------------|------------------------------|----------------|-----------------|
| 12 | 0.144 ± 0.013 | 0.0% | Moderate |
| 16 | **0.116 ± 0.002** | 0.0% | Hard |
| 20 | **0.091 ± 0.017** | 0.0% | Very Hard |

**Observation**: Performance degrades significantly with length!
- 12→16: -19% relative drop
- 16→20: -22% relative drop
- Random baseline: 0.0 Kendall-τ

### PoH Performance vs Length (HRM, 4 iterations)

| Array Length | PoH Kendall-τ (mean±CI) | Δ vs Baseline | Advantage |
|--------------|-------------------------|---------------|-----------|
| 12 | 0.146 ± 0.008 | +0.002 | +1.4% |
| 16 | **RUNNING** | TBD | TBD |
| 20 | **PENDING** | TBD | TBD |

## Expected Outcomes

### Scenario A: PoH Degrades Similarly
- Len 16: ~0.12 Kendall-τ
- Len 20: ~0.09 Kendall-τ
- **Interpretation**: Both architectures hit same bottleneck

### Scenario B: PoH Degrades Less
- Len 16: ~0.13-0.14 Kendall-τ (+10-20% over baseline)
- Len 20: ~0.11-0.12 Kendall-τ (+20-30% over baseline)
- **Interpretation**: ✅ Iterative refinement helps on longer sequences!

### Scenario C: PoH Degrades More
- Len 16: ~0.10-0.11 Kendall-τ
- Len 20: ~0.07-0.08 Kendall-τ
- **Interpretation**: ❌ HRM gradient truncation hurts more on long sequences

## Why This Matters

**Longer sequences are the IDEAL test for iterative architectures:**
- Single-pass attention has limited receptive field
- Iterative refinement can propagate information further
- Multi-hop reasoning becomes more important

**If PoH shows advantages on length 16-20**, this is a **strong publication result** because:
1. Task is genuinely hard (baseline ~10-12%)
2. Length generalization is critical for real applications
3. Clear architectural difference in scaling behavior

## Analysis Plan

Once results complete:

1. **Plot Kendall-τ vs Length** for both models
2. **Compute relative advantage** at each length
3. **Statistical testing**: Does advantage increase with length?
4. **Error analysis**: Where does each model fail?

## Publication Framing

### If PoH Shows Scaling Advantage

> "On partial observability sorting, PoH maintains higher correlation with ground truth as sequence length increases (Kendall-τ: 0.14@len12, 0.13@len16, 0.12@len20) compared to baseline's steeper degradation (0.14@len12, 0.12@len16, 0.09@len20), demonstrating that iterative refinement provides better length generalization under uncertainty."

### If Both Degrade Similarly

> "We characterize the scaling behavior of pointer-based sorting under partial observability, showing both architectures degrade similarly with length (~20-25% per +4 elements), indicating the information bottleneck is task-intrinsic rather than architectural."

---

**Status**: Length 16 and 20 experiments running...

**ETA**: ~15-20 minutes for both to complete

**Next**: Compare results and determine if scaling shows PoH advantages

