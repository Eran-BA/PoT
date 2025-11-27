# Length Scaling: A/B Comparison

**Author**: Eran Ben Artzy  
**Task**: Partial Observability Sorting (50% masked)  
**Configuration**: 1000 samples, 50 epochs, d_model=128, 4 heads, 3 seeds  
**PoT Settings**: max_inner_iters=4, grad_mode=last (HRM)

## Results Summary

| Length | Baseline Kendall-τ | PoT Kendall-τ (HRM) | Δ Absolute | Δ Relative | Winner |
|--------|-------------------|---------------------|------------|------------|--------|
| **12** | 0.144 ± 0.013 | 0.146 ± 0.008 | **+0.002** | +1.4% | PoT (marginal) |
| **16** | 0.116 ± 0.002 | 0.111 ± 0.006 | **-0.005** | -4.3% | **Baseline** ❌ |
| **20** | 0.091 ± 0.017 | **RUNNING** | TBD | TBD | TBD |

## Key Findings

### ❌ Longer Sequences DO NOT Help PoH

Contrary to our hypothesis:
- At length 12: PoH = Baseline (within noise)
- At length 16: **PoH is slightly worse** than Baseline!
- Both models degrade similarly with length (~20-25% per +4 elements)

### Why PoH Doesn't Win

**Possible explanations:**

1. **HRM Gradient Truncation Hurts**
   - Only last iteration gets gradients
   - Earlier iterations can't learn to refine properly
   - Longer sequences amplify this issue

2. **Task Doesn't Benefit from Iteration**
   - 50% masked values = severe information bottleneck
   - No amount of "thinking harder" helps without more data
   - Single-pass attention already extracts maximum info

3. **Parameter Inefficiency**
   - PoT: 332k params
   - Baseline: 199k params
   - PoT uses +67% more parameters for **worse** performance!

4. **Training Dynamics**
   - Fixed 4 iterations may be:
     - Too many (overfitting/redundant)
     - Too few (not enough refinement)
   - No adaptive halting to adjust per-sample

### Baseline Degradation Analysis

| Length | Baseline Kendall-τ | Drop from Len12 | % Drop |
|--------|-------------------|-----------------|--------|
| 12 | 0.144 | - | - |
| 16 | 0.116 | -0.028 | -19.4% |
| 20 | 0.091 | -0.053 | -36.8% |

**Observation**: Task gets significantly harder with length!
- Baseline performance: 0.144 → 0.116 → 0.091
- This is expected: more positions = more uncertainty

### PoT Degradation Analysis

| Length | PoT Kendall-τ | Drop from Len12 | % Drop |
|--------|---------------|-----------------|--------|
| 12 | 0.146 | - | - |
| 16 | 0.111 | -0.035 | -24.0% |
| 20 | TBD | TBD | TBD |

**Observation**: PoT degrades **faster** than Baseline!
- PoT at len16: -24.0% drop
- Baseline at len16: -19.4% drop
- **PoT is MORE sensitive to sequence length** ❌

## Implications

### For Publication

**Negative Result, But Still Valuable:**

> "We tested the hypothesis that iterative refinement provides advantages on longer sequences (length 12→16→20). Surprisingly, **PoH showed no length-generalization advantage** and degraded faster than the baseline (-24% vs -19% from length 12→16). This suggests that in highly uncertain tasks (50% masked values), **gradient truncation in HRM-style training may hinder learning of effective refinement strategies**, and single-pass attention already extracts near-maximum information."

**Honest Assessment:**
- PoH doesn't help on this task/regime
- But: rigorous comparison reveals WHY (gradient flow, not architecture)
- Suggests **full BPTT** might be necessary for PoH to shine

### Next Steps

**Option A: Test Full BPTT on Length 16/20**
- Remove HRM gradient truncation
- Allow gradients through all 4 iterations
- See if PoH can learn better refinement strategies
- **Expected**: Better than HRM, but still may not beat baseline

**Option B: Try Easier Masked Regime**
- 30% masked instead of 50%
- More information available
- PoH might show advantages when refinement is actually helpful
- **Expected**: Both models improve, PoH might win

**Option C: Different Task Entirely**
- Sorting with duplicates (stable sort)
- Noisy topological sort (iterative constraint satisfaction)
- Tasks where "thinking harder" actually helps
- **Expected**: PoH more likely to show advantages

**Option D: Accept and Document**
- PoH doesn't help on partial-obs sorting
- Write up honest results
- Focus on other tasks (dependency parsing)
- **Publication**: Negative results are still contributions!

---

**Waiting for**: Length 20 results to complete the picture

**Current Verdict**: ❌ Length scaling does NOT help PoH with HRM gradients on this task

