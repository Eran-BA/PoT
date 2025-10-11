# Journey Summary: Building Fair A/B Comparisons for PoT

**Author**: Eran Ben Artzy  
**Date**: January 2025  
**License**: Apache 2.0

## Executive Summary

We conducted extensive experiments to find tasks where **Pointer-over-Heads (PoT) Transformer shows measurable advantages** over baseline architectures. This document chronicles the journey, results, and lessons learned.

---

## The Journey

### Phase 1: Initial Sorting Experiments (Arrays)

**Goal**: Test PoH on simple array sorting  
**Result**: ❌ **Task too easy** - both models hit ~99% accuracy

| Task | Baseline | PoH | Outcome |
|------|----------|-----|---------|
| Clean sort (8 elements) | 99.9% | 99.9% | Saturated |
| Duplicates (100 samples) | 93.6% | 93.6% | Tied |
| Duplicates (50 samples) | 80.7% | 83.2% | **+2.4%** ⚠️ Marginal |

**Key Finding**: Easy tasks don't differentiate architectures.

---

### Phase 2: Fixing the 1/12 Trap (Decoder Semantics)

**Problem**: Initial sorting attempt stuck at ~8.3% (1/12 random guessing)  
**Root Cause**: Using dependency parsing semantics (token → head) instead of pointer decoder semantics (rank → position with coverage)

**Fix**: Implemented proper decoder:
- Rank-conditioned queries: `q_t = rank_embed(t)`
- Coverage masking: prevent re-selection
- Stable argsort: deterministic tie-breaking

**Result**: ✅ Both models immediately hit 100% on clean data

**Lesson**: **Semantics matter more than architecture!**

---

### Phase 3: Finding Genuinely Hard Tasks

**Goal**: Push into regimes where architectural differences matter

#### Task: Partial Observability (50% masked)

**Setup**: Sort array where 50% of values are randomly masked (-999 sentinel)

**Single-seed results:**
- Baseline: 26.6% Kendall-τ
- PoH (4 iters): 33.8% Kendall-τ
- **Advantage**: +7.2% (+27% relative) ✅

**Why this worked:**
- Genuinely hard (both struggle ~25-35%)
- Requires reasoning under uncertainty
- Iterative refinement should help

**Issue**: Single seed, no statistical validation yet

---

### Phase 4: Fair A/B Comparison Protocol

**Goal**: Rigorous, reproducible comparison with statistical validation

**Protocol Design:**
- ✅ Same data/task (controlled by seeds)
- ✅ Parameter-matched models
- ✅ Identical training (LR, epochs, optimizer, clip)
- ✅ PoT config: max_inner_iters=2, grad_mode=last (HRM)
- ✅ 5 seeds for statistical robustness
- ✅ Best dev Kendall-τ → test metrics
- ✅ Report: mean ± 95% CI

**Implementation:**
- `fair_ab_comparison.py`: Single runner for both models
- `compare_ab_results.py`: Statistical analysis
- `run_fair_ab.sh`: One-command execution
- `FAIR_AB_PROTOCOL.md`: Complete documentation

---

## Final Results (Multi-Seed)

### Partial Observability Sorting (Array Length 12, 50% Masked)

| Configuration | Iterations | Grad Mode | Kendall-τ (mean±CI) | Δ vs Baseline | Significant? |
|---------------|------------|-----------|---------------------|---------------|--------------|
| **Baseline** | N/A | N/A | **0.144 ± 0.013** | - | - |
| PoT | 2 | HRM (last) | 0.149 ± 0.021 | +0.005 (+3.6%) | ❌ No |
| PoT | 4 | HRM (last) | 0.146 ± 0.008 | +0.002 (+1.1%) | ❌ No |
| PoT | 4 | Full BPTT | **RUNNING** | TBD | TBD |

### Statistical Assessment

**All HRM configurations show NO significant improvement:**
- Confidence intervals overlap substantially
- Differences are within noise
- p > 0.05 for all comparisons

---

## Key Findings

### 1. Task Selection is Critical

**Goldilocks Problem:**
- Too easy (>95% accuracy) → No room for differentiation
- Too hard (<15% accuracy) → Both models fail similarly
- Just right (25-75%) → Can see differences, but still hard

### 2. HRM Gradients May Limit Learning

**Observation**: More iterations don't help with HRM
- 2 iters: 0.149 ± 0.021
- 4 iters: 0.146 ± 0.008 (worse!)

**Hypothesis**: Last-iterate gradients don't provide enough signal

**Test**: Full BPTT currently running

### 3. Absolute Scores Matter

With both models at ~14-15% Kendall-τ:
- Signal-to-noise ratio is low
- Architectural differences are harder to detect
- Statistical power is limited

### 4. Single-Seed Results Can Mislead

**Single seed** (preliminary): PoH +7.2% advantage  
**Multi-seed** (rigorous): PoH +0.2-0.5% advantage (not significant)

**Lesson**: Always use multiple seeds!

---

## Lessons Learned

### What Worked

✅ **Honest science**: Report negative results  
✅ **Rigorous protocol**: Multi-seed, parameter matching, identical training  
✅ **Comprehensive testing**: Multiple configurations (iterations, gradient modes)  
✅ **Proper decoder semantics**: Fixed the 1/12 trap  
✅ **Genuinely hard tasks**: Partial observability is a valid stress test  

### What Didn't Work

❌ **Easy tasks**: Clean sorting saturates too quickly  
❌ **Very hard tasks**: Compositional f(x) - both fail (~14%)  
❌ **HRM with many iterations**: Performance degraded  
❌ **Small model scale**: d_model=128 may be insufficient  

### What We Don't Know Yet

❓ **Full BPTT**: Will proper gradient flow help?  
❓ **Larger models**: Would d_model=256+ show clearer differences?  
❓ **Original domain**: Would dependency parsing work better?  
❓ **Easier partial obs**: Would 30% masking show clearer advantages?  

---

## Next Steps (Pending Full BPTT Results)

### Option 1: Full BPTT Shows Improvement (Δ > 0.02)

✅ **Publish as-is**
- Frame around gradient flow importance
- Emphasize rigorous methodology
- Report multi-seed statistics

### Option 2: Full BPTT Shows No Improvement (Δ < 0.01)

**Path A: Accept Results (Recommended)**
- Report honest negative result
- Valuable for community (what doesn't work)
- Shows rigorous methodology

**Path B: Move to Dependency Parsing**
- Original PoT domain
- Likely to show clearer advantages
- Use existing UD English-EWT

**Path C: Easier Partial Observability**
- 30% masking instead of 50%
- Higher absolute scores
- Clearer differentiation possible

---

## Contributions to Science

1. **Fair comparison methodology**: Detailed protocol for architectural A/B testing
2. **Task difficulty spectrum**: Characterized easy/hard/just-right regimes
3. **Decoder semantics**: Fixed critical bug in sorting formulation
4. **Gradient flow insights**: HRM may not be universal
5. **Statistical rigor**: Multi-seed validation with confidence intervals

---

## Files Created

### Core Framework
- `experiments/fair_ab_comparison.py`: Main A/B runner
- `experiments/compare_ab_results.py`: Statistical analysis
- `experiments/run_fair_ab.sh`: Automated execution

### Results
- `experiments/results/fair_ab_baseline.csv` + `_summary.json`
- `experiments/results/fair_ab_pot.csv` + `_summary.json` (HRM 2-iter)
- `experiments/results/fair_ab_pot_full_bptt.csv` + `_summary.json` (HRM 4-iter)
- `experiments/results/fair_ab_pot_bptt_4iters.csv` + `_summary.json` (Full BPTT - running)

### Documentation
- `experiments/FAIR_AB_PROTOCOL.md`: Complete methodology
- `experiments/FINAL_AB_RESULTS.md`: Results summary
- `experiments/REALISTIC_ASSESSMENT.md`: Honest analysis
- `experiments/COMPREHENSIVE_EXPERIMENTS.md`: Experimental design

### Task Implementations
- `experiments/sort_pointer_fixed.py`: Proper pointer decoder
- `experiments/genuinely_hard_tasks.py`: Partial obs, compositional, etc.
- `experiments/noisy_toposort.py`: Topological sort with noise
- `experiments/high_performance_partial_obs.py`: Scaled experiments

---

## Publication Strategy

### If Full BPTT Works

> "On partial observability sorting, Pointer-over-Heads with full gradient flow (4 iterations) achieves X.XX±X.XX Kendall-τ vs baseline's 0.144±0.013 (p<0.05, n=5 seeds), demonstrating that iterative refinement benefits from proper backpropagation."

### If Full BPTT Doesn't Work

> "We conducted rigorous A/B testing on partial observability sorting (5 seeds, parameter-matched, identical training). PoT with various configurations shows no significant advantage (Kendall-τ ~0.14-0.15, CIs overlap), suggesting architectural differences may not overcome the task's information bottleneck at current model scales. We establish baselines (~15% correlation) for future work and identify gradient flow, model scale, and task design as critical factors for architectural comparisons."

---

## Acknowledgments

This journey demonstrated that:
- **Honest science > cherry-picked results**
- **Methodology matters as much as results**
- **Negative results are still valuable**
- **Rigorous testing reveals truth**

The fair A/B comparison framework we built is a contribution in itself, regardless of the specific PoT results.

---

## Current Status

**Awaiting**: Full BPTT results (4 iterations, full gradient flow)

**Expected completion**: ~10-15 minutes

**Next decision point**: Compare full BPTT vs baseline to determine final path

---

**The journey continues...**

