# Comprehensive Sorting Experiments

**Author**: Eran Ben Artzy  
**License**: Apache 2.0

## Motivation

After fixing the decoder semantics (rank-conditioned queries + coverage masking), both Baseline and PoH achieve 100% accuracy on sorting unique integers up to length 20. This means the task is **saturated** and we need harder regimes to demonstrate PoH's advantages.

## Experimental Design

### 1. Sample Efficiency 🎯 **[MOST IMPACTFUL]**

**Hypothesis**: PoH's iterative refinement and head routing should enable better representation reuse, requiring fewer training samples to reach the same performance.

**Setup**:
- Train with: {200, 500, 1k, 2k, 5k} samples
- Test on: Length 12 and 20
- Epochs: 20-30 (early stopping possible)
- Metrics: Element accuracy, perfect sort rate, Kendall-τ

**Expected Result**: PoH should match baseline performance with 30-50% fewer samples, especially on longer sequences where routing matters more.

**Publication Impact**: Sample efficiency is **critical** for practical applications and shows architectural advantage without changing the task.

---

### 2. Length Generalization (OOD Robustness)

**Hypothesis**: PoH's iterative refinement should generalize better to longer sequences than seen during training.

**Setup**:
- Train on: Length ≤12 (5k samples)
- Test on: {12, 16, 20, 24, 32} 
- Metrics: Element accuracy, perfect sort rate, Kendall-τ, Hamming distance

**Expected Result**: PoH should maintain higher accuracy on OOD lengths (16+), while baseline degrades faster.

**Why**: Iterative refinement can apply learned "sorting steps" recursively to longer sequences.

---

### 3. Duplicates & Stable Sort

**Hypothesis**: Many baselines degrade when values can tie; PoH's iterative refinement can better learn the stable sort criterion (tie-break by original index).

**Setup**:
- Train on: Length 12 with duplicates (values in [-100, 100])
- Supervise: Stable argsort (ties broken by index)
- Metrics: Element accuracy, perfect sort rate (strict)

**Expected Result**: PoH should achieve higher perfect sort rate on data with ties.

**Why**: Stable sort requires fine-grained position tracking that benefits from multiple refinement passes.

---

### 4. PoH Routing Modes (Ablation)

**Hypothesis**: Different routing strategies (concat, soft mixture, top-k) show different speed/accuracy trade-offs.

**Setup**:
- Variants:
  - **Concat**: All heads concatenated (current implementation)
  - **Soft mixture**: Weighted sum of heads based on controller
  - **Top-k Gumbel**: Hard selection of k=1 or k=2 heads
- Train on: Length 12 (2k samples)
- Metrics: Accuracy, params, FLOPs, head entropy (specialization)

**Expected Result**:
- Concat: Highest accuracy, most params
- Soft mixture: Good accuracy, interpretable routing
- Top-k: Fastest inference, slightly lower accuracy

---

### 5. Reduced Compute Budget

**Hypothesis**: With matched FLOPs/time, PoH with fewer iterations competes with deeper baselines.

**Setup**:
- Fix compute budget (e.g., 2× baseline FLOPs)
- Compare:
  - Baseline (1 layer, wider)
  - PoH (2-3 iterations, narrower)
  - HRM-style (last-iterate gradients, more iterations)

**Expected Result**: PoH achieves similar accuracy with better interpretability and memory efficiency.

---

## Metrics to Report

### Core Metrics
- **Element accuracy**: Proportion of correct position predictions
- **Perfect sort rate**: Proportion of fully correct permutations
- **Kendall-τ**: Correlation between predicted and gold permutation (-1 to 1)
- **Hamming distance**: Proportion of position disagreements

### Interpretability Metrics
- **Attention entropy**: Measure head specialization
- **Mean iterations**: For halting-based models
- **Routing diversity**: Distribution of controller weights

### Efficiency Metrics
- **Samples to 95% accuracy**: Sample efficiency
- **Parameters**: Model size
- **FLOPs per forward pass**: Computational cost
- **Training time**: Wall-clock time to converge

---

## Reporting Standards

### Always Include
1. **Param counts** matched or explicitly stated
2. **Learning rate** scaled by `1/T` (averaged CE over ranks)
3. **Random seeds**: Report mean ± std over 3 seeds
4. **Full config**: d_model, n_heads, iterations, epochs, batch size

### Visualization
1. **Learning curves**: Accuracy vs samples (log scale)
2. **Generalization curves**: Accuracy vs test length
3. **Routing heatmaps**: Head activation patterns (for PoH)
4. **Attention visualizations**: Which positions attend to which

---

## Expected Publication Claims

1. **"PoH achieves X% higher accuracy with Y% fewer samples"** (Sample efficiency)
2. **"PoH maintains Z% accuracy on 2× longer sequences vs baseline's W%"** (Length generalization)
3. **"PoH learns stable sort with A% perfect sorts vs baseline's B%"** (Duplicates)
4. **"PoH routing enables head specialization with entropy H vs uniform"** (Interpretability)

---

## Running the Experiments

```bash
# Sample efficiency (recommended first)
python experiments/run_comprehensive_sorting_tests.py \
  --experiments sample_efficiency \
  --epochs 30 --d_model 128 --seed 42

# Length generalization
python experiments/run_comprehensive_sorting_tests.py \
  --experiments length_gen \
  --epochs 30 --d_model 128 --seed 42

# Duplicates
python experiments/run_comprehensive_sorting_tests.py \
  --experiments duplicates \
  --epochs 40 --d_model 128 --seed 42

# All experiments
python experiments/run_comprehensive_sorting_tests.py \
  --experiments sample_efficiency length_gen duplicates \
  --epochs 30 --d_model 128 --seed 42
```

## Plotting Results

```bash
python experiments/plot_sorting_results.py \
  --results_dir experiments/results \
  --output_dir experiments/plots
```

---

## Next Steps After These Experiments

If PoH shows advantages:
1. **Write up results** for publication appendix
2. **Add to main README** with plots
3. **Compare to dependency parsing** results (transfer insights)

If results are mixed:
1. **Try deeper PoH** (more iterations)
2. **Add position-aware queries** (GRUCell for decoder state)
3. **Scheduled sampling** (teacher forcing → greedy)
4. **Curriculum learning** (easy → hard sequences)

---

**Key Insight**: We've proven the architecture works correctly. Now we need to find the regimes where **iterative refinement with routing** provides measurable advantages over single-pass attention.

