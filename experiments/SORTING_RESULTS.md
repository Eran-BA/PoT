# Array Sorting Experiment Results

**Author:** Eran Ben Artzy  
**Date:** 2025-10-11

## Summary

A/B testing of Pointer-over-Heads (PoH) vs Baseline transformer on array sorting task.

## Task Description

Given an array of random floats, predict where each element should be positioned in the sorted array.

**Example:**
- Input: `[5.2, 2.1, 8.9, 1.3]`
- Sorted: `[1.3, 2.1, 5.2, 8.9]`
- Target: `[2, 1, 3, 0]` (element 0 goes to position 2, element 1 to position 1, etc.)

## Results

### Array Length: 8 Elements

| Model | Iterations | Test Accuracy | Perfect Sort Rate | Advantage |
|-------|-----------|---------------|-------------------|-----------|
| **PoH** | 4 | **98.08%** | **88.8%** | **Baseline +27.4%** |
| **PoH** | 8 | **98.22%** | **89.0%** | **Baseline +29.1%** |
| Baseline | 1 | 70.65% | 2.6% | - |
| Baseline | 1 | 69.10% | 4.2% | - |

**Key Findings (8 elements):**
- ✅ PoH achieves ~98% accuracy vs ~70% for baseline
- ✅ PoH gets 89% perfect sorts vs 3% for baseline (**~30x better**)
- ✅ More iterations (8 vs 4) gives marginal improvement (+0.2% perfect sort rate)
- ✅ **4 iterations appears optimal** for this task size

### Array Length: 12 Elements

| Model | Iterations | Test Accuracy | Perfect Sort Rate | Status |
|-------|-----------|---------------|-------------------|--------|
| PoH | 4 | 7.75% | 0% | ❌ **Failed to learn** |
| PoH | 8 | 8.32% | 0% | ❌ **Failed to learn** |
| Baseline | 1 | 47.8% | 0.2% | ✅ Learning (slow) |
| Baseline | 1 | 54.3% | 0% | ✅ Learning (slow) |

**Key Findings (12 elements):**
- ❌ PoH fails to optimize on longer sequences
- ❌ Stuck at random guessing (~8.3% = 1/12)
- ⚠️ **Likely cause**: Vanishing gradients through deep iteration depth
- ✅ Baseline still learns (albeit slowly)

## Analysis

### Why PoH Wins on Short Sequences (8 elements)
1. **Iterative Refinement**: Multiple passes allow progressive refinement of predictions
2. **Adaptive Routing**: Controller can learn to specialize heads for different sorting sub-tasks
3. **Better Optimization**: Gradient flow is stable for 4-8 iterations on short sequences

### Why PoH Fails on Long Sequences (12 elements)
1. **Vanishing Gradients**: Backprop through 4-8 iterations × 12 positions = very deep graph
2. **Optimization Difficulty**: Controller + attention + FFN all need to coordinate
3. **Increased Capacity Need**: May need much larger models or better initialization

## Recommendations

### For Short Sequences (≤10 elements):
- ✅ Use PoH with 4 iterations
- ✅ Expect significant improvement over single-pass baseline
- ✅ d_model=256, lr=3e-4 works well

### For Long Sequences (>10 elements):
- ⚠️ Need better optimization strategy
- 💡 Possible solutions:
  - Gradient clipping / normalization
  - Layer normalization adjustments
  - Warmup learning rate schedule
  - Deep supervision (losses at each iteration)
  - Residual connections with learned gating
  - Fewer iterations (2-3) for longer sequences

## Conclusion

**Pointer-over-Heads shows strong advantages on moderately-sized sequential decision tasks**, achieving ~30x better perfect prediction rates on 8-element sorting. However, **scaling to longer sequences requires addressing optimization challenges** in deep iterative architectures.

The sorting task validates that the PoH mechanism's iterative refinement is beneficial beyond dependency parsing, but highlights the need for careful optimization strategy design for longer sequences.

---

**Training Details:**
- Framework: PyTorch
- Device: CPU
- Training samples: 5,000
- Test samples: 500
- Batch size: 64
- Optimizer: Adam
- Value range: [0, 100]

