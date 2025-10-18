# Maze Benchmark Summary

## Overview
This document summarizes maze-solving benchmarks across different model configurations and maze difficulties.

---

## 30x30 Maze Benchmark (Colab, Wall Prob 0.45, Medium Difficulty)

**Dataset**: 300 training mazes, 60 test mazes, avg optimal path length: 30.1  
**Training**: 60 epochs  
**Device**: CUDA (A100-40GB)

| Model | Parameters | Accuracy | Optimality | Training Time |
|-------|-----------|----------|------------|---------------|
| **Baseline Transformer** | 4.90M | **100.00%** | **100.00%** | 6.25 min |
| **BERT** | 10.68M | **100.00%** | **100.00%** | 4.60 min |
| **PoH-HRM (R=4, T=4)** | 3.66M | **100.00%** | **100.00%** | 18.25 min |

**Key Findings:**
- ✅ All three models achieve perfect performance on this task
- ✅ PoH-HRM achieves this with **25% fewer parameters** than Baseline
- ⚠️ Task may be too easy to differentiate model capabilities at this difficulty level
- 📊 PoH-HRM requires 3x longer training time due to refinement iterations (R=4)

---

## 12x12 Maze Benchmark (Local, Higher Difficulty)

**Dataset**: 500 training mazes, 100 test mazes  
**Training**: 50 epochs, HRM-inspired features  
**Device**: MPS (Apple Silicon)

### Standard Training (O(R) Memory)
| Model | Accuracy | Optimality | Notes |
|-------|----------|------------|-------|
| **Baseline** | ~95%+ | ~90%+ | Full gradient through all steps |
| **PoH-HRM (O(R))** | ~95%+ | ~90%+ | Standard backprop through R=4 iterations |

### O(1) Memory Experiments (TBPTT-style)

| Configuration | Accuracy | Optimality | Features |
|--------------|----------|------------|----------|
| **O(1) + TBPTT + State Reset + Temp Annealing** | 46.91% | 34.57% | Last-iter-only backprop, HRM reset, temperature 2.0→1.0 |

**Key Findings:**
- ⚠️ O(1) memory mode shows degraded performance compared to full O(R) backprop
- 🔬 Temperature annealing and state reset partially recover performance
- 💡 Suggests that gradient flow through all refinement iterations is important for learning

---

## HRM (Original) on 30x30 Hard Mazes (Colab)

**Dataset**: `sapientinc/maze-30x30-hard-1k` (1000 train, 1000 test from HuggingFace)  
**Training**: 100 epochs, batch_size=32  
**Device**: CUDA (A100-80GB)  
**Status**: ✅ Training completed (3125 iterations, ~9 minutes)

**Notes:**
- HRM paper reports **70%+ accuracy** on Maze 30x30 Hard
- Our run completed successfully with HRM-specific features:
  - FlashAttention
  - AdamATan2 optimizer
  - Sparse embeddings
  - HRM's ACT-style pondering
- Evaluation metrics not yet extracted from HRM logs

---

## Comparison: PoT-HRM vs Original HRM

### Architecture Alignment
| Feature | PoT-HRM | Original HRM |
|---------|---------|--------------|
| **HRM Controller** | ✅ f_L + f_H modules | ✅ |
| **Input Normalization** | ✅ [-1, 1] mapping, pre-LayerNorm | ✅ |
| **Per-sample Standardization** | ✅ Controller inputs | ✅ |
| **Logit Clamping** | ✅ [-10, 10] | ✅ |
| **FlashAttention** | ❌ (standard attention) | ✅ |
| **AdamATan2** | ❌ (AdamW) | ✅ |
| **Sparse Embeddings** | ❌ | ✅ |
| **ACT/Pondering** | ⚠️ Approximation (`ponder_weight`) | ✅ Full ACT |
| **Hard Routing (ST)** | ✅ Optional (`--hard-route`) | ✅ |
| **Temperature Annealing** | ✅ 2.0→1.0 | ✅ |

### Parameter Efficiency
- **PoT-HRM**: 3.66M parameters
- **Baseline Transformer**: 4.90M parameters (25% more)
- **BERT**: 10.68M parameters (192% more)

**PoH-HRM achieves competitive performance with significantly fewer parameters.**

---

## Next Steps & Recommendations

1. **Increase Maze Difficulty for 30x30**:
   - Use higher `wall_prob` (0.6-0.7) or longer `min_path_length`
   - Test on HRM's actual `maze-30x30-hard` dataset for direct comparison

2. **Extract HRM Evaluation Metrics**:
   - Parse HRM checkpoint logs to get final accuracy/optimality
   - Compare directly against PoT-HRM on the same test set

3. **O(1) Memory Investigation**:
   - Current O(1) implementation shows significant performance drop
   - Consider hybrid approaches: backprop through last K iterations instead of just 1
   - Investigate if HRM paper's O(1) claim uses different training dynamics

4. **Feature Parity**:
   - Add FlashAttention support to PoT-HRM (requires CUDA)
   - Implement full ACT-style pondering (vs. current approximation)
   - Test if AdamATan2 vs AdamW makes a difference

5. **Scaling Experiments**:
   - Test on even larger mazes (40x40, 50x50)
   - Test on other sequential reasoning tasks (graph traversal, planning)

---

## Artifacts

- **30x30 Colab Results**: `/content/PoT/experiments/results/colab_maze30.json`
- **HRM Training Logs**: `/content/PoT/vendor/hrm/` (checkpoints in subdirs)
- **12x12 O(1) Results**: `experiments/results/comparison_O1_memory_*/results.json`
- **Experiment Documentation**: `experiments/EXPERIMENT_O1_OR_SPARSE.md`, `experiments/HRM_VS_PoT_REPORT.md`

---

**Generated**: October 18, 2025  
**Branch**: `scaling_parameter_size`

