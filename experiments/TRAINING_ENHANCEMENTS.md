# Training Enhancements Summary

## Overview

Enhanced PoH training with MLM-U inspired techniques for improved performance on maze solving tasks.

## Completed Enhancements

### 1. ✅ Label Smoothing + Cosine LR Warmup
- **Files**: `run_12x12_8m_benchmark.py`, `parameter_scaling_benchmark.py`
- **Implementation**:
  - `nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)`
  - Linear warmup to peak LR over warmup steps
  - Cosine decay to 0.1×LR over remaining steps
- **CLI Flags**:
  - `--lr` (default: 1e-3)
  - `--label-smoothing` (default: 0.1)
  - `--warmup-steps` (default: 500 for 12x12, 2000 for scaling)

### 2. ✅ Multi-Horizon Supervision (k-step)
- **Files**: `run_12x12_8m_benchmark.py`, `parameter_scaling_benchmark.py`
- **Implementation**:
  - For each position i, supervise k-step predictions (k ∈ [1..K])
  - Current index = path[:, i], Target = path[:, i+k]
  - Gather logits at current index, predict target
  - Average loss over all valid horizons
- **CLI Flag**: `--multi-horizon` (default: 1 for 12x12, 3 for scaling)

### 3. ✅ CNN Maze Encoder (Global Conditioning)
- **Files**: `run_12x12_8m_benchmark.py`, `parameter_scaling_benchmark.py`
- **Implementation**:
  - Small Conv2d stack: 1→16→32 channels with ReLU + AdaptiveAvgPool
  - Projects to d_model and broadcasts to all tokens: `h = h + maze_cond`
  - Applied to both Baseline and PoH models
- **CLI Flag**: `--maze-enc`
- **Note**: Currently enabled by default in `parameter_scaling_benchmark.py`

### 4. ✅ Validity-Aware Loss and Decoding
- **Files**: `run_12x12_8m_benchmark.py`, `parameter_scaling_benchmark.py`
- **Implementation**:
  - **Training**: Build per-sample valid neighbor mask (4-neighbors that are passable)
  - Set invalid positions to -1e4 before computing CE loss
  - Uses numerically stable log_softmax + NLL for masked CE
  - **Decoding**: Already uses greedy decoding with valid neighbor filtering
- **CLI Flag**: `--validity-mask`

### 5. ✅ Routing Entropy Regularization (PoH only)
- **Files**: `run_12x12_8m_benchmark.py`, `parameter_scaling_benchmark.py`
- **Implementation**:
  - Call refiner with `return_inner_stats=True`
  - Extract `route_entropy_mean` from stats
  - Add `λ * entropy` to loss (encourages sharper routing)
  - Optional annealing: linearly reduce λ over training
- **CLI Flags**:
  - `--route-ent-weight` (default: 0 for 12x12, 5e-4 for scaling)
  - `--ent-anneal` (flag to enable annealing)

### 6. ✅ Depth-First Parameter Parity
- **Files**: `run_12x12_8m_benchmark.py`, `parameter_scaling_benchmark.py`
- **Strategy**: Keep PoH depth constant, reduce width (d_model, n_heads) for parameter parity
- **Rationale**: Preserves hierarchical reasoning capability while matching parameter count
- **Results**: PoH with depth=4 (63.16% acc) >> PoH with depth=3 (50.88% acc)

### 7. ✅ Device Robustness
- **Files**: Both scripts
- **Implementation**: CUDA → MPS → CPU fallback with `--cpu` override
- **MPS Support**: Added for Apple Silicon GPUs

## Validated Results

### 12x12 Maze (~8M params, 30 epochs)

| Model | Params | Config | Accuracy | Optimality |
|-------|--------|--------|----------|------------|
| Baseline | 7.21M | depth=4 | 57.89% | 52.63% |
| **PoH-HRM** | **7.04M** | **d_model=320, n_heads=5, depth=4** | **63.16%** ✅ | **54.39%** ✅ |

**Key Insight**: PoH outperforms baseline by +5.27% accuracy with fewer parameters when depth is preserved.

## Usage Examples

### 12x12 Benchmark (All Features)
```bash
python -u experiments/run_12x12_8m_benchmark.py \
  --train 400 --test 80 --epochs 30 --R 4 --T 4 --cpu \
  --lr 1e-3 --label-smoothing 0.1 --warmup-steps 500 \
  --multi-horizon 1 --maze-enc --validity-mask \
  --route-ent-weight 0 \
  --output experiments/results/benchmark_12x12_8m
```

### Large/XL Scaling (All Features)
```bash
python -u experiments/parameter_scaling_benchmark.py \
  --maze-size 16 --train 1000 --test 100 --epochs 50 \
  --R 4 --T 4 --seed 42 \
  --lr 1e-3 --label-smoothing 0.1 --warmup-steps 2000 \
  --multi-horizon 3 --validity-mask \
  --route-ent-weight 5e-4 --ent-anneal \
  --output experiments/results/parameter_scaling
```

## References

- MLM-U paper: https://github.com/facebookresearch/maze_navigation_MLMU
- Inspired by multi-horizon supervision, validity masking, and CNN maze conditioning
- Depth-first parity: Our novel contribution showing depth > width for hierarchical models

## Next Steps

- [ ] Run Large/XL benchmarks with enhanced training
- [ ] Document CLI flags in script docstrings
- [ ] Update README with training enhancement guide
- [ ] Consider adding to Colab notebooks

