# O(R) vs O(1) vs Sparse Supervision (12x12 Mazes)

This document summarizes three controlled experiments on 12×12 maze solving with parameter parity (~8M params), comparing:
- Standard O(R) training (backprop through all refinement iterations)
- O(1) last-iteration-only training
- Sparse supervision (every 3rd step)

All runs use identical data generation, model parity, and enhanced training features.

## Configuration
- Maze size: 12×12
- Train/Test: 500 / 100 mazes (filtered; some runs yielded fewer with strict filters)
- Epochs: 50
- PoH: R=4, T=4
- Optimizer: AdamW, lr=1e-3, label_smoothing=0.1
- Scheduler: Cosine with warmup (warmup_steps=500)
- Objectives: validity masking, multi-horizon=1
- Device: Apple Silicon GPU (MPS)
- Parameter parity: PoH depth=4, width reduced to match baseline (~7.04M vs 7.21M)

## Results

### 1) Standard O(R) (all iterations)
- File: `experiments/results/comparison_OR_memory_proper/results.json`
- Command:
```
python -u experiments/run_12x12_8m_benchmark.py \
  --train 500 --test 100 --epochs 50 --R 4 --T 4 \
  --lr 1e-3 --label-smoothing 0.1 --warmup-steps 500 \
  --multi-horizon 1 --validity-mask \
  --output experiments/results/comparison_OR_memory_proper
```
- Outcome:
  - Baseline: 55.56% acc, 44.44% opt
  - PoH-HRM: 67.90% acc, 51.85% opt
  - PoH > Baseline

### 2) O(1) (last iteration only)
- File: `experiments/results/comparison_O1_memory_proper/results.json`
- Command:
```
python -u experiments/run_12x12_8m_benchmark.py \
  --train 500 --test 100 --epochs 50 --R 4 --T 4 \
  --lr 1e-3 --label-smoothing 0.1 --warmup-steps 500 \
  --multi-horizon 1 --validity-mask --last-iter-only \
  --output experiments/results/comparison_O1_memory_proper
```
- Outcome:
  - Baseline: 67.90% acc, 55.56% opt
  - PoH-HRM: 39.51% acc, 28.40% opt
  - O(1) significantly hurts PoH in this setting

### 3) Sparse supervision (every 3rd step)
- File: `experiments/results/comparison_sparse_supervision_proper/results.json`
- Command:
```
python -u experiments/run_12x12_8m_benchmark.py \
  --train 500 --test 100 --epochs 50 --R 4 --T 4 \
  --lr 1e-3 --label-smoothing 0.1 --warmup-steps 500 \
  --multi-horizon 1 --validity-mask --supervision-interval 3 \
  --output experiments/results/comparison_sparse_supervision_proper
```
- Outcome:
  - Baseline: 49.38% acc, 38.27% opt
  - PoH-HRM: 53.09% acc, 39.51% opt
  - Sparse supervision works; trails full O(R)

## Logs and Artifacts
- OR logs: `experiments/results/comparison_OR_memory_proper.log`
- O1 logs: `experiments/results/comparison_O1_memory_proper.log`
- Sparse logs: `experiments/results/comparison_sparse_supervision_proper.log`

## Takeaways
- O(R) is best for PoH: consistent win over baseline.
- O(1) reduces memory but degrades PoH performance substantially here.
- Sparse supervision reduces labeling cost with modest performance tradeoff.

## Next Steps
- Try O(1) with higher R (e.g., R=6–8) to compensate for shallower gradients.
- Combine sparse supervision with O(R) for larger mazes (16×16+).
- Scale to 16×16 using depth-first parity; reproduce trends.
