# HRM (Sapient) vs PoT-HRM (This Repo)

References: [HRM Paper](https://arxiv.org/pdf/2506.21734), [HRM Repo](https://github.com/sapientinc/HRM/tree/main)

## Summary
- Both architectures employ two-timescale recurrence (fast L, slow H) and iterative refinement over R inner steps with an H update period T.
- Our maze experiments integrate HRM into a PoH routing controller inside a Transformer-like stack; the original HRM paper uses a unified recurrent architecture tailored to puzzle datasets (ARC, Sudoku, Maze) with curated training and infrastructure.
- Key differences are practical rather than conceptual: learned compute (ACT/pondering), temperature/hard-routing schedules, scale, and training infra.

## Overlap (What Matches)
- Two-timescale controller: fast/slow modules with configurable period T. [Paper §Model]
- Iterative refinement (R) via inner loops; hierarchical control by H. [Paper Fig.1]
- O(R) training with BPTT.
- O(1) truncated BPTT variant (now implemented as TBPTT-style: outer residual accumulation + detach between inner steps).
- Stabilizers: AdamW, cosine LR with warmup, label smoothing, gradient clipping (mirrors best practices hinted in repo configs).
- Regularizers: routing-entropy penalties and annealing available.
- Curriculum-ready data pipeline (we use `maze-dataset`; HRM repo ships dataset builders).

## Differences (Gaps + Our Recent Additions)
- Learned compute (ACT/pondering):
  - HRM: employs learned halting/pondering signals (or equivalent scheduling) in full runs.
  - PoT: added ACT-style ponder regularization option; fixed R by default. (New: `--ponder-weight`).
- Routing schedule:
  - HRM: uses temperature schedules and (likely) phase-dependent routing sharpness.
  - PoT: added temperature anneal (2.0→1.0) and optional hard routing with straight-through. (New: `--hard-route`).
- Dynamic period T / gating of H-updates:
  - HRM: hints at learned/dynamic control of high-level updates and hierarchical convergence.
  - PoT: T is user-specified; no learned gating yet (roadmap item).
- Architecture scale and infra:
  - HRM: 27M+ params, multi-GPU, FA2/FA3, W&B pipelines, specific puzzle encoders.
  - PoT: parity-controlled experiments (~7–1000M elsewhere), MPS/CUDA; simpler maze encoder; A/B tooling.
- Task heads and decoders:
  - HRM: task-specific puzzle modules; optimized evaluation harness.
  - PoT: unified maze policy decoder with validity masking and greedy rollouts.

## Empirical Finding Here (12×12 Maze)
- O(R) (full BPTT): PoH > Baseline (67.90% acc / 51.85% opt vs 55.56% / 44.44%).
- O(1) (last-iter TBPTT): initially degraded; we aligned with TBPTT residuals + router reset + temp anneal + optional ST/ponder runs are in progress.
- Sparse supervision works but trails O(R).

## Action Items to Reach Feature Parity
1) Add learned T-gating / dynamic H updates (matching hierarchical convergence policy).
2) Integrate ACT halting (probabilistic) rather than scalar ponder regularization only.
3) Enable EMA and curriculum (min_path schedule) for stability/throughput.
4) Port HRM puzzle evaluation harness to unify metrics.

## Commands (O(1) variants now running)
```
# TBPTT-style O(1)
python experiments/run_12x12_8m_benchmark.py ... --last-iter-only --output experiments/results/comparison_O1_memory_TBPTT

# TBPTT + Router reset
python experiments/run_12x12_8m_benchmark.py ... --last-iter-only --output experiments/results/comparison_O1_memory_TBPTT_reset

# TBPTT + Router reset + Temperature anneal
python experiments/run_12x12_8m_benchmark.py ... --last-iter-only --output experiments/results/comparison_O1_memory_TBPTT_reset_temp

# TBPTT + ST hard routing + ponder regularization
env OMP_NUM_THREADS=8 python experiments/run_12x12_8m_benchmark.py ... --last-iter-only --hard-route --ponder-weight 0.01 \
  --output experiments/results/comparison_O1_memory_TBPTT_ST_ponder
```

## Bottom Line
- Conceptual alignment is strong; practical differences are mostly in learned compute and scheduling.
- We’ve implemented the key HRM-inspired tricks (TBPTT O(1), temperature schedules, hard routing, ponder penalty) and are validating them.
