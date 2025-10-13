# PoH Architecture Summary

**Author:** Eran Ben Artzy  
**Year:** 2025  
**Status:** Production-ready ✅

## Overview

This document describes the modular, production-ready **Pointer-over-Heads (PoH)** architecture implemented in this repository. The design follows a clean hierarchy: **PoHBlock → PoHStack → IterRefiner**, making it a drop-in replacement for PyTorch's `TransformerEncoder` while adding head-wise routing and iterative refinement.

---

## Architecture Hierarchy

```
IterRefiner                # Top-level: K inner iterations + optional ACT halting
  ↓
PoHStack                   # Stack of N PoH blocks (like TransformerEncoder)
  ↓
PoHBlock (×N)              # Single block with head-wise routing
  ├─ MultiheadAttention    # Standard PyTorch MHA
  ├─ HeadRouter            # Lightweight routing: [B, T] → [B, T, H]
  ├─ head_gain             # Learnable per-head scaling (H params)
  └─ FeedForward           # Standard FFN (optionally bias-free for param parity)
```

---

## Core Components

### 1. PoHBlock (Drop-in for TransformerEncoderLayer)

**File:** `src/pot/modules/block.py`

A single transformer block with **head-wise routing**:

```python
cfg = PoHConfig(
    d_model=512,
    n_heads=8,
    d_ff=2048,
    route_mode="soft",  # or "topk"
    route_topk=2,       # for topk mode
    route_temp=1.0,     # for soft mode
)

block = PoHBlock(cfg)
out, stats = block(x)  # x: [B, T, d_model]
```

**Key features:**
- Standard MHA + FFN with pre/post normalization
- **HeadRouter** produces per-token, per-head routing logits: `[B, T, H]`
- Routing modes:
  - **Soft:** Temperature-annealed softmax (differentiable, sums to 1)
  - **Top-k:** Sparse binary mask (selects top-k heads per token)
- Learned per-head gains: `head_gain` (H scalar parameters)
- Returns `stats` dict with:
  - `route_logits`: Raw routing scores
  - `route`: Routing weights (soft) or mask (top-k)
  - `route_entropy_mean`: Routing entropy across tokens/heads
  - `attn_entropy_mean`: Attention entropy

**Parameter parity:**
- HeadRouter: `d_model → d_model/4 → n_heads` (tiny MLP)
- Optional bias stripping in FFN to keep params tight
- **Measured delta vs baseline: 0.27% for depth=6** ✅

---

### 2. PoHStack (Like TransformerEncoder)

**File:** `src/pot/modules/block.py`

Stack of `N` PoH blocks with optional **shared router**:

```python
stack = PoHStack(cfg, depth=6)
out, stats_all = stack(x)  # stats_all: list of per-block stats
```

**Key features:**
- If `cfg.share_router=True`, all blocks use the same `HeadRouter` (saves params)
- If `cfg.share_router=False`, each block has its own router
- Returns list of stats dicts (one per block)

---

### 3. IterRefiner (Inner Refinement + ACT)

**File:** `src/pot/modules/block.py`

Wraps a stack and applies **K inner refinement steps** with optional **Adaptive Computation Time (ACT)** halting:

```python
refiner = IterRefiner(stack, max_inner_iters=3, act=True, threshold=0.99, penalty=0.01)
out, inner_stats = refiner(x, return_inner_stats=True)
```

**Key features:**
- **Without ACT:** Runs exactly `K` iterations (simple loop)
- **With ACT:** Dynamically halts per token when confidence threshold reached
  - Halting controlled by learned `halt_proj: [d_model] → [1]`
  - Ponder cost penalty encourages early halting
  - Returns weighted sum of states
- `return_inner_stats=True` collects per-iteration metrics:
  - `inner_step`: iteration number
  - `route_entropy_mean`: averaged across blocks
  - `attn_entropy_mean`: averaged across blocks
  - `halted_frac`: fraction of tokens halted (ACT only)
  - `ponder_cost`: ACT penalty term

---

## HeadRouter Design

**File:** `src/pot/modules/block.py`

Lightweight MLP that produces per-head routing logits:

```python
class HeadRouter(nn.Module):
    def __init__(self, d_model, n_heads, share_proj=True):
        hid = max(32, d_model // 4) if share_proj else d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, hid, bias=True),
            nn.ReLU(),
            nn.Linear(hid, n_heads, bias=False),
        )
    
    def forward(self, x):  # x: [B, T, d_model]
        return self.proj(x)  # [B, T, H]
```

**Why it's lightweight:**
- Hidden dim = `max(32, d_model/4)` by default
- Only 2 linear layers + ReLU
- Total params: `d_model × hid + hid × n_heads ≈ 0.27% overhead`

---

## Routing Modes

### Soft Routing (Differentiable)

```python
def soft_route(scores, temperature=1.0):
    return F.softmax(scores / temperature, dim=-1)  # [B, T, H]
```

- **Properties:**
  - Differentiable end-to-end
  - Weights sum to 1 over heads
  - Temperature control: lower = sharper
- **Use case:** When you want gradient flow through all heads

### Top-k Routing (Sparse)

```python
def topk_route(scores, k):
    topk_vals, topk_idx = scores.topk(k, dim=-1)
    mask = torch.zeros_like(scores)
    mask.scatter_(-1, topk_idx, 1.0)
    return mask  # [B, T, H]
```

- **Properties:**
  - Binary mask (0 or 1)
  - Exactly `k` heads active per token
  - Sparse computation (can save FLOPs in custom kernels)
- **Use case:** When you want hard routing and sparsity

---

## ACT Halting (Adaptive Computation Time)

Inspired by [Graves 2016](https://arxiv.org/abs/1603.08983), adapted for PoH:

**Algorithm:**
1. Initialize halting probability `p = 0` for each token
2. At each inner step `t`:
   - Compute halt signal: `sigmoid(halt_proj(h_t))` → `[B, T]`
   - Update `p` based on remaining probability budget
   - Accumulate weighted state: `sum += weight * h_t`
   - Early stop if all tokens halted (`p >= threshold`)
3. Add ponder cost penalty to loss: `penalty * mean(p)`

**Benefits:**
- Tokens can halt early if confident → saves compute
- Differentiable (straight-through estimator for thresholding)
- Encourages efficiency via ponder cost

**Hyperparameters:**
- `act_threshold` (default: 0.99): Cumulative probability to stop
- `act_penalty` (default: 0.01): Weight of ponder cost in loss

---

## Inner-Loop Logging

**File:** `src/pot/logging/innerloop.py`

CSV logger for per-inner-step telemetry:

```python
from src.pot.logging import InnerLoopLogger, InnerStepRow, grad_global_norm

with InnerLoopLogger("results/run1/innerloop.csv") as logger:
    for step in range(num_steps):
        out, inner_stats = refiner(x, return_inner_stats=True)
        loss.backward()
        grad_norm = grad_global_norm(model)
        
        for s in inner_stats:
            row = InnerStepRow(
                run_id="run1",
                epoch=epoch,
                global_step=step,
                inner_step=s["inner_step"],
                batch_size=batch_size,
                loss=float(loss.item()),
                grad_norm=float(grad_norm),
                attn_entropy_mean=s.get("attn_entropy_mean"),
                halted_frac=s.get("halted_frac"),
                uas_probe=uas_probe,  # optional
                ms_forward=ms_forward,  # optional
            )
            logger.log(row)
```

**Logged metrics:**
- `loss`: Loss at each inner step
- `grad_norm`: Global gradient norm
- `attn_entropy_mean`: Attention entropy (averaged across blocks)
- `route_entropy_mean`: Routing entropy
- `halted_frac`: Fraction halted (ACT only)
- `uas_probe`: Quick UAS on tiny batch (optional sanity check)
- `ms_forward`: Forward pass timing

**Why it's useful:**
- Diagnose diminishing returns (does inner step 2→3 help?)
- Track convergence dynamics (inner vs outer)
- Detect issues (entropy collapse, gradient vanishing)
- Safe to tail during training (`tail -f innerloop.csv`)

---

## Visualization

**Script:** `scripts/plot_inner_vs_outer.py`

Generates publication-quality plots:

```bash
python scripts/plot_inner_vs_outer.py --csv results/run1/innerloop.csv --epochs 1,10,20,40
```

**Plots generated:**
1. **Inner convergence per epoch:** Does loss drop across inner steps?
   - Shows marginal improvements (e.g., step 1→2: -0.5, step 2→3: -0.05)
   - Efficiency ratio: `(last_drop / first_drop) * 100%`
2. **Outer learning curve:** Loss at last inner step vs global step
3. **UAS probe curve:** Sanity check on tiny batch (if enabled)
4. **Attention entropy:** Detect over-confidence (entropy collapse)
5. **Timing per inner step:** Forward pass latency (helps cost/benefit analysis)

**Output:**
- `inner_convergence_epoch{E}.png` for each epoch
- `outer_learning_curve.png`
- `uas_probe_curve.png` (if available)
- `attention_entropy.png` (if available)
- `timing_per_inner_step.png` (if available)

---

## Parameter Parity

**Comparison:** PoH vs baseline TransformerEncoder (d=512, h=8, ff=2048, depth=6)

| Model | Params | Delta |
|-------|--------|-------|
| Baseline TransformerEncoder | 18,914,304 | — |
| PoH (shared router) | 18,965,680 | **+0.27%** ✅ |

**Breakdown of added params:**
- HeadRouter: `512 × 128 + 128 × 8 = 66,560` (shared across 6 blocks)
- head_gain: `8 × 6 = 48` (per-block scalars)
- Total overhead: **51,376 params (0.27%)**

**Design choices for parity:**
- Shared router across blocks
- Small hidden dim in router (`d_model/4`)
- Optional bias stripping in FFN
- Minimal per-head gains (H scalars)

**Test:** `tests/test_poh_modules.py::test_poh_vs_baseline_param_count`
- Asserts delta ≤ 1%
- Prints exact param counts

---

## Usage Examples

**File:** `examples/poh_usage.py`

Run all examples:
```bash
PYTHONPATH=/Users/rnbnrzy/Desktop/PoT:$PYTHONPATH python examples/poh_usage.py
```

**Examples included:**
1. **Basic usage:** Drop-in for TransformerEncoder
2. **Top-k routing:** Sparse head selection
3. **Inner refinement:** K=3 iterations
4. **ACT halting:** Adaptive computation
5. **Inner-loop logging:** Integration with CSV logger
6. **Param parity check:** Verify <1% delta

---

## Tests

**File:** `tests/test_poh_modules.py`

Run tests:
```bash
pytest tests/test_poh_modules.py -v
```

**Test suite (11/11 passed):**

### Parameter Parity (2 tests)
- ✅ `test_poh_vs_baseline_param_count`: Delta ≤ 1%
- ✅ `test_shared_router_reduces_params`: Shared < separate

### Routing (3 tests)
- ✅ `test_soft_routing_sums_to_one`: Soft weights sum to 1
- ✅ `test_topk_routing_sparsity`: Exactly k heads per token
- ✅ `test_temperature_affects_sharpness`: Lower temp → sharper

### ACT Halting (2 tests)
- ✅ `test_act_reduces_computation`: Halting happens
- ✅ `test_no_act_runs_all_iters`: Without ACT, runs K iters

### Inner Refinement (2 tests)
- ✅ `test_refinement_changes_output`: K>1 changes output
- ✅ `test_stats_collection`: Stats contain entropy/routing

### Drop-in Compatibility (2 tests)
- ✅ `test_forward_signature_compatibility`: Interface matches
- ✅ `test_gradient_flow`: Gradients flow correctly

---

## Ablation Guide

The architecture is designed for easy ablation studies:

| Ablation | How to configure |
|----------|-----------------|
| Soft vs top-k routing | `cfg.route_mode = "soft"` or `"topk"` |
| Number of heads | `cfg.route_topk = k` (for topk mode) |
| Temperature annealing | `cfg.route_temp = T` (for soft mode) |
| Shared vs separate router | `cfg.share_router = True/False` |
| ACT halting | `cfg.act_halting = True/False` |
| ACT threshold | `cfg.act_threshold = 0.99` (default) |
| ACT penalty | `cfg.act_penalty = 0.01` (default) |
| Inner iterations | `IterRefiner(..., max_inner_iters=K)` |
| Pre vs post norm | `cfg.norm_type = "pre"` or `"post"` |

**Recommended ablation matrix:**

```python
# Fair A/B comparison (all else equal)
configs = [
    # Baseline
    {"route_mode": "soft", "max_inner_iters": 1, "act_halting": False},
    
    # Inner refinement
    {"route_mode": "soft", "max_inner_iters": 3, "act_halting": False},
    
    # Top-k routing
    {"route_mode": "topk", "route_topk": 2, "max_inner_iters": 1},
    
    # ACT halting
    {"route_mode": "soft", "max_inner_iters": 5, "act_halting": True, "act_threshold": 0.99},
    
    # Full PoH
    {"route_mode": "topk", "route_topk": 2, "max_inner_iters": 3, "act_halting": True},
]
```

---

## Integration with Training

To use in your training loop:

```python
from src.pot.modules import PoHConfig, PoHStack, IterRefiner
from src.pot.logging import InnerLoopLogger, InnerStepRow, grad_global_norm

# 1. Build model
cfg = PoHConfig(d_model=512, n_heads=8, d_ff=2048, route_mode="topk", route_topk=2)
stack = PoHStack(cfg, depth=6)
refiner = IterRefiner(stack, max_inner_iters=3)

# 2. Training loop with logging
with InnerLoopLogger("results/run1/innerloop.csv") as logger:
    for epoch in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Forward with inner stats
            out, inner_stats = refiner(batch.x, return_inner_stats=True)
            loss = criterion(out, batch.y)
            loss.backward()
            
            grad_norm = grad_global_norm(refiner)
            optimizer.step()
            
            # Log each inner step
            for s in inner_stats:
                row = InnerStepRow(
                    run_id=args.run_id,
                    epoch=epoch,
                    global_step=global_step,
                    inner_step=s["inner_step"],
                    batch_size=batch.size,
                    loss=float(loss.item()),
                    grad_norm=float(grad_norm),
                    attn_entropy_mean=s.get("attn_entropy_mean"),
                    halted_frac=s.get("halted_frac"),
                )
                logger.log(row)
```

---

## File Structure

```
src/pot/
  modules/
    __init__.py          # Exports: PoHConfig, PoHBlock, PoHStack, IterRefiner, HeadRouter
    block.py             # Core architecture implementation
  logging/
    __init__.py          # Exports: InnerLoopLogger, InnerStepRow, grad_global_norm
    innerloop.py         # CSV logging for inner-loop telemetry

scripts/
  plot_inner_vs_outer.py  # Visualization script

examples/
  poh_usage.py           # 6 usage examples

tests/
  test_poh_modules.py    # 11 tests (all passing)
```

---

## Next Steps

**Immediate:**
1. Wire this into your existing training scripts (e.g., `fair_ab_comparison.py`)
2. Run experiments with the CSV logger enabled
3. Generate plots with `plot_inner_vs_outer.py`
4. Check for diminishing returns in inner iterations

**Medium-term:**
1. Add strong baselines (Dozat-Manning biaffine parser)
2. Run multi-seed experiments for variance
3. Measure latency/throughput (tokens/sec)
4. Generate comparison tables (UAS/LAS, params, speed)

**Long-term:**
1. Custom CUDA kernel for top-k routing (save actual FLOPs)
2. Routing heatmap visualization
3. Error analysis (which sentences benefit from routing?)
4. Non-projective languages (Czech, Ancient Greek)

---

## Citation

If you use this architecture, please cite:

```bibtex
@misc{benartzy2025poh,
  title={Pointer-over-Heads: Iterative Refinement with Head-Wise Routing},
  author={Eran Ben Artzy},
  year={2025},
  url={https://github.com/Eran-BA/PoT}
}
```

---

## License

Apache 2.0

---

**Status:** Production-ready ✅  
**Test coverage:** 11/11 passing ✅  
**Parameter parity:** 0.27% delta ✅  
**Documentation:** Complete ✅

