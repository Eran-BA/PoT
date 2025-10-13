# HRM Controller Testing Guide

**Author**: Eran Ben Artzy  
**Date**: 2025  
**License**: Apache 2.0

---

## Quick Start

```bash
# Run diagnostic smoke test
make smoke-hrm

# Run full HRM unit tests (requires pytest)
make test-hrm

# Run all tests
make test
```

---

## Test Hierarchy

### Level 1: Unit Tests (`tests/test_hrm_pointer_controller.py`)

**Purpose**: Validate core HRM controller functionality in isolation

**Tests**:
1. **`test_basic_forward_shapes`** - Correct output shapes and aux dict
2. **`test_high_level_updates_every_T_steps`** - Multi-timescale H-module updates
3. **`test_topk_sparsity_and_temperature`** - Sparse routing and temperature effects
4. **`test_temperature_schedule`** - Temperature annealing and clamping
5. **`test_gradients_flow`** - Backprop through controller
6. **`test_state_persistence`** - State carries across iterations
7. **`test_entropy_regularization`** - Entropy computation
8. **`test_different_input_shapes`** - Pooled vs sequence inputs
9. **`test_masked_pooling`** - Variable-length sequence handling
10. **`test_topk_variants`** - Different sparsity levels

**Run**:
```bash
PYTHONPATH=. pytest tests/test_hrm_pointer_controller.py -v
```

**Expected**: All tests pass (10/10)

---

### Level 2: Diagnostic Smoke Test (`tools/hrm_diag_smoke.py`)

**Purpose**: Quick sanity check of expected HRM behavior

**Checks**:
- ✅ Entropy decreases over iterations (routing sharpens)
- ✅ Max probability increases (sharper routing with temperature annealing)
- ✅ Routing shows head preference (non-uniform specialization)
- ✅ H-module updates at correct intervals (multiples of T)

**Run**:
```bash
PYTHONPATH=. python tools/hrm_diag_smoke.py
```

**Expected Output**:
```
================================================================================
RUNNING 8 ITERATIONS WITH TEMPERATURE ANNEALING
================================================================================

Iter 0: entropy=1.0874, max_prob=0.4278, temp=2.000, top_heads=[1, 2, 6] ← H-UPDATE
Iter 1: entropy=1.0876, max_prob=0.4307, temp=1.900, top_heads=[1, 5, 2] 
Iter 2: entropy=1.0852, max_prob=0.4433, temp=1.805, top_heads=[1, 5, 2] 
Iter 3: entropy=1.0891, max_prob=0.4159, temp=1.715, top_heads=[1, 5, 2] ← H-UPDATE
...
```

**What to look for**:
- H-UPDATE markers appear every T=3 steps (0, 3, 6)
- Entropy trend is generally downward
- Routing stabilizes on preferred heads (e.g., head 1 appears consistently)
- Temperature decreases from 2.0 → 0.7+

---

### Level 3: Integration Tests

**Purpose**: Validate HRM controller within PointerBlock

Create `tests/test_pointer_block_hrm.py`:

```python
import torch
import pytest
from src.models.pointer_block import PointerMoHTransformerBlock
from src.models.layers import HRMState

def test_pointer_block_with_hrm():
    """Test PoH block with HRM controller."""
    B, L, D, H = 2, 11, 64, 8
    block = PointerMoHTransformerBlock(
        d_model=D,
        n_heads=H,
        use_hrm=True,
        hrm_T=3,
        routing_topk=4
    )
    x = torch.randn(B, L, D)
    state = None
    
    x_out, aux = block(x, mask=None, iters=4)
    
    assert x_out.shape == (B, L, D)
    assert "routing_entropy" in aux
    assert "routing_logits" in aux
```

**Run**:
```bash
PYTHONPATH=. pytest tests/test_pointer_block_hrm.py -v
```

---

### Level 4: End-to-End Training Tests

**Purpose**: Verify HRM works in full training pipeline

#### Quick Smoke Training

```bash
python scripts/train.py \
  --model pot \
  --use_hrm \
  --hrm_T 4 \
  --routing_topk 3 \
  --max_inner_iters 2 \
  --epochs 3 \
  --train_samples 100 \
  --batch_size 8 \
  --output_csv experiments/results/smoke_hrm.csv
```

**Expected**:
- Training completes without errors
- CSV contains columns: `model, epoch, uas, entropy, temperature`
- Entropy decreases across epochs
- Model trains (loss decreases)

#### A/B Comparison

```bash
# Baseline (standard controller)
python scripts/train.py \
  --model pot \
  --controller standard \
  --max_inner_iters 4 \
  --epochs 20 \
  --seeds 1 2 3 \
  --output_csv experiments/results/ab_standard.csv

# HRM controller
python scripts/train.py \
  --model pot \
  --use_hrm \
  --hrm_T 4 \
  --routing_topk 3 \
  --max_inner_iters 4 \
  --epochs 20 \
  --seeds 1 2 3 \
  --temperature_schedule \
  --output_csv experiments/results/ab_hrm.csv

# Compare
python experiments/compare_ab_results.py \
  --baseline experiments/results/ab_standard.csv \
  --treatment experiments/results/ab_hrm.csv
```

**Expected**:
- HRM shows competitive or better performance
- HRM entropy starts high, decreases over epochs
- Head usage shows specialization (non-uniform)

---

## Troubleshooting

### Issue: `test_high_level_updates_every_T_steps` fails

**Symptom**: z_H changes at wrong intervals

**Debug**:
```python
# Add to test
for t in range(T + 2):
    state = step_once(state)
    print(f"Step {t}, z_H norm: {state.z_H.norm().item():.6f}")
```

**Fix**: Check `_maybe_update_H` logic, ensure `(step % T) == 0` condition

---

### Issue: Gradients don't flow

**Symptom**: `test_gradients_flow` fails

**Debug**:
```python
# Check each module
for name, param in ctrl.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm().item():.6f}")
```

**Fix**: Ensure loss depends on alphas, check for detached states

---

### Issue: Routing collapses to single head

**Symptom**: All probability mass on one head, entropy → 0

**Debug**:
```bash
PYTHONPATH=. python tools/hrm_diag_smoke.py
# Check entropy progression
```

**Fix**:
```python
# Increase temperature
controller.set_temperature(1.5)

# Add entropy regularization
loss += 1e-3 * aux['entropy']

# Reduce top-k sparsity
topk=4  # instead of 2
```

---

### Issue: H-module never updates

**Symptom**: z_H stays constant

**Debug**:
```python
print(f"Step: {state.step[0].item()}, T: {ctrl.T}")
print(f"Update needed: {(state.step[0] % ctrl.T) == 0}")
```

**Fix**: Ensure state is threaded correctly across iterations

---

## Performance Benchmarks

### Expected Timings (CPU, MacBook M1)

| Test | Time | Status |
|------|------|--------|
| Unit tests (all) | ~5s | ✅ |
| Diagnostic smoke | ~0.5s | ✅ |
| Integration test | ~1s | ✅ |
| E2E smoke train (3 epochs, 100 samples) | ~30s | ✅ |

### Memory Usage

| Configuration | Parameters | Memory (forward) |
|--------------|-----------|------------------|
| d_ctrl=64, n_heads=8 | ~71K | ~50MB |
| d_ctrl=128, n_heads=8 | ~150K | ~80MB |
| d_ctrl=256, n_heads=16 | ~600K | ~150MB |

---

## Continuous Integration

### GitHub Actions Workflow

Add to `.github/workflows/ci.yml`:

```yaml
- name: Test HRM Controller
  run: |
    PYTHONPATH=. pytest tests/test_hrm_pointer_controller.py -v
    PYTHONPATH=. python tools/hrm_diag_smoke.py
```

---

## Test Coverage

Run with coverage:

```bash
PYTHONPATH=. pytest tests/test_hrm_pointer_controller.py --cov=src.models.layers --cov-report=html
```

**Target**: >90% coverage for `HRMPointerController` class

---

## Next Tests to Add

1. **Stress test**: Very long sequences (L=1000), check memory
2. **Numerical stability**: Extreme temperatures (0.01, 100.0)
3. **Batch size variations**: B=1, B=128
4. **Device transfer**: CPU → GPU → CPU
5. **Mixed precision**: FP16 training compatibility
6. **Checkpointing**: Save/load state correctly
7. **Distributed training**: DDP compatibility

---

## References

- HRM Paper: [arXiv:2506.21734](https://arxiv.org/abs/2506.21734)
- PyTorch Testing Best Practices: [PyTorch Docs](https://pytorch.org/docs/stable/testing.html)
- Pytest Documentation: [pytest.org](https://docs.pytest.org/)

---

**Last Updated**: 2025  
**Maintainer**: Eran Ben Artzy

