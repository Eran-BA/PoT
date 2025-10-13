# HRM-Style Controller Integration Guide

**Author**: Eran Ben Artzy  
**Date**: 2025  
**License**: Apache 2.0

---

## Overview

The **HRMPointerController** is a drop-in replacement for the standard `PointerOverHeadsController` that implements **Hierarchical Reasoning Model** (HRM) principles:

- **Two-timescale recurrent modules** (fast L-module, slow H-module)
- **Separate persistent states** (z_L, z_H)
- **Multi-timescale updates**: H updates every T steps, L updates every step
- **Cross-module conditioning**: L is conditioned by H's context

This enables **deeper iterative reasoning** while maintaining computational efficiency.

---

## Key Differences from Standard Controller

| Feature | Standard Controller | HRM Controller |
|---------|-------------------|----------------|
| **State** | Single recurrent state | Dual states (z_L, z_H) |
| **Timescales** | Single-speed updates | Multi-timescale (fast L, slow H) |
| **Context** | Local per-iteration | H provides persistent context to L |
| **Updates** | Every iteration | L every step, H every T steps |
| **Complexity** | Lower parameters | ~2x parameters (two GRU cells) |

---

## Architecture

```
┌─────────────────────────────────────────┐
│         HRM Controller                  │
├─────────────────────────────────────────┤
│                                         │
│  Input x → inp_proj → x_ctrl           │
│                                         │
│  ┌─────────────────────────┐           │
│  │ High-Level (Slow) f_H   │           │
│  │ - Updates every T steps │           │
│  │ - Provides context      │           │
│  │ z_H ← GRU(x_ctrl, z_H)  │           │
│  └──────────┬──────────────┘           │
│             │ context                  │
│             ↓                           │
│  ┌─────────────────────────┐           │
│  │ Low-Level (Fast) f_L    │           │
│  │ - Updates every step    │           │
│  │ - Conditioned by z_H    │           │
│  │ z_L ← GRU([x, z_H], z_L)│           │
│  └──────────┬──────────────┘           │
│             │                           │
│             ↓                           │
│  z_L_cond = z_L + mix_gate(z_H)        │
│             │                           │
│             ↓                           │
│  logits = router(z_L_cond)             │
│  alphas = softmax(logits / T)          │
│                                         │
└─────────────────────────────────────────┘
```

---

## Usage

### Basic Initialization

```python
from src.models import HRMPointerController, HRMState

controller = HRMPointerController(
    d_model=128,
    n_heads=8,
    d_ctrl=64,           # Controller hidden dimension
    T=4,                 # H-module update period (multi-timescale)
    topk=4,              # Sparse routing (top-k heads)
    temperature_init=2.0, # Start soft
    temperature_min=0.7,  # Anneal to sharper routing
    entropy_reg=1e-3,    # Entropy regularization weight
    use_layernorm=True,
    dropout=0.1
)
```

### Forward Pass

```python
# Initialize state
state = controller.init_state(batch_size, device)

# Inner iteration loop
for t in range(max_inner_iters):
    alphas, state, aux = controller(
        x=x,                    # [B, L, d_model]
        head_outputs=head_feats, # [B, n_heads, L, d_head]
        state=state,             # Persistent HRM state
        return_aux=True
    )
    
    # Mix heads with routing weights
    mixed = (alphas.unsqueeze(-1).unsqueeze(-1) * head_feats).sum(dim=1)
    
    # Residual update
    x = x + mixed
```

### Temperature Scheduling

```python
# In trainer, schedule temperature per epoch
for epoch in range(num_epochs):
    T_epoch = max(T_min, T_init * (decay ** epoch))
    controller.set_temperature(T_epoch)
```

---

## Integration into Pointer Block

### Before (Standard Controller)

```python
class PointerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        self.controller = PointerOverHeadsController(d_model, n_heads)
    
    def forward(self, x):
        for _ in range(iters):
            logits = self.controller(x)
            alphas = F.softmax(logits, dim=-1)
            # ... mix heads
```

### After (HRM Controller)

```python
from src.models import HRMPointerController, HRMState

class PointerBlock(nn.Module):
    def __init__(self, d_model, n_heads, **kw):
        self.controller = HRMPointerController(
            d_model=d_model,
            n_heads=n_heads,
            d_ctrl=kw.get("d_ctrl", d_model),
            T=kw.get("hrm_T", 4),
            topk=kw.get("routing_topk", None),
            temperature_init=kw.get("temp_init", 2.0),
        )
    
    def forward(self, x, state=None):
        if state is None:
            state = self.controller.init_state(x.size(0), x.device)
        
        for t in range(iters):
            alphas, state, aux = self.controller(
                x, head_feats, state=state, return_aux=True
            )
            # ... mix heads
            
        return x, state, aux
```

---

## Hyperparameter Guidelines

### H-Module Period (T)

- **T=2**: Very frequent H-updates, almost synchronized
- **T=4**: Recommended default (balanced multi-timescale)
- **T=8**: Slower H-updates, stronger hierarchy
- **T > 10**: May lose context, H becomes too slow

**Rule of thumb**: `T ≈ sqrt(max_inner_iters)`

### Controller Dimension (d_ctrl)

- **d_ctrl = d_model**: Full expressiveness (default)
- **d_ctrl = d_model / 2**: More efficient, still effective
- **d_ctrl = 64**: Lightweight, good for smaller models

### Temperature Schedule

```python
# Exponential decay
T(epoch) = max(T_min, T_init * decay^epoch)

# Recommended:
T_init = 2.0      # Soft routing early
T_min = 0.7       # Sharper routing late
decay = 0.95      # Gradual annealing
```

### Top-K Routing

- **topk=None**: Dense routing (all heads)
- **topk=n_heads/2**: Moderate sparsity
- **topk=2-4**: Strong sparsity, faster, may lose flexibility

---

## Training Tweaks

### 1. Entropy Regularization

```python
# In trainer, add to loss
loss_task = ...  # your parsing/sorting loss
loss_entropy = aux['entropy']
loss = loss_task + λ * loss_entropy

# Decay λ over training
λ(epoch) = λ_init * 0.9^epoch  # e.g., 1e-3 → 1e-4
```

### 2. Deep Supervision

```python
# Collect alphas/logits across iterations
alphas_seq = []
for t in range(iters):
    alphas, state, aux = controller(...)
    alphas_seq.append(aux['router_logits'])

# Supervise each iteration
losses = [loss_fn(logits, target) for logits in alphas_seq]
loss = sum(losses) / len(losses)
```

### 3. Gradient Flow

- **HRM with last-iterate gradients**: Detach H, L states except final iteration
- **Full BPTT**: Keep gradients through all iterations (memory-intensive)

```python
# In inner loop
if use_hrm_style and t < max_iters - 1:
    state = HRMState(
        z_L=state.z_L.detach(),
        z_H=state.z_H.detach(),
        step=state.step
    )
```

---

## Comparison with Standard PoH

### When to Use HRM Controller

✅ **Use HRM when**:
- Long sequences (> 16 elements)
- Hard tasks (high uncertainty)
- Many inner iterations (≥ 8)
- Need multi-timescale reasoning

❌ **Use Standard when**:
- Short sequences
- Simple tasks
- Few iterations (≤ 4)
- Want minimal parameters

### Performance Expectations

Based on our experiments:

| Task | Standard PoH | HRM PoH | Advantage |
|------|-------------|---------|-----------|
| Sorting (len 12) | 0.146 | **TBD** | ? |
| Sorting (len 20, 12 iters) | 0.108 | **TBD** | ? |
| UD Parsing | **TBD** | **TBD** | ? |

**Next experiments**: A/B test HRM vs standard on length 20 sorting

---

## Implementation Details

### State Management

```python
@dataclass
class HRMState:
    z_L: torch.Tensor  # [B, d_ctrl] - fast module state
    z_H: torch.Tensor  # [B, d_ctrl] - slow module state
    step: torch.Tensor  # [B] - iteration counter
```

- **z_L**: Updated every iteration, conditioned by z_H
- **z_H**: Updated every T iterations, provides context
- **step**: Tracks when to update H (step % T == 0)

### Multi-Timescale Logic

```python
def _maybe_update_H(self, x_ctrl, state):
    needs = (state.step % self.T) == 0
    if needs.any():
        z_H_new = self.f_H(x_ctrl, state.z_H)
        state = HRMState(z_L=state.z_L, z_H=self.ln_H(z_H_new), step=state.step)
    return state
```

Only updates H when `step` is a multiple of `T`.

---

## Troubleshooting

### Issue: Routing collapses to single head

**Solution**: Increase temperature, add entropy regularization
```python
controller.set_temperature(1.5)  # softer routing
loss += 1e-3 * aux['entropy']    # encourage diversity
```

### Issue: H-module never updates

**Check**: Is `T` too large? Is state being properly threaded?
```python
print(f"Step: {state.step}, T: {controller.T}")
# Should see H-updates at steps 0, T, 2T, ...
```

### Issue: NaN in training

**Solution**: Clip gradients, reduce learning rate for controller
```python
# In trainer
torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=1.0)

# Or use differentiated LR
optimizer = AdamW([
    {'params': parser.encoder.parameters(), 'lr': 1e-5},
    {'params': controller.parameters(), 'lr': 5e-6},  # slower for recurrent
    {'params': other_params, 'lr': 3e-4}
])
```

---

## References

- **HRM Paper**: [Hierarchical Reasoning Model (arXiv:2506.21734)](https://arxiv.org/abs/2506.21734)
- **PoT Paper**: (Eran Ben Artzy, 2025)
- **Multi-Timescale RNNs**: Hierarchical recurrence enables longer dependencies

---

## Example Scripts

- `examples/hrm_controller_demo.py` - Basic usage demo
- `experiments/hrm_vs_standard_ab.py` - A/B comparison (TODO)
- `scripts/train.py --use_hrm` - Train with HRM controller (TODO)

---

**Ready to integrate?** See `examples/hrm_controller_demo.py` for a working example!

