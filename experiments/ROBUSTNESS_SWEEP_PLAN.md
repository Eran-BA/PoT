# HRM Robustness & Production Readiness Plan

**Author**: Eran Ben Artzy  
**Status**: Implementation in progress  
**Goal**: Lock down stable HRM defaults and comprehensive validation

---

## ✅ Completed

### 1. HRM Trainer with Stable Defaults (`src/training/hrm_trainer.py`)

**Features implemented**:
- ✅ Two optimizers (encoder + controller) with different LRs
  - Encoder: 3e-4, Controller: 1e-4, Other: 3e-4
- ✅ Separate gradient clipping per module
  - Encoder: 1.0, Controller: 0.5, Other: 1.0
- ✅ Deep supervision (average loss over inner steps)
- ✅ Temperature scheduling (2.0 → 0.7, decay 0.95/epoch)
- ✅ Entropy regularization with decay (1e-3, halves every 5 epochs)
- ✅ Controller warm-up (frozen first 5 epochs)
- ✅ AMP support for speed
- ✅ Comprehensive diagnostics logging

### 2. ACT-style Halting (`src/models/hrm_act.py`)

**Features implemented**:
- ✅ Learned halting gate: p_t = sigmoid(w^T z_H + b)
- ✅ Adaptive H-updates (when accumulated R >= 1.0)
- ✅ Ponder cost tracking for loss
- ✅ Per-sample update decisions
- ✅ Safety limits (max_ponders)

---

## 🔄 In Progress

### 3. Cacheable Head Features

**File**: `src/models/pointer_block.py` (needs modification)

```python
class PointerBlock(nn.Module):
    def forward(self, x, cache_heads=True, **kwargs):
        if cache_heads:
            # Compute once
            head_feats = self.attn_heads(x, mask)
        
        for t in range(iters):
            if not cache_heads:
                head_feats = self.attn_heads(x, mask)
            
            alphas, state, aux = self.controller(
                x, head_feats, state=state
            )
            mixed = (alphas[...,None,None] * head_feats).sum(dim=1)
            x = x + self.mix_proj(mixed)
```

---

### 4. Profilers & Speed Optimizations

**File**: `src/training/hrm_trainer.py` (extend)

**Add**:
```python
# Torch compile (PyTorch 2+)
model = torch.compile(model, mode="reduce-overhead")

# Memory tracking
if device.type == 'cuda':
    torch.cuda.reset_peak_memory_stats()
    # ... training ...
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    metrics['peak_memory_gb'] = peak_mem

# Timing
metrics.update({
    'epoch_wallclock_s': elapsed,
    'tokens_per_s': total_tokens / elapsed,
    'batches_per_s': num_batches / elapsed
})
```

---

### 5. Robustness Sweep Scripts

**File**: `experiments/run_robustness_sweeps.py`

**Sweeps**:
1. **Inner iterations**: `{0, 1, 2, 4, 8, 12}`
2. **HRM period T**: `{2, 3, 4, 6, 8}`
3. **Top-k**: `{None, 1, 2, 4, 8}`
4. **Controller LR**: `{5e-5, 1e-4, 2e-4, 5e-4}`
5. **Entropy reg**: `{0, 5e-4, 1e-3, 2e-3, 5e-3}`
6. **ACT vs Fixed-T**: both modes

**Output**: CSV per config with:
- `config_id, seed, iters, T, topk, lr_ctl, ent_reg, use_act`
- `train_time_s, tokens_per_s, peak_mem_gb`
- `kendall, perfect, accuracy`
- `routing_entropy, head_concentration, n_H_updates`

---

### 6. OOD & Scaling Tests

**File**: `experiments/run_ood_tests.py`

**Tests**:

A. **Length Extrapolation**
```python
# Train at 12, eval at 16, 20, 24
train_loader = create_data(length=12, samples=5000)
test_loaders = {
    12: create_data(length=12, samples=500),
    16: create_data(length=16, samples=500),
    20: create_data(length=20, samples=500),
    24: create_data(length=24, samples=500),
}
```

B. **Observability Stress**
```python
# Train at 50%, eval at 20%, 50%, 80%
mask_rates = [0.2, 0.5, 0.8]
for rate in mask_rates:
    eval_performance(model, mask_rate=rate)
```

C. **Low-Data Regime**
```python
# Train with 500, 1k, 2k, 5k, 10k samples
for n_samples in [500, 1000, 2000, 5000, 10000]:
    model = train(data[:n_samples], epochs=50)
    results[n_samples] = eval(model)
```

---

### 7. Controller Diagnostics

**Add to training loop** (`src/training/hrm_trainer.py`):

```python
def compute_diagnostics(self, alphas_batch):
    """
    Compute routing diagnostics.
    
    Returns:
        routing_entropy_mean: avg entropy across batch
        routing_entropy_std: std entropy across batch
        head_herfindahl: sum of squared probs (concentration)
        top1_share: max head probability (dominance)
    """
    # Entropy per sample
    entropies = -(alphas_batch * torch.log(alphas_batch + 1e-12)).sum(dim=-1)
    
    # Herfindahl index (concentration)
    herfindahl = (alphas_batch ** 2).sum(dim=-1).mean()
    
    # Top-1 share
    top1_share = alphas_batch.max(dim=-1)[0].mean()
    
    return {
        'routing_entropy_mean': entropies.mean().item(),
        'routing_entropy_std': entropies.std().item(),
        'head_herfindahl': herfindahl.item(),
        'top1_share': top1_share.item()
    }
```

**Log to JSONL**:
```python
# Save sample routing patterns
with open(f'routing_log_epoch{epoch}.jsonl', 'w') as f:
    for i in range(min(10, len(batch))):
        json.dump({
            'epoch': epoch,
            'sample_id': i,
            'topk_heads': topk_idx[i].tolist(),
            'alphas': alphas[i].tolist()
        }, f)
        f.write('\n')
```

---

### 8. Statistical Validation

**File**: `experiments/analyze_with_stats.py`

```python
import scipy.stats as stats
import numpy as np

def welch_ttest(baseline_scores, hrm_scores):
    """
    Welch's t-test (unequal variances).
    
    Returns:
        t_stat: t-statistic
        p_value: two-tailed p-value
        cohens_d: effect size
    """
    t_stat, p_value = stats.ttest_ind(
        baseline_scores,
        hrm_scores,
        equal_var=False
    )
    
    # Cohen's d
    mean_diff = np.mean(hrm_scores) - np.mean(baseline_scores)
    pooled_std = np.sqrt(
        (np.var(baseline_scores) + np.var(hrm_scores)) / 2
    )
    cohens_d = mean_diff / pooled_std
    
    return t_stat, p_value, cohens_d

# Usage
baseline = [0.091, 0.095, 0.088]  # 3 seeds
hrm = [0.108, 0.110, 0.106]       # 3 seeds

t, p, d = welch_ttest(baseline, hrm)
print(f"HRM vs Baseline: p={p:.4f}, d={d:.3f}")
if p < 0.05:
    print(f"✅ Significant (p<0.05, effect size d={d:.3f})")
```

---

### 9. Failure-Case Triage Guide

**File**: `docs/TROUBLESHOOTING_HRM.md`

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| **Uniform alphas (no specialization)** | Temperature too high, entropy reg too strong | Lower `temp_init` (1.5), reduce `entropy_reg` (5e-4), lower `lr_controller` |
| **Collapse to one head** | Temperature too low, no diversity | Raise `temp_min` (0.9), add entropy floor, increase `topk` |
| **Training oscillations** | Controller gradients too large | Stricter `clip_controller` (0.3), use `betas=(0.9,0.95)` on controller |
| **No advantage at long lengths** | Not enough iterations, H too slow | Increase `iters` (12+), enable ACT, reduce `T` (2-3) |
| **HRM worse than baseline** | Task too easy, overhead not justified | Use baseline for easy tasks, or reduce `iters` to 2 |
| **Out of memory** | Too many iterations, large batch | Reduce `iters`, enable gradient checkpointing, smaller batch |
| **Slow training** | Too many H-updates, no caching | Increase `T`, enable `cache_heads`, use torch.compile |

---

### 10. API & CLI Flags

**File**: `scripts/train_hrm.py` (new unified training script)

```python
parser.add_argument('--controller', choices=['classical', 'hrm', 'hrm_act'], default='hrm')

# HRM-specific
parser.add_argument('--hrm_T', type=int, default=4, help='H-module period')
parser.add_argument('--hrm_act', action='store_true', help='Use ACT halting')
parser.add_argument('--ponder_tau', type=float, default=0.01, help='Ponder cost weight')

# Routing
parser.add_argument('--routing_topk', type=int, default=None, help='Top-k heads')
parser.add_argument('--iters', type=int, default=4, help='Inner iterations')
parser.add_argument('--cache_heads', action='store_true', help='Cache head features')

# Training
parser.add_argument('--deep_supervision', action='store_true', default=True)
parser.add_argument('--lr_encoder', type=float, default=3e-4)
parser.add_argument('--lr_controller', type=float, default=1e-4)
parser.add_argument('--clip_encoder', type=float, default=1.0)
parser.add_argument('--clip_controller', type=float, default=0.5)

# Temperature
parser.add_argument('--temp_init', type=float, default=2.0)
parser.add_argument('--temp_min', type=float, default=0.7)
parser.add_argument('--temp_decay', type=float, default=0.95)

# Entropy
parser.add_argument('--entropy_reg', type=float, default=1e-3)
parser.add_argument('--entropy_decay_epochs', type=int, default=5)

# Warm-up
parser.add_argument('--controller_warmup_epochs', type=int, default=5)

# Speed
parser.add_argument('--use_amp', action='store_true', help='Mixed precision')
parser.add_argument('--compile', action='store_true', help='torch.compile')
```

---

## 📋 **Implementation Checklist**

- [x] HRM Trainer with stable defaults
- [x] ACT-style halting controller
- [ ] Cacheable head features
- [ ] Profilers & memory tracking
- [ ] Robustness sweep script
- [ ] OOD test suite
- [ ] Controller diagnostics
- [ ] Statistical validation script
- [ ] Troubleshooting guide
- [ ] Unified CLI & config

---

## 🚀 **Next Actions**

1. **Immediate**: Finish cacheable heads + profilers
2. **Today**: Run robustness sweep on length 12, 20
3. **This week**: Complete OOD tests, statistical validation
4. **For paper**: Generate all tables, significance tests, failure analysis

---

**Status**: 40% complete, on track for production deployment.

**ETA**: 2-3 days for full implementation and validation.

