# Full HRM Implementation Plan

Based on the [official HRM repository](https://github.com/Hemanth21k/HRM) and paper analysis.

## Current Status
- ✅ Basic HRM controller (f_L and f_H with period T)
- ✅ Routing weights applied to attention heads
- ❌ Missing critical HRM components (see below)
- ❌ Only 100 epochs vs 20,000 required
- **Result:** Stuck at 87% token accuracy (0% grid accuracy)

## Missing Components for Full HRM

### 1. Puzzle-Specific Embeddings
**File:** `models/hrm/hrm_act_v1.py:116-120`
```python
self.puzzle_emb = CastedSparseEmbedding(
    num_puzzle_identifiers, 
    puzzle_emb_ndim,
    batch_size=batch_size, 
    init_std=0,  # Zero init!
    cast_to=forward_dtype
)
```
- Each puzzle gets a **learned embedding** prepended to input
- Allows model to specialize per puzzle type
- Critical for few-shot learning (1000 examples)

### 2. Adaptive Halting (ACT + Q-Learning)
**File:** `models/hrm/hrm_act_v1.py:195-230`
```python
# Q-learning for when to halt
q_continue_logits = self.q_head(hidden_states[:, 0, :])[:, 0]  # [B]
q_halt_logits = q_continue_logits[:, 1]  # [B]

# Exploration-exploitation
should_halt = (torch.rand_like(q_halt_logits) < exploration_prob) | (q_halt_logits >= 0)
```
- Model learns **when to stop thinking** (not fixed R iterations)
- Uses Q-learning with bootstrapping targets
- `halt_max_steps=8` in their config

### 3. Multiple Reasoning Cycles
**File:** `models/hrm/hrm_act_v1.py:38-39`
```python
H_cycles: int  # High-level cycles
L_cycles: int  # Low-level cycles per H-cycle
```
- Not just one pass through layers
- H-level and L-level alternate multiple times
- Default: `L_cycles=8` for maze task

### 4. Post-Norm Architecture
**File:** `models/hrm/hrm_act_v1.py:77-83`
```python
# Post Norm (not Pre-Norm!)
hidden_states = rms_norm(
    hidden_states + self.self_attn(hidden_states), 
    variance_epsilon=rms_norm_eps
)
hidden_states = rms_norm(
    hidden_states + self.mlp(hidden_states), 
    variance_epsilon=rms_norm_eps
)
```
- RMSNorm after residual (not before)
- More stable for deep reasoning

### 5. SwiGLU FFN
**File:** `models/layers.py` (SwiGLU)
```python
class SwiGLU(nn.Module):
    # Gated activation: x * swish(gate(x))
    # More expressive than ReLU
```

### 6. Proper Loss Function
**File:** `models/losses.py:34-37`
```python
def softmax_cross_entropy(logits, labels, ignore_index=-100):
    return F.cross_entropy(
        logits.to(torch.float32).view(-1, logits.shape[-1]), 
        labels.view(-1), 
        ignore_index=ignore_index, 
        reduction="none"
    ).view(labels.shape)
```
- Per-token loss (not per-grid)
- With Q-halt loss: `F.binary_cross_entropy_with_logits(q_halt_logits, is_correct)`
- With Q-continue loss for bootstrapping

### 7. Training Hyperparameters
**From official repo:**
```bash
epochs=20000  # Not 100!
eval_interval=2000
lr=1e-4
puzzle_emb_lr=1e-4
weight_decay=1.0  # High!
puzzle_emb_weight_decay=1.0
global_batch_size=384  # On 8 GPUs
```

### 8. Learned Positional Encodings
```python
if pos_encodings == "learned":
    self.embed_pos = CastedEmbedding(
        seq_len + puzzle_emb_len, 
        hidden_size, 
        init_std=embed_init_std
    )
```
- Not just sinusoidal
- Trained end-to-end

## Why Our Implementation Fails

| Component | HRM (Official) | Our Implementation | Impact |
|-----------|----------------|-------------------|--------|
| Epochs | 20,000 | 100 | ❌ Massive underfitting |
| Puzzle embeddings | ✅ | ❌ | ❌ No task specialization |
| Adaptive halting | ✅ Q-learning | ❌ Fixed R=4 | ❌ Inefficient computation |
| Reasoning cycles | L_cycles=8 | R=4 (different!) | ❌ Less depth |
| Architecture | Post-norm + SwiGLU | Pre-norm + ReLU | ⚠️ Suboptimal |
| Loss | Multi-component | Simple CE | ⚠️ No halt learning |
| Weight decay | 1.0 | Default (0.01) | ⚠️ Overfitting |

## Architecture Diagram

```
Input Tokens [B, SeqLen]
    ↓
Puzzle Embedding [B, puzzle_emb_len, D] (prepended)
    ↓
Token Embeddings [B, SeqLen+puzzle_emb_len, D]
    ↓
Position Embeddings (learned)
    ↓
┌─────────────── ACT Loop (until halt) ───────────────┐
│                                                      │
│  ┌────── H-level (slow, abstract) ──────┐          │
│  │  H_init → H_layers → z_H             │          │
│  └────────────────────────────────────────┘          │
│            ↓ (inject into L-level)                  │
│  ┌────── L_cycles iterations ──────────┐            │
│  │  z_L + z_H → L_layers → z_L'        │  ×8       │
│  └────────────────────────────────────────┘          │
│            ↓                                        │
│  LM Head → logits [B, SeqLen, vocab]               │
│  Q Head → [q_continue, q_halt]                     │
│            ↓                                        │
│  if q_halt >= 0: break                             │
│                                                      │
└──────────────────────────────────────────────────────┘
    ↓
Loss = CE(logits, labels) + BCE(q_halt, is_correct) + BCE(q_continue, target_q)
```

## Implementation Priority

### Phase 1: Core Architecture (High Priority)
1. ✅ Add puzzle embeddings
2. ✅ Implement Q-learning halting
3. ✅ Add L_cycles loop
4. ✅ Switch to post-norm + SwiGLU

### Phase 2: Training Setup (High Priority)
5. ✅ Implement proper loss function
6. ✅ Scale to 20,000 epochs
7. ✅ Add high weight decay (1.0)
8. ✅ Add learned positional encodings

### Phase 3: Optimization (Medium Priority)
9. ⚠️ Add FlashAttention support
10. ⚠️ Multi-GPU training (8 GPUs)
11. ⚠️ Gradient checkpointing for memory

## Expected Results

After full implementation:
- **Token accuracy:** 95%+ (from 87%)
- **Grid accuracy:** 70-74% (from 0%)
- **Training time:** ~1 hour on 8x A100
- **Halting behavior:** Model learns to stop at ~5-7 steps on average

## References

1. [HRM Paper](https://arxiv.org/abs/2506.21734)
2. [Official HRM Repository](https://github.com/sapientinc/HRM)
3. [HRM Models Code](https://github.com/sapientinc/HRM/tree/main/models)

---

**Conclusion:** Our current implementation is a **simplified proof-of-concept**, not the full HRM architecture. To match the paper's 74% grid accuracy, we need to implement all the components above, especially puzzle embeddings, Q-learning halting, and train for 20,000 epochs.

