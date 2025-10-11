# Pointer Decoder Results: Proper Implementation

**Author**: Eran Ben Artzy  
**License**: Apache 2.0

## Key Insight: The 1/12 Trap Was a Semantic Bug

The original PoH model stuck at **~8.3% accuracy** (≈ 1/12) on 12-element sorting because it used **dependency parsing semantics** (each token picks its head) instead of **pointer decoder semantics** (each output rank picks an input position with coverage masking).

## The Three Critical Fixes

1. **Proper Decoder Loop**: Rank-conditioned queries, not token-conditioned
2. **Coverage Masking**: Prevent re-selection of already-chosen positions  
3. **Stable Argsort**: Handle ties deterministically

With these fixes, the model **immediately escapes the 1/12 trap** and learns perfectly.

## Results: Proper Pointer Decoder

| Array Length | Model | Test Accuracy | Perfect Sort Rate | Parameters |
|--------------|-------|---------------|-------------------|------------|
| 6 | Baseline | **100.00%** | **100.00%** | 199,552 |
| 12 | Baseline | **100.00%** | **100.00%** | 199,552 |
| 12 | PoH (4 iters) | **100.00%** | **100.00%** | 332,548 |
| 20 | Baseline | **100.00%** | **100.00%** | 199,552 |
| 20 | PoH (4 iters) | **100.00%** | **100.00%** | 332,548 |

### Training Speed Comparison (12 elements, 5000 samples)

- **Baseline**: 100% accuracy by epoch 10
- **PoH (4 iterations)**: 100% accuracy by epoch 10

Both converge at similar rates on this simple task.

## What Changed From Previous Implementation

### Before (Broken)
```python
# Wrong: Each INPUT token chooses a head
for i in range(N):
    logits[i] = biaffine(h[i], h)  # Token i picks its head
    # No coverage mask, no decoder loop
```
- Accuracy: ~8.3% (random guessing)
- Perfect sorts: 0%

### After (Fixed)
```python
# Correct: Each OUTPUT rank chooses an input position
for t in range(N):
    q_t = rank_embed(t)  # Query for rank t
    logits[t] = pointer(q_t, z)  # Rank t picks input position
    mask[chosen] = -inf  # Coverage: prevent re-selection
```
- Accuracy: 100%
- Perfect sorts: 100%

## PoH Integration Details

The PoH iterative refinement is applied **before the decoder loop**, not between ranks:

```python
# Encode input
z = encoder(x)  # [B, N, d_model]

# Optional: Refine latent with PoH inner iterations
if use_poh:
    z = poh_block(z)  # Refine z, returns pre-FFN latent

# Decode with coverage masking (NO detach between ranks)
for t in range(N):
    logits[t] = pointer(rank_t, z, mask)
    mask[chosen] = 0
```

**Key**: HRM-style gradients (if used) apply **inside PoH's inner loop**, not between decoder ranks.

## Next Steps

1. **OOD Generalization**: Train on small arrays (6-10), test on large (15-20)
2. **Sample Efficiency**: Compare learning curves (baseline vs PoH)
3. **Curriculum Learning**: Gradually increase sequence length
4. **Noisy Data**: Add duplicate values, missing elements
5. **Deep Supervision**: Add losses at intermediate PoH iterations

## Key Takeaway

**The semantics matter more than the architecture.**  
A simple baseline with correct decoder semantics (rank queries + coverage) outperforms a complex architecture with wrong semantics (token queries, no coverage).

With proper decoder semantics, PoH can now be fairly evaluated on:
- Sample efficiency
- OOD generalization  
- Long-range dependencies
- Curriculum learning

---

**Training Details:**
- Framework: PyTorch
- Data: Unique random integers per array
- Optimizer: Adam (lr=1e-3)
- Batch size: 32
- Gradient clipping: 1.0

