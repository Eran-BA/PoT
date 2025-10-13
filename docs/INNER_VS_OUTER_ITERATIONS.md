# Inner vs Outer Iterations: Clarification

## 🎯 Quick Summary

**Inner Iterations** = Refinement steps **within the model** during a single forward pass  
**Outer Iterations** = Training steps (gradient updates) during optimization

---

## 📖 Detailed Explanation

### Inner Iterations (Refinement)

**What they are:**
- The number of times the model **refines its representation** in a single forward pass
- Controlled by `max_inner_iters` parameter
- Happens **inside** the model's forward() method

**Example:**
```python
# max_inner_iters = 12
model = IterRefiner(stack, max_inner_iters=12)

# Single forward pass
x = torch.randn(2, 10, 512)
output = model(x)  # ← This does 12 refinement iterations internally!

# Pseudocode of what happens inside:
# h = x
# for t in range(12):  # ← 12 inner iterations
#     h = stack(h)     # Refine representation
#     # h feeds back as input
# return h
```

**Visualization:**
```
Input x
  ↓
[Iteration 1] → h₁
  ↓ (feed back)
[Iteration 2] → h₂
  ↓ (feed back)
[Iteration 3] → h₃
  ↓ (feed back)
  ...
  ↓ (feed back)
[Iteration 12] → h₁₂ (final output)
  ↓
Output

All 12 iterations happen in ONE forward pass!
```

---

### Outer Iterations (Training Steps)

**What they are:**
- The number of gradient descent steps during training
- Controlled by `max_steps` or `epochs` in training config
- Each step does: forward → loss → backward → optimizer.step()

**Example:**
```python
# max_steps = 2000
trainer = Trainer(model, cfg={"train": {"max_steps": 2000}})

# Training loop (pseudocode)
for step in range(2000):  # ← 2000 outer iterations (training steps)
    output = model(batch)  # Each forward does 12 inner iters
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

**Visualization:**
```
Training Step 1 (outer iteration 1):
  Forward (12 inner iters) → Loss → Backward → Update weights

Training Step 2 (outer iteration 2):
  Forward (12 inner iters) → Loss → Backward → Update weights

...

Training Step 2000 (outer iteration 2000):
  Forward (12 inner iters) → Loss → Backward → Update weights
```

---

## 🔍 Key Differences

| Aspect | Inner Iterations | Outer Iterations |
|--------|------------------|------------------|
| **Happens during** | Single forward pass | Entire training |
| **Purpose** | Refine representation | Optimize weights |
| **Parameter** | `max_inner_iters` | `max_steps` or `epochs` |
| **Controlled by** | Model architecture | Training loop |
| **Gradient flow** | Through all iterations | Once per step |
| **Typical count** | 1-20 | 1000s-100000s |
| **Cost per unit** | ~1x forward pass / K | 1 forward + 1 backward |

---

## 📊 Example: NLI Benchmark

**Configuration:**
```yaml
model:
  max_inner_iters: 12  # ← INNER (refinement steps)

train:
  max_steps: 20000     # ← OUTER (training steps)
  batch_size: 32
```

**What happens:**
```
Training step 1:
  Batch of 32 samples
  → Forward pass with 12 inner refinement iterations
  → Compute loss
  → Backward pass (gradients through all 12 iterations)
  → Update weights

Training step 2:
  New batch of 32 samples
  → Forward pass with 12 inner refinement iterations
  → Compute loss
  → Backward pass
  → Update weights

... (repeat 20000 times)

Training step 20000:
  Final batch
  → Forward pass with 12 inner refinement iterations
  → Compute loss
  → Backward pass
  → Update weights
```

**Total iterations:**
- **Inner iterations per forward:** 12
- **Outer iterations (training steps):** 20,000
- **Total inner iterations across training:** 12 × 20,000 = 240,000

---

## 💡 Why This Matters

### Inner Iterations (12 is optimal)

**Too few (K=1-3):**
- Model doesn't refine enough
- Performance suffers
- Fast but less accurate

**Optimal (K=12):**
- ✅ Sufficient refinement
- ✅ Diminishing returns plateau not reached
- ✅ Good performance/cost tradeoff

**Too many (K=20+):**
- Marginal gains (<0.5%)
- 2x more compute for 0.1% improvement
- Wasteful

### Outer Iterations (20,000 for SNLI)

**Too few (< 5,000):**
- Model underfits
- Weights not converged

**Optimal (10,000-20,000):**
- ✅ Full convergence
- ✅ Best validation accuracy
- ✅ Reasonable training time

**Too many (> 50,000):**
- Overfitting risk
- Wasted time (performance plateaus)

---

## 🎯 Practical Guidelines

### When configuring PoH:

**Inner iterations (`max_inner_iters`):**
```python
# Quick prototyping / sanity checks
max_inner_iters = 2-3

# Production benchmarks (recommended)
max_inner_iters = 12  # ← Optimal from empirical analysis

# Research / maximum performance
max_inner_iters = 20
```

**Outer iterations (`max_steps`):**
```python
# Quick test
max_steps = 100-500

# Development
max_steps = 2000-5000

# Production / publication
max_steps = 10000-20000  # Depends on dataset size
```

---

## 📝 Code Examples

### Setting Inner Iterations

```python
from src.pot.modules import PoHConfig, PoHStack, IterRefiner

cfg = PoHConfig(d_model=512, n_heads=8)
stack = PoHStack(cfg, depth=6)

# Option 1: Directly in IterRefiner
refiner = IterRefiner(stack, max_inner_iters=12)  # ← 12 inner iters

# Option 2: In PoHConfig for PoHGPT
cfg_gpt = PoHConfig(
    d_model=512,
    n_heads=8,
    depth=6,
    max_inner_iters=12,  # ← 12 inner iters
)
model = PoHGPT(vocab_size=32000, cfg=cfg_gpt)
```

### Setting Outer Iterations

```yaml
# experiments/configs/nli/poh.yaml
train:
  max_steps: 20000      # ← 20,000 outer iters (training steps)
  batch_size: 32
  eval_interval: 500
```

```python
# In training script
trainer = Trainer(model, cfg)
trainer.train()  # Runs for max_steps outer iterations
```

---

## ✅ Summary

**Inner iterations (12):**
- Model refinement steps
- Happen in **one forward pass**
- Set via `max_inner_iters`
- 12 is optimal (empirically proven)

**Outer iterations (20,000):**
- Training steps
- Gradient updates
- Set via `max_steps` or `epochs`
- Depends on dataset size

**Both are important, but independent!**

---

**Last Updated:** October 2025  
**See also:** [POH_ITERATION_GUIDE.md](POH_ITERATION_GUIDE.md) for choosing optimal inner iteration counts

