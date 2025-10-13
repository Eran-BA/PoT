# PoH Terminology Guide (HRM-Aligned)

**Last Updated:** October 2025  
**Status:** Official terminology for v1.0.0+

---

## 🎯 Quick Reference

| Term | Meaning | Code | Updates |
|------|---------|------|---------|
| **HRM Inner Loop** | f_L (low-level, fast) | `self.f_L` | Every refinement step |
| **HRM Outer Loop** | f_H (high-level, slow) | `self.f_H` | Every T refinement steps |
| **Refinement Iterations** | Multi-step processing | `max_inner_iters=12` | R times per forward pass |
| **Training Steps** | Gradient descent | `max_steps=20000` | Optimization loop |

---

## 📚 Official Terminology (HRM-Aligned)

### 1. **HRM Controller Terminology** (From HRM Paper)

**HRM Inner Loop (f_L):**
- Low-level, fast module
- Updates **every refinement step**
- Maintains state `z_L`
- Receives context from f_H
- Handles reactive, immediate processing

**HRM Outer Loop (f_H):**
- High-level, slow module  
- Updates **every T refinement steps** (T=4 default)
- Maintains state `z_H`
- Provides strategic guidance to f_L
- Handles abstract planning

**Code:**
```python
# In src/pot/core/hrm_controller.py
self.f_L = nn.GRUCell(...)  # HRM inner loop
self.f_H = nn.GRUCell(...)  # HRM outer loop
self.T = 4                  # f_H update period
```

---

### 2. **PoH Refinement Terminology** (Our Extension)

**Refinement Iterations (R):**
- Number of times the model refines its representation
- Happens **within a single forward pass**
- R=12 is optimal (from empirical analysis)
- Also called: "refinement steps" or "multi-step processing"

**Code:**
```python
# In src/pot/modules/block.py
class IterRefiner(nn.Module):
    def __init__(self, stack, max_inner_iters=12):
        self.R = max_inner_iters  # R = refinement iterations
    
    def forward(self, x):
        h = x
        for t in range(self.R):  # R refinement steps
            h = self.stack(h)
        return h
```

**Parameter Name:**
- `max_inner_iters` (kept for backward compatibility)
- Documented as "refinement iterations" in all prose
- Internal variable: `self.R` (not `self.K` to avoid confusion)

---

### 3. **Training Terminology**

**Training Steps:**
- Number of gradient descent iterations
- Controlled by `max_steps` or `epochs`
- Each step: forward (with R refinement iterations) + backward + update

**Code:**
```python
# In training configs
max_steps: 20000  # Training steps

# In training loop
for step in range(max_steps):  # Training steps
    output = refiner(x)         # Does R refinement iterations
    loss.backward()
    optimizer.step()
```

---

## 🔄 Three Nested Loops

PoH combines three distinct loop mechanisms:

```
1. Training Loop (20,000 steps)
   ↓
   2. Refinement Loop (12 iterations per forward)
      ↓
      3. HRM Timescales (f_L every step, f_H every T steps)
```

### Example

```python
# Level 1: Training loop
for step in range(20000):  # Training steps
    
    # Level 2: Refinement loop
    h = x
    for t in range(12):  # Refinement iterations (R=12)
        
        # Level 3: HRM controller updates
        if t % 4 == 0:
            z_H = f_H(x, z_H)  # HRM outer loop update
        z_L = f_L(x, z_H, z_L)  # HRM inner loop update
        
        h = PoHBlock(h, routing=z_L)
    
    # Backward pass
    loss.backward()
    optimizer.step()
```

---

## ❌ Deprecated / Avoid

### DO NOT SAY:
- ❌ "Inner iterations" (ambiguous - HRM inner loop or refinement?)
- ❌ "Outer iterations" (ambiguous - HRM outer loop or training?)
- ❌ "K iterations" (use R for refinement)

### INSTEAD SAY:
- ✅ "Refinement iterations" or "refinement steps" (R=12)
- ✅ "HRM inner loop" (f_L, fast)
- ✅ "HRM outer loop" (f_H, slow)
- ✅ "Training steps" (gradient descent)

---

## 📖 Usage in Documentation

### When Writing Docs:

**Correct:**
```markdown
The model performs 12 refinement iterations per forward pass.
During each refinement step, the HRM inner loop (f_L) updates,
and every 4 steps, the HRM outer loop (f_H) updates.
```

**Incorrect:**
```markdown
The model performs 12 inner iterations per forward pass.
The inner loop updates every step, the outer loop every 4 steps.
```

### When Writing Code Comments:

**Correct:**
```python
# Refinement iterations (R=12)
for t in range(self.R):
    # HRM inner loop (f_L) updates every step
    z_L = self.f_L(...)
    
    # HRM outer loop (f_H) updates every T steps
    if t % self.T == 0:
        z_H = self.f_H(...)
```

**Incorrect:**
```python
# Inner iterations
for t in range(self.K):
    # Inner loop updates
    z_L = self.f_L(...)
```

---

## 🎓 Teaching Guide

### For New Users:

**Explain in this order:**

1. **HRM Controller** (from paper):
   - "The HRM has two loops: f_L (inner, fast) and f_H (outer, slow)"
   - "These are controller timescales for routing decisions"

2. **Refinement** (our contribution):
   - "PoH applies the transformer stack multiple times (R=12)"
   - "This allows multi-step reasoning within one forward pass"

3. **Training** (standard):
   - "Like any model, we train with gradient descent"
   - "Each training step does R refinement iterations"

### Analogy:

```
HRM loops = "thinking fast vs thinking slow" (Kahneman)
  - Inner loop (f_L) = fast, intuitive reactions
  - Outer loop (f_H) = slow, deliberate planning

Refinement iterations = "iterative problem solving"
  - Like editing an essay multiple times
  - Each pass refines the solution

Training steps = "learning from experience"
  - Standard gradient descent
  - Improve weights over many examples
```

---

## 🔍 Parameter Names

| Parameter | Type | Meaning | Default |
|-----------|------|---------|---------|
| `max_inner_iters` | int | Refinement iterations (R) | 12 |
| `T` | int | HRM outer loop period | 4 |
| `max_steps` | int | Training steps | 20000 |
| `epochs` | int | Training epochs | 50 |

**Why `max_inner_iters` kept?**
- Backward compatibility with existing code
- Widely used in configs and scripts
- Documented clearly as "refinement iterations"
- Internal variable renamed to `self.R` for clarity

---

## 📊 Statistics Naming

### Code Variable Names:

```python
# Good (HRM-aligned)
refinement_stats = []         # Stats per refinement step
hrm_inner_updates = count_L   # f_L updates
hrm_outer_updates = count_H   # f_H updates
training_steps = step         # Gradient descent steps

# Bad (ambiguous)
inner_stats = []              # Inner what? HRM or refinement?
outer_updates = count         # Outer what? HRM or training?
iterations = step             # Which iterations?
```

---

## ✅ Checklist for Contributors

Before submitting code/docs:

- [ ] Use "refinement iterations" not "inner iterations"
- [ ] Use "HRM inner loop (f_L)" and "HRM outer loop (f_H)"
- [ ] Use "training steps" not "outer iterations"
- [ ] Variable `R` for refinement count, not `K`
- [ ] Comments clarify which "loop" you mean
- [ ] No ambiguous use of "inner/outer"

---

## 📚 Related Documentation

- [HRM_VS_REFINEMENT_LOOPS.md](HRM_VS_REFINEMENT_LOOPS.md) - Detailed disambiguation
- [INNER_VS_OUTER_ITERATIONS.md](INNER_VS_OUTER_ITERATIONS.md) - Original explanation (deprecated terminology)
- [POH_ITERATION_GUIDE.md](POH_ITERATION_GUIDE.md) - Why R=12 is optimal
- [src/pot/core/hrm_controller.py](../src/pot/core/hrm_controller.py) - HRM implementation
- [src/pot/modules/block.py](../src/pot/modules/block.py) - Refinement implementation

---

## 🔄 Migration from Old Terminology

### Old Docs (Pre-v1.0.0):

| Old Term | New Term (HRM-Aligned) |
|----------|------------------------|
| Inner iterations | Refinement iterations |
| Outer iterations | Training steps |
| K iterations | R refinement steps |
| Inner loop updates | HRM inner loop (f_L) OR refinement steps |
| Outer loop updates | HRM outer loop (f_H) OR training steps |

### Code Changes:

- `self.K` → `self.R` (internal variable)
- `max_inner_iters` parameter kept (backward compat)
- Comments updated to say "refinement"
- Docstrings clarified

---

**Last Updated:** October 2025  
**Approved by:** Repository maintainers  
**Status:** Official terminology standard for all PoH documentation and code

