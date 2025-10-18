# CRITICAL BUGS FIXED: Q-Learning and Adaptive Halting

## 🚨 Problem: Model Stuck at 87.5% Token Accuracy

After 16+ epochs, the model was completely stuck:
- **Token Acc**: 87.5% (no improvement)
- **Grid Acc**: 0.0% (no perfect solutions)
- **Avg Steps**: 16.0 (always maxing out)
- **Q_halt Loss**: 0.0002 (learned nothing)
- **Q_continue Loss**: 0.0000 (never computed!)

## 🔍 Root Causes Identified

### Bug #1: Q-continue Loss Never Computed ❌

**Location**: `experiments/maze_grid2grid_hrm.py:301`

**Original Code**:
```python
if steps < model.max_halting_steps and model.training:
    # Compute Q-continue bootstrap loss
    ...
```

**Problem**:
- Model always runs to `max_halting_steps=16` (never halts early)
- Condition `steps < 16` is **always False** (because `steps == 16`)
- Q-continue loss **NEVER gets computed**
- Model can't learn when to continue/halt!

**Fix**: Remove the `steps < max_halting_steps` condition entirely:
```python
if model.training:
    # Always compute Q-continue bootstrap loss
    ...
```

**Impact**: Q-learning bootstrap target is now trained on every batch, allowing the model to learn value predictions.

---

### Bug #2: Exploration Logic Inverted ❌

**Location**: `src/pot/models/adaptive_halting.py:139`

**Original Code**:
```python
if torch.rand(1).item() < self.exploration_prob:
    min_steps = torch.randint(2, self.max_steps + 1, (1,)).item()
    halt = halt & (step >= min_steps)  # BUG: AND prevents halting!
```

**Problem**:
- Uses `&` (AND) operator: "halt only if model wants to halt AND past min_steps"
- During exploration (10% of time), if `min_steps=10`, model **can't halt** until step 10
- This **prevents** early halting exploration, the opposite of intended behavior!

**Fix**: Use `|` (OR) operator to **force** halting during exploration:
```python
if is_training and torch.rand(1).item() < self.exploration_prob:
    min_steps = torch.randint(2, self.max_steps + 1, (1,)).item()
    halt = halt | (step >= min_steps)  # OR: force halt to explore shorter paths
```

**Impact**: Model can now explore different computation depths (2-16 steps) during training.

---

### Bug #3: Inference Always Maxed Out ❌

**Location**: `src/pot/models/adaptive_halting.py:124-125`

**Original Code**:
```python
# Inference: always run max_steps (HRM approach for batching)
if not is_training:
    return torch.zeros(batch_size, dtype=torch.bool, device=device)
```

**Problem**:
- During evaluation, model **forced** to run all 16 steps
- Learned halting policy (`q_halt > q_continue`) is never used!
- Can't measure if adaptive halting is working
- Wastes computation on easy examples

**Fix**: Use learned policy during both training and inference:
```python
# Halt when q_halt > q_continue (learned policy)
halt = (q_halt > q_continue)
```

**Impact**: Model can now use its learned halting policy to save computation and demonstrate adaptive behavior.

---

### Bug #4: Puzzle Embedding Initialization 🔧

**Location**: `experiments/maze_grid2grid_hrm.py:96`

**Original Code**:
```python
self.puzzle_emb = PuzzleEmbedding(num_puzzles, puzzle_emb_dim, init_std=0.0)
```

**Problem**:
- Caller explicitly passes `init_std=0.0`
- Relies on `PuzzleEmbedding` internal logic to convert 0.0 → 0.02
- Unclear and error-prone

**Fix**: Explicitly set `init_std=0.02` at call site:
```python
self.puzzle_emb = PuzzleEmbedding(num_puzzles, puzzle_emb_dim, init_std=0.02)
```

**Impact**: Clear initialization, puzzle embeddings start with small random values for better gradient flow.

---

## 📊 Why Model Was Stuck at 87.5%

The combination of these bugs created a **learning deadlock**:

1. **Q-continue never trained** → Model can't learn value predictions
2. **Exploration prevented early halting** → Model never explores shorter computation paths
3. **Inference forced max steps** → Can't measure or use adaptive behavior
4. **Weak puzzle embeddings** → Can't specialize to individual mazes

Result: Model learns a **one-size-fits-all** solution that gets 87.5% token accuracy but **0% grid accuracy** (no perfect solutions).

---

## ✅ Expected Results After Fix

With these bugs fixed, the model should now:

1. **Learn Q-values properly**:
   - `q_halt` predicts correctness (should improve over epochs)
   - `q_continue` learns bootstrap targets (should stabilize around 0.5-0.8)

2. **Adapt computation dynamically**:
   - Easy mazes: halt early (2-4 steps)
   - Hard mazes: use more steps (8-16 steps)
   - Average steps should be < 16 during training

3. **Break through 87.5% plateau**:
   - Token accuracy should exceed 90%
   - Grid accuracy should start improving (1%, 5%, 10%+)
   - Perfect solutions should appear!

4. **Learn per-maze specialization**:
   - Puzzle embeddings should diverge from initialization
   - Different mazes should get different embeddings
   - Model should "remember" individual mazes

---

## 🧪 How to Test

Run the updated code with:
```bash
!wget -q https://raw.githubusercontent.com/Eran-BA/PoT/scaling_parameter_size/RUN_POH_COLAB_DIRECT.py && python RUN_POH_COLAB_DIRECT.py
```

**Look for these signs of success**:

1. **Q_continue Loss > 0**: Should be 0.1-0.5 (was 0.0000 before)
2. **Avg Steps < 16**: Should vary between 6-14 during training
3. **Token Acc > 87.5%**: Should reach 90%+ within 50 epochs
4. **Grid Acc > 0%**: Should see first perfect solution within 100 epochs
5. **Q_halt Loss decreasing**: Should drop from 0.69 → 0.3-0.5

---

## 📈 Diagnostic Commands

After training starts, monitor:

```python
# Check if Q-values are learning
print(f"Q_halt loss: {train_q_halt_loss:.4f}")  # Should decrease
print(f"Q_continue loss: {train_q_continue_loss:.4f}")  # Should be > 0!

# Check if adaptive halting is working
print(f"Avg steps: {train_steps:.1f}")  # Should be < 16

# Check if puzzle embeddings are learning
emb_std = model.puzzle_emb.embeddings.weight.std().item()
print(f"Puzzle emb std: {emb_std:.4f}")  # Should increase from 0.02
```

---

## 🎯 Target Performance

**HRM Paper Baseline**: 74% grid accuracy on 30x30 mazes

**Our Goal**: Match or exceed this with PoH + Q-halting + Puzzle embeddings

If these fixes work, we should see:
- **50-100 epochs**: Token acc 90%+, first perfect grids
- **500 epochs**: Grid acc 10-20%
- **2000+ epochs**: Grid acc 50-70% (approaching HRM)

---

## 🔧 Files Modified

1. `experiments/maze_grid2grid_hrm.py`
   - Fixed Q-continue loss computation
   - Fixed puzzle embedding initialization

2. `src/pot/models/adaptive_halting.py`
   - Fixed exploration logic (& → |)
   - Fixed inference halting behavior
   - Improved docstrings

---

## 📝 Next Steps

1. ✅ Push fixes to GitHub
2. 🔄 Re-run training in Colab
3. 📊 Monitor Q-losses and avg steps
4. 🎉 Celebrate when we see Grid Acc > 0%!

---

**Commit**: df2f16e
**Branch**: scaling_parameter_size
**Date**: 2025-01-19

