# Investigation: PoH-HRM 16×16 Maze Optimality Failure

## Problem

During maze scaling benchmark, PoH-HRM showed **48% optimality** on 16×16 mazes, while achieving 84-88% on other sizes:

| Maze Size | Baseline Opt | BERT Opt | PoH Opt | Notes |
|-----------|--------------|----------|---------|-------|
| 8×8   | 84% | 82.5% | **87.5%** ✅ | PoH wins |
| 12×12 | 84.5% | 80% | **84.5%** ✅ | PoH ties baseline |
| 16×16 | **81%** | 76.5% | **48%** ❌ | PoH FAILS |
| 20×20 | **83%** | 80% | 77% ⚠️ | PoH recovering |
| 24×24 | **77%** | ? | ? | Continuing... |

## Observations

### 1. Training Time
- **PoH**: ~30 min per maze size
- **Baseline**: ~15 min per maze size
- **2x slower** suggests more computation but may not be converging

### 2. Accuracy vs Optimality
- **Accuracy**: 100% (finds valid path)
- **Optimality**: 48% (path is suboptimal)
- This means PoH is learning to solve mazes but not finding shortest paths

### 3. Pattern Across Sizes
- Works well on 8×8, 12×12 (small)
- **Fails on 16×16** (medium)
- Recovering on 20×20 (large)
- This U-shape is unusual!

## Hypotheses

### H1: Insufficient Refinement Iterations (R=4)
**Theory**: 16×16 mazes need more iterations than R=4 provides.

**Test**: 
```python
# Try R=6 or R=8 for 16×16
poh = MazeSolver(..., max_inner_iters=8, ...)
```

**Evidence**:
- 20×20 starts recovering (more tokens, maybe triggers different behavior)
- 8×8 works (few enough tokens)

### H2: HRM Period Mismatch (T=4)
**Theory**: T=4 (f_H updates every 4 steps) doesn't align well with 16×16 maze structure.

**Test**:
```python
# Try T=8 or T=2 for 16×16
poh = MazeSolver(..., T=8, ...)
```

**Evidence**:
- 16×16 = 256 tokens
- 256 / 4 = 64 f_H updates
- Maybe needs more or fewer high-level updates

### H3: Overfitting to "Any Path" Early in Training
**Theory**: Model learns to find ANY valid path quickly, then stops improving toward optimal paths.

**Test**:
- Add **curriculum learning**: start with optimality loss weight=0.1, increase to 1.0
- Use **two-stage training**: stage 1 (validity), stage 2 (optimality)

**Evidence**:
- 100% accuracy means it learned path-finding
- 48% optimality means it didn't learn shortest-path

### H4: Gradient Flow Issues
**Theory**: HRM controller gradients not flowing properly for medium-sized sequences.

**Test**:
- Add gradient norm monitoring
- Check HRM state evolution during training
- Verify f_L and f_H are both updating

**Evidence**:
- U-shaped performance curve suggests something specific to 16×16

### H5: Attention Head Saturation
**Theory**: With n_heads=4, HRM routing might saturate for 256-token sequences.

**Test**:
```python
# Try n_heads=8 for 16×16
poh = MazeSolver(..., n_heads=8, ...)
```

**Evidence**:
- 16×16 = 256 tokens = 2^8
- Maybe needs more heads for this size

## Recommended Actions

### Immediate (Quick Tests)
1. ✅ **Make mazes harder** (wall_prob=0.45) - DONE
2. 🔄 **Re-run 16×16 with harder mazes** - see if problem persists
3. 📊 **Add logging**:
   ```python
   # Log path length distribution
   # Log HRM entropy over iterations
   # Log convergence metrics
   ```

### Short-term (Targeted Experiments)
1. **Vary R** for 16×16: Test R=2, 4, 6, 8
2. **Vary T** for 16×16: Test T=2, 4, 8, 16
3. **Vary n_heads** for 16×16: Test n_heads=2, 4, 8
4. **Two-stage training**: First learn path-finding, then optimize

### Long-term (Deep Investigation)
1. **Visualize attention patterns** for 16×16 vs 8×8 and 20×20
2. **Ablation study**: PoH without HRM on 16×16
3. **Architecture search**: Find optimal R, T, n_heads per maze size
4. **Curriculum learning**: Gradually increase maze size during training

## Expected Outcomes

**If H1 (R=4 insufficient)**: Increasing R should improve 16×16 optimality
**If H2 (T=4 mismatch)**: Changing T should show non-monotonic improvement
**If H3 (overfitting)**: Two-stage training should recover optimality
**If H4 (gradient flow)**: Gradient monitoring will show HRM not updating
**If H5 (head saturation)**: Increasing n_heads should improve 16×16

## Status

- ✅ Fixed BERT for large mazes
- ✅ Made mazes harder (wall_prob=0.45)
- 🔄 Re-running benchmark with harder mazes
- ⏳ Waiting for results to confirm if problem persists

## Next Steps

1. Wait for current harder maze run to complete
2. If 16×16 still fails, run quick experiments with varied R, T, n_heads
3. Add diagnostic logging to understand HRM behavior
4. Consider per-size hyperparameter tuning if problem is systematic

