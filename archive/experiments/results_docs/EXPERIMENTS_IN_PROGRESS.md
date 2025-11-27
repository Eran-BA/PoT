# Experiments in Progress

## 🎯 Completed Results

### ✅ 12x12 Maze Benchmark (Depth-First Parity)
**File**: `experiments/results/benchmark_12x12_depth_first/results.json`

**Configuration**:
- Training: 306 mazes (path length: 64.5±11.9)
- Testing: 57 mazes (path length: 64.4±11.6)
- Epochs: 50
- PoH: R=4, T=4

**Results**:
```
Baseline: 52.63% acc, 33.33% opt (7.21M params)
PoH-HRM:  56.14% acc, 47.37% opt (7.04M params, depth=4)
Winner: PoH by +3.51% acc, +14.04% opt ✅
```

**Key Finding**: Depth-first parameter parity (keep depth=4, reduce width) preserves PoH's hierarchical advantage!

---

## 🔬 Running Experiments

### 1️⃣ O(R) vs O(1) Memory Comparison

**Goal**: Quantify memory-convergence tradeoff

**Experiments**:
- **O(R) Standard**: Backprop through all R=4 iterations (memory: O(R))
- **O(1) Mode**: Backprop only through last iteration (memory: O(1))

**Configuration**:
- Training: 200 mazes
- Testing: 40 mazes
- Epochs: 30
- Both: R=4, T=4, same hyperparameters

**Hypothesis**: O(1) uses less memory but converges slower

**Output Files**:
- `experiments/results/comparison_OR_memory/results.json`
- `experiments/results/comparison_O1_memory/results.json`

---

### 2️⃣ Dense vs Sparse Supervision

**Goal**: Test if sparse supervision improves long-term planning

**Experiments**:
- **Dense** (interval=1): Supervise every step
- **Sparse** (interval=3): Supervise every 3rd step

**Configuration**:
- Training: 200 mazes
- Testing: 40 mazes
- Epochs: 30
- PoH: R=4, T=4

**Hypothesis**: Sparse supervision forces model to learn multi-step planning

**Output File**:
- `experiments/results/comparison_sparse_supervision/results.json`

---

### 3️⃣ 16x16 Maze (In Progress)

**Goal**: Test if PoH advantage scales to larger mazes

**Configuration**:
- Size: 16×16 (vs 12×12)
- Training: 500 mazes
- Testing: 100 mazes
- Epochs: 50

**Status**: Running in background

**Output File**:
- `experiments/results/benchmark_16x16_depth_first/results.json`

---

## 📊 Monitoring

Run `./monitor_experiments.sh` to check progress:
```bash
chmod +x monitor_experiments.sh
./monitor_experiments.sh
```

Or manually check logs:
```bash
tail -f experiments/results/comparison_OR_memory.log
tail -f experiments/results/comparison_O1_memory.log
tail -f experiments/results/comparison_sparse_supervision.log
```

---

## 🧪 Key Innovations Implemented

### 1. **O(1) Memory Mode** (NEW!)
```python
# Standard O(R) mode
h, _ = self.refiner(x)  # Backprop through all R iterations

# O(1) mode
for i in range(R):
    h = self.stack(h)
    if i < R - 1:
        h = h.detach()  # Break gradient flow
```

**CLI Flag**: `--last-iter-only`

### 2. **Sparse Supervision** (NEW!)
```python
# Dense supervision (default)
for i in range(0, path_len-1, 1):  # Every step
    loss += criterion(pred[i], target[i+1])

# Sparse supervision  
for i in range(0, path_len-1, 3):  # Every 3rd step
    loss += criterion(pred[i], target[i+1])
```

**CLI Flag**: `--supervision-interval N`

### 3. **Depth-First Parameter Parity**
- Keep PoH depth constant (e.g., depth=4)
- Reduce width (d_model, d_ff) to match baseline params
- **Why**: Preserves hierarchical advantage vs reducing depth

### 4. **Enhanced Training**
- AdamW optimizer
- Label smoothing (0.1)
- Cosine LR with warmup
- Validity masking (only valid moves)
- Multi-horizon supervision
- Gradient clipping (1.0)

---

## 📈 Expected Timeline

- **O(R)/O(1) comparison**: ~30 min per experiment = 60 min total
- **Sparse supervision**: ~30 min
- **16x16 benchmark**: ~60 min

**Total**: ~2.5 hours

---

## 🎯 Next Steps

After experiments complete:
1. Analyze O(1) vs O(R) convergence curves
2. Compare sparse vs dense supervision
3. Evaluate 16x16 scaling results
4. Generate comparison plots
5. Update paper/README with findings

---

## 📝 Implementation Details

**Files Modified**:
- `experiments/run_12x12_8m_benchmark.py`: Added O(1) mode + sparse supervision
- `test_o1_memory.py`: Validation script for O(1) mode
- `experiments/diagnose_poh_gradients.py`: Gradient flow diagnostic

**Git Branch**: `scaling_parameter_size`
**Latest Commit**: `8395a7e` - "feat: Add O(1) memory mode + sparse supervision"

