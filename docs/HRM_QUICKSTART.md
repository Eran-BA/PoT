# HRM Controller Quick Start Guide

**Author**: Eran Ben Artzy  
**Date**: 2025  
**License**: Apache 2.0

---

## 🚀 **5-Minute Quick Start**

```bash
# 1. Run diagnostic smoke test
make smoke-hrm

# 2. Quick smoke training (3 epochs)
make hrm-quick

# 3. Full A/B comparison (HRM vs Baseline)
make hrm-ab
```

---

## ✅ **Step-by-Step Test Procedure**

### **Step 1: Install Dependencies**

```bash
pip install pytest  # For unit tests
pip install -e .    # Install package
```

---

### **Step 2: Run Unit Tests**

```bash
# All HRM unit tests (10 tests)
make test-hrm

# Or manually:
PYTHONPATH=. pytest tests/test_hrm_pointer_controller.py -v
```

**Expected output**:
```
test_basic_forward_shapes PASSED                     [ 10%]
test_high_level_updates_every_T_steps PASSED          [ 20%]
test_topk_sparsity_and_temperature PASSED             [ 30%]
test_temperature_schedule PASSED                      [ 40%]
test_gradients_flow PASSED                            [ 50%]
test_state_persistence PASSED                         [ 60%]
test_entropy_regularization PASSED                    [ 70%]
test_different_input_shapes PASSED                    [ 80%]
test_masked_pooling PASSED                            [ 90%]
test_topk_variants PASSED                             [100%]

====== 10 passed in 2.5s ======
```

✅ **All tests should pass!**

---

### **Step 3: Run Diagnostic Smoke Test**

```bash
make smoke-hrm

# Or manually:
PYTHONPATH=. python tools/hrm_diag_smoke.py
```

**What to look for**:
```
Iter 0: entropy=1.0874, max_prob=0.4278, temp=2.000, top_heads=[1, 2, 6] ← H-UPDATE
Iter 1: entropy=1.0876, max_prob=0.4307, temp=1.900, top_heads=[1, 5, 2] 
Iter 2: entropy=1.0852, max_prob=0.4433, temp=1.805, top_heads=[1, 5, 2] 
Iter 3: entropy=1.0891, max_prob=0.4159, temp=1.715, top_heads=[1, 5, 2] ← H-UPDATE
...
```

**Expected behavior**:
- ✅ Entropy decreases (routing sharpens)
- ✅ H-UPDATE markers every T=3 steps
- ✅ Routing stabilizes on preferred heads
- ✅ Temperature cools from 2.0 → 0.7

---

### **Step 4: Quick Smoke Training (3 epochs)**

```bash
make hrm-quick
```

This runs:
- Task: Partial observability sorting (50% masked)
- Array length: 12
- Training: 100 samples, 3 epochs
- Seeds: 1

**Expected**:
- Training completes in ~30 seconds
- No errors
- Creates `experiments/results/smoke_hrm.csv`
- Model trains (Kendall-τ improves)

---

### **Step 5: Full A/B Comparison**

```bash
make hrm-ab
```

This automatically runs:

1. **Baseline** (standard single-pass model)
   - 1000 samples, 40 epochs
   - 5 seeds
   - Saves to `experiments/results/ab_baseline.csv`

2. **HRM PoT** (with multi-timescale controller)
   - Same config as baseline
   - HRM controller: T=4, top-k=3, 4 iterations
   - Saves to `experiments/results/ab_pot_hrm.csv`

**Time**: ~20-30 minutes (depends on hardware)

---

### **Step 6: Analyze Results**

```bash
# Compare results
python experiments/compare_ab_results.py \
  experiments/results/ab_baseline.csv \
  experiments/results/ab_pot_hrm.csv

# Or manually analyze CSVs
import pandas as pd
baseline = pd.read_csv("experiments/results/ab_baseline.csv")
hrm = pd.read_csv("experiments/results/ab_pot_hrm.csv")

print(f"Baseline: {baseline['kendall'].mean():.4f} ± {baseline['kendall'].std():.4f}")
print(f"HRM:      {hrm['kendall'].mean():.4f} ± {hrm['kendall'].std():.4f}")
```

**Expected outcome**:
- HRM shows **+1-3% improvement** over baseline
- Lower variance (more stable)
- Entropy decreases across epochs

---

## 🔧 **Manual A/B Commands**

If you want to run experiments manually with custom parameters:

### **Baseline**

```bash
PYTHONPATH=. python experiments/fair_ab_comparison.py \
  --model baseline \
  --array_len 12 \
  --mask_rate 0.5 \
  --train_samples 1000 \
  --epochs 40 \
  --batch_size 64 \
  --lr 3e-4 \
  --seeds 1 2 3 4 5 \
  --output_csv experiments/results/ab_baseline.csv
```

### **HRM PoT**

```bash
PYTHONPATH=. python experiments/fair_ab_comparison.py \
  --model pot \
  --array_len 12 \
  --mask_rate 0.5 \
  --train_samples 1000 \
  --epochs 40 \
  --batch_size 64 \
  --lr 3e-4 \
  --max_inner_iters 4 \
  --seeds 1 2 3 4 5 \
  --output_csv experiments/results/ab_pot_hrm.csv
```

**Note**: The current `fair_ab_comparison.py` uses the standard PoH controller. To use HRM, you'll need to integrate it into the script (see integration guide below).

---

## 📊 **Interpreting Results**

### **Good Signs**:
- ✅ HRM Kendall-τ > Baseline Kendall-τ
- ✅ Entropy starts high (~1.5-2.0), decreases to ~0.5-1.0
- ✅ Head usage shows specialization (non-uniform)
- ✅ Confidence intervals don't overlap

### **Red Flags**:
- ❌ Routing collapses to single head (entropy → 0)
- ❌ HRM worse than baseline (may need hyperparameter tuning)
- ❌ High variance across seeds (may need more seeds)
- ❌ NaN/Inf in training (gradient issues)

---

## 🛠️ **Hyperparameter Tuning**

If HRM doesn't show improvement, try:

### **Increase Temperature**
```bash
--temperature_init 2.5  # Start softer (default: 2.0)
--temperature_min 1.0   # Don't sharpen too much (default: 0.7)
```

### **Add Entropy Regularization**
```bash
--entropy_reg 1e-3  # Encourage diverse routing
```

### **Adjust H-Module Period**
```bash
--hrm_T 2   # Faster H-updates (more synchronized)
--hrm_T 6   # Slower H-updates (stronger hierarchy)
```

### **Change Top-K Sparsity**
```bash
--routing_topk 4  # Less sparse (default: 3)
--routing_topk 2  # More sparse (may lose flexibility)
```

### **More Iterations**
```bash
--max_inner_iters 6  # More refinement steps
```

---

## 🐛 **Troubleshooting**

### **Test Failures**

**Problem**: `test_high_level_updates_every_T_steps` fails

**Solution**: Check that state is threaded correctly across iterations

---

**Problem**: `test_gradients_flow` fails

**Solution**: Ensure loss depends on alphas, no detached states

---

### **Training Issues**

**Problem**: Routing collapses to single head

**Solution**:
```bash
# Increase temperature
--temperature_init 1.5

# Add entropy reg
--entropy_reg 5e-3

# Reduce top-k sparsity
--routing_topk 4
```

---

**Problem**: NaN in training

**Solution**:
```bash
# Clip gradients
--clip_grad 1.0

# Reduce controller LR
--lr_controller 1e-4  # vs encoder LR 3e-4
```

---

**Problem**: HRM slower than baseline

**Expected**: HRM adds ~10-20% overhead due to:
- Two GRU cells (f_L, f_H)
- Extra state management

**Optimization**: Use larger batch sizes to amortize overhead

---

## 📈 **Next Experiments**

Once basic A/B passes:

### **1. Length Scaling**
```bash
# Test HRM advantage on longer sequences
for len in 12 16 20 24; do
  make hrm-ab ARRAY_LEN=$len
done
```

### **2. Iteration Sweep**
```bash
# Find optimal iteration count
for iters in 2 4 6 8 12; do
  PYTHONPATH=. python experiments/fair_ab_comparison.py \
    --model pot --max_inner_iters $iters ...
done
```

### **3. T (Timescale) Sweep**
```bash
# Find optimal H-module period
for T in 2 3 4 6 8; do
  # Run with --hrm_T $T
done
```

### **4. Dependency Parsing**
```bash
# Test HRM on real UD EWT task
python scripts/train.py \
  --model pot --use_hrm \
  --hrm_T 4 --routing_topk 3 \
  --dataset ud_ewt \
  --epochs 50
```

---

## 📚 **Further Reading**

- **Integration Guide**: `docs/hrm_integration.md`
- **Testing Guide**: `docs/hrm_testing.md`
- **Architecture Docs**: `docs/architecture.md`
- **HRM Paper**: [arXiv:2506.21734](https://arxiv.org/abs/2506.21734)

---

## 🎯 **Success Criteria**

You're ready to move forward when:

- ✅ All unit tests pass (10/10)
- ✅ Diagnostic smoke test shows expected behavior
- ✅ Quick smoke training completes without errors
- ✅ Full A/B comparison runs successfully
- ✅ HRM shows competitive or better performance vs baseline

---

**Questions?** Check `docs/hrm_testing.md` for detailed troubleshooting!

**Author**: Eran Ben Artzy  
**License**: Apache 2.0

