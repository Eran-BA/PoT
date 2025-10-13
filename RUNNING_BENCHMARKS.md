# 🚀 Running NLI Benchmarks

**Date:** October 13, 2025  
**Status:** ✅ Benchmarks Running

---

## 📊 Active Benchmarks

### 1. Synthetic NLI Benchmark (RUNNING)
**Script:** `experiments/fair_ab_nli.py`  
**Status:** ✅ Running in background  
**Duration:** ~30-60 minutes  
**Steps:** 10,000  
**Data:** Synthetic random NLI pairs  

**Monitor progress:**
```bash
tail -f logs/fair_ab_nli.log
```

**Results will be saved to:**
```
experiments/results/nli/ab_results.csv
```

---

### 2. Real NLI Benchmark (READY)
**Script:** `experiments/real_nli_benchmark.py`  
**Status:** ⏳ Ready to run (requires `datasets` library)  
**Duration:** ~2-3 hours (full) or ~30 min (quick test)  
**Steps:** 20,000 (configurable)  
**Data:** SNLI or MultiNLI from Hugging Face  

**Prerequisites:**
```bash
pip install datasets
```

**Quick test (5K samples, 2K steps, ~30 minutes):**
```bash
cd /Users/rnbnrzy/Desktop/PoT
PYTHONPATH=$PWD python3 experiments/real_nli_benchmark.py \
  --dataset snli \
  --max_train_samples 5000 \
  --max_steps 2000 \
  --batch_size 32
```

**Full benchmark (20K steps, ~2-3 hours):**
```bash
PYTHONPATH=$PWD python3 experiments/real_nli_benchmark.py \
  --dataset snli \
  --max_steps 20000 \
  --batch_size 32
```

**Results will be saved to:**
```
experiments/results/real_nli/benchmark_results.csv
experiments/results/real_nli/BERT_best.pt
experiments/results/real_nli/PoH_best.pt
```

---

## 📁 Output Files

### Synthetic Benchmark
- **CSV Results:** `experiments/results/nli/ab_results.csv`
- **Checkpoints:** `experiments/results/nli/bert/`, `experiments/results/nli/poh/`
- **Logs:** `logs/fair_ab_nli.log`

### Real Benchmark
- **CSV Results:** `experiments/results/real_nli/benchmark_results.csv`
- **Checkpoints:** `experiments/results/real_nli/BERT_best.pt`, `experiments/results/real_nli/PoH_best.pt`
- **Logs:** (console output or redirect to file)

---

## 🔍 Monitoring Progress

### Check if benchmarks are running:
```bash
ps aux | grep -E "(fair_ab_nli|real_nli_benchmark)"
```

### Monitor synthetic benchmark:
```bash
tail -f logs/fair_ab_nli.log
```

### Monitor real benchmark (if redirected):
```bash
tail -f logs/real_nli_benchmark.log
```

---

## 📊 Expected Results

### Synthetic Data (fair_ab_nli.py)
```
BERT Baseline: acc=~0.35-0.40
PoH:           acc=~0.30-0.45
Δ improvement: Variable (synthetic data not representative)
```

**Note:** Synthetic random data doesn't show PoH's true strengths.

### Real Data (real_nli_benchmark.py)
```
BERT Baseline: acc=~0.85-0.88 (on SNLI)
PoH:           acc=~0.87-0.90 (on SNLI)
Δ improvement: +2-5% expected
```

**Note:** Real semantic data benefits from PoH's adaptive routing and iterative refinement.

---

## ⏱️ Estimated Times

| Benchmark | Steps | Data | Device | Time |
|-----------|-------|------|--------|------|
| Synthetic (quick) | 100 | Synthetic | CPU | ~3 min |
| Synthetic (full) | 10K | Synthetic | CPU | ~30-60 min |
| Real (quick test) | 2K | SNLI (5K samples) | CPU | ~30 min |
| Real (quick test) | 2K | SNLI (5K samples) | GPU | ~10 min |
| Real (full) | 20K | SNLI (full) | CPU | ~2-3 hours |
| Real (full) | 20K | SNLI (full) | GPU | ~30-60 min |

---

## 🛠️ Troubleshooting

### Benchmark not running?
Check process:
```bash
ps aux | grep fair_ab_nli
```

### Out of memory?
Reduce batch size in configs:
```yaml
train:
  batch_size: 16  # or 8
```

### Datasets library not found?
Install:
```bash
pip install datasets
```

### Slow on CPU?
Expected. GPU recommended for real benchmarks:
- Enable GPU in system
- PyTorch will auto-detect CUDA
- Or reduce steps: `--max_steps 1000`

---

## 📈 Next Steps After Benchmarks Complete

1. **Check Results:**
   ```bash
   cat experiments/results/nli/ab_results.csv
   cat experiments/results/real_nli/benchmark_results.csv
   ```

2. **Generate Plots:**
   ```bash
   python3 scripts/plot_nli_results.py
   ```

3. **Analyze Results:**
   - Compare BERT vs PoH accuracy
   - Per-class performance (entailment/neutral/contradiction)
   - Training time comparison
   - Plot learning curves

4. **Run Ablations:**
   - Different inner iteration counts (1, 2, 3, 5)
   - Top-k routing vs soft routing
   - With/without outer residuals

---

## 🎯 Commands Summary

```bash
# Monitor running benchmark
tail -f logs/fair_ab_nli.log

# Install datasets for real benchmark
pip install datasets

# Run real NLI quick test
PYTHONPATH=$PWD python3 experiments/real_nli_benchmark.py \
  --max_train_samples 5000 --max_steps 2000

# Run real NLI full benchmark
PYTHONPATH=$PWD python3 experiments/real_nli_benchmark.py \
  --dataset snli --max_steps 20000

# Check results
cat experiments/results/nli/ab_results.csv
cat experiments/results/real_nli/benchmark_results.csv
```

---

**✅ Synthetic benchmark is running in background!**  
**📊 Real benchmark ready to run after installing `datasets`**  
**⏳ Check back in 30-60 minutes for synthetic results**

---

**Author:** Eran Ben-Artzy  
**License:** Apache 2.0  
**Year:** 2025

