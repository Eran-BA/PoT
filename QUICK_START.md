# 🚀 Quick Start: Run NLI Benchmarks

## ✅ Simple Commands (Copy & Paste)

### Step 1: Activate Virtual Environment
```bash
cd /Users/rnbnrzy/Desktop/PoT
source venv/bin/activate
```

### Step 2: Install Missing Dependencies
```bash
pip install pyyaml datasets --quiet
```

### Step 3: Choose a Benchmark

#### Option A: Quick Test (3 minutes)
```bash
PYTHONPATH=$PWD python experiments/quick_nli_test.py
```

#### Option B: Synthetic Benchmark (30-60 minutes)
```bash
PYTHONPATH=$PWD python experiments/fair_ab_nli.py
```

#### Option C: Real NLI Quick Test (30 minutes, recommended)
```bash
PYTHONPATH=$PWD python experiments/real_nli_benchmark.py \
  --dataset snli \
  --max_train_samples 5000 \
  --max_steps 2000 \
  --batch_size 32
```

#### Option D: Real NLI Full Benchmark (2-3 hours)
```bash
PYTHONPATH=$PWD python experiments/real_nli_benchmark.py \
  --dataset snli \
  --max_steps 20000 \
  --batch_size 32
```

---

## 🎯 Recommended: Start with Quick Test

```bash
cd /Users/rnbnrzy/Desktop/PoT
source venv/bin/activate
pip install pyyaml datasets --quiet
PYTHONPATH=$PWD python experiments/quick_nli_test.py
```

This runs in 3 minutes and verifies everything works!

---

## 📊 Check Results

### After Quick Test:
```bash
# See console output for summary
```

### After Full Benchmarks:
```bash
# Synthetic results
cat experiments/results/nli/ab_results.csv

# Real NLI results  
cat experiments/results/real_nli/benchmark_results.csv
```

---

## 🐛 Troubleshooting

### Error: "No module named 'yaml'"
**Solution:** You forgot to activate the virtual environment!
```bash
source venv/bin/activate
pip install pyyaml
```

### Error: "No module named 'torch'"
**Solution:** Activate venv first!
```bash
source venv/bin/activate
```

### Error: "No module named 'datasets'"
**Solution:** Install datasets:
```bash
pip install datasets
```

---

## ⚡ One-Line Commands (After activating venv)

```bash
# Quick test (3 min)
source venv/bin/activate && PYTHONPATH=$PWD python experiments/quick_nli_test.py

# Real NLI quick (30 min)  
source venv/bin/activate && PYTHONPATH=$PWD python experiments/real_nli_benchmark.py --max_train_samples 5000 --max_steps 2000

# Full synthetic (60 min)
source venv/bin/activate && PYTHONPATH=$PWD python experiments/fair_ab_nli.py
```

---

## 📝 Using the Interactive Script

```bash
cd /Users/rnbnrzy/Desktop/PoT
./run_nli_benchmarks.sh
```

Then select from menu:
1. Quick synthetic test (3 min)
2. Full synthetic benchmark (30-60 min)
3. Real NLI quick test (30 min, 5K samples)
4. Real NLI full benchmark (2-3 hours)
5. Run both synthetic and real (quick)

---

## ✅ Recommended First Run

```bash
cd /Users/rnbnrzy/Desktop/PoT
source venv/bin/activate
pip install pyyaml datasets --quiet
echo "✅ Dependencies ready!"
echo ""
echo "Starting quick test..."
PYTHONPATH=$PWD python experiments/quick_nli_test.py
```

**This will:**
- Take 3 minutes
- Show you BERT vs PoH (12 iterations) comparison
- Verify everything works
- Then you can run longer benchmarks!

**Note:** PoH uses 12 inner iterations (optimal from empirical analysis).  
See `docs/POH_ITERATION_GUIDE.md` for details on iteration count selection.

---

**🎉 Ready to go! Just copy-paste the recommended commands above!**

