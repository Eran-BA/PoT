# PoH Notebooks

This directory contains Jupyter/Colab notebooks for interactive exploration and benchmarking of the Pointer-over-Heads (PoH) transformer.

## 📓 Available Notebooks

### 1. PoT_Colab.ipynb
**Main demonstration notebook**
- Complete PoH architecture walkthrough
- Dependency parsing examples
- Interactive visualizations
- Quick start for beginners

**Run in Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eran-BA/PoT/blob/main/notebooks/PoT_Colab.ipynb)

### 2. PoH_GPT_AB_Test.ipynb
**Autoregressive language modeling benchmark**
- PoH-GPT vs Baseline GPT comparison
- Quick 2-minute A/B test
- Perplexity comparison
- Generation examples

**Run in Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eran-BA/PoT/blob/main/notebooks/PoH_GPT_AB_Test.ipynb)

**Features:**
- Synthetic language modeling dataset
- Fair parameter matching
- Side-by-side training
- Results visualization

### 3. PoH_NLI_Benchmark.ipynb
**Natural Language Inference benchmark**
- PoH vs BERT on NLI tasks
- SNLI/MultiNLI support
- Quick test (3 minutes)
- Full benchmark option

**Run in Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eran-BA/PoT/blob/main/notebooks/PoH_NLI_Benchmark.ipynb)

**Features:**
- Real dataset support (Hugging Face datasets)
- Synthetic data option (no dependencies)
- 12-iteration optimal configuration
- Accuracy comparison

---

## 🚀 Quick Start

### Running Locally

```bash
# Install Jupyter
pip install jupyter notebook

# Launch notebook server
cd notebooks
jupyter notebook
```

### Running in Colab

Click any of the "Open in Colab" badges above!

---

## 📋 Notebook Contents Summary

| Notebook | Task | Duration | Datasets | Key Features |
|----------|------|----------|----------|--------------|
| **PoT_Colab** | Dependency Parsing | 10-15 min | UD English | Architecture demo, visualization |
| **PoH_GPT_AB_Test** | Language Modeling | 2-15 min | Synthetic | Quick A/B test, generation |
| **PoH_NLI_Benchmark** | NLI | 3 min - 4 hours | SNLI/Synthetic | BERT comparison, real datasets |

---

## 🎯 Recommended Order

1. **Start with PoT_Colab.ipynb**
   - Understand PoH architecture
   - See routing in action
   - Learn basic usage

2. **Try PoH_GPT_AB_Test.ipynb**
   - Quick 2-minute test
   - See autoregressive PoH
   - Compare to baseline

3. **Run PoH_NLI_Benchmark.ipynb**
   - Real NLP task
   - Production-like setup
   - State-of-the-art comparison

---

## 💡 Tips

**For Quick Testing:**
- All notebooks have "quick test" options (2-3 minutes)
- Perfect for verifying setup works
- Shows key results without full training

**For Full Benchmarks:**
- PoH-GPT: 15 minutes
- PoH-NLI: 4 hours (real SNLI)
- Use GPU for faster training

**For Colab:**
- Enable GPU runtime (Runtime → Change runtime type → GPU)
- Notebooks install dependencies automatically
- Results saved to Colab session

---

## 📚 Related Documentation

- [Quick Start Guide](../QUICK_START.md) - Command-line quickstart
- [Architecture Guide](../docs/architecture/POH_ARCHITECTURE_SUMMARY.md) - Technical details
- [Iteration Guide](../docs/POH_ITERATION_GUIDE.md) - Why 12 iterations?
- [Running Benchmarks](../docs/guides/RUNNING_BENCHMARKS.md) - Full benchmark guide

---

## 🐛 Troubleshooting

**Notebook won't start:**
```bash
pip install jupyter ipykernel
python -m ipykernel install --user
```

**Import errors:**
```python
# Run this in the first notebook cell
!pip install torch numpy matplotlib seaborn pyyaml
```

**Colab GPU not available:**
- Go to Runtime → Change runtime type
- Select "GPU" from Hardware accelerator dropdown

---

## 📊 Expected Results

### PoH-GPT Quick Test
```
Baseline GPT: ppl=1009.67
PoH-GPT:      ppl=1000.03
Δ improvement: +0.95%
```

### PoH-NLI Quick Test
```
BERT-Small: acc=0.450
PoH-Small:  acc=0.465
Δ improvement: +3.3%
```

---

**Questions?** See [docs/README.md](../docs/README.md) or open an issue!

