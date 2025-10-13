# 🧠 NLI Benchmark: BERT vs PoH

**Date:** October 13, 2025  
**Status:** ✅ Complete & Ready to Run

---

## 🎯 Overview

This benchmark compares **BERT** (standard transformer encoder) against **PoH** (Pointer-over-Heads transformer) on the **Natural Language Inference (NLI)** task.

### Task: Natural Language Inference

Given a premise and hypothesis, classify their relationship:
- ✅ **Entailment**: hypothesis follows from premise
- ⚖️ **Neutral**: hypothesis could be true  
- ❌ **Contradiction**: hypothesis contradicts premise

**Example:**
```
Premise: "A man is playing guitar"
Hypothesis: "A musician is performing"
Label: Entailment ✅
```

---

## 🏗️ Models

### BERT Baseline (`src/models/bert_baseline.py`)
- Standard BERT-base architecture
- 12 transformer layers
- 768 hidden dimensions
- 12 attention heads
- 3072 FFN dimensions
- ~110M parameters (full), ~11M (small version for testing)

### PoH-NLI (`src/pot/models/poh_nli.py`)
- Same base architecture as BERT
- **+ Adaptive head routing** (dynamic attention head selection)
- **+ Iterative refinement** (3 inner reasoning steps)
- **+ Outer residuals** (ReZero-style skip connections)
- ~110M parameters (matched with BERT for fair comparison)

---

## 📊 What Makes This Benchmark Fair?

| Aspect | BERT | PoH | Notes |
|--------|------|-----|-------|
| **Parameters** | ~110M | ~110M | Matched |
| **Architecture** | 12 layers, 768 dim | 12 layers, 768 dim | Identical base |
| **Embeddings** | Token + Position + Segment | Token + Position + Segment | Same |
| **Optimizer** | AdamW (lr=2e-5) | AdamW (lr=2e-5) | Same |
| **Batch Size** | 32 | 32 | Same |
| **Training Steps** | 10,000 | 10,000 | Same |
| **Data** | Synthetic NLI | Synthetic NLI | Same |
| **Random Seed** | 42 | 42 | Reproducible |

**Key Difference:** PoH adds adaptive routing + iterative refinement with minimal parameter overhead.

---

## 🚀 Quick Start

### 1. Quick Test (3 minutes)
```bash
cd PoT
python experiments/quick_nli_test.py
```

**What it does:**
- Trains small versions (4 layers, 256 dim) for 100 steps
- Compares BERT-Small vs PoH-Small
- Shows accuracy improvement

**Expected output:**
```
BERT-Small: acc=0.XXX, time=XX.Xs
PoH-Small:  acc=0.XXX, time=XX.Xs
Δ improvement: +X.XX%
```

### 2. Full Benchmark (30 minutes)
```bash
python experiments/fair_ab_nli.py
```

**What it does:**
- Trains full BERT-base and PoH (12 layers, 768 dim)
- 10,000 training steps with evaluation every 500 steps
- Logs results to `experiments/results/nli/ab_results.csv`

### 3. Google Colab
Upload `PoH_NLI_Benchmark.ipynb` to Colab and run all cells.

---

## 📁 File Structure

```
PoT/
├── src/
│   ├── models/
│   │   └── bert_baseline.py         # BERT implementation
│   └── pot/
│       ├── models/
│       │   └── poh_nli.py           # PoH-NLI implementation
│       └── tasks/
│           └── nli.py               # NLI task adapter
├── experiments/
│   ├── configs/nli/
│   │   ├── bert_baseline.yaml       # BERT config
│   │   └── poh.yaml                 # PoH config
│   ├── quick_nli_test.py            # Quick smoke test
│   └── fair_ab_nli.py               # Full benchmark
└── PoH_NLI_Benchmark.ipynb          # Colab notebook
```

---

## 🔬 Technical Details

### Input Format
Both models use BERT-style sequence packing:
```
[CLS] premise_tokens [SEP] hypothesis_tokens [SEP]
```

### Training Details
- **Loss:** CrossEntropyLoss (3-way classification)
- **Optimizer:** AdamW with weight decay 0.01
- **LR Schedule:** Linear warmup (1000 steps) + linear decay
- **Gradient Clipping:** Max norm 1.0
- **Evaluation:** Every 500 steps on validation set

### PoH-Specific Settings
- **Inner Iterations:** 3 (configurable)
- **Routing Mode:** Soft (all heads weighted)
- **Outer Residual:** Enabled with ReZero init
- **Shared Router:** True (reduces parameters)

---

## 📈 Expected Results

### Hypothesis
PoH should outperform BERT because:

1. **Adaptive Routing** 
   - Dynamically selects relevant attention heads
   - Focuses on premise-hypothesis relationships
   - Reduces noise from irrelevant heads

2. **Iterative Refinement**
   - Multiple reasoning steps (3 iterations)
   - Refines representations progressively
   - Better captures complex entailment patterns

3. **Outer Residuals**
   - Stable gradient flow across iterations
   - Prevents degradation from deep refinement
   - Enables effective multi-step reasoning

### Metrics
- **Primary:** Accuracy (3-way classification)
- **Secondary:** Per-class accuracy (entailment/neutral/contradiction)
- **Efficiency:** Training time, tokens/sec

---

## 🧪 Synthetic Data

For quick testing without external dependencies, we use synthetic NLI data:
- Random token IDs for premise and hypothesis
- Random labels (0, 1, 2)
- Configurable sequence lengths
- Proper padding and masking

**Note:** For real benchmarks, use SNLI or MultiNLI datasets via Hugging Face `datasets` library.

---

## 🎓 Why This Matters

### Research Value
- **Fair Comparison:** Matched parameters eliminate confounds
- **Ablation-Ready:** Easy to disable routing or refinement
- **Reproducible:** Fixed seeds, deterministic training
- **Extensible:** Add real datasets, more metrics, etc.

### Practical Value
- **Production Template:** Clean, modular code
- **Easy Integration:** Drop-in BERT replacement
- **Well-Documented:** Comprehensive comments and docs
- **Colab-Ready:** Instant cloud execution

---

## 📝 Next Steps

### P1: Real Data
- Integrate SNLI dataset (550K examples)
- Add MultiNLI (433K examples)
- Report test set performance

### P2: Analysis
- Visualize routing patterns
- Plot inner-loop convergence
- Error analysis by label type

### P3: Optimization
- Mixed precision training (AMP)
- Gradient checkpointing for larger models
- Distributed training support

### P4: Ablations
- Routing mode (soft vs top-k)
- Number of inner iterations (1-5)
- Outer residual vs no residual
- Shared vs per-block routers

---

## ✅ Checklist

- [x] BERT baseline implementation
- [x] PoH-NLI implementation
- [x] NLI task adapter
- [x] Synthetic data generator
- [x] Training scripts (quick + full)
- [x] YAML configs
- [x] Colab notebook
- [x] Documentation
- [x] Git commit & push
- [ ] Run full benchmark (user's machine)
- [ ] Generate plots
- [ ] Update main README

---

## 🙏 Credits

**Author:** Eran Ben-Artzy  
**License:** Apache 2.0  
**Year:** 2025

**Inspired by:**
- BERT (Devlin et al., 2019)
- Hierarchical Reasoning Models
- Adaptive Computation Time
- ReZero transformers

---

**🎉 Ready to benchmark! Run `python experiments/quick_nli_test.py` to get started.**

