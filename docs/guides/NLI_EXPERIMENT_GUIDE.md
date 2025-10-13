# NLI Experiments: Testing PoH on Natural Language Understanding

**Task**: Natural Language Inference (SNLI)  
**Dataset**: 570k premise-hypothesis pairs  
**Goal**: Test if PoH improves compositional reasoning in NLP

---

## 🎯 Why Natural Language Inference?

### What is NLI?

Given a **premise** and a **hypothesis**, classify their relationship:
- **Entailment**: Hypothesis is definitely true given the premise
- **Contradiction**: Hypothesis is definitely false given the premise
- **Neutral**: Can't determine from the premise alone

**Examples:**

| Premise | Hypothesis | Label |
|---------|-----------|-------|
| "A man is playing guitar" | "A person is making music" | **Entailment** |
| "A man is playing guitar" | "A woman is playing piano" | **Contradiction** |
| "A man is playing guitar" | "The man is skilled" | **Neutral** |

### Why This Task Tests PoH Well

1. **Compositional Reasoning Required**
   - Must understand both premise AND hypothesis
   - Requires composing word meanings, syntax, semantics

2. **Multiple Reasoning Patterns**
   - Lexical overlap detection ("guitar" → "music")
   - Negation/contradiction patterns ("man" ≠ "woman")
   - Semantic similarity ("playing" → "making music")
   - Syntactic alignment
   - World knowledge ("playing guitar" → making music)

3. **Head Specialization Opportunity**
   - Different heads can learn different reasoning patterns
   - PoH routing can select relevant heads per example
   - Iterative refinement for hard cases

4. **Established Baselines**
   - BERT baseline: ~84-86% accuracy
   - Human performance: ~87-89%
   - Clear room for improvement

---

## 🏗️ Architecture

### Baseline (BERT + Classifier)

```
Input: [CLS] premise [SEP] hypothesis [SEP]
  ↓
BERT Encoder (12 layers, 768-dim)
  ↓
[CLS] representation
  ↓
Classification Head → 3-way softmax
```

**Parameters**: ~110M (BERT) + ~0.5M (classifier)

---

### PoH Enhanced

```
Input: [CLS] premise [SEP] hypothesis [SEP]
  ↓
BERT Encoder (12 layers, 768-dim)
  ↓
PoH Layer (iterative refinement with head routing)
  ├─ Controller: Learns routing weights α over heads
  ├─ Multi-head attention: Specialized heads
  ├─ Iterative refinement: 3-5 iterations
  └─ Temperature annealing: Soft → sharp routing
  ↓
Cross-Attention (hypothesis ← premise)
  ↓
Pooling ([CLS], mean, max, cross-attention)
  ↓
Classification Head → 3-way softmax
```

**Parameters**: ~110M (BERT) + ~2M (PoH) + ~0.5M (classifier)  
**Overhead**: ~2% parameters, ~20% time

---

## 📊 Experiment Design

### Quick Sanity Test (5-10 minutes)

```bash
# Test that everything works
PYTHONPATH=. python experiments/nli_poh_experiment.py \
  --encoder bert-base-uncased \
  --use_poh \
  --max_inner_iters 3 \
  --epochs 3 \
  --batch_size 32 \
  --num_train_samples 10000 \
  --num_val_samples 2000 \
  --seed 42
```

**Expected:**
- Baseline: ~75-78% accuracy (limited data)
- PoH: ~76-80% accuracy (+1-2% absolute)

---

### Full Experiment (2-4 hours per run)

**Configuration:**
- Training samples: 50,000
- Validation: 5,000
- Test: Full test set (~10k)
- Epochs: 5
- Seeds: 3 (42, 43, 44)

**Variants:**
1. **Baseline**: BERT + classifier
2. **PoH (3 iters)**: BERT + PoH (3 iterations) + classifier
3. **PoH (5 iters)**: BERT + PoH (5 iterations) + classifier

**Run:**
```bash
chmod +x experiments/run_nli_experiments.sh
./experiments/run_nli_experiments.sh
```

**Expected Results:**

| Model | Test Accuracy | vs Baseline |
|-------|--------------|-------------|
| Baseline | 84.0-86.0% | - |
| PoH (3 iters) | 85.0-87.5% | +1.0-1.5% ✅ |
| PoH (5 iters) | 85.5-88.0% | +1.5-2.0% ✅ |

---

### Frozen Encoder Test (Isolates PoH Contribution)

Freeze BERT, only train PoH layer + classifier.

**Purpose**: Show that PoH layer itself adds value, not just fine-tuning.

```bash
PYTHONPATH=. python experiments/nli_poh_experiment.py \
  --encoder bert-base-uncased \
  --freeze_encoder \
  --use_poh \
  --max_inner_iters 3 \
  --epochs 5 \
  --num_train_samples 50000
```

**Expected:**
- Baseline (frozen BERT): ~70-75%
- PoH (frozen BERT): ~73-78% (+3-5% with just PoH layer)

---

## 🔍 What to Look For

### 1. Accuracy Improvements

**Success Criteria:**
- **Minimum**: +0.5% absolute on test set
- **Target**: +1.0-1.5% absolute
- **Stretch**: +2.0%+ absolute

**Why This Matters:**
- On NLI, even +1% is significant (hard to improve over BERT)
- Consistent gains across seeds = robust
- Improvements on hard examples = PoH helps with complex reasoning

---

### 2. Routing Patterns

Check the diagnostics to see if heads specialize:

```python
import json

# Load results
with open('experiments/results_nli/nli_poh_seed42.json') as f:
    results = json.load(f)

# Check entropy
entropy_per_iter = results['results_per_epoch'][-1]['mean_entropy']
print(f"Routing entropy: {entropy_per_iter:.3f}")

# Expected:
# - Epoch 1: ~1.5-2.0 (soft routing)
# - Epoch 5: ~0.8-1.2 (specialized but not collapsed)
```

**Good Signs:**
- ✅ Entropy starts high (~1.5-2.0), gradually decreases
- ✅ Not collapsed to 0 (would mean one head dominates)
- ✅ Stable across iterations (no wild fluctuations)

**Bad Signs:**
- ❌ Entropy → 0 (routing collapsed)
- ❌ Entropy stays at ~2.3 (uniform, no specialization)
- ❌ Unstable (bouncing between 0 and 2)

---

### 3. Per-Class Performance

NLI has known difficulty patterns:
- **Entailment**: Usually easiest (~88-90%)
- **Neutral**: Hardest (~78-82%)
- **Contradiction**: Medium (~84-86%)

**Check if PoH helps with hard class (neutral):**

```python
# Compute per-class accuracy
# If PoH improves neutral more than others → strong evidence for compositional reasoning
```

---

### 4. Iteration Convergence

**Plot accuracy vs iteration:**

```python
# Expected pattern:
# Iter 1: 84.0%
# Iter 2: 85.2%
# Iter 3: 85.8%
# Iter 4: 86.0%
# Iter 5: 86.0% (plateau)
```

If accuracy **decreases** with iterations → over-refining (reduce max_iters)

---

## 📈 Expected Findings

### Main Result

**PoH should achieve +1-2% absolute accuracy improvement over BERT baseline**

This would be significant because:
1. BERT is already very strong on NLI (~84-86%)
2. Hard to improve without massive models or ensembles
3. +1% = ~1000 more correct predictions on 10k test set

---

### Secondary Findings

1. **Head Specialization**
   - Different heads focus on different reasoning patterns
   - Can visualize with attention heatmaps
   - Routing entropy shows adaptation

2. **Iterative Improvement**
   - Accuracy increases with iterations (up to ~3-5)
   - Diminishing returns after that
   - Some examples need more iterations (hard cases)

3. **Frozen Encoder Test**
   - PoH layer alone adds +3-5% over frozen BERT
   - Shows PoH contributes beyond just fine-tuning
   - Validates architectural improvement

4. **Efficiency**
   - 2% more parameters
   - 20% more time per forward pass
   - Worth it for +1-2% accuracy

---

## 🚀 Quick Start

### Option 1: Quick Test (5 minutes)

```bash
cd /Users/rnbnrzy/Desktop/PoT
source venv/bin/activate  # If using venv

# Run quick sanity test
PYTHONPATH=. python experiments/nli_poh_experiment.py \
  --use_poh --max_inner_iters 3 --epochs 3 \
  --num_train_samples 5000 --num_val_samples 1000 \
  --seed 42
```

---

### Option 2: Full Experiment (2-4 hours)

```bash
chmod +x experiments/run_nli_experiments.sh
./experiments/run_nli_experiments.sh 2>&1 | tee nli_experiments.log
```

---

### Option 3: Single Full Run (30-60 minutes)

```bash
PYTHONPATH=. python experiments/nli_poh_experiment.py \
  --encoder bert-base-uncased \
  --use_poh \
  --max_inner_iters 3 \
  --epochs 5 \
  --batch_size 32 \
  --lr 2e-5 \
  --num_train_samples 50000 \
  --num_val_samples 5000 \
  --seed 42 \
  --output_dir experiments/results_nli
```

---

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'experiments.sort_pointer_improved'"

**Fix:**
```bash
export PYTHONPATH=/Users/rnbnrzy/Desktop/PoT:$PYTHONPATH
# Or run with: PYTHONPATH=. python ...
```

---

### Issue: Out of Memory (GPU)

**Fix 1**: Reduce batch size
```bash
--batch_size 16  # or 8
```

**Fix 2**: Reduce max iterations
```bash
--max_inner_iters 2  # instead of 3 or 5
```

**Fix 3**: Use gradient checkpointing
```python
# In model __init__:
self.encoder.gradient_checkpointing_enable()
```

---

### Issue: SNLI dataset download fails

**Fix**: Download manually
```python
from datasets import load_dataset
dataset = load_dataset("snli", cache_dir="./data")
```

Then update script to use local cache.

---

## 📚 Background Reading

### SNLI Dataset
- Paper: Bowman et al. (2015) "A large annotated corpus for learning natural language inference"
- 570k human-written sentence pairs
- 3-way classification (entailment, contradiction, neutral)
- Standard benchmark for NLI

### Why NLI is Hard
1. **Compositional**: Must understand both sentences and their relationship
2. **Lexical gaps**: Synonym detection, paraphrasing
3. **World knowledge**: "playing guitar" → making music
4. **Negation**: Subtle contradictions
5. **Neutral class**: Hardest to learn (requires saying "don't know")

### BERT Baseline
- Pre-trained on 3.3B words (Books + Wikipedia)
- 12 layers, 768-dim, 12 heads
- Fine-tuning on SNLI: ~84-86% accuracy
- State-of-art (non-ensemble): ~89-91% with RoBERTa-large

---

## 📊 Analysis Scripts

After running experiments, analyze results:

```python
import json
import glob
import pandas as pd

# Load all results
results = []
for file in glob.glob('experiments/results_nli/nli_*.json'):
    with open(file) as f:
        data = json.load(f)
        results.append({
            'model': 'PoH' if data['args']['use_poh'] else 'Baseline',
            'iters': data['args'].get('max_inner_iters', 1),
            'seed': data['args']['seed'],
            'test_acc': data['test_acc'],
            'best_val_acc': data['best_val_acc'],
        })

df = pd.DataFrame(results)

# Group by model
summary = df.groupby(['model', 'iters']).agg({
    'test_acc': ['mean', 'std'],
    'best_val_acc': ['mean', 'std'],
})

print(summary)
```

---

## 🎯 Success Criteria

| Metric | Minimum | Target | Stretch |
|--------|---------|--------|---------|
| **Test Accuracy (PoH)** | 85.0% | 86.0% | 87.0% |
| **vs Baseline** | +0.5% | +1.0% | +2.0% |
| **Routing Entropy** | 0.5-2.0 | 0.8-1.5 | 1.0-1.3 |
| **Consistent across seeds** | 2/3 | 3/3 | 3/3 + low variance |

---

**Ready to test PoH on a challenging NLP task!** 🚀

All code is implemented, documented, and ready to run.
