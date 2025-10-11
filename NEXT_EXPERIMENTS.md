# 🔬 Next Experiments: Unlocking Multi-Iteration Benefits

**Author:** Eran Ben Artzy  
**Date:** October 11, 2025  

---

## 🎯 Current Finding

**PoH achieves 97.95% UAS with 1 iteration, but 2-7 iterations show NO improvement.**

Why? The model learns to "solve early" because:
1. ✅ Task saturation (~97.5% baseline leaves little headroom)
2. ✅ No new evidence per iteration (same context, just re-mixed)
3. ✅ Training pressure to solve in pass 1 (only final logits supervised)
4. ✅ Weak controller feedback (mean/max is coarse signal)
5. ✅ Parsing is mostly local (self-attention already captures long-range)

---

## 🔬 Proposed Experiments (Priority Order)

### 1. Distance-Bucket Analysis (Diagnostic)
**Goal:** Prove iterations don't help long-range dependencies

```bash
python ab_ud_pointer_vs_baseline.py \
  --data_source conllu --conllu_dir ud_data \
  --epochs 3 --batch_size 32 --lr 3e-5 \
  --max_inner_iters 1 --log_distance_buckets
  
python ab_ud_pointer_vs_baseline.py \
  --data_source conllu --conllu_dir ud_data \
  --epochs 3 --batch_size 32 --lr 3e-5 \
  --max_inner_iters 3 --log_distance_buckets
```

**Expected:** If 3 iters don't improve UAS on long-range arcs (>6 tokens), 
they're not adding multi-hop reasoning.

---

### 2. Entropy Halting Ablation (Efficiency)
**Goal:** Achieve ~1.2-1.5 avg iterations with same UAS

```bash
for threshold in 0.6 0.65 0.7; do
  python ab_ud_pointer_vs_baseline.py \
    --data_source conllu --conllu_dir ud_data \
    --epochs 3 --batch_size 32 --lr 3e-5 \
    --halting_mode entropy --max_inner_iters 3 \
    --ent_threshold $threshold \
    --log_csv entropy_ablation.csv
done
```

**Expected:** Lower threshold → more early stopping → fewer avg iters with same UAS
**Best config:** threshold that gives mean_iters ~1.3 with 97.9% UAS

---

### 3. Deep Supervision (Training Fix)
**Goal:** Force model to improve across iterations via auxiliary losses

**Modification:** Add per-pass losses with increasing weights:
```python
# In forward pass:
losses = []
for t, logits_t in enumerate(pass_logits):
    weight = 0.3 + 0.35 * t  # [0.3, 0.65, 1.0]
    losses.append(weight * CE_loss(logits_t, targets))
total_loss = sum(losses)
```

```bash
python ab_ud_pointer_vs_baseline.py \
  --data_source conllu --conllu_dir ud_data \
  --epochs 5 --batch_size 32 --lr 3e-5 \
  --max_inner_iters 3 --deep_supervision \
  --aux_loss_weights 0.3,0.5,1.0 \
  --log_csv deep_supervision.csv
```

**Expected:** 
- 1 iter: 97.95% UAS (baseline)
- 3 iter w/ deep sup: 98.0-98.2% UAS (+0.05-0.25%)

---

### 4. Stronger Controller Feedback (Architecture Fix)
**Goal:** Give controller richer signal for re-routing

**Modification:** Replace mean/max with attention-based summary:
```python
# Instead of: summary = heads_out.mean(dim=-1)
# Use: summary = attend(query=token_ctx, key=heads_out, value=heads_out)
```

```bash
python ab_ud_pointer_vs_baseline.py \
  --data_source conllu --conllu_dir ud_data \
  --epochs 5 --batch_size 32 --lr 3e-5 \
  --max_inner_iters 3 --controller_feedback attention \
  --log_csv attention_feedback.csv
```

**Expected:** Better feedback → routing changes across iterations → potential gain

---

### 5. Curriculum Training (Training Strategy)
**Goal:** Prevent "solve early" collapse

**Strategy:**
1. Train 3 epochs with `max_inner_iters=1` (learn basic routing)
2. Unfreeze to `max_inner_iters=3` with deep supervision
3. Train 2 more epochs

```bash
# Phase 1: Single iteration
python ab_ud_pointer_vs_baseline.py \
  --data_source conllu --conllu_dir ud_data \
  --epochs 3 --batch_size 32 --lr 3e-5 \
  --max_inner_iters 1 \
  --save_checkpoint phase1.pt

# Phase 2: Multi-iteration refinement
python ab_ud_pointer_vs_baseline.py \
  --data_source conllu --conllu_dir ud_data \
  --epochs 2 --batch_size 32 --lr 1e-5 \
  --max_inner_iters 3 --deep_supervision \
  --load_checkpoint phase1.pt \
  --log_csv curriculum.csv
```

**Expected:** Curriculum prevents early collapse, unlocks iteration benefits

---

## 📊 Diagnostic Metrics to Log

For each experiment, log:

### Distance-Bucket UAS
```python
buckets = {
    '1-2': [],    # Local (adjacent/nearby)
    '3-5': [],    # Medium-range
    '6-10': [],   # Long-range
    '>10': []     # Very long-range
}

for sent_idx, (heads_gold, heads_pred) in enumerate(zip(all_heads_gold, all_heads_pred)):
    for dep_idx, (gold, pred) in enumerate(zip(heads_gold, heads_pred)):
        distance = abs(gold - dep_idx)
        if distance <= 2:
            bucket = '1-2'
        elif distance <= 5:
            bucket = '3-5'
        elif distance <= 10:
            bucket = '6-10'
        else:
            bucket = '>10'
        
        buckets[bucket].append(1 if gold == pred else 0)

# Report UAS per bucket
for bucket, correct in buckets.items():
    uas = sum(correct) / len(correct) if correct else 0
    print(f"  Distance {bucket}: {uas:.4f} ({len(correct)} arcs)")
```

### Iteration Dynamics
```python
# Log per iteration:
- routing_entropy[t]: Entropy of routing weights at iteration t
- representation_change[t]: ||x^{t+1} - x^t||
- prediction_change[t]: KL(logits^{t+1} || logits^t)
```

If these values → 0 after iteration 1, later iterations are inert.

### Per-Pass Accuracy
```python
# With deep supervision, log:
- pass_1_uas: Accuracy of first-pass predictions
- pass_2_uas: Accuracy of second-pass predictions  
- pass_3_uas: Accuracy of final predictions

# Shows if model refines over iterations
```

---

## 🎯 Success Criteria

### Experiment 1 (Distance Buckets)
✅ If 3 iters improve long-range (>6 tokens) by +0.5-1% → iterations add reasoning
❌ If all buckets show <0.1% difference → iterations don't help any range

### Experiment 2 (Entropy Halting)
✅ Find threshold giving mean_iters=1.3 with 97.9% UAS → efficiency win
✅ Adaptive computation working as intended

### Experiment 3 (Deep Supervision)
✅ If 3 iters reach 98.0-98.2% (vs 97.95% for 1 iter) → unlocks iteration benefit
✅ Per-pass UAS increases: pass_1=97.0%, pass_2=97.7%, pass_3=98.1%

### Experiment 4 (Attention Feedback)
✅ If routing weights change significantly across iterations → feedback is active
✅ If UAS improves +0.1-0.3% → better signal helps

### Experiment 5 (Curriculum)
✅ If phase 2 reaches 98.1-98.3% → prevents early collapse
✅ Shows iterations can help with right training strategy

---

## 📝 Minimal Code Patches Needed

1. **Distance-bucket logging** (~20 lines in `epoch` function)
2. **Deep supervision flag** (~15 lines in forward pass)
3. **Per-iteration metrics** (~10 lines to log dynamics)
4. **Attention-based feedback** (~30 lines in controller)

All are **drop-in additions** - no major refactoring needed!

---

## 🚀 Quick Win: Run Entropy Halting First

**Why:** No code changes needed! Just sweep thresholds.

**Time:** ~30 minutes in Colab
**Result:** Shows if adaptive computation helps (likely yes for efficiency)

```bash
# In Colab:
for threshold in [0.6, 0.65, 0.7]:
    !python ab_ud_pointer_vs_baseline.py \
      --data_source conllu --conllu_dir ud_data \
      --epochs 3 --batch_size 32 --lr 3e-5 \
      --halting_mode entropy --max_inner_iters 3 \
      --ent_threshold {threshold} \
      --log_csv entropy_sweep.csv
```

Then analyze:
```python
import pandas as pd
df = pd.read_csv('entropy_sweep.csv')
poh = df[df['model'] == 'PoH']

for threshold in [0.6, 0.65, 0.7]:
    subset = poh[(poh['ent_threshold'] == threshold) & (poh['epoch'] == 3)]
    if not subset.empty:
        row = subset.iloc[-1]
        print(f"Threshold {threshold}: "
              f"UAS={row['dev_uas']:.4f}, "
              f"avg_iters={row['mean_inner_iters']:.2f}")
```

---

## 💡 Expected Paper Narrative (After Experiments)

### If Deep Supervision Works:
"We find that PoH's iterative refinement requires deep supervision to prevent 
early task saturation. With auxiliary losses on intermediate passes, 3 iterations 
achieve 98.1% UAS (+0.15% over single-pass), demonstrating that multi-step 
reasoning can be unlocked through appropriate training signals. Distance-bucket 
analysis reveals the gain primarily benefits long-range dependencies (>6 tokens), 
validating the multi-hop reasoning hypothesis."

### If Entropy Halting Wins:
"We demonstrate that entropy-based early stopping enables adaptive computation, 
achieving 97.9% UAS with an average of 1.3 iterations (vs 3.0 for fixed halting), 
providing a 2.3× speedup with negligible accuracy loss. This validates PoH as 
an efficient architecture that allocates computation where needed."

### If Both Work:
"Our architecture combines deep supervision for training-time refinement with 
entropy-based halting for inference-time efficiency, achieving both improved 
accuracy (+0.15%) and reduced computational cost (−30% FLOPs)."

---

**Ready to implement these patches?** I can add the code for distance buckets 
and deep supervision right now!

