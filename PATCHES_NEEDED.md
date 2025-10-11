# 🔧 Minimal Patches to Enable Diagnostic Experiments

**Goal:** Add 3 diagnostic features with minimal code changes (~50 lines total)

---

## Patch 1: Expose `ent_threshold` as CLI argument (5 lines)

### File: `ab_ud_pointer_vs_baseline.py`

**Location:** In `main()`, around line 552 (after `--routing_topk` argument)

```python
# Add this CLI argument
ap.add_argument("--ent_threshold", type=float, default=0.8,
                help="Entropy threshold for early stopping (halting_mode=entropy)")
```

**Location:** Around line 620 (where `PoHParser` is created)

```python
# Change from:
poh = PoHParser(d_model=d_model, n_heads=args.heads, d_ff=d_ff_poh,
                halting_mode=args.halting_mode, max_inner_iters=args.max_inner_iters,
                routing_topk=args.routing_topk, combination=args.combination,
                n_labels=n_labels, use_labels=use_labels)

# To:
poh = PoHParser(d_model=d_model, n_heads=args.heads, d_ff=d_ff_poh,
                halting_mode=args.halting_mode, max_inner_iters=args.max_inner_iters,
                routing_topk=args.routing_topk, combination=args.combination,
                ent_threshold=args.ent_threshold,  # ← ADD THIS
                n_labels=n_labels, use_labels=use_labels)
```

**Location:** `PoHParser.__init__()`, around line 286

```python
# Change signature from:
def __init__(self, enc_name="distilbert-base-uncased", d_model=768, n_heads=8, d_ff=2048,
             halting_mode="entropy", max_inner_iters=3, routing_topk=2, combination="mask_concat",
             n_labels=50, use_labels=True):

# To:
def __init__(self, enc_name="distilbert-base-uncased", d_model=768, n_heads=8, d_ff=2048,
             halting_mode="entropy", max_inner_iters=3, routing_topk=2, combination="mask_concat",
             ent_threshold=0.8,  # ← ADD THIS
             n_labels=50, use_labels=True):
```

**Location:** Inside `PoHParser.__init__()`, around line 293

```python
# Change from:
self.block = PointerMoHTransformerBlock(
    d_model=d_model, n_heads=n_heads, d_ff=d_ff,
    halting_mode=halting_mode, max_inner_iters=max_inner_iters,
    min_inner_iters=1, ent_threshold=0.8,  # ← hardcoded!
    routing_topk=routing_topk, combination=combination,
    controller_recurrent=True, controller_summary="mean")

# To:
self.block = PointerMoHTransformerBlock(
    d_model=d_model, n_heads=n_heads, d_ff=d_ff,
    halting_mode=halting_mode, max_inner_iters=max_inner_iters,
    min_inner_iters=1, ent_threshold=ent_threshold,  # ← use parameter!
    routing_topk=routing_topk, combination=combination,
    controller_recurrent=True, controller_summary="mean")
```

**✅ Result:** Can now sweep entropy thresholds from command line!

```bash
for t in 0.6 0.65 0.7 0.75 0.8; do
    python ab_ud_pointer_vs_baseline.py \
      --halting_mode entropy --ent_threshold $t \
      --log_csv entropy_sweep.csv
done
```

---

## Patch 2: Add Distance-Bucket Logging (15 lines)

### File: `ab_ud_pointer_vs_baseline.py`

**Location:** In `main()`, add CLI argument around line 540

```python
ap.add_argument("--log_distance_buckets", action="store_true",
                help="Log UAS breakdown by dependency distance")
```

**Location:** In `epoch()` function, after eval metrics are computed (around line 470)

```python
# After computing total_correct_uas, add:

if not train and log_distance_buckets:
    # Collect all predictions
    all_heads_gold = []
    all_heads_pred = []
    
    # Re-run eval to collect predictions (or save them earlier in the loop)
    # For now, add this inside the batch loop above:
    # Inside: for batch_idx in range(0, len(data), bs):
    #   After computing metrics, add:
    #   batch_heads_gold = heads_gold  # Already have this
    #   batch_heads_pred = pred_heads[pad].cpu().tolist()  # Get predictions
    #   all_heads_gold.extend(batch_heads_gold)
    #   all_heads_pred.extend(batch_heads_pred)
    
    # After loop:
    from utils.diagnostics import log_distance_buckets
    log_distance_buckets(all_heads_gold, all_heads_pred, prefix="  ")
```

**Better approach:** Pass `log_distance_buckets` flag to `epoch()` and collect predictions:

```python
# In epoch() signature:
def epoch(model, data, tokenizer, device, label_vocab=None, bs=8, train=True, 
          lr=5e-5, weight_decay=0.01, scheduler=None, 
          emit_conllu=False, conllu_path=None, ignore_punct=False,
          log_distance_buckets=False):  # ← ADD THIS
    # ... existing code ...
    
    # Inside eval loop, after computing metrics:
    if not train and log_distance_buckets:
        if not hasattr(epoch, 'all_heads_gold'):
            epoch.all_heads_gold = []
            epoch.all_heads_pred = []
        
        # Collect this batch
        for b in range(len(heads_gold)):
            epoch.all_heads_gold.append(heads_gold[b])
            epoch.all_heads_pred.append(pred_heads[b, :len(heads_gold[b])].cpu().tolist())
    
    # After loop, log distance buckets
    if not train and log_distance_buckets and hasattr(epoch, 'all_heads_gold'):
        from utils.diagnostics import log_distance_buckets
        log_distance_buckets(epoch.all_heads_gold, epoch.all_heads_pred, prefix="  ")
        # Clear for next epoch
        epoch.all_heads_gold = []
        epoch.all_heads_pred = []
```

**In main():** Pass the flag to `epoch()`:

```python
# Around line 650, when calling epoch():
tr_b = epoch(baseline, train, tokenizer, device, label_vocab, 
             args.batch_size, True, args.lr, args.weight_decay, 
             log_distance_buckets=args.log_distance_buckets)  # ← ADD
dv_b = epoch(baseline, dev, tokenizer, device, label_vocab,
             args.batch_size, False,
             log_distance_buckets=args.log_distance_buckets)  # ← ADD
```

**✅ Result:** See which distance ranges benefit from iterations!

```bash
python ab_ud_pointer_vs_baseline.py \
  --max_inner_iters 3 --log_distance_buckets
```

---

## Patch 3: CSV Logging for Entropy Threshold (5 lines)

### File: `ab_ud_pointer_vs_baseline.py`

**Location:** Around line 700, where CSV logging happens

```python
# In the CSV logging section, add ent_threshold:
from utils.logger import append_row, flatten_cfg

if args.log_csv:
    row = flatten_cfg(
        seed=args.seed,
        # ... existing fields ...
        ent_threshold=args.ent_threshold,  # ← ADD THIS
        # ... rest of fields ...
    )
    append_row(args.log_csv, row)
```

**✅ Result:** Can analyze which threshold works best!

---

## Summary of Changes

| Feature | Lines | Difficulty | Impact |
|---------|-------|------------|--------|
| Entropy threshold CLI | 5 | Easy | High - enables key experiment |
| Distance buckets | 15 | Medium | High - validates hypothesis |
| CSV logging | 5 | Easy | Medium - better analysis |
| **Total** | **25** | **Easy-Medium** | **High** |

---

## Quick Test After Patching

```bash
# Test 1: Entropy sweep (should now work)
for t in 0.6 0.65 0.7; do
  python ab_ud_pointer_vs_baseline.py \
    --data_source conllu --conllu_dir ud_data \
    --epochs 2 --batch_size 32 --lr 3e-5 \
    --halting_mode entropy --max_inner_iters 3 \
    --ent_threshold $t --log_csv entropy_test.csv
done

# Analyze results
python -c "
import pandas as pd
df = pd.read_csv('entropy_test.csv')
poh = df[df['model'] == 'PoH']
for t in [0.6, 0.65, 0.7]:
    subset = poh[(poh['ent_threshold'] == t) & (poh['epoch'] == 2)]
    if not subset.empty:
        row = subset.iloc[-1]
        print(f'Threshold {t}: UAS={row[\"dev_uas\"]:.4f}, avg_iters={row.get(\"mean_inner_iters\", \"N/A\")}')
"

# Test 2: Distance buckets (should now log)
python ab_ud_pointer_vs_baseline.py \
  --data_source conllu --conllu_dir ud_data \
  --epochs 1 --batch_size 32 \
  --max_inner_iters 3 --log_distance_buckets
```

---

## Optional: Deep Supervision (More Complex)

Requires architecture changes to `PoHParser.forward()` to collect intermediate logits.
See `utils/diagnostics.py::compute_deep_supervision_loss()` for the helper function.

**Complexity:** Medium-High (30-40 lines, architectural change)  
**Benefit:** Potentially unlocks multi-iteration gains

**Defer this until entropy + distance experiments show promise!**

---

## 🚀 Recommended Order

1. **Patch 1 first** (entropy threshold) - enables immediate experiment
2. Run entropy sweep in Colab (~10 minutes)
3. If you find good threshold, **add Patch 2** (distance buckets)
4. Run distance analysis to validate hypothesis
5. If distances show promise, consider deep supervision

---

**All patches are minimal, non-breaking, and can be added incrementally!**

