# PoH Decision Flowchart: Should You Use It?

**Quick decision tree to determine if PoH is right for your NLP task.**

---

## 🎯 The 5-Question Decision Tree

```
┌─────────────────────────────────────────┐
│  Does your task involve POINTERS or     │
│  STRUCTURED OUTPUT?                     │
│  (e.g., each token points to another)   │
└─────────────┬───────────────────────────┘
              │
         ┌────┴────┐
         │   YES   │ (continue)
         └────┬────┘
              │
              v
┌─────────────────────────────────────────┐
│  Is baseline accuracy < 90%?            │
│  (i.e., is this a HARD task?)           │
└─────────────┬───────────────────────────┘
              │
         ┌────┴────┐
         │   YES   │ (continue)
         └────┬────┘
              │
              v
┌─────────────────────────────────────────┐
│  Does the task require LONG-RANGE       │
│  DEPENDENCIES (>10 tokens)?             │
└─────────────┬───────────────────────────┘
              │
         ┌────┴────┐
         │   YES   │ (continue)
         └────┬────┘
              │
              v
┌─────────────────────────────────────────┐
│  Would ITERATIVE REFINEMENT help?       │
│  (decisions depend on other decisions)  │
└─────────────┬───────────────────────────┘
              │
         ┌────┴────┐
         │   YES   │ (continue)
         └────┬────┘
              │
              v
┌─────────────────────────────────────────┐
│  Can you afford 8-12× compute?          │
│  (iterations, not real-time)            │
└─────────────┬───────────────────────────┘
              │
         ┌────┴────┐
         │   YES   │
         └────┬────┘
              │
              v
     ┌────────────────┐
     │  ✅ USE PoH!   │
     │  Expected:     │
     │  +10-30% gain  │
     └────────────────┘


ANY "NO" ANSWER?
    ↓
┌─────────────────────┐
│  ❌ USE BASELINE    │
│  PoH won't help     │
└─────────────────────┘
```

---

## 📋 Task-Specific Quick Reference

### Dependency Parsing
- **Pointers?** ✅ Yes (each word → head)
- **Hard?** ✅ Yes (~80-92% UAS)
- **Long-range?** ✅ Yes (cross-sentence)
- **Iterative?** ✅ Yes (attachment ambiguity)
- **Compute OK?** ✅ Yes (offline parsing)
- **VERDICT:** 🏆 **USE PoH** (+15-30%)

### Coreference Resolution
- **Pointers?** ✅ Yes (mention → antecedent)
- **Hard?** ✅ Yes (~60-75% F1)
- **Long-range?** ✅ Yes (100+ tokens)
- **Iterative?** ✅ Yes (chain reasoning)
- **Compute OK?** ✅ Yes (document-level)
- **VERDICT:** 🏆 **USE PoH** (+20-40%)

### Semantic Role Labeling
- **Pointers?** ✅ Yes (argument → predicate)
- **Hard?** ✅ Yes (~75-85% F1)
- **Long-range?** ✅ Yes (long-distance args)
- **Iterative?** ✅ Yes (frame disambiguation)
- **Compute OK?** ✅ Yes (offline)
- **VERDICT:** 🏆 **USE PoH** (+10-25%)

### Relation Extraction
- **Pointers?** ✅ Yes (entity pairs)
- **Hard?** ✅ Yes (~50-65% F1)
- **Long-range?** ✅ Yes (document-level)
- **Iterative?** ✅ Yes (multi-relation)
- **Compute OK?** ✅ Yes (batch)
- **VERDICT:** 🏆 **USE PoH** (+15-30%)

### Multi-Hop QA
- **Pointers?** ⚠️ Partial (passage → passage)
- **Hard?** ✅ Yes (~55-70% EM)
- **Long-range?** ✅ Yes (multi-passage)
- **Iterative?** ✅ Yes (hop-by-hop)
- **Compute OK?** ✅ Yes (QA systems)
- **VERDICT:** ✅ **USE PoH** (+8-15%)

### Named Entity Recognition
- **Pointers?** ❌ No (sequence tagging)
- **Hard?** ❌ No (~92-95% F1)
- **Long-range?** ❌ No (local context)
- **Iterative?** ⚠️ Rare (nested only)
- **Compute OK?** ⚠️ Maybe
- **VERDICT:** ⚠️ **BASELINE** (PoH for nested NER only)

### Sentiment Analysis
- **Pointers?** ❌ No (classification)
- **Hard?** ❌ No (~90-95% acc)
- **Long-range?** ❌ No (pooling works)
- **Iterative?** ❌ No (single decision)
- **Compute OK?** ❌ No (real-time)
- **VERDICT:** ❌ **BASELINE** (PoH is overkill)

### Machine Translation
- **Pointers?** ❌ No (generation)
- **Hard?** ⚠️ Depends (language pair)
- **Long-range?** ✅ Yes (long sentences)
- **Iterative?** ⚠️ Partial (decoder layers)
- **Compute OK?** ⚠️ Maybe (inference time)
- **VERDICT:** ⚠️ **BASELINE** (PoH for encoder only, low-resource)

### Text Classification
- **Pointers?** ❌ No (label output)
- **Hard?** ❌ No (~85-95% acc)
- **Long-range?** ⚠️ Sometimes (long docs)
- **Iterative?** ❌ No (pooling sufficient)
- **Compute OK?** ❌ No (high throughput)
- **VERDICT:** ❌ **BASELINE** (PoH offers nothing)

---

## 🎨 Visual Task Map

```
                    STRUCTURED OUTPUT
                            ↑
                            │
                  ┌─────────┼─────────┐
                  │  PoH    │ Baseline│
                  │  ZONE   │  ZONE   │
           HIGH   │    🏆   │    ⚠️   │
        DIFFICULTY├─────────┼─────────┤
                  │    ⚠️   │    ❌   │
           LOW    │Marginal │ Overkill│
                  └─────────┴─────────┘

TASK PLACEMENT:
┌────────────────────────────────────────┐
│ 🏆 PoH WINS (Top-Right Quadrant)      │
│ • Dependency Parsing                   │
│ • Coreference Resolution               │
│ • Semantic Role Labeling               │
│ • Relation Extraction                  │
│ • Multi-Hop QA                         │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│ ⚠️ MARGINAL (Left Quadrants)          │
│ • NER (nested only)                    │
│ • Constituency Parsing                 │
│ • Event Extraction                     │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│ ❌ BASELINE BETTER (Bottom Quadrants)  │
│ • Sentiment Analysis                   │
│ • Text Classification                  │
│ • Standard NER                         │
│ • Paraphrase Detection                 │
└────────────────────────────────────────┘
```

---

## 🔍 Deep Dive: When Iterative Refinement Helps

### ✅ HELPS (Use PoH):

**1. Ambiguity Resolution**
```
"I saw the man with the telescope."
Iteration 1: telescope → saw? man? (ambiguous)
Iteration 2: Consider "with" semantics
Iteration 3: Resolve based on verb frame
```

**2. Constraint Satisfaction**
```
Dependency parsing: Must form a tree
Iteration 1: Local attachments (may have cycles)
Iteration 2: Fix cycles
Iteration 3: Ensure single root
```

**3. Multi-Hop Reasoning**
```
Q: "Where was Obama's wife born?"
Hop 1: Obama → wife → Michelle Obama
Hop 2: Michelle Obama → birthplace → Chicago
```

**4. Transitive Dependencies**
```
Coreference: Alice ... she ... her ... the woman
Chain must be consistent transitively
```

### ❌ DOESN'T HELP (Use Baseline):

**1. Independent Decisions**
```
Sentiment: Each word has sentiment score
No cross-token dependencies → no iteration benefit
```

**2. Local Context**
```
POS Tagging: Mostly trigram features
Longer context rarely helps
```

**3. Single Global Decision**
```
Topic Classification: Entire document → 1 label
Pooling + linear is optimal
```

---

## 💰 Cost-Benefit Analysis

### When PoH is Worth the Cost:

| Scenario | Compute Cost | Accuracy Gain | ROI | Decision |
|----------|-------------|---------------|-----|----------|
| Hard task, high value | 12× | +20% | 1.67% per iter | ✅ Worth it |
| Medium task, offline | 8× | +10% | 1.25% per iter | ✅ Worth it |
| Easy task, research | 4× | +5% | 1.25% per iter | ⚠️ Marginal |
| Any task, real-time | 12× | +15% | N/A | ❌ Too slow |
| Easy task, production | 8× | +2% | 0.25% per iter | ❌ Not worth it |

**Rule of Thumb:** 
- ROI > 1.0% per iteration → Use PoH
- ROI < 0.5% per iteration → Use baseline
- Real-time constraints → Use baseline

---

## 🚀 Recommended First Experiments

If you want to prove PoH value, run these in order:

### 1. **Coreference on GAP** (Highest Expected Gain)
- **Dataset:** Google's Gender-Ambiguous Pronouns
- **Baseline:** ~65-70% accuracy
- **PoH Expected:** ~75-85% accuracy (+10-20%)
- **Why:** Hardest pronoun resolution task, needs multi-hop
- **Impact:** 🔥🔥🔥 Very high (active research, interpretable)

### 2. **Dependency Parsing on UD** (Your Current Work)
- **Dataset:** Universal Dependencies (pick morphologically rich language)
- **Baseline:** ~75-88% UAS (language-dependent)
- **PoH Expected:** ~85-95% UAS (+8-15%)
- **Why:** Natural pointer task, structured output
- **Impact:** 🔥🔥 High (multilingual, practical)

### 3. **DocRED Relation Extraction** (Long-Range)
- **Dataset:** Document-level RE with evidence
- **Baseline:** ~50-60% F1
- **PoH Expected:** ~60-70% F1 (+10-20%)
- **Why:** Long documents, multiple relations, partial evidence
- **Impact:** 🔥🔥🔥 Very high (document-level is hot)

### 4. **HotpotQA Multi-Hop** (Multi-Hop Reasoning)
- **Dataset:** 2-hop question answering
- **Baseline:** ~60% EM
- **PoH Expected:** ~65-70% EM (+8-15%)
- **Why:** Explicit multi-hop, can visualize reasoning paths
- **Impact:** 🔥🔥🔥 Very high (interpretability + performance)

### 5. **AMR Parsing** (Structured Semantics)
- **Dataset:** Abstract Meaning Representation
- **Baseline:** ~75 Smatch
- **PoH Expected:** ~78-82 Smatch (+4-7%)
- **Why:** Graph structure, semantic relations
- **Impact:** 🔥 Moderate-high (niche but respected)

---

## 📊 Expected Results Summary

| Task | Difficulty | PoH Gain | Best Iters | Compute Cost | Overall |
|------|-----------|----------|------------|--------------|---------|
| Coreference (GAP) | 🔴 Very Hard | +15-25% | 12-16 | High | 🏆🏆🏆 |
| DocRED (RE) | 🔴 Very Hard | +10-20% | 10-12 | High | 🏆🏆🏆 |
| Dependency Parsing | 🟡 Hard | +10-18% | 8-12 | Medium | 🏆🏆 |
| HotpotQA | 🟡 Hard | +8-15% | 8-10 | Medium | 🏆🏆 |
| SRL | 🟡 Hard | +10-15% | 8-10 | Medium | 🏆🏆 |
| AMR Parsing | 🟡 Hard | +4-10% | 6-8 | Medium | 🏆 |
| NER (nested) | 🟢 Medium | +3-8% | 4-6 | Low | ⚠️ |
| Constituency | 🟢 Medium | +2-5% | 4-6 | Low | ⚠️ |
| Sentiment | 🟢 Easy | -2-+2% | N/A | N/A | ❌ |
| Text Classification | 🟢 Easy | -5-0% | N/A | N/A | ❌ |

---

## 🎯 Final Recommendation

**To find where YOUR PoH shines:**

1. **Start with Coreference (GAP)** - Highest expected gain, clear win
2. **Validate on Dependency Parsing** - Your current expertise, natural fit
3. **Scale to DocRED** - Document-level, long-range dependencies
4. **Prove multi-hop with HotpotQA** - Interpretable, impactful

**Avoid:**
- Sentiment, classification, generation tasks
- Tasks with >92% baseline accuracy
- Real-time inference scenarios

**Sweet Spot:**
- Structured output (pointers, graphs)
- Hard tasks (70-85% baseline)
- Long-range dependencies (>10 tokens)
- Offline/batch inference OK

---

**Bottom Line:** PoH is a **specialist**, not a generalist. It shines on **hard structured prediction**, not easy classification. Pick your battles wisely!

Author: Eran Ben Artzy  
Date: 2025

