# PoT Architecture Documentation

**Pointer-over-Heads Transformer for Dependency Parsing**

Author: Eran Ben Artzy  
License: Apache 2.0

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [Training Modes](#training-modes)
6. [Extension Points](#extension-points)

---

## Overview

The Pointer-over-Heads (PoH) Transformer is a novel architecture for dependency parsing that dynamically routes information through attention heads using a learned controller. Unlike standard transformers that use all heads equally, PoH adaptively selects which heads to attend to based on the input.

### Key Innovation

**Dynamic Head Routing**: Instead of concatenating all attention head outputs, PoH uses a pointer-style controller to compute routing weights over heads, enabling adaptive computation and head specialization.

### Architecture Diagram

```
Input Tokens
     ↓
Encoder (DistilBERT)
     ↓
Word-level Pooling
     ↓
┌─────────────────────────────────────┐
│  PoH Transformer Block              │
│  ┌───────────────────────────────┐  │
│  │  Controller                   │  │
│  │  (computes routing logits)    │  │
│  └────────────┬──────────────────┘  │
│               ↓                      │
│  ┌─────────────────────────────┐   │
│  │  Multi-Head Self-Attention  │   │
│  │  (per-head outputs)         │   │
│  └─────────────────────────────┘   │
│               ↓                      │
│  ┌─────────────────────────────┐   │
│  │  Route & Combine Heads      │   │
│  │  (weighted mixture/top-k)   │   │
│  └─────────────────────────────┘   │
│               ↓                      │
│  ┌─────────────────────────────┐   │
│  │  Feed-Forward Network       │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
     ↓
Biaffine Head Prediction
     ↓
Biaffine Label Classification
     ↓
Dependency Tree
```

---

## System Architecture

### Module Organization

```
src/
├── models/          # Model architectures
│   ├── base.py      # ParserBase (shared encoder)
│   ├── baseline.py  # Vanilla MHA baseline
│   ├── poh.py       # PoH parser (main)
│   ├── pointer_block.py  # PoH transformer block
│   └── layers.py    # Building blocks
├── data/            # Data loading
│   ├── loaders.py   # HF, CoNLL-U, dummy
│   └── collate.py   # Batching
├── training/        # Training logic
│   ├── trainer.py   # Training manager
│   └── schedulers.py  # LR schedulers
└── utils/           # Utilities
    ├── helpers.py   # Pooling, padding
    ├── metrics.py   # UAS/LAS computation
    ├── logger.py    # CSV logging
    └── iterative_losses.py  # Deep supervision
```

---

## Core Components

### 1. PointerMoHTransformerBlock

**Location**: `src/models/pointer_block.py`

The main innovation - a transformer block with dynamic head routing.

**Key Parameters**:
- `halting_mode`: How to stop iterations ('fixed', 'entropy', 'halting')
- `max_inner_iters`: Maximum inner refinement steps
- `routing_topk`: Number of heads to select (0 = soft routing)
- `combination`: How to combine heads ('mask_concat', 'mixture')
- `grad_mode`: Gradient computation ('full' BPTT, 'last' iterate)

**Algorithm**:
```
for iteration in range(max_inner_iters):
    1. Compute attention with all heads → heads_out[H]
    2. Controller sees current state + head outputs → routing_logits[H]
    3. Convert to routing weights (soft or hard top-k) → alphas[H]
    4. Combine heads: output = Σ(alphas[h] * heads_out[h])
    5. Update state: state_next = output
    6. Check halting criterion (entropy, learned, etc.)
```

### 2. PointerOverHeadsController

**Location**: `src/models/layers.py`

Computes routing logits over attention heads.

**Modes**:
- **Static**: Routes based on current state only
- **Recurrent**: Routes based on state + provisional head outputs (feedback)

**Implementation**:
```python
logits = MLP_static(LayerNorm(state))

if recurrent:
    head_summaries = Mean(provisional_heads)  # [H]
    logits += MLP_recurrent(concat([state, head_summaries]))
```

### 3. BiaffinePointer

**Location**: `src/models/layers.py`

Computes head attachment scores using biaffine attention.

**Formula**:
```
score(dep, head) = dep^T W head + U^T head
```

Returns logits [B, T, T+1] where position 0 is ROOT.

### 4. PoHParser

**Location**: `src/models/poh.py`

Full dependency parser combining:
1. Pretrained encoder (DistilBERT)
2. PoH transformer block
3. Biaffine head prediction
4. Biaffine label classification

**Forward Modes**:
- `forward()`: Standard training/inference
- `forward_trm()`: TRM-style outer supervision steps

---

## Data Flow

### Training Pipeline

```
1. Input: List of sentences with tokens, heads, labels
          ↓
2. Tokenization: Words → subwords (BERT tokenizer)
          ↓
3. Encoding: Subwords → contextual embeddings
          ↓
4. Pooling: Subwords → word-level (mean pooling)
          ↓
5. PoH Block: Word reps → refined reps (iterative)
          ↓
6. Head Prediction: Biaffine attention → head logits
          ↓
7. Label Prediction: Biaffine classification → label logits
          ↓
8. Loss: Cross-entropy (heads + labels)
          ↓
9. Backprop: Gradients → update parameters
```

### Batch Processing

```python
# Input: List of examples
examples = [
    {"tokens": ["The", "cat"], "head": [0, 1], "deprel": ["det", "root"]},
    ...
]

# Collation
subwords, word_ids, heads, labels = collate_batch(examples, tokenizer, device)

# Forward pass
loss, metrics = parser(subwords, word_ids, heads, labels)

# Metrics: {"uas": 0.85, "las": 0.82, "tokens": 50, ...}
```

---

## Training Modes

### 1. Standard Mode (Default)

Single forward pass, loss on final output.

```python
parser = PoHParser(max_inner_iters=1)
loss, metrics = parser(subw, word_ids, heads, labels)
```

### 2. Deep Supervision

Apply loss at each iteration with ramped weights.

```python
parser = PoHParser(
    max_inner_iters=3,
    deep_supervision=True,
    ramp_strength=1.0
)
```

**Loss**: `Σ w_t * CE(logits_t, targets)` where `w_t ∈ [0.3, 1.0]`

### 3. ACT-Style Halting

Learned halting with ponder cost.

```python
parser = PoHParser(
    halting_mode="halting",
    act_halting=True,
    ponder_coef=1e-3
)
```

**Loss**: `Σ p_t * CE(logits_t, targets) + λ * Σ p_t`

### 4. Combined (Recommended)

ACT + deep supervision for best results.

```python
parser = PoHParser(
    halting_mode="halting",
    deep_supervision=True,
    act_halting=True,
    ponder_coef=1e-3,
    ramp_strength=1.0
)
```

### 5. TRM Mode

Outer supervision steps with inner refinement loops.

```python
# Enable TRM mode via args
loss, metrics = parser.forward_trm(
    subw, word_ids, heads, labels,
    args=args  # Contains trm_supervision_steps, trm_inner_updates
)
```

---

## Extension Points

### Adding New Models

Inherit from `ParserBase`:

```python
from src.models.base import ParserBase

class MyParser(ParserBase):
    def __init__(self, enc_name="distilbert-base-uncased", **kwargs):
        super().__init__(enc_name, d_model=768)
        # Add your custom layers
        self.my_layer = nn.Linear(768, 768)
        self.pointer = BiaffinePointer(768)
    
    def forward(self, subw, word_ids, heads, labels):
        # Implement your forward pass
        ...
```

### Adding New Data Sources

Extend `src/data/loaders.py`:

```python
def load_my_dataset(split: str) -> List[Dict]:
    """Load from custom data source."""
    data = []
    # Load your data
    for sentence in my_data_loader:
        data.append({
            "tokens": sentence.tokens,
            "head": sentence.heads,
            "deprel": sentence.labels
        })
    return data
```

### Adding New Loss Functions

Add to `src/utils/iterative_losses.py`:

```python
def my_custom_loss(
    routed_seq: torch.Tensor,  # [B, iters, T, D]
    pointer_fn: Callable,
    targets: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """Implement your custom loss."""
    total_loss = 0.0
    for t in range(routed_seq.size(1)):
        logits_t = pointer_fn(routed_seq[:, t], ...)
        loss_t = F.cross_entropy(...)
        total_loss += loss_t
    return total_loss
```

### Adding New Halting Criteria

Modify `PointerMoHTransformerBlock.forward()`:

```python
elif self.halting_mode == "my_halting":
    # Implement your halting logic
    should_halt = my_halting_criterion(token_ctx, alphas, ...)
    if should_halt:
        break
```

---

## Performance Considerations

### Memory Usage

- **Full BPTT** (`grad_mode="full"`): Memory scales with `max_inner_iters`
- **Last-iterate** (`grad_mode="last"`): Constant memory (one iteration)
- **collect_all**: Stores all intermediate states (for deep supervision)

### Computational Cost

- **Soft routing** (`routing_topk=0`): Uses all heads (standard MHA cost)
- **Hard top-k** (`routing_topk=2`): Only computes selected heads (faster)
- **Iterations**: Cost scales linearly with `max_inner_iters`

### Optimal Settings (UD English EWT)

Based on empirical experiments:
- `max_inner_iters=1`: Best UAS (surprising result!)
- `routing_topk=0`: Soft routing outperforms hard routing
- `grad_mode="full"`: Better than last-iterate for this task
- `combination="mask_concat"`: Standard concatenation works well

---

## References

- **Deep Supervision**: "Deeply-Supervised Nets" (Lee et al., 2015)
- **ACT**: "Adaptive Computation Time" (Graves, 2016)
- **HRM**: "Hyper-Relational Networks" (Ma et al., 2021)
- **TRM**: "Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871)
- **Biaffine Attention**: "Deep Biaffine Attention" (Dozat & Manning, 2017)

---

## Summary

The PoT architecture provides:
- ✅ Dynamic head routing for adaptive computation
- ✅ Multiple halting modes for efficiency
- ✅ Deep supervision for iterative refinement
- ✅ Modular design for easy extension
- ✅ Production-ready implementation

For usage examples, see `docs/usage_guide.md`.  
For API details, see `docs/api_reference.md`.

