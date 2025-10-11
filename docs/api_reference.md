# PoT API Reference

**Complete API documentation for all public classes and functions**

Author: Eran Ben Artzy  
License: Apache 2.0

---

## Table of Contents

1. [Models](#models)
2. [Training](#training)
3. [Data](#data)
4. [Utilities](#utilities)

---

## Models

### ParserBase

**Location**: `src.models.base`

Base class for all dependency parsers.

```python
class ParserBase(nn.Module):
    def __init__(self, enc_name: str, d_model: int)
```

**Parameters**:
- `enc_name`: HuggingFace model identifier (e.g., "distilbert-base-uncased")
- `d_model`: Model hidden dimension

**Attributes**:
- `encoder`: Pretrained transformer encoder
- `d_model`: Hidden dimension size

---

### PoHParser

**Location**: `src.models.poh`

Main Pointer-over-Heads dependency parser.

```python
class PoHParser(ParserBase):
    def __init__(
        self,
        enc_name: str = "distilbert-base-uncased",
        d_model: int = 768,
        n_heads: int = 8,
        d_ff: int = 2048,
        halting_mode: str = "fixed",
        max_inner_iters: int = 3,
        routing_topk: int = 2,
        combination: str = "mask_concat",
        ent_threshold: float = 0.8,
        n_labels: int = 50,
        use_labels: bool = True,
        deep_supervision: bool = False,
        act_halting: bool = False,
        ponder_coef: float = 1e-3,
        ramp_strength: float = 1.0,
        grad_mode: str = "full"
    )
```

**Parameters**:
- `enc_name`: Encoder model name
- `d_model`: Hidden dimension
- `n_heads`: Number of attention heads
- `d_ff`: Feed-forward network dimension
- `halting_mode`: Stopping criterion ("fixed", "entropy", "halting")
- `max_inner_iters`: Maximum inner iterations
- `routing_topk`: Number of heads to select (0 = soft routing)
- `combination`: Head combination mode ("mask_concat", "mixture")
- `ent_threshold`: Entropy threshold for early stopping
- `n_labels`: Number of dependency relation labels
- `use_labels`: Whether to predict labels (LAS) or just heads (UAS)
- `deep_supervision`: Enable deep supervision mode
- `act_halting`: Enable ACT-style differentiable halting
- `ponder_coef`: Coefficient for ACT ponder cost
- `ramp_strength`: Strength of deep supervision ramp (0-1)
- `grad_mode`: Gradient mode ("full" or "last")

**Methods**:

#### forward()
```python
def forward(
    self,
    subw: Dict[str, torch.Tensor],
    word_ids: List[List[Optional[int]]],
    heads_gold: List[List[int]],
    labels_gold: Optional[List[List[int]]] = None
) -> Tuple[torch.Tensor, Dict[str, float]]
```

Standard forward pass for training/evaluation.

**Returns**: `(loss, metrics)` where metrics include:
- `uas`: Unlabeled Attachment Score
- `las`: Labeled Attachment Score
- `tokens`: Number of tokens
- `inner_iters_used`: Iterations executed
- (eval only) `pred_heads`: Predicted heads
- (eval only) `pred_labels`: Predicted labels

#### forward_trm()
```python
def forward_trm(
    self,
    subw: Dict[str, torch.Tensor],
    word_ids: List[List[Optional[int]]],
    heads_gold: List[List[int]],
    labels_gold: Optional[List[List[int]]] = None,
    args = None
) -> Tuple[torch.Tensor, Dict[str, float]]
```

TRM-style forward with outer supervision steps.

**Args Parameter Fields**:
- `trm_supervision_steps`: Number of outer steps
- `trm_inner_updates`: Inner updates per step
- `trm_ramp_strength`: Deep supervision ramp

---

### BaselineParser

**Location**: `src.models.baseline`

Vanilla MHA baseline parser for comparison.

```python
class BaselineParser(ParserBase):
    def __init__(
        self,
        enc_name: str = "distilbert-base-uncased",
        d_model: int = 768,
        n_heads: int = 8,
        d_ff: int = 2048,
        n_labels: int = 50,
        use_labels: bool = True
    )
```

**Methods**:

#### forward()
```python
def forward(
    self,
    subw: Dict[str, torch.Tensor],
    word_ids: List[List[Optional[int]]],
    heads_gold: List[List[int]],
    labels_gold: Optional[List[List[int]]] = None
) -> Tuple[torch.Tensor, Dict[str, float]]
```

Same interface as PoHParser.forward().

---

### PointerMoHTransformerBlock

**Location**: `src.models.pointer_block`

Core PoH transformer block with adaptive routing.

```python
class PointerMoHTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        use_pre_norm: bool = True,
        routing_tau: float = 0.7,
        routing_topk: int = 0,
        controller_recurrent: bool = True,
        controller_summary: str = "mean",
        combination: str = "mask_concat",
        halting_mode: str = "fixed",
        max_inner_iters: int = 2,
        min_inner_iters: int = 1,
        ent_threshold: float = 0.7,
        ponder_coef: float = 0.001,
        grad_mode: str = "full"
    )
```

**Methods**:

#### forward()
```python
def forward(
    self,
    x: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    return_aux: bool = True,
    collect_all: bool = False,
    return_final_z: bool = False
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]
```

**Parameters**:
- `x`: Input [B, T, D]
- `attn_mask`: Optional mask [B, H, T, T]
- `return_aux`: Return auxiliary outputs
- `collect_all`: Collect all iteration states (for deep supervision)
- `return_final_z`: Return final latent (for TRM mode)

**Returns**: `(output, aux)` where aux includes:
- `alphas`: Routing weights [B, iters, T, H]
- `logits`: Routing logits [B, iters, T, H]
- `attn_probs`: Attention probabilities [B, H, T, T]
- `inner_iters_used`: Number of iterations
- (if collect_all) `routed`: Per-iteration states [B, iters, T, D]
- (if collect_all & halting) `halt_logits`: Halting logits
- (if return_final_z) `z_final`: Final latent [B, T, D]

---

## Training

### Trainer

**Location**: `src.training.trainer`

Training manager with automatic optimizer setup and metric tracking.

```python
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        device: torch.device,
        label_vocab: Optional[Dict[str, int]] = None
    )
```

**Methods**:

#### train_epoch()
```python
def train_epoch(
    self,
    data: List[Dict],
    batch_size: int = 32,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    scheduler: Optional[Any] = None,
    args: Optional[Any] = None
) -> Dict[str, float]
```

Run one training epoch.

**Returns**: Dictionary with:
- `loss`: Average training loss
- `mean_inner_iters`: Average iterations (PoH only)
- `time`: Elapsed time in seconds
- `tokens`: Total tokens processed

#### eval_epoch()
```python
def eval_epoch(
    self,
    data: List[Dict],
    batch_size: int = 32,
    emit_conllu: bool = False,
    conllu_path: Optional[str] = None,
    ignore_punct: bool = False,
    args: Optional[Any] = None
) -> Dict[str, float]
```

Run evaluation epoch.

**Returns**: Dictionary with:
- `loss`: Average loss
- `uas`: Unlabeled Attachment Score
- `las`: Labeled Attachment Score
- `mean_inner_iters`: Average iterations
- `time`: Elapsed time
- `tokens`: Total tokens

---

### Schedulers

**Location**: `src.training.schedulers`

#### get_linear_schedule_with_warmup()
```python
def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int
) -> LambdaLR
```

Create LR scheduler with linear warmup then linear decay.

**Example**:
```python
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=1000
)

for batch in dataloader:
    loss.backward()
    optimizer.step()
    scheduler.step()  # Update LR after each step
```

---

## Data

### Loaders

**Location**: `src.data.loaders`

#### get_dataset()
```python
def get_dataset(
    source: str,
    split: str,
    conllu_dir: Optional[str] = None
) -> List[Dict]
```

Unified interface for loading data.

**Parameters**:
- `source`: "hf", "conllu", or "dummy"
- `split`: "train", "validation", or "test"
- `conllu_dir`: Directory for CoNLL-U files (if source="conllu")

**Returns**: List of examples with keys:
- `tokens`: List of word strings
- `head`: List of head indices (0 = ROOT)
- `deprel`: List of dependency relation strings

#### load_hf_dataset()
```python
def load_hf_dataset(split: str) -> Optional[List[Dict]]
```

Load UD English EWT from HuggingFace.

#### load_conllu_files()
```python
def load_conllu_files(path: str) -> List[Dict]
```

Load from local CoNLL-U files.

#### create_dummy_dataset()
```python
def create_dummy_dataset(n_samples: int = 64) -> List[Dict]
```

Generate synthetic data for testing.

#### build_label_vocab()
```python
def build_label_vocab(data: List[Dict]) -> Dict[str, int]
```

Build label vocabulary from data.

**Returns**: Dict mapping label strings to indices.

---

### Collation

**Location**: `src.data.collate`

#### collate_batch()
```python
def collate_batch(
    examples: Union[List[Dict], Dict],
    tokenizer: AutoTokenizer,
    device: torch.device,
    label_vocab: Optional[Dict[str, int]] = None,
    return_deprels: bool = False
) -> Union[
    Tuple[Dict, List, List, List],
    Tuple[Dict, List, List, List, List]
]
```

Collate examples into batched tensors.

**Returns**:
- `enc`: Tokenized inputs (dict with 'input_ids', etc.)
- `word_ids`: Word indices for each subword
- `heads`: Gold heads
- `labels`: Gold labels
- (if return_deprels) `deprels_str`: Original string labels

---

## Utilities

### Helpers

**Location**: `src.utils.helpers`

#### mean_pool_subwords()
```python
def mean_pool_subwords(
    last_hidden: torch.Tensor,
    word_ids: List[List[int]]
) -> List[torch.Tensor]
```

Pool subword tokens to word-level representations.

**Parameters**:
- `last_hidden`: Subword embeddings [B, S, D]
- `word_ids`: Word indices for each subword

**Returns**: List of word tensors [num_words, D]

#### pad_words()
```python
def pad_words(
    word_batches: List[torch.Tensor],
    pad_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]
```

Pad variable-length word sequences.

**Returns**: `(padded_tensor, mask)`
- `padded_tensor`: [B, max_len, D]
- `mask`: [B, max_len] bool

#### make_targets()
```python
def make_targets(
    heads: List[List[int]],
    max_len: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]
```

Create padded target tensors.

**Returns**: `(targets, mask)`
- `targets`: [B, max_len] head indices
- `mask`: [B, max_len] bool

---

### Metrics

**Location**: `src.utils.metrics`

#### compute_uas_las()
```python
def compute_uas_las(
    pred_heads: torch.Tensor,
    gold_heads: torch.Tensor,
    pred_labels: Optional[torch.Tensor],
    gold_labels: Optional[torch.Tensor],
    mask: torch.Tensor,
    deprels: Optional[List[List[str]]] = None,
    ignore_punct: bool = False
) -> Tuple[float, float]
```

Compute UAS and LAS with optional punctuation masking.

**Returns**: `(uas, las)`

#### build_masks_for_metrics()
```python
def build_masks_for_metrics(
    heads: List[List[int]],
    deprels: Optional[List[List[str]]] = None
) -> Tuple[torch.Tensor, torch.Tensor]
```

Build masks for metric computation.

**Returns**: `(is_token, is_eval)` both [B, T] bool

---

### Logger

**Location**: `src.utils.logger`

#### append_row()
```python
def append_row(csv_path: str, row: Dict[str, Any]) -> None
```

Append row to CSV file with timestamp.

**Example**:
```python
append_row("results.csv", {
    "epoch": 1,
    "uas": 0.85,
    "las": 0.82
})
```

---

### CoNLL-U Writer

**Location**: `src.utils.conllu_writer`

#### write_conllu()
```python
def write_conllu(
    path: str,
    tokens: List[List[str]],
    heads_gold: Optional[List[List[int]]] = None,
    deprels_gold: Optional[List[List[str]]] = None,
    heads_pred: Optional[List[List[int]]] = None,
    deprels_pred: Optional[List[List[str]]] = None
) -> None
```

Write predictions to CoNLL-U format for official evaluation.

---

### Iterative Losses

**Location**: `src.utils.iterative_losses`

#### deep_supervision_loss()
```python
def deep_supervision_loss(
    routed_seq: torch.Tensor,  # [B, iters, T, D]
    pointer_fn: Callable,
    targets: torch.Tensor,
    mask: torch.Tensor,
    weight_schedule: str = "linear"
) -> torch.Tensor
```

Apply weighted loss at each iteration.

#### act_expected_loss()
```python
def act_expected_loss(
    routed_seq: torch.Tensor,
    halt_logits: torch.Tensor,
    pointer_fn: Callable,
    targets: torch.Tensor,
    mask: torch.Tensor,
    ponder_coef: float = 1e-3,
    per_token: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]
```

ACT-style expected loss with ponder cost.

**Returns**: `(loss, ponder)`

#### act_deep_supervision_loss()
```python
def act_deep_supervision_loss(
    routed_seq: torch.Tensor,
    halt_logits: torch.Tensor,
    pointer_fn: Callable,
    targets: torch.Tensor,
    mask: torch.Tensor,
    ponder_coef: float = 1e-3,
    ramp_strength: float = 1.0
) -> Tuple[torch.Tensor, Dict]
```

Combined ACT + deep supervision (recommended).

**Returns**: `(loss, diagnostics)`

---

## Type Aliases

Common types used throughout the API:

```python
# Data
Example = Dict[str, Any]  # {"tokens": [...], "head": [...], "deprel": [...]}
Dataset = List[Example]
LabelVocab = Dict[str, int]

# Tensors
Tensor3D = torch.Tensor  # [B, T, D]
Tensor4D = torch.Tensor  # [B, I, T, D]
Mask = torch.Tensor  # [B, T] bool

# Metrics
Metrics = Dict[str, float]  # {"uas": 0.85, "las": 0.82, ...}
```

---

## Constants

**Model Defaults**:
- `d_model`: 768 (DistilBERT dimension)
- `n_heads`: 8
- `d_ff`: 2048
- `max_inner_iters`: 1 (optimal for UD EWT)
- `routing_topk`: 0 (soft routing)

**Training Defaults**:
- `batch_size`: 32
- `lr`: 3e-5
- `weight_decay`: 0.01
- `warmup_ratio`: 0.1

---

## Usage Examples

### Basic Training
```python
from src.models import PoHParser
from src.training import Trainer
from src.data.loaders import get_dataset, build_label_vocab
from transformers import AutoTokenizer
import torch

# Setup
device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Data
train_data = get_dataset("conllu", "train", "data/")
dev_data = get_dataset("conllu", "validation", "data/")
label_vocab = build_label_vocab(train_data + dev_data)

# Model
parser = PoHParser(n_labels=len(label_vocab)).to(device)

# Train
trainer = Trainer(parser, tokenizer, device, label_vocab)
for epoch in range(5):
    train_metrics = trainer.train_epoch(train_data, batch_size=32)
    dev_metrics = trainer.eval_epoch(dev_data, batch_size=32)
    print(f"Epoch {epoch+1}: UAS={dev_metrics['uas']:.3f}")
```

---

## See Also

- **Architecture**: `docs/architecture.md`
- **Usage Guide**: `docs/usage_guide.md`
- **Theory**: `docs/GRADIENT_MODES_THEORY.md`, `docs/DEEP_SUPERVISION_GUIDE.md`

