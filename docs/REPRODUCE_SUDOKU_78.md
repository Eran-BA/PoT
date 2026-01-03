# Reproducing the 78.9% Sudoku-Extreme Result

This guide documents how to reproduce the **78.9% grid accuracy** on Sudoku-Extreme
achieved with the `HybridPoHHRMSolver` using a Transformer depth controller.

## Quick Summary

| Metric | Value |
|--------|-------|
| Grid Accuracy | 78.9% |
| Cell Accuracy | 97.8% |
| Model | HybridPoHHRMSolver |
| Controller | CausalDepthTransformerRouter |
| Parameters | 20,799,516 |
| Dataset | Sudoku-Extreme (10k puzzles) |
| HuggingFace | [Eran92/pot-sudoku-78](https://huggingface.co/Eran92/pot-sudoku-78) |

## Model Configuration

The exact configuration used in the W&B run (extracted from checkpoint):

```python
config = {
    # Architecture
    "d_model": 512,
    "n_heads": 8,
    "H_layers": 2,
    "L_layers": 2,
    "d_ff": 2048,
    "H_cycles": 2,
    "L_cycles": 6,
    "T": 4,
    "dropout": 0.039,
    "hrm_grad_style": True,
    "halt_max_steps": 2,
    
    # Controller (Transformer over depth axis)
    "controller_type": "transformer",
    "controller_kwargs": {
        "n_ctrl_layers": 2,
        "n_ctrl_heads": 4,
        "d_ctrl": 256,
        "max_depth": 32,
        "token_conditioned": True,
    },
    
    # Feature injection (disabled for this run)
    "injection_mode": "none",
    
    # Sudoku-specific
    "vocab_size": 10,
    "num_puzzles": 1,
    "puzzle_emb_dim": 512,
}
```

## Training Details

### Training Configuration

The model was trained with the following hyperparameters (from HPO):

- **Epochs**: 2001
- **Batch size**: 768
- **Learning rate**: 3.7e-4
- **Weight decay**: 0.108
- **Beta2**: 0.968
- **Warmup steps**: 2000
- **LR min ratio**: 0.1
- **Optimizer**: AdamW

### Special Features

- **Async batching**: True (HRM-style)
- **HRM grad style**: True (only last L+H get gradients)
- **Puzzle optimizer**: Separate optimizer for puzzle embeddings
  - Puzzle LR multiplier: 53.85
  - Puzzle weight decay: 1.78

### Data Augmentation

On-the-fly augmentation applied during training:
- Digit permutation (1-9 → shuffled)
- Transpose (50% probability)
- Row/column block shuffling
- Individual row/column shuffling within blocks

## Checkpoint Loading

The pre-trained checkpoint is available on **HuggingFace** (recommended) and W&B:

- **HuggingFace**: https://huggingface.co/Eran92/pot-sudoku-78
- **W&B**: `eranbt92-open-university-of-israel/sudoku-controller-transformer-finetune/sudoku-finetune-best:v34`

### Loading from HuggingFace (Recommended)

```python
import torch
from huggingface_hub import hf_hub_download
from src.pot.models.sudoku_solver import HybridPoHHRMSolver

# Download checkpoint
checkpoint_path = hf_hub_download("Eran92/pot-sudoku-78", "best_model.pt")

# Create model with exact config
model = HybridPoHHRMSolver(
    vocab_size=10,
    d_model=512,
    n_heads=8,
    H_layers=2,
    L_layers=2,
    d_ff=2048,
    dropout=0.039,
    H_cycles=2,
    L_cycles=6,
    T=4,
    num_puzzles=1,
    puzzle_emb_dim=512,
    hrm_grad_style=True,
    halt_max_steps=2,
    controller_type="transformer",
    controller_kwargs={
        "n_ctrl_layers": 2,
        "n_ctrl_heads": 4,
        "d_ctrl": 256,
        "max_depth": 32,
        "token_conditioned": True,
    },
    injection_mode="none",
)

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

### Downloading from W&B

```python
import wandb

api = wandb.Api()
artifact = api.artifact(
    "eranbt92-open-university-of-israel/sudoku-controller-transformer-finetune/sudoku-finetune-best:v34"
)
artifact_dir = artifact.download()
```

## Compatibility Notes

### New Parameters (Post W&B Run)

The following parameters were added to the codebase after the W&B run:

1. **`use_rope`** (default: `False`)
   - Enables Rotary Position Embeddings
   - The W&B checkpoint was trained with `use_rope=False`
   - Set to `False` when loading the checkpoint

2. **`use_flash_attn`** (default: `True`)
   - Runtime optimization for attention
   - Does not affect the state dict
   - Can be set to any value when loading

3. **`injection_mode`** and **`injection_kwargs`**
   - Feature injection was disabled (`"none"`) in the W&B run
   - The checkpoint has **no injector parameters**
   - Set `injection_mode="none"` when loading

### Model Structure Verification

The checkpoint contains **120 state dict keys** with **7,919,388 parameters**:

| Component | Parameters |
|-----------|------------|
| L_level | 3,946,248 |
| H_level | 3,946,248 |
| pos_embed | 20,736 |
| output_proj | 2,570 |
| input_embed | 2,560 |
| q_head | 514 |
| final_norm | 256 |
| puzzle_emb | 256 |

## Running the Compatibility Test

To verify your environment can load the checkpoint:

```bash
# Architecture-only test (no W&B login required)
python tools/test_wandb_checkpoint_compatibility.py --architecture-only

# Full test with checkpoint
wandb login
python tools/test_wandb_checkpoint_compatibility.py

# With local checkpoint
python tools/test_wandb_checkpoint_compatibility.py --checkpoint path/to/checkpoint.pt
```

## Training From Scratch

### Dataset Setup

```bash
# Download Sudoku-Extreme dataset
cd data
python scripts/download_sudoku_extreme.py
```

### Training Command

```bash
python scripts/train_sudoku_transformer.py \
    --d-model 256 \
    --n-heads 8 \
    --H-layers 2 \
    --L-layers 2 \
    --d-ff 1024 \
    --H-cycles 4 \
    --L-cycles 4 \
    --controller-type transformer \
    --epochs 100 \
    --batch-size 64 \
    --lr 3e-4 \
    --lr-min 1e-5 \
    --warmup-epochs 20 \
    --weight-decay 0.01 \
    --augment \
    --data-dir data/sudoku-extreme-10k-aug-100
```

## Expected Results

After training:
- **Grid accuracy**: 78-80% on validation set
- **Cell accuracy**: 97-98%
- **Training time**: ~4 hours on A100 GPU

## Troubleshooting

### "Missing keys" when loading checkpoint

Ensure you're using the exact configuration above. Common issues:
- Wrong `controller_type` (must be `"transformer"`)
- Wrong `injection_mode` (must be `"none"`)
- Wrong dimensions (`d_model=256`, `d_ff=1024`)

### "Unexpected keys" when loading checkpoint

The checkpoint may include optimizer state. Use:
```python
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
```

### Different results after loading

Ensure:
- `model.eval()` is called for inference
- Random seed is set for reproducibility
- Same augmentation pipeline (or no augmentation for eval)

## W&B Run Details

- **Project**: sudoku-controller-transformer-finetune
- **Artifact**: sudoku-finetune-best:v34
- **Run URL**: [W&B Dashboard](https://wandb.ai/eranbt92-open-university-of-israel/sudoku-controller-transformer-finetune)

## Citation

If you use this model or reproduce these results, please cite:

```bibtex
@misc{pot-transformer-2025,
  author = {Eran Ben Artzy},
  title = {Pointer-over-Heads Transformer: Dynamic Routing for Structured Prediction},
  year = {2025},
  url = {https://github.com/eranbt92/PoT}
}
```

