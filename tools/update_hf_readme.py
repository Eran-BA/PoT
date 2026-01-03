#!/usr/bin/env python3
"""Update README on HuggingFace Hub."""

from huggingface_hub import HfApi
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()

README = '''# PoT Sudoku 78.9% - Pre-trained Checkpoint

This repository contains a pre-trained `HybridPoHHRMSolver` model achieving **78.9% grid accuracy** on Sudoku-Extreme.

## Model Details

| Metric | Value |
|--------|-------|
| Grid Accuracy | 78.9% |
| Cell Accuracy | 97.8% |
| Parameters | 20,799,516 |
| Architecture | HybridPoHHRMSolver |
| Controller | CausalDepthTransformerRouter |

## Configuration

```python
config = {
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
    "controller_type": "transformer",
    "controller_kwargs": {
        "n_ctrl_layers": 2,
        "n_ctrl_heads": 4,
        "d_ctrl": 256,
        "max_depth": 32,
        "token_conditioned": True,
    },
    "injection_mode": "none",
    "vocab_size": 10,
    "num_puzzles": 1,
    "puzzle_emb_dim": 512,
}
```

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from src.pot.models.sudoku_solver import HybridPoHHRMSolver

# Download checkpoint
checkpoint_path = hf_hub_download("Eran92/pot-sudoku-78", "best_model.pt")

# Create model with exact config
model = HybridPoHHRMSolver(
    d_model=512,
    n_heads=8,
    H_layers=2,
    L_layers=2,
    d_ff=2048,
    H_cycles=2,
    L_cycles=6,
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

# Load weights
checkpoint = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Solve a puzzle
puzzle = torch.tensor([[5, 3, 0, 0, 7, 0, 0, 0, 0,
                        6, 0, 0, 1, 9, 5, 0, 0, 0,
                        0, 9, 8, 0, 0, 0, 0, 6, 0,
                        8, 0, 0, 0, 6, 0, 0, 0, 3,
                        4, 0, 0, 8, 0, 3, 0, 0, 1,
                        7, 0, 0, 0, 2, 0, 0, 0, 6,
                        0, 6, 0, 0, 0, 0, 2, 8, 0,
                        0, 0, 0, 4, 1, 9, 0, 0, 5,
                        0, 0, 0, 0, 8, 0, 0, 7, 9]])
solution = model.solve(puzzle)
print(solution.reshape(9, 9))
```

## Training

Trained on Sudoku-Extreme (10k puzzles) with:
- Batch size: 768
- Learning rate: 3.7e-4
- Epochs: 2001
- On-the-fly augmentation (digit permutation, transpose, shuffling)

## Links

- [PoT Repository](https://github.com/rnbnrzy/PoT)
- [W&B Run](https://wandb.ai/eranbt92-open-university-of-israel/sudoku-controller-transformer-finetune)
'''

api = HfApi(token=os.getenv('HF_TOKEN'))

# Save README to temp file and upload
with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
    f.write(README)
    readme_path = f.name

api.upload_file(
    path_or_fileobj=readme_path,
    path_in_repo='README.md',
    repo_id='Eran92/pot-sudoku-78',
    repo_type='model',
)

os.unlink(readme_path)
print('README updated on HuggingFace!')

