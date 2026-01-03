---
license: apache-2.0
tags:
- sudoku
- transformer
- pointer-over-heads
- reasoning
library_name: pot-transformer
---

# PoT Sudoku Solver - 78.9% Grid Accuracy

This is a pre-trained **HybridPoHHRMSolver** model for solving Sudoku puzzles,
achieving **78.9% grid accuracy** on the Sudoku-Extreme benchmark.

## Model Details

| Metric | Value |
|--------|-------|
| Grid Accuracy | 78.9% |
| Cell Accuracy | 97.8% |
| Parameters | 20.8M |
| Architecture | HybridPoHHRMSolver |
| Controller | CausalDepthTransformerRouter |

## Usage

```python
import torch
from huggingface_hub import hf_hub_download

# Download checkpoint
checkpoint_path = hf_hub_download(
    repo_id="Eran92/pot-sudoku-78",
    filename="best_model.pt"
)

# Load model
from src.pot.models.sudoku_solver import HybridPoHHRMSolver

model = HybridPoHHRMSolver(
    d_model=512,
    n_heads=8,
    H_layers=2,
    L_layers=2,
    d_ff=2048,
    H_cycles=4,
    L_cycles=4,
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

checkpoint = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Solve a puzzle (0=blank, 1-9=given digits)
puzzle = torch.tensor([[
    5, 3, 0, 0, 7, 0, 0, 0, 0,
    6, 0, 0, 1, 9, 5, 0, 0, 0,
    0, 9, 8, 0, 0, 0, 0, 6, 0,
    8, 0, 0, 0, 6, 0, 0, 0, 3,
    4, 0, 0, 8, 0, 3, 0, 0, 1,
    7, 0, 0, 0, 2, 0, 0, 0, 6,
    0, 6, 0, 0, 0, 0, 2, 8, 0,
    0, 0, 0, 4, 1, 9, 0, 0, 5,
    0, 0, 0, 0, 8, 0, 0, 7, 9,
]])
puzzle_ids = torch.zeros(1, dtype=torch.long)

with torch.no_grad():
    logits = model(puzzle, puzzle_ids)[0]
    solution = logits.argmax(dim=-1)
    print(solution.reshape(9, 9))
```

## Training

The model was trained on Sudoku-Extreme with on-the-fly augmentation:
- Digit permutation
- Transpose
- Row/column band shuffling

## Citation

```bibtex
@misc{pot-transformer-2025,
  author = {Eran Ben Artzy},
  title = {Pointer-over-Heads Transformer: Dynamic Routing for Structured Prediction},
  year = {2025},
  url = {https://github.com/eranbt92/PoT}
}
```
