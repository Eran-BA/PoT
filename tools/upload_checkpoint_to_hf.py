#!/usr/bin/env python3
"""
Upload Sudoku checkpoint to HuggingFace Hub.

Usage:
    python tools/upload_checkpoint_to_hf.py --repo-id YOUR_USERNAME/pot-sudoku-78

Requirements:
    pip install huggingface_hub

Author: Eran Ben Artzy
Year: 2025
"""

import argparse
import json
import os
from pathlib import Path


def load_dotenv():
    """Load environment variables from .env file."""
    env_paths = [
        Path(__file__).parent.parent / ".env",
        Path.cwd() / ".env",
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            print(f"Loading environment from: {env_path}")
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and value:
                            os.environ[key] = value
            return True
    return False


# Load .env
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Upload checkpoint to HuggingFace Hub")
    parser.add_argument("--checkpoint", type=str, 
                        default="wandb_artifacts/eranbt92-open-university-of-israel_sudoku-controller-transformer-finetune_sudoku-finetune-best_v34/best_model.pt",
                        help="Path to checkpoint file")
    parser.add_argument("--repo-id", type=str, required=True,
                        help="HuggingFace repo ID (e.g., 'username/pot-sudoku-78')")
    parser.add_argument("--private", action="store_true",
                        help="Make the repo private")
    args = parser.parse_args()
    
    from huggingface_hub import HfApi, create_repo, login
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 1
    
    # Login with HF token
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        print("✓ Found HF_TOKEN in environment")
        login(token=hf_token)
    else:
        print("⚠️ No HF_TOKEN found, using cached credentials")
    
    print(f"📦 Uploading checkpoint to: {args.repo_id}")
    
    # Create repo if it doesn't exist
    api = HfApi()
    try:
        create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        print(f"✓ Repository created/verified: {args.repo_id}")
    except Exception as e:
        print(f"Note: {e}")
    
    # Model config (extracted from checkpoint)
    config = {
        "model_type": "HybridPoHHRMSolver",
        "architecture": "Pointer-over-Heads Transformer with Depth Controller",
        "task": "sudoku",
        "dataset": "sudoku-extreme-10k",
        "metrics": {
            "grid_accuracy": 0.789,
            "cell_accuracy": 0.978,
        },
        "model_config": {
            "d_model": 512,
            "n_heads": 8,
            "H_layers": 2,
            "L_layers": 2,
            "d_ff": 2048,
            "H_cycles": 2,  # From checkpoint
            "L_cycles": 6,  # From checkpoint
            "dropout": 0.039,
            "hrm_grad_style": True,
            "halt_max_steps": 2,  # Gradually increased during training
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
        },
        "training_config": {
            "epochs": 2001,
            "batch_size": 768,
            "lr": 3.7e-4,
            "async_batch": True,
            "augment": True,
        },
        "total_parameters": 20_799_516,
    }
    
    # Create README
    readme_content = f"""---
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
    repo_id="{args.repo_id}",
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
    H_cycles=2,  # From checkpoint
    L_cycles=6,  # From checkpoint
    hrm_grad_style=True,
    halt_max_steps=2,
    controller_type="transformer",
    controller_kwargs={{
        "n_ctrl_layers": 2,
        "n_ctrl_heads": 4,
        "d_ctrl": 256,
        "max_depth": 32,
        "token_conditioned": True,
    }},
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
@misc{{pot-transformer-2025,
  author = {{Eran Ben Artzy}},
  title = {{Pointer-over-Heads Transformer: Dynamic Routing for Structured Prediction}},
  year = {{2025}},
  url = {{https://github.com/eranbt92/PoT}}
}}
```
"""
    
    # Save config and README
    config_path = checkpoint_path.parent / "config.json"
    readme_path = checkpoint_path.parent / "README.md"
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    with open(readme_path, "w") as f:
        f.write(readme_content)
    
    # Upload files
    print("Uploading files...")
    
    api.upload_file(
        path_or_fileobj=str(checkpoint_path),
        path_in_repo="best_model.pt",
        repo_id=args.repo_id,
        repo_type="model",
    )
    print("  ✓ best_model.pt")
    
    api.upload_file(
        path_or_fileobj=str(config_path),
        path_in_repo="config.json",
        repo_id=args.repo_id,
        repo_type="model",
    )
    print("  ✓ config.json")
    
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
    )
    print("  ✓ README.md")
    
    print(f"\n🎉 Upload complete!")
    print(f"   View at: https://huggingface.co/{args.repo_id}")
    
    return 0


if __name__ == "__main__":
    exit(main())

