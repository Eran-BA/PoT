#!/usr/bin/env python3
"""
Load pre-trained Sudoku model from HuggingFace Hub.

Example:
    python tools/load_from_hf.py --repo-id eranbt92/pot-sudoku-78 --solve "530070000600195000098000060800060003400803001700020006060000280000419005000080079"

Author: Eran Ben Artzy
Year: 2025
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_sudoku_model(repo_id: str = "Eran92/pot-sudoku-78", device: str = "cpu"):
    """
    Load pre-trained Sudoku solver from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repo ID
        device: Device to load model on
        
    Returns:
        model: Loaded HybridPoHHRMSolver in eval mode
    """
    import torch
    from huggingface_hub import hf_hub_download
    from src.pot.models.sudoku_solver import HybridPoHHRMSolver
    
    # Download checkpoint
    print(f"Downloading from: {repo_id}")
    checkpoint_path = hf_hub_download(
        repo_id=repo_id,
        filename="best_model.pt"
    )
    
    # Create model with exact config from checkpoint
    model = HybridPoHHRMSolver(
        d_model=512,
        n_heads=8,
        H_layers=2,
        L_layers=2,
        d_ff=2048,
        H_cycles=2,  # From checkpoint
        L_cycles=6,  # From checkpoint
        dropout=0.039,
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
        vocab_size=10,
        num_puzzles=1,
        puzzle_emb_dim=512,
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")
    return model


def solve_puzzle(model, puzzle_str: str, device: str = "cpu"):
    """
    Solve a Sudoku puzzle.
    
    Args:
        model: Loaded model
        puzzle_str: 81-character string (0=blank, 1-9=given)
        device: Device
        
    Returns:
        solution: 9x9 numpy array
    """
    import torch
    
    # Parse puzzle string
    puzzle = torch.tensor([int(c) for c in puzzle_str], dtype=torch.long).unsqueeze(0)
    puzzle_ids = torch.zeros(1, dtype=torch.long)
    
    puzzle = puzzle.to(device)
    puzzle_ids = puzzle_ids.to(device)
    
    # Solve
    with torch.no_grad():
        logits = model(puzzle, puzzle_ids)[0]
        solution = logits.argmax(dim=-1)
    
    return solution.cpu().numpy().reshape(9, 9)


def print_grid(grid, title=""):
    """Pretty-print a Sudoku grid."""
    if title:
        print(f"\n{title}")
    print("+" + "-"*7 + "+" + "-"*7 + "+" + "-"*7 + "+")
    for i in range(9):
        row = ""
        for j in range(9):
            if j % 3 == 0:
                row += "| "
            row += str(grid[i, j]) + " "
        row += "|"
        print(row)
        if i % 3 == 2:
            print("+" + "-"*7 + "+" + "-"*7 + "+" + "-"*7 + "+")


def main():
    parser = argparse.ArgumentParser(description="Load and use pre-trained Sudoku solver")
    parser.add_argument("--repo-id", type=str, default="Eran92/pot-sudoku-78",
                        help="HuggingFace repo ID")
    parser.add_argument("--solve", type=str, default=None,
                        help="81-char puzzle string to solve (0=blank)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu/cuda)")
    args = parser.parse_args()
    
    import numpy as np
    
    # Load model
    model = load_sudoku_model(args.repo_id, args.device)
    
    # Solve puzzle if provided
    if args.solve:
        if len(args.solve) != 81:
            print(f"❌ Puzzle must be 81 characters, got {len(args.solve)}")
            return 1
        
        puzzle = np.array([int(c) for c in args.solve]).reshape(9, 9)
        print_grid(puzzle, "Input Puzzle:")
        
        solution = solve_puzzle(model, args.solve, args.device)
        print_grid(solution, "Solution:")
        
        # Check if valid
        blanks = (puzzle == 0).sum()
        matches = ((puzzle == 0) | (puzzle == solution)).all()
        print(f"\nFilled {blanks} blanks")
        print(f"Preserves given digits: {'✓' if matches else '❌'}")
    else:
        # Demo puzzle
        demo = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        print("No puzzle provided. Using demo puzzle:")
        
        puzzle = np.array([int(c) for c in demo]).reshape(9, 9)
        print_grid(puzzle, "Demo Puzzle:")
        
        solution = solve_puzzle(model, demo, args.device)
        print_grid(solution, "Solution:")
    
    return 0


if __name__ == "__main__":
    exit(main())

