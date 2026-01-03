"""
Sudoku Validity Check.

Utility function for checking Sudoku constraint satisfaction:
- Each digit 1-9 appears exactly once per row
- Each digit 1-9 appears exactly once per column
- Each digit 1-9 appears exactly once per 3x3 box

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import torch


def sudoku_validity_check(predictions: torch.Tensor) -> torch.Tensor:
    """
    Check if Sudoku predictions are valid (satisfy all constraints).
    
    This is a hard check on discrete predictions, not differentiable.
    Use for evaluation, not training.
    
    Args:
        predictions: [B, 81] - predicted digits (1-9)
        
    Returns:
        valid: [B] - boolean tensor, True if grid is valid
    """
    B = predictions.size(0)
    grids = predictions.view(B, 9, 9)
    valid = torch.ones(B, dtype=torch.bool, device=predictions.device)
    
    for b in range(B):
        grid = grids[b]
        
        # Check rows
        for r in range(9):
            row = grid[r]
            if row.unique().numel() != 9 or (row < 1).any() or (row > 9).any():
                valid[b] = False
                break
        
        if not valid[b]:
            continue
            
        # Check columns
        for c in range(9):
            col = grid[:, c]
            if col.unique().numel() != 9:
                valid[b] = False
                break
        
        if not valid[b]:
            continue
            
        # Check 3x3 boxes
        for br in range(3):
            for bc in range(3):
                box = grid[br*3:(br+1)*3, bc*3:(bc+1)*3].flatten()
                if box.unique().numel() != 9:
                    valid[b] = False
                    break
            if not valid[b]:
                break
    
    return valid

