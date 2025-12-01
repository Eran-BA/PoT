"""
Sudoku Constraint Loss.

Auxiliary loss functions that encourage Sudoku constraint satisfaction:
- Each digit 1-9 appears exactly once per row
- Each digit 1-9 appears exactly once per column
- Each digit 1-9 appears exactly once per 3x3 box

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import torch
import torch.nn.functional as F


def sudoku_constraint_loss(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Compute auxiliary loss for Sudoku constraint violations.
    
    Encourages each row, column, and 3x3 box to have each digit exactly once
    by penalizing duplicate predictions via soft entropy-like loss.
    
    This loss operates on soft probabilities, making it differentiable and
    suitable for end-to-end training. It complements the cross-entropy loss
    by providing global structural guidance.
    
    Args:
        logits: [B, 81, vocab_size] - raw logits (vocab 0-9, but we focus on 1-9)
        temperature: Softmax temperature for soft constraints (default: 1.0)
        
    Returns:
        constraint_loss: Scalar loss penalizing constraint violations
        
    Example:
        >>> logits = model(input_seq, puzzle_ids)[0]  # [B, 81, 10]
        >>> c_loss = sudoku_constraint_loss(logits)
        >>> total_loss = ce_loss + 0.5 * c_loss
    """
    B = logits.size(0)
    
    # Get probabilities for digits 1-9 only (ignore 0=blank)
    probs = F.softmax(logits[:, :, 1:10] / temperature, dim=-1)  # [B, 81, 9]
    probs = probs.view(B, 9, 9, 9)  # [B, row, col, digit]
    
    constraint_loss = 0.0
    
    # Row constraint: sum of probs for each digit in each row should be ~1
    # If the model predicts the same digit in multiple columns, sum > 1
    row_sums = probs.sum(dim=2)  # [B, 9, 9] - sum over columns
    row_loss = ((row_sums - 1.0) ** 2).mean()
    constraint_loss += row_loss
    
    # Column constraint: sum of probs for each digit in each column should be ~1
    col_sums = probs.sum(dim=1)  # [B, 9, 9] - sum over rows
    col_loss = ((col_sums - 1.0) ** 2).mean()
    constraint_loss += col_loss
    
    # Box constraint: sum of probs for each digit in each 3x3 box should be ~1
    for box_row in range(3):
        for box_col in range(3):
            box = probs[:, box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3, :]  # [B, 3, 3, 9]
            box_sums = box.sum(dim=(1, 2))  # [B, 9] - sum over box cells
            box_loss = ((box_sums - 1.0) ** 2).mean()
            constraint_loss += box_loss
    
    # Normalize by number of constraint groups (9 rows + 9 cols + 9 boxes = 27)
    return constraint_loss / 27.0


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

