"""
Core task-agnostic architecture components.
"""

from .hrm_controller import HRMPointerController, HRMState
from .pointer_block import PointerBlock
from .losses import ranknet_loss, soft_sort_loss
from .metrics import compute_mask_aware_kendall_tau
from .sudoku_loss import sudoku_constraint_loss, sudoku_validity_check

__all__ = [
    "HRMPointerController",
    "HRMState",
    "PointerBlock",
    "ranknet_loss",
    "soft_sort_loss",
    "compute_mask_aware_kendall_tau",
    # Sudoku
    "sudoku_constraint_loss",
    "sudoku_validity_check",
]

