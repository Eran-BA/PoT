"""
Core task-agnostic architecture components.
"""

from .hrm_controller import HRMPointerController, HRMState
from .depth_transformer_controller import CausalDepthTransformerRouter, DepthControllerCache
from .pointer_block import PointerBlock
from .losses import ranknet_loss, soft_sort_loss
from .metrics import compute_mask_aware_kendall_tau
from .sudoku_loss import sudoku_validity_check

__all__ = [
    # Controllers
    "HRMPointerController",
    "HRMState",
    "CausalDepthTransformerRouter",
    "DepthControllerCache",
    # Blocks
    "PointerBlock",
    # Losses
    "ranknet_loss",
    "soft_sort_loss",
    # Metrics
    "compute_mask_aware_kendall_tau",
    # Sudoku
    "sudoku_validity_check",
]

