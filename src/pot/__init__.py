"""
PoT: Pointer-over-Heads Transformer
Dynamic Routing Architecture for Structured NLP

A task-agnostic framework for iterative refinement via learned attention routing.
"""

__version__ = "0.2.1"
__author__ = "Eran Ben Artzy"

from .core.hrm_controller import HRMPointerController, HRMState
from .core.pointer_block import PointerBlock
from .core.losses import ranknet_loss, soft_sort_loss
from .core.metrics import compute_mask_aware_kendall_tau

__all__ = [
    "HRMPointerController",
    "HRMState",
    "PointerBlock",
    "ranknet_loss",
    "soft_sort_loss",
    "compute_mask_aware_kendall_tau",
]

