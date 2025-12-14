"""
Core task-agnostic architecture components.
"""

from .hrm_controller import HRMPointerController, HRMState
from .depth_transformer_controller import CausalDepthTransformerRouter, DepthControllerCache
from .pot_transformer_controller import PoTDepthTransformerRouter
from .lstm_controllers import (
    LSTMDepthController,
    LSTMDepthState,
    xLSTMDepthController,
    xLSTMDepthState,
    minGRUDepthController,
)
from .controller_factory import create_controller, get_controller_info, CONTROLLER_TYPES
from .pointer_block import PointerBlock
from .losses import ranknet_loss, soft_sort_loss
from .metrics import compute_mask_aware_kendall_tau
from .sudoku_loss import sudoku_validity_check

__all__ = [
    # Controller Factory
    "create_controller",
    "get_controller_info",
    "CONTROLLER_TYPES",
    # Controllers - GRU based (default)
    "HRMPointerController",
    "HRMState",
    # Controllers - Transformer based
    "CausalDepthTransformerRouter",
    "DepthControllerCache",
    # Controllers - PoT Transformer (Nested PoT)
    "PoTDepthTransformerRouter",
    # Controllers - LSTM based
    "LSTMDepthController",
    "LSTMDepthState",
    # Controllers - xLSTM based (exponential gating)
    "xLSTMDepthController",
    "xLSTMDepthState",
    # Controllers - minGRU (simplified)
    "minGRUDepthController",
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

