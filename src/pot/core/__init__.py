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
from .swin_depth_controller import SwinDepthController, SwinDepthCache
from .mamba_controller import MambaDepthController, MambaDepthState
from .diffusion_controller import DiffusionDepthController, DiffusionDepthState
from .controller_factory import create_controller, get_controller_info, CONTROLLER_TYPES
from .feature_injection import FeatureInjector, INJECTION_MODES
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
    # Controllers - Swin (hierarchical window attention)
    "SwinDepthController",
    "SwinDepthCache",
    # Controllers - Mamba (SSM-based, O(N) complexity)
    "MambaDepthController",
    "MambaDepthState",
    # Controllers - Diffusion (iterative denoising)
    "DiffusionDepthController",
    "DiffusionDepthState",
    # Feature Injection
    "FeatureInjector",
    "INJECTION_MODES",
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

