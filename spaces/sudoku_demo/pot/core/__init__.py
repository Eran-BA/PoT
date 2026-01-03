"""
Core task-agnostic architecture components.

Simplified for HuggingFace Spaces deployment.
"""

from .hrm_controller import HRMPointerController, HRMState
from .depth_transformer_controller import CausalDepthTransformerRouter, DepthControllerCache
from .lstm_controllers import LSTMDepthState, xLSTMDepthState
from .controller_factory import create_controller, CONTROLLER_TYPES
from .feature_injection import FeatureInjector, INJECTION_MODES

__all__ = [
    "create_controller",
    "CONTROLLER_TYPES",
    "HRMPointerController",
    "HRMState",
    "CausalDepthTransformerRouter",
    "DepthControllerCache",
    "LSTMDepthState",
    "xLSTMDepthState",
    "FeatureInjector",
    "INJECTION_MODES",
]
