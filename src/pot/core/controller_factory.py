"""
Controller Factory

Factory function for creating different depth controller types.
Provides a unified interface for switching between GRU, LSTM, xLSTM,
minGRU, and Transformer-based controllers.

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

from typing import Optional, Literal

import torch.nn as nn

from .hrm_controller import HRMPointerController
from .depth_transformer_controller import CausalDepthTransformerRouter
from .lstm_controllers import (
    LSTMDepthController,
    xLSTMDepthController,
    minGRUDepthController,
)


ControllerType = Literal["gru", "lstm", "xlstm", "mingru", "transformer"]

CONTROLLER_TYPES = ["gru", "lstm", "xlstm", "mingru", "transformer"]


def create_controller(
    controller_type: ControllerType,
    d_model: int,
    n_heads: int,
    d_ctrl: Optional[int] = None,
    dropout: float = 0.0,
    T: int = 4,
    token_conditioned: bool = True,
    temperature: float = 1.0,
    topk: Optional[int] = None,
    max_depth: int = 32,
    n_ctrl_layers: int = 2,
    n_ctrl_heads: int = 4,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create a depth controller.
    
    All controllers operate across the DEPTH axis (refinement iterations),
    not across the input sequence length. They produce routing weights α
    that determine how attention heads are combined.
    
    Args:
        controller_type: Type of controller to create
            - "gru": HRMPointerController (default, two-timescale GRU)
            - "lstm": LSTMDepthController (standard LSTM)
            - "xlstm": xLSTMDepthController (exponential gating)
            - "mingru": minGRUDepthController (simplified GRU)
            - "transformer": CausalDepthTransformerRouter (attention over depth)
        d_model: Model hidden dimension
        n_heads: Number of attention heads to route over
        d_ctrl: Controller hidden dimension (default: d_model)
        dropout: Dropout probability
        T: HRM period for GRU controller (only used for "gru")
        token_conditioned: If True, routing varies per token
        temperature: Softmax temperature for routing
        topk: Optional top-k sparsification
        max_depth: Maximum refinement steps (for transformer controller)
        n_ctrl_layers: Number of layers in transformer controller
        n_ctrl_heads: Number of heads in transformer controller
        **kwargs: Additional arguments passed to controller
        
    Returns:
        Controller module with compatible step()/forward() API
        
    Example:
        >>> controller = create_controller("xlstm", d_model=512, n_heads=8)
        >>> state = None
        >>> for t in range(16):
        ...     alpha, state, aux = controller.step(X, state=state)
    """
    controller_type = controller_type.lower()
    d_ctrl = d_ctrl or d_model
    
    if controller_type == "gru":
        # Default HRM-style two-timescale GRU controller
        return HRMPointerController(
            d_model=d_model,
            n_heads=n_heads,
            d_ctrl=d_ctrl,
            T=T,
            topk=topk,
            temperature_init=temperature,
            dropout=dropout,
            **kwargs,
        )
    
    elif controller_type == "lstm":
        return LSTMDepthController(
            d_model=d_model,
            n_heads=n_heads,
            d_ctrl=d_ctrl,
            dropout=dropout,
            token_conditioned=token_conditioned,
            temperature=temperature,
            topk=topk,
            **kwargs,
        )
    
    elif controller_type == "xlstm":
        return xLSTMDepthController(
            d_model=d_model,
            n_heads=n_heads,
            d_ctrl=d_ctrl,
            dropout=dropout,
            token_conditioned=token_conditioned,
            temperature=temperature,
            topk=topk,
            **kwargs,
        )
    
    elif controller_type == "mingru":
        return minGRUDepthController(
            d_model=d_model,
            n_heads=n_heads,
            d_ctrl=d_ctrl,
            dropout=dropout,
            token_conditioned=token_conditioned,
            temperature=temperature,
            topk=topk,
            **kwargs,
        )
    
    elif controller_type == "transformer":
        return CausalDepthTransformerRouter(
            d_model=d_model,
            n_heads=n_heads,
            d_ctrl=d_ctrl,
            n_ctrl_layers=n_ctrl_layers,
            n_ctrl_heads=n_ctrl_heads,
            dropout=dropout,
            max_depth=max_depth,
            token_conditioned=token_conditioned,
            temperature=temperature,
            topk=topk,
            **kwargs,
        )
    
    else:
        valid_types = ", ".join(CONTROLLER_TYPES)
        raise ValueError(
            f"Unknown controller_type: '{controller_type}'. "
            f"Valid options are: {valid_types}"
        )


def get_controller_info(controller_type: ControllerType) -> dict:
    """
    Get information about a controller type.
    
    Returns:
        Dict with 'name', 'description', 'paper' (if applicable)
    """
    info = {
        "gru": {
            "name": "HRM GRU Controller",
            "description": "Two-timescale GRU (f_L fast, f_H slow) from HRM paper",
            "paper": "https://arxiv.org/abs/2506.21734",
        },
        "lstm": {
            "name": "LSTM Depth Controller",
            "description": "Standard LSTM with stronger gating than GRU",
            "paper": None,
        },
        "xlstm": {
            "name": "xLSTM Depth Controller",
            "description": "Extended LSTM with exponential gating (sLSTM variant)",
            "paper": "https://arxiv.org/abs/2405.04517",
        },
        "mingru": {
            "name": "minGRU Depth Controller",
            "description": "Simplified GRU with single gate, fewer parameters",
            "paper": None,
        },
        "transformer": {
            "name": "Causal Depth Transformer",
            "description": "Transformer with causal attention over depth axis",
            "paper": None,
        },
    }
    return info.get(controller_type.lower(), {"name": "Unknown", "description": ""})
