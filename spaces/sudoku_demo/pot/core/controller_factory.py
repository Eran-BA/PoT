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
from .pot_transformer_controller import PoTDepthTransformerRouter
from .lstm_controllers import (
    LSTMDepthController,
    xLSTMDepthController,
    minGRUDepthController,
)
from .swin_depth_controller import SwinDepthController
from .mamba_controller import MambaDepthController
from .diffusion_controller import DiffusionDepthController


ControllerType = Literal["gru", "lstm", "xlstm", "mingru", "transformer", "pot_transformer", "swin", "mamba", "diffusion"]

CONTROLLER_TYPES = ["gru", "lstm", "xlstm", "mingru", "transformer", "pot_transformer", "swin", "mamba", "diffusion"]


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
    
    elif controller_type == "pot_transformer":
        # Nested PoT: Transformer controller with gated MHA internally
        return PoTDepthTransformerRouter(
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
    
    elif controller_type == "swin":
        # Swin-style hierarchical controller with local window attention
        window_size = kwargs.pop("window_size", 7)
        n_stages = kwargs.pop("n_stages", 3)
        stage_depths = kwargs.pop("stage_depths", None)
        stage_heads = kwargs.pop("stage_heads", None)
        
        return SwinDepthController(
            d_model=d_model,
            n_heads=n_heads,
            d_ctrl=d_ctrl,
            window_size=window_size,
            n_stages=n_stages,
            stage_depths=stage_depths,
            n_ctrl_heads=stage_heads,
            dropout=dropout,
            max_depth=max_depth,
            token_conditioned=token_conditioned,
            temperature=temperature,
            topk=topk,
            **kwargs,
        )
    
    elif controller_type == "mamba":
        # Mamba-style SSM controller with O(N) complexity
        d_state = kwargs.pop("d_state", 16)
        dt_rank = kwargs.pop("dt_rank", None)
        use_fast_path = kwargs.pop("use_fast_path", False)
        
        controller = MambaDepthController(
            d_model=d_model,
            n_heads=n_heads,
            d_ctrl=d_ctrl,
            d_state=d_state,
            dt_rank=dt_rank,
            dropout=dropout,
            token_conditioned=token_conditioned,
            temperature=temperature,
            topk=topk,
            use_fast_path=use_fast_path,
            **kwargs,
        )
        
        # Enable fast path with torch.compile if requested
        if use_fast_path:
            controller.enable_fast_path()
        
        return controller
    
    elif controller_type == "diffusion":
        # Diffusion-based controller with iterative denoising
        n_denoise_layers = kwargs.pop("n_denoise_layers", 2)
        noise_schedule = kwargs.pop("noise_schedule", "cosine")
        
        return DiffusionDepthController(
            d_model=d_model,
            n_heads=n_heads,
            d_ctrl=d_ctrl,
            n_denoise_layers=n_denoise_layers,
            noise_schedule=noise_schedule,
            max_depth=max_depth,
            dropout=dropout,
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
        "pot_transformer": {
            "name": "PoT Depth Transformer (Nested PoT)",
            "description": "Transformer with GATED MHA internally (nested PoT architecture)",
            "paper": None,
        },
        "swin": {
            "name": "Swin Depth Controller",
            "description": "Hierarchical controller with local window attention and shifting",
            "paper": "https://arxiv.org/abs/2103.14030",
        },
        "mamba": {
            "name": "Mamba Depth Controller",
            "description": "Selective SSM with O(N) linear complexity and input-dependent transitions",
            "paper": "https://arxiv.org/abs/2312.00752",
        },
        "diffusion": {
            "name": "Diffusion Depth Controller",
            "description": "Iterative denoising controller inspired by diffusion transformers",
            "paper": "https://arxiv.org/abs/2212.09748",
        },
    }
    return info.get(controller_type.lower(), {"name": "Unknown", "description": ""})
