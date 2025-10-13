"""
Modular PoH architecture components.

Clean hierarchy:
- PoHBlock: Single transformer-style block with head-wise routing
- PoHStack: Stack of PoH blocks (like TransformerEncoder)
- IterRefiner: Wraps stack with inner-loop refinement + optional ACT halting
- PositionalEncoding: Config-switchable positional encoding (none/absolute/rotary)

Author: Eran Ben Artzy
Year: 2025
"""

from .block import (
    PoHConfig,
    PoHBlock,
    PoHStack,
    IterRefiner,
    HeadRouter,
    topk_route,
    soft_route,
)
from .positional import (
    PositionalEncoding,
    SinusoidalPositionalEncoding,
    apply_rotary_pos_emb,
)

__all__ = [
    "PoHConfig",
    "PoHBlock",
    "PoHStack",
    "IterRefiner",
    "HeadRouter",
    "topk_route",
    "soft_route",
    "PositionalEncoding",
    "SinusoidalPositionalEncoding",
    "apply_rotary_pos_emb",
]

