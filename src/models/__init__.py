"""
PoT dependency parsing models.

Exports:
    ParserBase: Base parser class
    BaselineParser: Vanilla MHA baseline parser
    PoHParser: Pointer-over-Heads parser with adaptive routing
    PointerMoHTransformerBlock: Main PoH transformer block

    BiaffinePointer: Biaffine head prediction layer
    BiaffineLabeler: Biaffine label classification layer

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

# Only import layers that don't require transformers at module load time.
# This keeps `models.layers` usable when `pot` is installed as a dependency.
from .layers import (
    BiaffinePointer,
    BiaffineLabeler,
    MultiHeadSelfAttention,
    PointerOverHeadsController,
    HRMPointerController,
    HRMState,
    entropy_from_logits,
    gumbel_softmax_topk,
)

__all__ = [
    "ParserBase",
    "BaselineParser",
    "VanillaBlock",
    "PoHParser",
    "PointerMoHTransformerBlock",
    "BiaffinePointer",
    "BiaffineLabeler",
    "MultiHeadSelfAttention",
    "PointerOverHeadsController",
    "HRMPointerController",
    "HRMState",
    "entropy_from_logits",
    "gumbel_softmax_topk",
    "PoHGPT",
    "BaselineGPT",
]

def __getattr__(name):
    """Lazy import models that require transformers to avoid mandatory dependency."""
    if name == "ParserBase":
        from .base import ParserBase
        return ParserBase
    if name == "BaselineParser":
        from .baseline import BaselineParser
        return BaselineParser
    if name == "VanillaBlock":
        from .baseline import VanillaBlock
        return VanillaBlock
    if name == "PoHParser":
        from .poh import PoHParser
        return PoHParser
    if name == "PointerMoHTransformerBlock":
        from .pointer_block import PointerMoHTransformerBlock
        return PointerMoHTransformerBlock
    if name == "PoHGPT":
        from pot.models.poh_gpt import PoHGPT
        return PoHGPT
    if name == "BaselineGPT":
        from .baseline_gpt import BaselineGPT
        return BaselineGPT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
