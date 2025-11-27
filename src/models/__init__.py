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

# Only import layers that don't require transformers at module load time
# This allows src.pot.modules to import MultiHeadSelfAttention without
# triggering the transformers dependency
from src.models.layers import (
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
        from src.models.base import ParserBase
        return ParserBase
    if name == "BaselineParser":
        from src.models.baseline import BaselineParser
        return BaselineParser
    if name == "VanillaBlock":
        from src.models.baseline import VanillaBlock
        return VanillaBlock
    if name == "PoHParser":
        from src.models.poh import PoHParser
        return PoHParser
    if name == "PointerMoHTransformerBlock":
        from src.models.pointer_block import PointerMoHTransformerBlock
        return PointerMoHTransformerBlock
    if name == "PoHGPT":
        from src.pot.models.poh_gpt import PoHGPT
        return PoHGPT
    if name == "BaselineGPT":
        from src.models.baseline_gpt import BaselineGPT
        return BaselineGPT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
