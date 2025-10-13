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

from src.models.base import ParserBase
from src.models.baseline import BaselineParser, VanillaBlock
from src.models.poh import PoHParser
from src.models.pointer_block import PointerMoHTransformerBlock
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
from src.pot.models.poh_gpt import PoHGPT
from src.models.baseline_gpt import BaselineGPT

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
