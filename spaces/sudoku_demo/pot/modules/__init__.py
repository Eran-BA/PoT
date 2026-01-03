"""
Modular PoH architecture components.

Simplified for HuggingFace Spaces deployment - only importing what's needed.
"""

# Only import what's actually used by the sudoku solver
from .rope import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
    CosSin,
)

__all__ = [
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
    "rotate_half",
    "CosSin",
]
