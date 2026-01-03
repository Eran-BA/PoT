"""
PoH model implementations.

Simplified for HuggingFace Spaces deployment - only importing sudoku solver.
"""

from .sudoku_solver import HybridPoHHRMSolver

__all__ = [
    "HybridPoHHRMSolver",
]
