"""
PoH model implementations.

This package contains task-specific models and architectures built on top
of the core PoH modules.

Author: Eran Ben Artzy
Year: 2025
"""

from .poh_gpt import PoHGPT
from .reasoning_module import ReasoningModule
from .hybrid_hrm import HybridHRMBase
from .puzzle_embedding import PuzzleEmbedding
from .hrm_layers import RMSNorm, SwiGLU
from .adaptive_halting import QHaltingController
from .sudoku_solver import PoHSudokuSolver, HybridPoHHRMSolver, BaselineSudokuSolver

__all__ = [
    "PoHGPT",
    "ReasoningModule",
    "HybridHRMBase",
    "PuzzleEmbedding",
    "RMSNorm",
    "SwiGLU",
    "QHaltingController",
    # Sudoku solvers
    "PoHSudokuSolver",
    "HybridPoHHRMSolver",
    "BaselineSudokuSolver",
]

