"""
PoH model implementations.

This package contains task-specific models and architectures built on top
of the core PoH modules.

Author: Eran Ben Artzy
Year: 2025
"""

from .poh_gpt import PoHGPT
from .reasoning_module import ReasoningModule
from .hybrid_hrm import HybridHRMBase, ACTCarry, PoTAsyncCarry
from .puzzle_embedding import PuzzleEmbedding
from .hrm_layers import RMSNorm, SwiGLU
from .adaptive_halting import QHaltingController
from .sudoku_solver import PoHSudokuSolver, HybridPoHHRMSolver, BaselineSudokuSolver
from .hybrid_nli import HybridPoHHRMForNLI
from .arc_solver import HybridPoHARCSolver, BaselineARCSolver

__all__ = [
    "PoHGPT",
    "ReasoningModule",
    "HybridHRMBase",
    "ACTCarry",
    "PoTAsyncCarry",
    "PuzzleEmbedding",
    "RMSNorm",
    "SwiGLU",
    "QHaltingController",
    # Sudoku solvers
    "PoHSudokuSolver",
    "HybridPoHHRMSolver",
    "BaselineSudokuSolver",
    # NLI
    "HybridPoHHRMForNLI",
    # ARC solvers
    "HybridPoHARCSolver",
    "BaselineARCSolver",
]

