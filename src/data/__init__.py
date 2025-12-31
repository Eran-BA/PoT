"""
Data loading utilities.

Author: Eran Ben Artzy
Year: 2025
"""

from .sudoku import SudokuDataset, download_sudoku_dataset, shuffle_sudoku
from .arc import ARCDataset, download_arc_dataset, ARC_VOCAB_SIZE, ARC_SEQ_LEN

__all__ = [
    # Sudoku
    "SudokuDataset",
    "download_sudoku_dataset",
    "shuffle_sudoku",
    # ARC
    "ARCDataset",
    "download_arc_dataset",
    "ARC_VOCAB_SIZE",
    "ARC_SEQ_LEN",
]

