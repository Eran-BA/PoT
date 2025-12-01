"""
Data loading utilities.

Author: Eran Ben Artzy
Year: 2025
"""

from .sudoku import SudokuDataset, download_sudoku_dataset, shuffle_sudoku

__all__ = [
    "SudokuDataset",
    "download_sudoku_dataset",
    "shuffle_sudoku",
]

