"""
Training utilities.

Author: Eran Ben Artzy
Year: 2025
"""

from .sudoku_trainer import (
    train_epoch,
    train_epoch_async,
    evaluate,
    debug_gradients,
    debug_activations,
    debug_predictions,
    run_debug,
)

__all__ = [
    "train_epoch",
    "train_epoch_async",
    "evaluate",
    "debug_gradients",
    "debug_activations",
    "debug_predictions",
    "run_debug",
]

