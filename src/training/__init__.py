"""
Training utilities.

Author: Eran Ben Artzy
Year: 2025
"""

from .sudoku_trainer import (
    train_epoch,
    train_epoch_async,
    evaluate,
    log_halt_histogram_to_wandb,
    debug_gradients,
    debug_activations,
    debug_predictions,
    run_debug,
)
from .ppo_blocksworld import (
    BlocksworldPPOTrainer,
    PPOConfig as BlocksworldPPOConfig,
    RolloutBuffer as BlocksworldRolloutBuffer,
)
from .sokoban_ppo import (
    PPOConfig as SokobanPPOConfig,
    SokobanPPOTrainer,
    SokobanEnv,
    VectorizedSokobanEnv,
    train_ppo as train_sokoban_ppo,
)

__all__ = [
    "train_epoch",
    "train_epoch_async",
    "evaluate",
    "log_halt_histogram_to_wandb",
    "debug_gradients",
    "debug_activations",
    "debug_predictions",
    "run_debug",
    # PPO Blocksworld
    "BlocksworldPPOTrainer",
    "BlocksworldPPOConfig",
    "BlocksworldRolloutBuffer",
    # Sokoban PPO
    "SokobanPPOConfig",
    "SokobanPPOTrainer",
    "SokobanEnv",
    "VectorizedSokobanEnv",
    "train_sokoban_ppo",
]

