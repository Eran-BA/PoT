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
from .stability_probes import (
    compute_stability_probes,
)
from .sokoban_supervised import (
    train_epoch as train_sokoban_epoch,
    train_epoch_async as train_sokoban_epoch_async,
    evaluate as evaluate_sokoban,
    train_supervised as train_sokoban_supervised,
    InfiniteDataLoader as SokobanInfiniteDataLoader,
    compute_solve_rate as compute_sokoban_solve_rate,
    rollout_episode as rollout_sokoban_episode,
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
    # Stability Probes
    "compute_stability_probes",
    # Sokoban PPO
    "SokobanPPOConfig",
    "SokobanPPOTrainer",
    "SokobanEnv",
    "VectorizedSokobanEnv",
    "train_sokoban_ppo",
    # Sokoban Supervised
    "train_sokoban_epoch",
    "train_sokoban_epoch_async",
    "evaluate_sokoban",
    "train_sokoban_supervised",
    "SokobanInfiniteDataLoader",
    "compute_sokoban_solve_rate",
    "rollout_sokoban_episode",
]

