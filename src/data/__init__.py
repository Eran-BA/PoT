"""
Data loading utilities.

Author: Eran Ben Artzy
Year: 2025
"""

from .sudoku import SudokuDataset, download_sudoku_dataset, shuffle_sudoku
from .arc import ARCDataset, download_arc_dataset, ARC_VOCAB_SIZE, ARC_SEQ_LEN
from .blocksworld import (
    BlocksworldDataset,
    BlocksworldTrajectoryDataset,
    BlocksworldPPODataset,
    download_blocksworld_dataset,
)
from .blocksworld_bad_trajectories import (
    BadTrajectoryGenerator,
    BadTrajectoryInfo,
    is_valid_single_move,
    validate_bad_trajectory,
)
from .blocksworld_planner import (
    FastDownwardPlanner,
    TrajectoryGenerator,
    extract_sub_trajectories,
    generate_problem_pddl,
    BLOCKSWORLD_DOMAIN_PDDL,
)

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
    # Blocksworld
    "BlocksworldDataset",
    "BlocksworldTrajectoryDataset",
    "BlocksworldPPODataset",
    "download_blocksworld_dataset",
    # Blocksworld Bad Trajectories
    "BadTrajectoryGenerator",
    "BadTrajectoryInfo",
    "is_valid_single_move",
    "validate_bad_trajectory",
    # Blocksworld Planner
    "FastDownwardPlanner",
    "TrajectoryGenerator",
    "extract_sub_trajectories",
    "generate_problem_pddl",
    "BLOCKSWORLD_DOMAIN_PDDL",
]

