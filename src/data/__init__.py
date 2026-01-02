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
from .sokoban import (
    SokobanDataset,
    load_boxoban_levels,
    download_boxoban_dataset,
    board_to_onehot,
    onehot_to_board,
    board_to_string,
    parse_level_string,
    board_hash,
    augment_board,
    TILE_WALL, TILE_FLOOR, TILE_PLAYER, TILE_BOX, TILE_TARGET,
    TILE_BOX_ON_TARGET, TILE_PLAYER_ON_TARGET, NUM_TILE_TYPES,
    BOARD_HEIGHT, BOARD_WIDTH, NUM_ACTIONS,
)
from .sokoban_rules import (
    legal_actions,
    step,
    is_solved,
    is_deadlock,
    is_action_legal,
    get_player_pos,
    random_walk,
)
from .sokoban_hf import (
    SokobanHFDataset,
    download_sokoban_hf_dataset,
    parse_hf_board,
    augment_board_and_action,
)
from .sokoban_generator import (
    SokobanGenerator,
    SokobanGeneratedDataset,
    bfs_solve,
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
    # Sokoban
    "SokobanDataset",
    "load_boxoban_levels",
    "download_boxoban_dataset",
    "board_to_onehot",
    "onehot_to_board",
    "board_to_string",
    "parse_level_string",
    "board_hash",
    "augment_board",
    "TILE_WALL",
    "TILE_FLOOR",
    "TILE_PLAYER",
    "TILE_BOX",
    "TILE_TARGET",
    "TILE_BOX_ON_TARGET",
    "TILE_PLAYER_ON_TARGET",
    "NUM_TILE_TYPES",
    "BOARD_HEIGHT",
    "BOARD_WIDTH",
    "NUM_ACTIONS",
    # Sokoban Rules
    "legal_actions",
    "step",
    "is_solved",
    "is_deadlock",
    "is_action_legal",
    "get_player_pos",
    "random_walk",
    # Sokoban HuggingFace (Supervised)
    "SokobanHFDataset",
    "download_sokoban_hf_dataset",
    "parse_hf_board",
    "augment_board_and_action",
    # Sokoban Generator (for hard tasks)
    "SokobanGenerator",
    "SokobanGeneratedDataset",
    "bfs_solve",
]

