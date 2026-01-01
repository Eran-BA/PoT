"""
Sokoban Dataset for Boxoban.

Loads Boxoban levels from DeepMind's dataset and provides:
- 7-way categorical tile encoding
- State sampling via random walks
- Geometric augmentations (flip, rotate)

Tile encoding (Sudoku-style):
- 0: WALL
- 1: FLOOR
- 2: PLAYER
- 3: BOX
- 4: TARGET
- 5: BOX_ON_TARGET
- 6: PLAYER_ON_TARGET

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import os
import re
import hashlib
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from torch.utils.data import Dataset
from tqdm import tqdm


# =============================================================================
# Tile Encoding Constants
# =============================================================================

# Tile types (7-way categorical, Sudoku-style)
TILE_WALL = 0
TILE_FLOOR = 1
TILE_PLAYER = 2
TILE_BOX = 3
TILE_TARGET = 4
TILE_BOX_ON_TARGET = 5
TILE_PLAYER_ON_TARGET = 6

NUM_TILE_TYPES = 7

# Boxoban character mapping
CHAR_TO_TILE = {
    '#': TILE_WALL,
    ' ': TILE_FLOOR,
    '@': TILE_PLAYER,
    '$': TILE_BOX,
    '.': TILE_TARGET,
    '*': TILE_BOX_ON_TARGET,
    '+': TILE_PLAYER_ON_TARGET,
}

TILE_TO_CHAR = {v: k for k, v in CHAR_TO_TILE.items()}

# Board dimensions for Boxoban medium
BOARD_HEIGHT = 10
BOARD_WIDTH = 10

# Actions: Up, Down, Left, Right
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
NUM_ACTIONS = 4

ACTION_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT']

# Direction deltas (row, col)
ACTION_DELTAS = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
}


# =============================================================================
# Board Encoding/Decoding
# =============================================================================

def parse_level_string(level_str: str) -> np.ndarray:
    """
    Parse a Boxoban level string into a board tensor.
    
    Args:
        level_str: Multi-line string representing the level
    
    Returns:
        np.ndarray of shape [10, 10] with tile indices
    """
    lines = level_str.strip().split('\n')
    
    # Initialize board with walls (safe default)
    board = np.full((BOARD_HEIGHT, BOARD_WIDTH), TILE_WALL, dtype=np.int64)
    
    for row, line in enumerate(lines):
        if row >= BOARD_HEIGHT:
            break
        for col, char in enumerate(line):
            if col >= BOARD_WIDTH:
                break
            if char in CHAR_TO_TILE:
                board[row, col] = CHAR_TO_TILE[char]
    
    return board


def board_to_string(board: np.ndarray) -> str:
    """
    Convert a board tensor back to string representation.
    
    Args:
        board: np.ndarray of shape [H, W] with tile indices
    
    Returns:
        Multi-line string representation
    """
    lines = []
    for row in range(board.shape[0]):
        line = ''.join(TILE_TO_CHAR.get(board[row, col], '?') 
                       for col in range(board.shape[1]))
        lines.append(line)
    return '\n'.join(lines)


def board_to_onehot(board: np.ndarray) -> np.ndarray:
    """
    Convert board to one-hot encoding.
    
    Args:
        board: np.ndarray of shape [H, W] with tile indices
    
    Returns:
        np.ndarray of shape [H, W, NUM_TILE_TYPES]
    """
    H, W = board.shape
    onehot = np.zeros((H, W, NUM_TILE_TYPES), dtype=np.float32)
    for t in range(NUM_TILE_TYPES):
        onehot[:, :, t] = (board == t).astype(np.float32)
    return onehot


def onehot_to_board(onehot: np.ndarray) -> np.ndarray:
    """
    Convert one-hot encoding back to board.
    
    Args:
        onehot: np.ndarray of shape [H, W, NUM_TILE_TYPES]
    
    Returns:
        np.ndarray of shape [H, W] with tile indices
    """
    return onehot.argmax(axis=-1).astype(np.int64)


def board_hash(board: np.ndarray) -> str:
    """
    Compute a hash for board deduplication.
    
    Args:
        board: np.ndarray of shape [H, W]
    
    Returns:
        Hash string
    """
    return hashlib.md5(board.tobytes()).hexdigest()


def get_board_masks(board: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract derived masks from board.
    
    Args:
        board: np.ndarray of shape [H, W]
    
    Returns:
        Dictionary with masks:
        - player_pos: (row, col) tuple
        - box_mask: boolean [H, W]
        - target_mask: boolean [H, W]
        - wall_mask: boolean [H, W]
    """
    player_mask = (board == TILE_PLAYER) | (board == TILE_PLAYER_ON_TARGET)
    player_pos = np.argwhere(player_mask)
    player_pos = tuple(player_pos[0]) if len(player_pos) > 0 else None
    
    box_mask = (board == TILE_BOX) | (board == TILE_BOX_ON_TARGET)
    target_mask = (board == TILE_TARGET) | (board == TILE_BOX_ON_TARGET) | (board == TILE_PLAYER_ON_TARGET)
    wall_mask = (board == TILE_WALL)
    
    return {
        'player_pos': player_pos,
        'box_mask': box_mask,
        'target_mask': target_mask,
        'wall_mask': wall_mask,
    }


# =============================================================================
# Dataset Loading
# =============================================================================

def load_boxoban_levels(
    data_dir: str,
    difficulty: str = 'medium',
    split: str = 'train',
) -> List[np.ndarray]:
    """
    Load Boxoban levels from the dataset.
    
    Args:
        data_dir: Path to boxoban-levels directory
        difficulty: 'medium', 'hard', or 'unfiltered'
        split: 'train', 'valid', or 'test'
    
    Returns:
        List of board tensors [H, W]
    """
    data_path = Path(data_dir)
    
    # Check for boxoban-levels subdirectory
    if (data_path / 'boxoban-levels').exists():
        data_path = data_path / 'boxoban-levels'
    
    split_dir = data_path / difficulty / split
    
    if not split_dir.exists():
        # Try alternative naming
        if split == 'val':
            split_dir = data_path / difficulty / 'valid'
        elif split == 'valid':
            split_dir = data_path / difficulty / 'val'
    
    if not split_dir.exists():
        raise FileNotFoundError(f"Could not find split directory: {split_dir}")
    
    levels = []
    
    # Find all .txt files
    txt_files = sorted(split_dir.glob('*.txt'))
    
    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            content = f.read()
        
        # Parse levels (separated by "; N" lines)
        level_blocks = re.split(r'^;\s*\d+\s*$', content, flags=re.MULTILINE)
        
        for block in level_blocks:
            block = block.strip()
            if not block:
                continue
            
            # Check it looks like a valid level
            lines = block.split('\n')
            if len(lines) >= BOARD_HEIGHT and all('#' in line or ' ' in line for line in lines[:BOARD_HEIGHT]):
                board = parse_level_string(block)
                levels.append(board)
    
    return levels


def download_boxoban_dataset(data_dir: str) -> None:
    """
    Download Boxoban dataset from GitHub if not present.
    
    Args:
        data_dir: Directory to store the dataset
    """
    data_path = Path(data_dir)
    boxoban_path = data_path / 'boxoban-levels'
    
    if boxoban_path.exists():
        print(f"Boxoban dataset already exists at {boxoban_path}")
        return
    
    print("Downloading Boxoban dataset from GitHub...")
    data_path.mkdir(parents=True, exist_ok=True)
    
    import subprocess
    result = subprocess.run(
        ['git', 'clone', 'https://github.com/deepmind/boxoban-levels.git'],
        cwd=str(data_path),
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to clone Boxoban: {result.stderr}")
    
    print(f"Downloaded to {boxoban_path}")


# =============================================================================
# Geometric Augmentations
# =============================================================================

def flip_horizontal(board: np.ndarray) -> np.ndarray:
    """Flip board horizontally (left-right)."""
    return np.flip(board, axis=1).copy()


def flip_vertical(board: np.ndarray) -> np.ndarray:
    """Flip board vertically (up-down)."""
    return np.flip(board, axis=0).copy()


def rotate_180(board: np.ndarray) -> np.ndarray:
    """Rotate board 180 degrees."""
    return np.rot90(board, k=2).copy()


def transform_action(action: int, transform: str) -> int:
    """
    Transform action according to board transformation.
    
    Args:
        action: Original action (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        transform: 'flip_h', 'flip_v', 'rot180', or 'identity'
    
    Returns:
        Transformed action
    """
    if transform == 'identity':
        return action
    elif transform == 'flip_h':
        # Left <-> Right
        if action == ACTION_LEFT:
            return ACTION_RIGHT
        elif action == ACTION_RIGHT:
            return ACTION_LEFT
        return action
    elif transform == 'flip_v':
        # Up <-> Down
        if action == ACTION_UP:
            return ACTION_DOWN
        elif action == ACTION_DOWN:
            return ACTION_UP
        return action
    elif transform == 'rot180':
        # All directions flip
        if action == ACTION_UP:
            return ACTION_DOWN
        elif action == ACTION_DOWN:
            return ACTION_UP
        elif action == ACTION_LEFT:
            return ACTION_RIGHT
        elif action == ACTION_RIGHT:
            return ACTION_LEFT
    return action


def augment_board(
    board: np.ndarray,
    action: Optional[int] = None,
) -> List[Tuple[np.ndarray, Optional[int], str]]:
    """
    Apply all geometric augmentations to a board.
    
    Args:
        board: Original board [H, W]
        action: Optional action label to transform
    
    Returns:
        List of (augmented_board, transformed_action, transform_name) tuples
    """
    augmentations = []
    
    # Identity
    augmentations.append((
        board.copy(),
        action,
        'identity'
    ))
    
    # Horizontal flip
    augmentations.append((
        flip_horizontal(board),
        transform_action(action, 'flip_h') if action is not None else None,
        'flip_h'
    ))
    
    # Vertical flip
    augmentations.append((
        flip_vertical(board),
        transform_action(action, 'flip_v') if action is not None else None,
        'flip_v'
    ))
    
    # 180 rotation
    augmentations.append((
        rotate_180(board),
        transform_action(action, 'rot180') if action is not None else None,
        'rot180'
    ))
    
    return augmentations


# =============================================================================
# Sokoban Dataset
# =============================================================================

class SokobanDataset(Dataset):
    """
    Dataset for Sokoban levels.
    
    Provides boards as tensors with optional one-hot encoding.
    
    Args:
        data_dir: Path to data directory (containing boxoban-levels)
        difficulty: 'medium', 'hard', or 'unfiltered'
        split: 'train', 'valid', or 'test'
        use_onehot: If True, return one-hot encoding
        augment: If True, apply geometric augmentations
    """
    
    def __init__(
        self,
        data_dir: str,
        difficulty: str = 'medium',
        split: str = 'train',
        use_onehot: bool = True,
        augment: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.difficulty = difficulty
        self.split = split
        self.use_onehot = use_onehot
        self.augment = augment
        
        # Load levels
        self.levels = load_boxoban_levels(data_dir, difficulty, split)
        print(f"[{split}] Loaded {len(self.levels)} {difficulty} levels")
        
        # Apply augmentations if requested
        if augment:
            augmented_levels = []
            for level in self.levels:
                for aug_level, _, _ in augment_board(level):
                    augmented_levels.append(aug_level)
            self.levels = augmented_levels
            print(f"[{split}] After augmentation: {len(self.levels)} levels")
    
    def __len__(self) -> int:
        return len(self.levels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        board = self.levels[idx]
        
        if self.use_onehot:
            board_tensor = torch.tensor(board_to_onehot(board), dtype=torch.float32)
        else:
            board_tensor = torch.tensor(board, dtype=torch.long)
        
        # Get masks
        masks = get_board_masks(board)
        
        return {
            'board': board_tensor,
            'board_indices': torch.tensor(board, dtype=torch.long),
            'player_pos': torch.tensor(masks['player_pos'] if masks['player_pos'] else [0, 0], dtype=torch.long),
            'box_mask': torch.tensor(masks['box_mask'], dtype=torch.bool),
            'target_mask': torch.tensor(masks['target_mask'], dtype=torch.bool),
        }


def get_dataset_stats(data_dir: str, difficulty: str = 'medium') -> Dict[str, int]:
    """
    Get statistics about the Boxoban dataset.
    
    Args:
        data_dir: Path to data directory
        difficulty: Difficulty level
    
    Returns:
        Dictionary with counts per split
    """
    stats = {}
    
    for split in ['train', 'valid', 'test']:
        try:
            levels = load_boxoban_levels(data_dir, difficulty, split)
            stats[split] = len(levels)
        except FileNotFoundError:
            stats[split] = 0
    
    return stats

