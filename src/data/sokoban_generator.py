"""
Sokoban Test Data Generator.

Generates Sokoban boards at different difficulty levels using gym-sokoban.
Each board comes with the BFS-solved optimal next action.

Difficulty Levels (from paper):
- SimpleSokoban: 6x6, 1 box (easy)
- LargerSokoban: 10x10, 1 box (medium)
- TwoBoxesSokoban: 6x6, 2 boxes (hard)
- ComplexSokoban: 10x10, 2 boxes (hardest)

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import Dataset
from collections import deque
import hashlib

try:
    from gym_sokoban.envs import SokobanEnv
    HAS_GYM_SOKOBAN = True
except ImportError:
    HAS_GYM_SOKOBAN = False
    print("Warning: gym-sokoban not installed. Run: pip install gym-sokoban")


# =============================================================================
# Constants
# =============================================================================

# Tile types (matching our format)
TILE_WALL = 0
TILE_FLOOR = 1
TILE_PLAYER = 2
TILE_BOX = 3
TILE_TARGET = 4
TILE_BOX_ON_TARGET = 5
TILE_PLAYER_ON_TARGET = 6
NUM_TILE_TYPES = 7

# gym-sokoban internal encoding
GYM_WALL = 0
GYM_FLOOR = 1
GYM_TARGET = 2
GYM_BOX_ON_TARGET = 3
GYM_BOX = 4
GYM_PLAYER = 5
GYM_PLAYER_ON_TARGET = 6

# Actions (0-indexed)
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
NUM_ACTIONS = 4

# gym-sokoban uses 1-indexed actions
GYM_ACTION_MAP = {
    1: ACTION_UP,
    2: ACTION_DOWN,
    3: ACTION_LEFT,
    4: ACTION_RIGHT,
}


# =============================================================================
# Conversion Functions
# =============================================================================

def gym_to_our_format(gym_board: np.ndarray) -> np.ndarray:
    """Convert gym-sokoban board to our format."""
    mapping = {
        GYM_WALL: TILE_WALL,
        GYM_FLOOR: TILE_FLOOR,
        GYM_TARGET: TILE_TARGET,
        GYM_BOX_ON_TARGET: TILE_BOX_ON_TARGET,
        GYM_BOX: TILE_BOX,
        GYM_PLAYER: TILE_PLAYER,
        GYM_PLAYER_ON_TARGET: TILE_PLAYER_ON_TARGET,
    }
    
    result = np.zeros_like(gym_board)
    for gym_val, our_val in mapping.items():
        result[gym_board == gym_val] = our_val
    
    return result


def board_to_onehot(board: np.ndarray) -> np.ndarray:
    """Convert board to one-hot encoding."""
    H, W = board.shape
    onehot = np.zeros((H, W, NUM_TILE_TYPES), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            onehot[i, j, board[i, j]] = 1.0
    return onehot


def board_hash(board: np.ndarray) -> str:
    """Get unique hash for a board state."""
    return hashlib.md5(board.tobytes()).hexdigest()


# =============================================================================
# BFS Solver
# =============================================================================

def get_player_pos(board: np.ndarray) -> Tuple[int, int]:
    """Get player position."""
    pos = np.argwhere((board == TILE_PLAYER) | (board == TILE_PLAYER_ON_TARGET))
    if len(pos) == 0:
        return None
    return tuple(pos[0])


def is_solved(board: np.ndarray) -> bool:
    """Check if puzzle is solved (all boxes on targets)."""
    # No standalone boxes should remain
    return not np.any(board == TILE_BOX)


def apply_action(board: np.ndarray, action: int) -> Tuple[np.ndarray, bool]:
    """Apply action and return new board state."""
    DELTAS = {
        ACTION_UP: (-1, 0),
        ACTION_DOWN: (1, 0),
        ACTION_LEFT: (0, -1),
        ACTION_RIGHT: (0, 1),
    }
    
    player_pos = get_player_pos(board)
    if player_pos is None:
        return board.copy(), False
    
    pr, pc = player_pos
    dr, dc = DELTAS[action]
    nr, nc = pr + dr, pc + dc
    
    H, W = board.shape
    if nr < 0 or nr >= H or nc < 0 or nc >= W:
        return board.copy(), False
    
    new_board = board.copy()
    next_tile = board[nr, nc]
    
    # Can't walk into wall
    if next_tile == TILE_WALL:
        return board.copy(), False
    
    # Push box?
    if next_tile in (TILE_BOX, TILE_BOX_ON_TARGET):
        # Check beyond
        br, bc = nr + dr, nc + dc
        if br < 0 or br >= H or bc < 0 or bc >= W:
            return board.copy(), False
        
        beyond = board[br, bc]
        if beyond not in (TILE_FLOOR, TILE_TARGET):
            return board.copy(), False
        
        # Move box
        if beyond == TILE_TARGET:
            new_board[br, bc] = TILE_BOX_ON_TARGET
        else:
            new_board[br, bc] = TILE_BOX
        
        # Clear box from destination
        if next_tile == TILE_BOX_ON_TARGET:
            new_board[nr, nc] = TILE_TARGET
        else:
            new_board[nr, nc] = TILE_FLOOR
    
    # Move player
    current = new_board[pr, pc]
    if current == TILE_PLAYER_ON_TARGET:
        new_board[pr, pc] = TILE_TARGET
    else:
        new_board[pr, pc] = TILE_FLOOR
    
    dest = new_board[nr, nc]
    if dest == TILE_TARGET:
        new_board[nr, nc] = TILE_PLAYER_ON_TARGET
    else:
        new_board[nr, nc] = TILE_PLAYER
    
    return new_board, True


def bfs_solve(board: np.ndarray, max_steps: int = 100) -> Optional[List[int]]:
    """
    BFS solver to find optimal action sequence.
    
    Returns:
        List of actions to solve, or None if unsolvable within max_steps
    """
    if is_solved(board):
        return []
    
    start_hash = board_hash(board)
    visited = {start_hash}
    queue = deque([(board, [])])
    
    while queue:
        current, path = queue.popleft()
        
        if len(path) >= max_steps:
            continue
        
        for action in range(NUM_ACTIONS):
            new_board, valid = apply_action(current, action)
            if not valid:
                continue
            
            h = board_hash(new_board)
            if h in visited:
                continue
            
            visited.add(h)
            new_path = path + [action]
            
            if is_solved(new_board):
                return new_path
            
            queue.append((new_board, new_path))
    
    return None  # Unsolvable within max_steps


# =============================================================================
# Board Generator
# =============================================================================

class SokobanGenerator:
    """
    Generate Sokoban boards at various difficulty levels.
    """
    
    DIFFICULTY_CONFIGS = {
        'simple': {'dim_room': (6, 6), 'num_boxes': 1, 'max_steps': 30},
        'larger': {'dim_room': (10, 10), 'num_boxes': 1, 'max_steps': 50},
        'two_boxes': {'dim_room': (6, 6), 'num_boxes': 2, 'max_steps': 50},
        'complex': {'dim_room': (10, 10), 'num_boxes': 2, 'max_steps': 100},
    }
    
    def __init__(self, difficulty: str = 'simple', seed: int = 42):
        if not HAS_GYM_SOKOBAN:
            raise ImportError("gym-sokoban not installed. Run: pip install gym-sokoban")
        
        self.difficulty = difficulty
        self.config = self.DIFFICULTY_CONFIGS[difficulty]
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Create gym environment
        self.env = SokobanEnv(
            dim_room=self.config['dim_room'],
            num_boxes=self.config['num_boxes'],
            max_steps=self.config['max_steps'],
        )
    
    def generate_board_with_action(self) -> Optional[Dict[str, Any]]:
        """
        Generate a board and its optimal first action.
        
        Returns:
            Dict with 'board', 'action', 'solution_length' or None if failed
        """
        # Reset environment with random seed
        seed = int(self.rng.integers(0, 2**31))
        self.env.reset()
        
        # Get room state (gym-sokoban internal representation)
        gym_board = self.env.room_state.copy()
        
        # Convert to our format
        board = gym_to_our_format(gym_board)
        
        # Solve with BFS
        solution = bfs_solve(board, max_steps=self.config['max_steps'])
        
        if solution is None or len(solution) == 0:
            return None
        
        return {
            'board': board,
            'action': solution[0],  # First action (0-indexed)
            'solution_length': len(solution),
        }
    
    def generate_dataset(self, n_samples: int, verbose: bool = True) -> List[Dict]:
        """Generate n_samples of (board, action) pairs."""
        samples = []
        attempts = 0
        max_attempts = n_samples * 10
        
        while len(samples) < n_samples and attempts < max_attempts:
            attempts += 1
            result = self.generate_board_with_action()
            
            if result is not None:
                samples.append(result)
                if verbose and len(samples) % 100 == 0:
                    print(f"Generated {len(samples)}/{n_samples} samples")
        
        if verbose:
            print(f"Generated {len(samples)} samples in {attempts} attempts")
        
        return samples


# =============================================================================
# Dataset Class
# =============================================================================

class SokobanGeneratedDataset(Dataset):
    """
    Dataset of generated Sokoban boards at specific difficulty.
    """
    
    def __init__(
        self,
        difficulty: str = 'simple',
        n_samples: int = 1000,
        seed: int = 42,
        augment: bool = False,
    ):
        self.difficulty = difficulty
        self.augment = augment
        
        print(f"Generating {n_samples} {difficulty} Sokoban boards...")
        generator = SokobanGenerator(difficulty=difficulty, seed=seed)
        self.samples = generator.generate_dataset(n_samples)
        
        print(f"[{difficulty}] Generated {len(self.samples)} samples")
        if self.samples:
            print(f"  Board shape: {self.samples[0]['board'].shape}")
            avg_len = np.mean([s['solution_length'] for s in self.samples])
            print(f"  Avg solution length: {avg_len:.1f}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        board = sample['board'].copy()
        action = sample['action']
        
        # Apply augmentation if enabled
        if self.augment:
            from .sokoban_hf import augment_board_and_action
            augmentations = augment_board_and_action(board, action)
            aug_idx = np.random.randint(len(augmentations))
            board, action = augmentations[aug_idx]
        
        board_onehot = board_to_onehot(board)
        
        return {
            'input': torch.tensor(board_onehot, dtype=torch.float32),
            'label': torch.tensor(action, dtype=torch.long),
            'board_indices': torch.tensor(board, dtype=torch.long),
        }
    
    @property
    def board_shape(self) -> Tuple[int, int]:
        if self.samples:
            return self.samples[0]['board'].shape
        return self.difficulty_to_shape(self.difficulty)
    
    @staticmethod
    def difficulty_to_shape(difficulty: str) -> Tuple[int, int]:
        shapes = {
            'simple': (6, 6),
            'larger': (10, 10),
            'two_boxes': (6, 6),
            'complex': (10, 10),
        }
        return shapes.get(difficulty, (6, 6))


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == '__main__':
    print("Testing Sokoban generator...")
    
    for difficulty in ['simple', 'complex']:
        print(f"\n{'=' * 40}")
        print(f"Difficulty: {difficulty}")
        print('=' * 40)
        
        ds = SokobanGeneratedDataset(
            difficulty=difficulty,
            n_samples=100,
            seed=42,
        )
        
        # Show a sample
        sample = ds[0]
        print(f"Sample input shape: {sample['input'].shape}")
        print(f"Sample action: {sample['label']} ({['Up', 'Down', 'Left', 'Right'][sample['label']]})")

