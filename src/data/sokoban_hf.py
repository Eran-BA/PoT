"""
Sokoban HuggingFace Dataset Loader.

Loads the Xiaofeng77/sokoban dataset for SUPERVISED learning.
Format: (board_state, next_action) pairs - exactly like Sudoku.

Dataset: https://huggingface.co/datasets/Xiaofeng77/sokoban

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import re
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import Dataset
from datasets import load_dataset


# =============================================================================
# Constants
# =============================================================================

# Tile types (same as sokoban.py)
TILE_WALL = 0
TILE_FLOOR = 1
TILE_PLAYER = 2
TILE_BOX = 3
TILE_TARGET = 4
TILE_BOX_ON_TARGET = 5
TILE_PLAYER_ON_TARGET = 6
NUM_TILE_TYPES = 7

# HuggingFace dataset uses different symbols
HF_CHAR_TO_TILE = {
    '#': TILE_WALL,
    '_': TILE_FLOOR,
    'P': TILE_PLAYER,
    'X': TILE_BOX,
    'O': TILE_TARGET,
    '√': TILE_BOX_ON_TARGET,  # Box on target (checkmark)
    'S': TILE_PLAYER_ON_TARGET,
}

# Actions (0-indexed in our model, 1-indexed in HF dataset)
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
NUM_ACTIONS = 4

# HF dataset uses 1-indexed actions (1=Up, 2=Down, 3=Left, 4=Right)
# Convert to 0-indexed for our model
HF_ACTION_MAP = {
    1: ACTION_UP,     # Up
    2: ACTION_DOWN,   # Down
    3: ACTION_LEFT,   # Left
    4: ACTION_RIGHT,  # Right
}

# Action names for debugging
ACTION_NAMES = ['Up', 'Down', 'Left', 'Right']


# =============================================================================
# Parsing Functions
# =============================================================================

def parse_hf_board(prompt: str) -> Optional[np.ndarray]:
    """
    Parse board from HuggingFace Sokoban prompt.
    
    The prompt contains a grid like:
    # \t # \t # \t # \t # \t # \t
    # \t P \t X \t _ \t O \t # \t
    ...
    
    Args:
        prompt: The full prompt string
    
    Returns:
        np.ndarray of shape [H, W] with tile indices, or None if parsing fails
    """
    # Find the board in the prompt - it's between [Current Observation]: and "Decide"
    match = re.search(r'\[Current Observation\]:\s*\n(.*?)\nDecide', prompt, re.DOTALL)
    if not match:
        return None
    
    board_text = match.group(1).strip()
    
    # Parse each row
    rows = []
    for line in board_text.split('\n'):
        if not line.strip():
            continue
        
        # Split by tab and parse each cell
        cells = [c.strip() for c in line.split('\t') if c.strip()]
        row = []
        for cell in cells:
            if cell in HF_CHAR_TO_TILE:
                row.append(HF_CHAR_TO_TILE[cell])
            else:
                # Unknown character, treat as floor
                row.append(TILE_FLOOR)
        
        if row:
            rows.append(row)
    
    if not rows:
        return None
    
    # Convert to numpy array
    # Pad rows to same length if needed
    max_len = max(len(row) for row in rows)
    for row in rows:
        while len(row) < max_len:
            row.append(TILE_WALL)
    
    return np.array(rows, dtype=np.int64)


def board_to_onehot(board: np.ndarray) -> np.ndarray:
    """
    Convert board to one-hot encoding.
    
    Args:
        board: [H, W] with tile indices (0-6)
    
    Returns:
        [H, W, NUM_TILE_TYPES] one-hot encoded
    """
    H, W = board.shape
    onehot = np.zeros((H, W, NUM_TILE_TYPES), dtype=np.float32)
    
    for i in range(H):
        for j in range(W):
            onehot[i, j, board[i, j]] = 1.0
    
    return onehot


def augment_board_and_action(
    board: np.ndarray,
    action: int,
) -> List[Tuple[np.ndarray, int]]:
    """
    Apply geometric augmentations to board and action.
    
    Actions must be transformed consistently with the board.
    
    Args:
        board: [H, W] board
        action: Action index (0-3: UP/DOWN/LEFT/RIGHT)
    
    Returns:
        List of (augmented_board, augmented_action) tuples
    """
    # Action transformations for each augmentation
    # Original: UP=0, DOWN=1, LEFT=2, RIGHT=3
    
    augmentations = []
    
    # Original
    augmentations.append((board.copy(), action))
    
    # Rotate 90 clockwise: UP->RIGHT, RIGHT->DOWN, DOWN->LEFT, LEFT->UP
    rot90_action = [3, 2, 0, 1][action]  # UP->RIGHT, DOWN->LEFT, LEFT->UP, RIGHT->DOWN
    augmentations.append((np.rot90(board, k=-1).copy(), rot90_action))
    
    # Rotate 180
    rot180_action = [1, 0, 3, 2][action]  # UP<->DOWN, LEFT<->RIGHT
    augmentations.append((np.rot90(board, k=2).copy(), rot180_action))
    
    # Rotate 270 (or 90 counter-clockwise)
    rot270_action = [2, 3, 1, 0][action]  # UP->LEFT, DOWN->RIGHT, LEFT->DOWN, RIGHT->UP
    augmentations.append((np.rot90(board, k=1).copy(), rot270_action))
    
    # Horizontal flip: LEFT<->RIGHT
    hflip_action = [0, 1, 3, 2][action]  # LEFT<->RIGHT
    augmentations.append((np.fliplr(board).copy(), hflip_action))
    
    # Vertical flip: UP<->DOWN
    vflip_action = [1, 0, 2, 3][action]  # UP<->DOWN
    augmentations.append((np.flipud(board).copy(), vflip_action))
    
    # Horizontal flip + rotate 90
    hflip_rot90_action = [3, 2, 0, 1][hflip_action]
    augmentations.append((np.rot90(np.fliplr(board), k=-1).copy(), hflip_rot90_action))
    
    # Vertical flip + rotate 90
    vflip_rot90_action = [3, 2, 0, 1][vflip_action]
    augmentations.append((np.rot90(np.flipud(board), k=-1).copy(), vflip_rot90_action))
    
    return augmentations


# =============================================================================
# Dataset Class
# =============================================================================

class SokobanHFDataset(Dataset):
    """
    Sokoban dataset from HuggingFace for SUPERVISED learning.
    
    Exactly like SudokuDataset:
    - Input: board state
    - Label: correct action
    - Loss: Cross-entropy (identical to Sudoku)
    
    Args:
        split: 'train' or 'test'
        augment: Whether to apply on-the-fly augmentation (default: True for train)
        cache_dir: Optional cache directory for HuggingFace datasets
    """
    
    def __init__(
        self,
        split: str = 'train',
        augment: Optional[bool] = None,
        cache_dir: Optional[str] = None,
    ):
        self.split = split
        self.augment = augment if augment is not None else (split == 'train')
        
        # Load from HuggingFace
        print(f"Loading Xiaofeng77/sokoban dataset ({split})...")
        dataset = load_dataset(
            "Xiaofeng77/sokoban",
            split=split,
            cache_dir=cache_dir,
        )
        
        # Parse all examples
        self.examples = []
        failed = 0
        
        for item in dataset:
            # Get prompt and ground truth
            prompt = item['prompt'][0]['content']
            ground_truth = item['reward_model']['ground_truth'][0]
            
            # Parse board
            board = parse_hf_board(prompt)
            if board is None:
                failed += 1
                continue
            
            # Convert action (1-indexed in HF dataset)
            action = HF_ACTION_MAP.get(ground_truth, 0)
            
            self.examples.append({
                'board': board,
                'action': action,
            })
        
        print(f"[{split}] Loaded {len(self.examples)} examples ({failed} failed to parse)")
        print(f"  Board size: {self.examples[0]['board'].shape if self.examples else 'N/A'}")
        print(f"  Augmentation: {'ON-THE-FLY' if self.augment else 'OFF'}")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        board = example['board'].copy()
        action = example['action']
        
        # Apply on-the-fly augmentation
        if self.augment:
            augmentations = augment_board_and_action(board, action)
            aug_idx = np.random.randint(len(augmentations))
            board, action = augmentations[aug_idx]
        
        # Convert to one-hot
        board_onehot = board_to_onehot(board)
        
        return {
            'input': torch.tensor(board_onehot, dtype=torch.float32),  # [H, W, 7]
            'label': torch.tensor(action, dtype=torch.long),  # scalar
            'board_indices': torch.tensor(board, dtype=torch.long),  # [H, W]
        }
    
    @property
    def board_shape(self) -> Tuple[int, int]:
        """Get board dimensions."""
        if self.examples:
            return self.examples[0]['board'].shape
        return (6, 6)  # Default for HF dataset
    
    @property
    def num_actions(self) -> int:
        """Number of possible actions."""
        return NUM_ACTIONS


def download_sokoban_hf_dataset(cache_dir: Optional[str] = None) -> None:
    """
    Download the Sokoban HuggingFace dataset.
    
    Args:
        cache_dir: Optional cache directory
    """
    print("Downloading Xiaofeng77/sokoban dataset from HuggingFace...")
    
    # Just load to trigger download
    for split in ['train', 'test']:
        dataset = load_dataset(
            "Xiaofeng77/sokoban",
            split=split,
            cache_dir=cache_dir,
        )
        print(f"  {split}: {len(dataset)} examples")
    
    print("Download complete!")


# =============================================================================
# Combined Dataset: HuggingFace + Generated
# =============================================================================

class SokobanCombinedDataset(Dataset):
    """
    Combined dataset: HuggingFace data + dynamically generated puzzles.
    
    This allows scaling up training data beyond the ~3k HuggingFace examples.
    Generated puzzles use gym-sokoban + BFS solver for optimal actions.
    
    Args:
        hf_split: HuggingFace split ('train' or 'test')
        n_generated: Number of additional puzzles to generate
        difficulty: Difficulty for generated puzzles ('simple', 'larger', 'two_boxes', 'complex')
        augment: Whether to apply on-the-fly augmentation
        seed: Random seed for generation
        cache_dir: Cache directory for HuggingFace
    """
    
    def __init__(
        self,
        hf_split: str = 'train',
        n_generated: int = 0,
        difficulty: str = 'simple',
        augment: Optional[bool] = None,
        seed: int = 42,
        cache_dir: Optional[str] = None,
    ):
        self.augment = augment if augment is not None else (hf_split == 'train')
        
        # Load HuggingFace dataset
        print(f"Loading HuggingFace Sokoban dataset ({hf_split})...")
        self.hf_dataset = SokobanHFDataset(
            split=hf_split,
            augment=False,  # We'll apply augmentation ourselves
            cache_dir=cache_dir,
        )
        
        # Generate additional puzzles if requested
        self.generated_examples = []
        if n_generated > 0:
            print(f"Generating {n_generated} additional {difficulty} puzzles...")
            from .sokoban_generator import SokobanGenerator
            
            generator = SokobanGenerator(difficulty=difficulty, seed=seed)
            generated = generator.generate_dataset(n_generated, verbose=True)
            
            for sample in generated:
                self.generated_examples.append({
                    'board': sample['board'],
                    'action': sample['action'],
                })
            
            print(f"  Generated {len(self.generated_examples)} puzzles")
        
        # Combined stats
        self.n_hf = len(self.hf_dataset)
        self.n_gen = len(self.generated_examples)
        
        print(f"\n[Combined Dataset]")
        print(f"  HuggingFace: {self.n_hf}")
        print(f"  Generated:   {self.n_gen}")
        print(f"  Total:       {len(self)}")
        print(f"  Augmentation: {'ON-THE-FLY (8x)' if self.augment else 'OFF'}")
    
    def __len__(self) -> int:
        return self.n_hf + self.n_gen
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Determine source
        if idx < self.n_hf:
            # From HuggingFace
            example = self.hf_dataset.examples[idx]
            board = example['board'].copy()
            action = example['action']
        else:
            # From generated
            gen_idx = idx - self.n_hf
            example = self.generated_examples[gen_idx]
            board = example['board'].copy()
            action = example['action']
        
        # Apply on-the-fly augmentation
        if self.augment:
            augmentations = augment_board_and_action(board, action)
            aug_idx = np.random.randint(len(augmentations))
            board, action = augmentations[aug_idx]
            board = board.copy()  # Fix negative stride issue
        
        # Convert to one-hot
        board_onehot = board_to_onehot(board)
        
        return {
            'input': torch.tensor(board_onehot, dtype=torch.float32),
            'label': torch.tensor(action, dtype=torch.long),
            'board_indices': torch.tensor(board, dtype=torch.long),
        }
    
    @property
    def board_shape(self) -> Tuple[int, int]:
        """Get board dimensions (from HuggingFace)."""
        return self.hf_dataset.board_shape
    
    @property
    def num_actions(self) -> int:
        """Number of possible actions."""
        return NUM_ACTIONS


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    # Test dataset loading
    train_ds = SokobanHFDataset(split='train', augment=True)
    test_ds = SokobanHFDataset(split='test', augment=False)
    
    print(f"\nTrain examples: {len(train_ds)}")
    print(f"Test examples: {len(test_ds)}")
    print(f"Board shape: {train_ds.board_shape}")
    
    # Test a sample
    sample = train_ds[0]
    print(f"\nSample:")
    print(f"  Input shape: {sample['input'].shape}")
    print(f"  Label: {sample['label']}")
    print(f"  Board indices shape: {sample['board_indices'].shape}")

