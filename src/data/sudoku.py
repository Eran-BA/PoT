"""
Sudoku Dataset and Augmentation Utilities.

Provides dataset loading and validity-preserving augmentations for Sudoku puzzles.

Key insight from HRM: iterate over PUZZLES (1000), not samples (1M).
Each epoch visits each puzzle once, sampling one augmentation per puzzle.

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import csv
import json
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional
from torch.utils.data import Dataset
from tqdm import tqdm


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment Sudoku by applying validity-preserving transformations.
    
    Transformations:
    - Digit permutation (1-9 -> random mapping)
    - Row permutation within bands
    - Column permutation within stacks
    - Transpose
    
    Args:
        board: 9x9 numpy array of the puzzle (0=blank)
        solution: 9x9 numpy array of the solution
        
    Returns:
        Tuple of (augmented_board, augmented_solution)
    """
    # Digit mapping: permute 1-9, keep 0 unchanged
    digit_map = np.zeros(10, dtype=np.uint8)
    digit_map[1:] = np.random.permutation(9) + 1
    
    # Transpose?
    transpose = np.random.rand() < 0.5
    
    # Row permutation within 3 bands
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])
    
    # Column permutation within 3 stacks
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])
    
    def transform(x):
        if transpose:
            x = x.T
        x = x[row_perm][:, col_perm]
        return digit_map[x]
    
    return transform(board.copy()), transform(solution.copy())


class SudokuDataset(Dataset):
    """
    Sudoku dataset loader with ON-THE-FLY augmentation.
    
    Key insight: Generate fresh random augmentations every access to prevent
    memorization. The model can never memorize specific augmentations.
    
    Training: iterate over puzzles, apply random augmentation each time
    Test/Val: no augmentation, use original puzzles
    
    Args:
        data_dir: Path to dataset directory
        split: 'train', 'val', or 'test'
        augment: Whether to apply on-the-fly augmentation (default: True for train)
    """
    
    def __init__(
        self, 
        data_dir: str, 
        split: str = 'train',
        augment: Optional[bool] = None,
    ):
        self.data_dir = Path(data_dir) / split
        self.split = split
        
        # Default: augment only for training
        self.augment = augment if augment is not None else (split == 'train')
        
        # Load numpy format
        inputs_path = self.data_dir / 'all__inputs.npy'
        labels_path = self.data_dir / 'all__labels.npy'
        
        if inputs_path.exists():
            self.inputs = np.load(inputs_path)
            self.labels = np.load(labels_path)
        else:
            raise FileNotFoundError(
                f"Dataset not found at {self.data_dir}. "
                f"Run with --download flag to fetch from HuggingFace."
            )
        
        # For training: we need unique puzzles only (no pre-computed augmentations)
        # Check if we have puzzle indices (old format with pre-computed augs)
        puzzle_idx_path = self.data_dir / 'all__puzzle_indices.npy'
        if puzzle_idx_path.exists():
            puzzle_indices = np.load(puzzle_idx_path)
            # Extract unique puzzles (first occurrence of each)
            unique_puzzles, first_idx = np.unique(puzzle_indices, return_index=True)
            self.num_puzzles = len(unique_puzzles)
            
            if split == 'train':
                # Use only original puzzles, ignore pre-computed augmentations
                self.inputs = self.inputs[first_idx]
                self.labels = self.labels[first_idx]
        else:
            self.num_puzzles = len(self.inputs)
        
        # Epoch indices for shuffling
        self._epoch_indices = np.arange(len(self.inputs))
        if split == 'train':
            np.random.shuffle(self._epoch_indices)
        
        print(f"[{split}] Loaded {len(self.inputs)} puzzles")
        print(f"  Augmentation: {'ON-THE-FLY' if self.augment else 'OFF'}")
    
    def __len__(self):
        return len(self._epoch_indices)
    
    def __getitem__(self, idx):
        real_idx = self._epoch_indices[idx]
        inp = self.inputs[real_idx].copy()
        label = self.labels[real_idx].copy()
        
        # Apply ON-THE-FLY augmentation for training
        if self.augment:
            inp_2d = inp.reshape(9, 9)
            label_2d = label.reshape(9, 9)
            inp, label = shuffle_sudoku(inp_2d, label_2d)
            inp = inp.flatten()
            label = label.flatten()
        
        # HRM uses puzzle_identifiers=0 for ALL puzzles (single shared embedding)
        puzzle_id = torch.tensor(0, dtype=torch.long)
        
        return {
            'input': torch.LongTensor(inp),
            'label': torch.LongTensor(label),
            'puzzle_id': puzzle_id,
        }
    
    def on_epoch_end(self):
        """Shuffle puzzle order for next epoch."""
        if self.split == 'train':
            # Use permutation instead of shuffle to handle read-only arrays
            # (e.g., when dataset is shared via Ray object store)
            self._epoch_indices = np.random.permutation(self._epoch_indices)


def download_sudoku_dataset(
    output_dir: str, 
    subsample_size: int = 10000, 
    num_aug: int = 100,  # Ignored - augmentation is now on-the-fly
    val_ratio: float = 0.1,
) -> None:
    """
    Download and build Sudoku dataset from HuggingFace.
    
    Creates train/val/test splits. Augmentation is done ON-THE-FLY during
    training (not pre-computed) to prevent memorization.
    
    Args:
        output_dir: Directory to save the dataset
        subsample_size: Number of training puzzles to use
        num_aug: IGNORED - kept for backward compatibility
        val_ratio: Fraction of puzzles to hold out for validation
    """
    from huggingface_hub import hf_hub_download
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Downloading Sudoku-Extreme dataset from HuggingFace...")
    print("Note: Augmentation is now ON-THE-FLY (not pre-computed)")
    
    # Download train CSV and create train/val split
    csv_file = hf_hub_download(
        repo_id="sapientinc/sudoku-extreme",
        filename="train.csv",
        repo_type="dataset"
    )
    
    # Parse all training puzzles
    all_puzzles = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            all_puzzles.append(row)
    
    # Subsample
    if subsample_size and subsample_size < len(all_puzzles):
        np.random.shuffle(all_puzzles)
        all_puzzles = all_puzzles[:subsample_size]
    
    # Split into train and val
    n_val = max(1, int(len(all_puzzles) * val_ratio))
    n_train = len(all_puzzles) - n_val
    train_puzzles = all_puzzles[:n_train]
    val_puzzles = all_puzzles[n_train:]
    
    print(f"  Train puzzles: {len(train_puzzles)}, Val puzzles: {len(val_puzzles)}")
    
    # Process train and val splits (NO pre-computed augmentations)
    for split, puzzles in [('train', train_puzzles), ('val', val_puzzles)]:
        split_dir = output_path / split
        split_dir.mkdir(exist_ok=True)
        
        inputs, labels = [], []
        
        for row in tqdm(puzzles, desc=f"Processing {split}"):
            source, puzzle, solution, rating = row[0], row[1], row[2], row[3]
            puzzle = puzzle.replace('.', '0')
            inp = np.array([int(c) for c in puzzle], dtype=np.uint8)
            sol = np.array([int(c) for c in solution], dtype=np.uint8)
            
            # Store ONLY original puzzles (augmentation is on-the-fly)
            inputs.append(inp)
            labels.append(sol)
        
        np.save(split_dir / 'all__inputs.npy', np.array(inputs))
        np.save(split_dir / 'all__labels.npy', np.array(labels))
        
        with open(split_dir / 'dataset.json', 'w') as f:
            json.dump({
                'seq_len': 81,
                'vocab_size': 10,
                'num_puzzles': len(inputs),
            }, f)
        
        print(f"  {split}: {len(inputs)} puzzles (augmentation: on-the-fly)")
    
    # Also download test set (original 422k puzzles for final evaluation)
    csv_file = hf_hub_download(
        repo_id="sapientinc/sudoku-extreme",
        filename="test.csv",
        repo_type="dataset"
    )
    
    split_dir = output_path / 'test'
    split_dir.mkdir(exist_ok=True)
    
    inputs, labels = [], []
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        
        for row in tqdm(reader, desc="Processing test"):
            source, puzzle, solution, rating = row[0], row[1], row[2], row[3]
            puzzle = puzzle.replace('.', '0')
            inp = np.array([int(c) for c in puzzle], dtype=np.uint8)
            sol = np.array([int(c) for c in solution], dtype=np.uint8)
            
            inputs.append(inp)
            labels.append(sol)
    
    np.save(split_dir / 'all__inputs.npy', np.array(inputs))
    np.save(split_dir / 'all__labels.npy', np.array(labels))
    
    with open(split_dir / 'dataset.json', 'w') as f:
        json.dump({
            'seq_len': 81,
            'vocab_size': 10,
            'num_puzzles': len(inputs),
        }, f)
    
    print(f"  test: {len(inputs)} puzzles")
    print(f"✓ Dataset saved to {output_dir}")

