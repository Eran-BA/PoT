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
    Sudoku dataset loader for HRM-format data.
    
    Key insight from HRM: iterate over PUZZLES (1000), not samples (1M).
    Each epoch visits each puzzle once, sampling one augmentation per puzzle.
    
    Args:
        data_dir: Path to dataset directory
        split: 'train' or 'test'
        samples_per_epoch: If set, sample this many per epoch
    """
    
    def __init__(
        self, 
        data_dir: str, 
        split: str = 'train',
        samples_per_epoch: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir) / split
        self.split = split
        
        # Load numpy format
        inputs_path = self.data_dir / 'all__inputs.npy'
        labels_path = self.data_dir / 'all__labels.npy'
        
        if inputs_path.exists():
            self.inputs = np.load(inputs_path)
            self.labels = np.load(labels_path)
            
            # Load puzzle indices for proper ID mapping
            puzzle_idx_path = self.data_dir / 'all__puzzle_indices.npy'
            if puzzle_idx_path.exists():
                self.puzzle_indices = np.load(puzzle_idx_path)
            else:
                self.puzzle_indices = np.arange(len(self.inputs))
        else:
            raise FileNotFoundError(
                f"Dataset not found at {self.data_dir}. "
                f"Run with --download flag to fetch from HuggingFace."
            )
        
        # Build puzzle -> sample indices mapping
        self.unique_puzzles = np.unique(self.puzzle_indices)
        self.num_puzzles = len(self.unique_puzzles)
        
        # For each puzzle, store indices of its augmentations
        self.puzzle_to_samples = {}
        for puzzle_id in self.unique_puzzles:
            self.puzzle_to_samples[puzzle_id] = np.where(self.puzzle_indices == puzzle_id)[0]
        
        # For training: epoch = iterate over puzzles, sample one augmentation each
        # For test: use all samples (no augmentation sampling)
        self.samples_per_epoch = samples_per_epoch
        if split == 'train' and samples_per_epoch is None:
            # Default: one sample per puzzle per epoch (like HRM)
            self.samples_per_epoch = self.num_puzzles
        
        self._epoch_indices = None
        self._resample_epoch()
        
        print(f"[{split}] Loaded {len(self.inputs)} total samples")
        print(f"  Unique puzzles: {self.num_puzzles}")
        print(f"  Samples per epoch: {len(self)}")
    
    def _resample_epoch(self):
        """Resample which augmentation to use for each puzzle this epoch."""
        if self.split == 'train':
            # Sample one augmentation per puzzle
            indices = []
            puzzle_order = np.random.permutation(self.unique_puzzles)
            for puzzle_id in puzzle_order:
                aug_indices = self.puzzle_to_samples[puzzle_id]
                # Randomly pick one augmentation
                idx = np.random.choice(aug_indices)
                indices.append(idx)
            self._epoch_indices = np.array(indices)
        else:
            # Test: use all samples
            self._epoch_indices = np.arange(len(self.inputs))
    
    def __len__(self):
        return len(self._epoch_indices)
    
    def __getitem__(self, idx):
        real_idx = self._epoch_indices[idx]
        inp = torch.LongTensor(self.inputs[real_idx])
        label = torch.LongTensor(self.labels[real_idx])
        # HRM uses puzzle_identifiers=0 for ALL puzzles (single shared embedding)
        # NOT puzzle_indices (which is unique per puzzle and causes memorization)
        puzzle_id = torch.tensor(0, dtype=torch.long)
        
        return {
            'input': inp,
            'label': label,
            'puzzle_id': puzzle_id,
        }
    
    def on_epoch_end(self):
        """Call this at the end of each epoch to resample augmentations."""
        if self.split == 'train':
            self._resample_epoch()


def download_sudoku_dataset(
    output_dir: str, 
    subsample_size: int = 1000, 
    num_aug: int = 1000
) -> None:
    """
    Download and build Sudoku dataset from HuggingFace.
    
    Args:
        output_dir: Directory to save the dataset
        subsample_size: Number of training puzzles to use
        num_aug: Number of augmentations per puzzle
    """
    from huggingface_hub import hf_hub_download
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Downloading Sudoku-Extreme dataset from HuggingFace...")
    
    # Download CSV files
    for split in ['train', 'test']:
        csv_file = hf_hub_download(
            repo_id="sapientinc/sudoku-extreme",
            filename=f"{split}.csv",
            repo_type="dataset"
        )
        
        split_dir = output_path / split
        split_dir.mkdir(exist_ok=True)
        
        # Parse CSV and convert to numpy
        inputs, labels, puzzle_indices = [], [], []
        
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            
            puzzles = list(reader)
            if subsample_size and split == 'train':
                puzzles = puzzles[:subsample_size]
            
            for puzzle_idx, row in enumerate(tqdm(puzzles, desc=f"Processing {split}")):
                # CSV format: source, puzzle, solution, rating
                source, puzzle, solution, rating = row[0], row[1], row[2], row[3]
                
                # Convert string to numpy array (replace '.' with '0' for blanks)
                puzzle = puzzle.replace('.', '0')
                inp = np.array([int(c) for c in puzzle], dtype=np.uint8)
                sol = np.array([int(c) for c in solution], dtype=np.uint8)
                
                # Add original
                inputs.append(inp)
                labels.append(sol)
                puzzle_indices.append(puzzle_idx)
                
                # Add augmentations (only for train)
                if split == 'train' and num_aug > 0:
                    for _ in range(num_aug):
                        aug_inp, aug_sol = shuffle_sudoku(
                            inp.reshape(9, 9), 
                            sol.reshape(9, 9)
                        )
                        inputs.append(aug_inp.flatten())
                        labels.append(aug_sol.flatten())
                        puzzle_indices.append(puzzle_idx)
        
        # Save as numpy
        np.save(split_dir / 'all__inputs.npy', np.array(inputs))
        np.save(split_dir / 'all__labels.npy', np.array(labels))
        np.save(split_dir / 'all__puzzle_indices.npy', np.array(puzzle_indices))
        
        # Save metadata
        with open(split_dir / 'dataset.json', 'w') as f:
            json.dump({
                'seq_len': 81,
                'vocab_size': 10,
                'num_puzzles': len(np.unique(puzzle_indices)),
                'num_samples': len(inputs),
            }, f)
        
        print(f"  {split}: {len(inputs)} samples from {len(np.unique(puzzle_indices))} puzzles")
    
    print(f"✓ Dataset saved to {output_dir}")

