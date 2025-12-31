"""
ARC (Abstraction and Reasoning Corpus) Dataset.

Provides dataset loading for ARC puzzles (ARC-1 and ARC-2).
Based on HRM's dataset format: https://github.com/sapientinc/HRM

Key properties:
- Max grid size: 30x30
- Vocab: 12 (PAD=0, EOS=1, digits 2-11 for cell values 0-9)
- Variable input/output sizes

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional, List
from torch.utils.data import Dataset
from tqdm import tqdm


# Constants from HRM's build_arc_dataset.py
ARC_MAX_GRID_SIZE = 30
ARC_SEQ_LEN = ARC_MAX_GRID_SIZE * ARC_MAX_GRID_SIZE  # 900
ARC_VOCAB_SIZE = 12  # PAD + EOS + digits 0-9


def dihedral_transform(grid: np.ndarray, trans_id: int) -> np.ndarray:
    """
    Apply one of 8 dihedral transformations (rotations + reflections).
    
    trans_id 0-3: rotations by 0, 90, 180, 270 degrees
    trans_id 4-7: same rotations after horizontal flip
    """
    if trans_id >= 4:
        grid = np.flip(grid, axis=1)  # Horizontal flip
        trans_id -= 4
    grid = np.rot90(grid, k=trans_id)
    return grid


def shuffle_arc(
    inp: np.ndarray, 
    out: np.ndarray, 
    do_color_perm: bool = True,
    do_transform: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment ARC grid pair with validity-preserving transformations.
    
    Transformations:
    - Color permutation (1-9 -> random mapping, 0 stays 0)
    - Dihedral transformations (rotate/flip)
    
    Args:
        inp: Input grid (2D numpy array, values 0-9)
        out: Output grid (2D numpy array, values 0-9)
        do_color_perm: Apply color permutation
        do_transform: Apply dihedral transformation
        
    Returns:
        Tuple of (augmented_inp, augmented_out)
    """
    if do_color_perm:
        # Permute colors 1-9, keep 0 unchanged
        mapping = np.zeros(10, dtype=np.uint8)
        mapping[1:] = np.random.permutation(np.arange(1, 10))
        inp = mapping[inp]
        out = mapping[out]
    
    if do_transform:
        trans_id = np.random.randint(0, 8)
        inp = dihedral_transform(inp, trans_id)
        out = dihedral_transform(out, trans_id)
    
    return inp.copy(), out.copy()


def grid_to_seq(
    grid: np.ndarray, 
    do_translation: bool = False,
    pad_offset: Tuple[int, int] = None,
) -> np.ndarray:
    """
    Convert 2D grid to 1D sequence with padding and EOS markers.
    
    HRM format:
    - PAD: 0
    - EOS: 1  
    - Digits: 2-11 (representing 0-9)
    
    Args:
        grid: 2D numpy array (H, W), values 0-9
        do_translation: If True, apply random translation within 30x30
        pad_offset: Optional (pad_r, pad_c) to use specific offset (for paired grids)
        
    Returns:
        1D array of length 900 (30*30)
    """
    nrow, ncol = grid.shape
    
    # Compute padding offset
    if pad_offset is not None:
        pad_r, pad_c = pad_offset
    elif do_translation:
        max_pad_r = ARC_MAX_GRID_SIZE - nrow
        max_pad_c = ARC_MAX_GRID_SIZE - ncol
        pad_r = np.random.randint(0, max_pad_r + 1)
        pad_c = np.random.randint(0, max_pad_c + 1)
    else:
        pad_r = pad_c = 0
    
    # Create padded grid with shifted values (+2 for PAD/EOS)
    padded = np.zeros((ARC_MAX_GRID_SIZE, ARC_MAX_GRID_SIZE), dtype=np.uint8)
    padded[pad_r:pad_r+nrow, pad_c:pad_c+ncol] = grid + 2
    
    # Add EOS markers at boundaries
    eos_row, eos_col = pad_r + nrow, pad_c + ncol
    if eos_row < ARC_MAX_GRID_SIZE:
        padded[eos_row, pad_c:eos_col] = 1
    if eos_col < ARC_MAX_GRID_SIZE:
        padded[pad_r:eos_row, eos_col] = 1
    
    return padded.flatten()


def grids_to_seq_pair(
    inp: np.ndarray, 
    out: np.ndarray, 
    do_translation: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert input/output grid pair to sequences with SAME translation.
    
    This matches HRM's approach: compute translation based on max of both
    grid sizes, then apply SAME offset to both.
    
    Args:
        inp: Input grid (H1, W1), values 0-9
        out: Output grid (H2, W2), values 0-9
        do_translation: If True, apply random translation
        
    Returns:
        Tuple of (inp_seq, out_seq), each length 900
    """
    if do_translation:
        # HRM-style: compute offset based on MAX of both grid sizes
        max_h = max(inp.shape[0], out.shape[0])
        max_w = max(inp.shape[1], out.shape[1])
        max_pad_r = ARC_MAX_GRID_SIZE - max_h
        max_pad_c = ARC_MAX_GRID_SIZE - max_w
        pad_r = np.random.randint(0, max_pad_r + 1)
        pad_c = np.random.randint(0, max_pad_c + 1)
        pad_offset = (pad_r, pad_c)
    else:
        pad_offset = (0, 0)
    
    # Apply SAME offset to both grids
    inp_seq = grid_to_seq(inp, pad_offset=pad_offset)
    out_seq = grid_to_seq(out, pad_offset=pad_offset)
    
    return inp_seq, out_seq


class ARCDataset(Dataset):
    """
    ARC dataset loader with ON-THE-FLY augmentation (like Sudoku).
    
    Always uses on-the-fly augmentation for training:
    - Color permutation (1-9 shuffled, 0 fixed)
    - Dihedral transforms (8 rotations/flips)
    - Translational augmentation (random position in 30x30)
    
    This gives infinite variety every epoch, better than pre-computed.
    
    Args:
        data_dir: Path to dataset directory
        split: 'train' or 'test'
        subset: Subset name (default 'all')
        augment: Whether to apply on-the-fly augmentation (default: True for train)
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        subset: str = 'all',
        augment: Optional[bool] = None,
    ):
        self.data_dir = Path(data_dir) / split
        self.split = split
        self.subset = subset
        
        # Try HRM format first (preprocessed .npy files)
        inputs_path = self.data_dir / f'{subset}__inputs.npy'
        labels_path = self.data_dir / f'{subset}__labels.npy'
        
        if inputs_path.exists():
            self._load_npy_format(inputs_path, labels_path)
        else:
            raise FileNotFoundError(
                f"Dataset not found at {self.data_dir}. "
                f"Run the dataset builder script first."
            )
        
        # On-the-fly augmentation (like Sudoku) - always for training
        if augment is not None:
            self.augment = augment
        else:
            self.augment = (split == 'train')  # Simple: train=augment, test=no augment
        
        # Epoch indices for shuffling
        self._epoch_indices = np.arange(len(self.inputs))
        if split == 'train':
            np.random.shuffle(self._epoch_indices)
        
        print(f"[{split}/{subset}] Loaded {len(self.inputs)} examples ({self.num_puzzles} puzzles)")
        print(f"  Augmentation: {'ON-THE-FLY' if self.augment else 'OFF'}")
    
    def _load_npy_format(self, inputs_path: Path, labels_path: Path):
        """Load HRM-style preprocessed format."""
        self.inputs = np.load(inputs_path)
        self.labels = np.load(labels_path)
        
        # Load puzzle indices if available (for deduplication)
        puzzle_idx_path = self.data_dir / f'{self.subset}__puzzle_indices.npy'
        if puzzle_idx_path.exists():
            puzzle_indices = np.load(puzzle_idx_path)
            
            if self.split == 'train':
                # For training, get unique puzzles (first of each group)
                unique_puzzles, first_idx = np.unique(puzzle_indices, return_index=True)
                # Keep augmented versions - HRM style iterates over puzzles
                # but we'll iterate over all examples with on-the-fly aug
                self.num_puzzles = len(unique_puzzles)
            else:
                self.num_puzzles = len(np.unique(puzzle_indices))
        else:
            self.num_puzzles = len(self.inputs)
        
        # Load puzzle identifiers if available
        identifiers_path = self.data_dir / f'{self.subset}__puzzle_identifiers.npy'
        if identifiers_path.exists():
            self.puzzle_identifiers = np.load(identifiers_path)
        else:
            self.puzzle_identifiers = np.zeros(len(self.inputs), dtype=np.int32)
    
    def __len__(self):
        return len(self._epoch_indices)
    
    def __getitem__(self, idx):
        real_idx = self._epoch_indices[idx]
        inp = self.inputs[real_idx].copy()
        label = self.labels[real_idx].copy()
        
        # Apply ON-THE-FLY augmentation for training
        if self.augment:
            inp, label = self._apply_augmentation(inp, label)
        
        # Get puzzle ID (clamped to 0 for single shared embedding like HRM)
        puzzle_id = 0  # HRM uses single shared embedding
        
        return {
            'input': torch.LongTensor(inp),
            'label': torch.LongTensor(label),
            'puzzle_id': torch.tensor(puzzle_id, dtype=torch.long),
        }
    
    def _apply_augmentation(self, inp_seq: np.ndarray, label_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply on-the-fly augmentation to ARC sequences.
        
        Since data is in HRM sequence format (PAD=0, EOS=1, digits=2-11),
        we need to:
        1. Reshape to 2D grid
        2. Find the actual content region (non-PAD)
        3. Apply color permutation and dihedral transforms
        4. Reshape back to sequence
        """
        # Reshape to 2D
        inp_2d = inp_seq.reshape(ARC_MAX_GRID_SIZE, ARC_MAX_GRID_SIZE)
        label_2d = label_seq.reshape(ARC_MAX_GRID_SIZE, ARC_MAX_GRID_SIZE)
        
        # Find content mask (non-PAD, non-EOS cells)
        content_mask = (inp_2d >= 2) | (label_2d >= 2)
        
        if not content_mask.any():
            return inp_seq, label_seq
        
        # Get bounding box of content
        rows_with_content = np.any(content_mask, axis=1)
        cols_with_content = np.any(content_mask, axis=0)
        r_min, r_max = np.where(rows_with_content)[0][[0, -1]]
        c_min, c_max = np.where(cols_with_content)[0][[0, -1]]
        
        # Extract content region (convert from token space to value space: subtract 2)
        inp_content = np.maximum(inp_2d[r_min:r_max+1, c_min:c_max+1].astype(np.int16) - 2, 0).astype(np.uint8)
        label_content = np.maximum(label_2d[r_min:r_max+1, c_min:c_max+1].astype(np.int16) - 2, 0).astype(np.uint8)
        
        # Apply augmentation (color permutation + dihedral transform)
        inp_aug, label_aug = shuffle_arc(inp_content, label_content, do_color_perm=True, do_transform=True)
        
        # Convert back to sequence format with SAME random translation for both
        # (HRM-style: offset computed from max of both grid sizes)
        inp_new, label_new = grids_to_seq_pair(inp_aug, label_aug, do_translation=True)
        
        return inp_new, label_new
    
    def on_epoch_end(self):
        """Shuffle example order for next epoch."""
        if self.split == 'train':
            self._epoch_indices = np.random.permutation(self._epoch_indices)


def download_arc_dataset(
    output_dir: str,
    version: str = 'arc-2',  # 'arc-1' or 'arc-2'
    num_aug: int = 100,
    seed: int = 42,
) -> None:
    """
    Download and build ARC dataset.
    
    This creates the HRM-compatible format from raw ARC JSON files.
    
    Args:
        output_dir: Directory to save the dataset
        version: 'arc-1' or 'arc-2'
        num_aug: Number of augmentations per puzzle (0 for on-the-fly only)
        seed: Random seed for reproducibility
    """
    import subprocess
    import sys
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(seed)
    
    print(f"Building {version.upper()} dataset...")
    
    # Determine source repos based on version
    # Official ARC-AGI repos: https://github.com/arcprize/ARC-AGI-2
    if version == 'arc-2':
        repo_url = "https://github.com/arcprize/ARC-AGI-2"
        data_subdir = "data"
    else:  # arc-1
        repo_url = "https://github.com/arcprize/ARC-AGI"
        data_subdir = "data"
    
    # Clone or use existing
    raw_dir = output_path / 'raw-data'
    repo_name = repo_url.split('/')[-1]
    repo_path = raw_dir / repo_name
    
    if not repo_path.exists():
        print(f"Cloning {repo_url}...")
        raw_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--depth', '1', repo_url, str(repo_path)], check=True)
    
    arc_data_path = repo_path / data_subdir
    
    # Process each split
    for split_name in ['training', 'evaluation']:
        split_path = arc_data_path / split_name
        if not split_path.exists():
            print(f"  Skipping {split_name} (not found)")
            continue
        
        out_split = 'train' if split_name == 'training' else 'test'
        out_dir = output_path / out_split
        out_dir.mkdir(exist_ok=True)
        
        inputs_list = []
        labels_list = []
        puzzle_indices = []
        puzzle_identifiers = []
        
        puzzle_files = list(split_path.glob('*.json'))
        print(f"  Processing {split_name}: {len(puzzle_files)} puzzles...")
        
        for puzzle_id, puzzle_file in enumerate(tqdm(puzzle_files)):
            with open(puzzle_file, 'r') as f:
                puzzle = json.load(f)
            
            # Process train examples
            for example in puzzle.get('train', []):
                inp_grid = np.array(example['input'], dtype=np.uint8)
                out_grid = np.array(example['output'], dtype=np.uint8)
                
                # Convert to sequence format
                inp_seq = grid_to_seq(inp_grid, do_translation=False)
                out_seq = grid_to_seq(out_grid, do_translation=False)
                
                inputs_list.append(inp_seq)
                labels_list.append(out_seq)
                puzzle_indices.append(len(puzzle_identifiers))
                puzzle_identifiers.append(puzzle_id)
            
            # For test split, also process test examples
            if out_split == 'test':
                for example in puzzle.get('test', []):
                    inp_grid = np.array(example['input'], dtype=np.uint8)
                    out_grid = np.array(example['output'], dtype=np.uint8)
                    
                    inp_seq = grid_to_seq(inp_grid, do_translation=False)
                    out_seq = grid_to_seq(out_grid, do_translation=False)
                    
                    inputs_list.append(inp_seq)
                    labels_list.append(out_seq)
                    puzzle_indices.append(len(puzzle_identifiers))
                    puzzle_identifiers.append(puzzle_id)
        
        # Save numpy arrays
        np.save(out_dir / 'all__inputs.npy', np.array(inputs_list))
        np.save(out_dir / 'all__labels.npy', np.array(labels_list))
        np.save(out_dir / 'all__puzzle_indices.npy', np.array(puzzle_indices))
        np.save(out_dir / 'all__puzzle_identifiers.npy', np.array(puzzle_identifiers))
        
        # Save metadata
        with open(out_dir / 'dataset.json', 'w') as f:
            json.dump({
                'seq_len': ARC_SEQ_LEN,
                'vocab_size': ARC_VOCAB_SIZE,
                'num_puzzles': len(puzzle_files),
                'num_examples': len(inputs_list),
            }, f, indent=2)
        
        print(f"    Saved {len(inputs_list)} examples from {len(puzzle_files)} puzzles")
    
    print(f"✓ Dataset saved to {output_dir}")

