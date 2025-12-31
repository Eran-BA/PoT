#!/usr/bin/env python3
"""
Build ARC dataset with HRM-style pre-computed augmentations.

Usage:
    # ARC-2 with 1000 augmentations per puzzle (HRM-style)
    python scripts/build_arc_dataset.py --version arc-2 --num-aug 1000 --output-dir data/arc-2-aug-1000
    
    # ARC-2 minimal (on-the-fly augmentation during training)
    python scripts/build_arc_dataset.py --version arc-2 --num-aug 0 --output-dir data/arc-2
    
This matches HRM's dataset/build_arc_dataset.py approach:
- Pre-compute color permutation + dihedral transforms
- Hash-based deduplication to ensure variety
- Translational augmentation baked into training examples
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# Constants matching HRM
ARC_MAX_GRID_SIZE = 30
ARC_SEQ_LEN = ARC_MAX_GRID_SIZE * ARC_MAX_GRID_SIZE  # 900
ARC_VOCAB_SIZE = 12  # PAD + EOS + digits 0-9
ARC_AUGMENT_RETRIES_FACTOR = 5


@dataclass
class ARCPuzzle:
    """A single ARC puzzle with all its examples."""
    id: str
    examples: List[Tuple[np.ndarray, np.ndarray]]  # (input, output) pairs


def dihedral_transform(grid: np.ndarray, trans_id: int) -> np.ndarray:
    """
    Apply one of 8 dihedral transformations.
    trans_id 0-3: rotations by 0, 90, 180, 270 degrees
    trans_id 4-7: same rotations after horizontal flip
    """
    if trans_id >= 4:
        grid = np.flip(grid, axis=1)
        trans_id -= 4
    return np.rot90(grid, k=trans_id).copy()


def arc_grid_to_np(grid: List[List[int]]) -> np.ndarray:
    """Convert JSON grid to numpy array."""
    arr = np.array(grid, dtype=np.uint8)
    assert arr.ndim == 2
    assert arr.shape[0] <= ARC_MAX_GRID_SIZE and arr.shape[1] <= ARC_MAX_GRID_SIZE
    assert np.all((arr >= 0) & (arr <= 9))
    return arr


def np_grid_to_seq_translational_augment(
    inp: np.ndarray, 
    out: np.ndarray, 
    do_translation: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert grid pair to sequence format with SAME translational augmentation.
    
    Format: PAD=0, EOS=1, digits=2-11
    """
    if do_translation:
        max_h = max(inp.shape[0], out.shape[0])
        max_w = max(inp.shape[1], out.shape[1])
        pad_r = np.random.randint(0, ARC_MAX_GRID_SIZE - max_h + 1)
        pad_c = np.random.randint(0, ARC_MAX_GRID_SIZE - max_w + 1)
    else:
        pad_r = pad_c = 0
    
    result = []
    for grid in [inp, out]:
        nrow, ncol = grid.shape
        # Pad with zeros, shift values by 2
        padded = np.zeros((ARC_MAX_GRID_SIZE, ARC_MAX_GRID_SIZE), dtype=np.uint8)
        padded[pad_r:pad_r+nrow, pad_c:pad_c+ncol] = grid + 2
        
        # Add EOS markers
        eos_row, eos_col = pad_r + nrow, pad_c + ncol
        if eos_row < ARC_MAX_GRID_SIZE:
            padded[eos_row, pad_c:eos_col] = 1
        if eos_col < ARC_MAX_GRID_SIZE:
            padded[pad_r:eos_row, eos_col] = 1
        
        result.append(padded.flatten())
    
    return result[0], result[1]


def puzzle_hash(puzzle_examples: List[Tuple[np.ndarray, np.ndarray]]) -> str:
    """Hash puzzle examples for deduplication."""
    def grid_hash(grid: np.ndarray) -> str:
        buffer = [x.to_bytes(1, 'little') for x in grid.shape]
        buffer.append(grid.tobytes())
        return hashlib.sha256(b"".join(buffer)).hexdigest()
    
    hashes = []
    for inp, out in puzzle_examples:
        hashes.append(f"{grid_hash(inp)}|{grid_hash(out)}")
    hashes.sort()
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()


def augment_puzzle(
    puzzle: ARCPuzzle,
    aug_count: int,
    seed: int = None,
) -> List[ARCPuzzle]:
    """
    Generate augmented versions of a puzzle.
    
    Returns list of [original, aug1, aug2, ..., aug_n].
    Uses hash-based deduplication to ensure variety.
    """
    if seed is not None:
        np.random.seed(seed)
    
    group = [puzzle]
    
    if aug_count <= 0:
        return group
    
    hashes = {puzzle_hash(puzzle.examples)}
    
    for _ in range(ARC_AUGMENT_RETRIES_FACTOR * aug_count):
        # Augmentation plan
        trans_id = np.random.randint(0, 8)
        # Color permutation: 0 stays 0, 1-9 shuffled
        mapping = np.zeros(10, dtype=np.uint8)
        mapping[1:] = np.random.permutation(np.arange(1, 10, dtype=np.uint8))
        
        aug_repr = f"t{trans_id}_{''.join(str(x) for x in mapping)}"
        
        def map_grid(grid: np.ndarray) -> np.ndarray:
            return dihedral_transform(mapping[grid], trans_id)
        
        # Create augmented puzzle
        aug_examples = [(map_grid(inp), map_grid(out)) for inp, out in puzzle.examples]
        h = puzzle_hash(aug_examples)
        
        if h not in hashes:
            hashes.add(h)
            group.append(ARCPuzzle(f"{puzzle.id}_{aug_repr}", aug_examples))
        
        if len(group) >= aug_count + 1:
            break
    
    if len(group) < aug_count + 1:
        print(f"  [Puzzle {puzzle.id}] augmentation not full, only {len(group)}")
    
    return group


def build_arc_dataset(
    dataset_dirs: List[str],
    output_dir: str,
    num_aug: int = 1000,
    seed: int = 42,
):
    """
    Build ARC dataset with pre-computed augmentations.
    
    Matches HRM's build_arc_dataset.py output format.
    """
    np.random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all puzzles
    all_puzzles = {"train": [], "test": []}
    puzzle_id_map = {}  # puzzle_name -> unique_id
    next_puzzle_id = 1  # 0 is reserved for blank
    
    for dataset_dir in dataset_dirs:
        dataset_path = Path(dataset_dir)
        
        for split_dir in ['training', 'evaluation']:
            split_path = dataset_path / split_dir
            if not split_path.exists():
                continue
            
            out_split = 'train' if split_dir == 'training' else 'test'
            puzzle_files = sorted(split_path.glob('*.json'))
            
            print(f"Processing {split_dir}: {len(puzzle_files)} puzzles...")
            
            for puzzle_file in tqdm(puzzle_files, desc=f"  {split_dir}"):
                with open(puzzle_file, 'r') as f:
                    data = json.load(f)
                
                puzzle_name = puzzle_file.stem
                
                # Assign unique ID
                if puzzle_name not in puzzle_id_map:
                    puzzle_id_map[puzzle_name] = next_puzzle_id
                    next_puzzle_id += 1
                
                # Create puzzle from train examples
                train_examples = [
                    (arc_grid_to_np(ex['input']), arc_grid_to_np(ex['output']))
                    for ex in data.get('train', [])
                ]
                
                if train_examples:
                    puzzle = ARCPuzzle(puzzle_name, train_examples)
                    
                    # Generate augmentations for training split only
                    if out_split == 'train':
                        aug_group = augment_puzzle(puzzle, num_aug)
                    else:
                        aug_group = [puzzle]
                    
                    all_puzzles[out_split].append(aug_group)
                
                # For test split, also include test examples (no augmentation)
                if out_split == 'test':
                    test_examples = [
                        (arc_grid_to_np(ex['input']), arc_grid_to_np(ex['output']))
                        for ex in data.get('test', [])
                    ]
                    if test_examples:
                        test_puzzle = ARCPuzzle(puzzle_name, test_examples)
                        all_puzzles[out_split].append([test_puzzle])
    
    print(f"\nTotal puzzle IDs: {next_puzzle_id}")
    
    # Save each split
    for split_name, puzzle_groups in all_puzzles.items():
        split_dir = output_path / split_name
        split_dir.mkdir(exist_ok=True)
        
        enable_translation = (split_name == 'train')
        
        inputs_list = []
        labels_list = []
        puzzle_identifiers = []
        puzzle_indices = [0]
        group_indices = [0]
        
        total_examples = 0
        puzzle_idx = 0
        
        for group in tqdm(puzzle_groups, desc=f"Building {split_name}"):
            for puzzle in group:
                base_name = puzzle.id.split('_t')[0]  # Remove aug suffix
                pid = puzzle_id_map.get(base_name, 0)
                
                # HRM-style: one example per puzzle gets no translation
                no_aug_idx = np.random.randint(0, len(puzzle.examples))
                
                for ex_idx, (inp, out) in enumerate(puzzle.examples):
                    do_trans = enable_translation and (ex_idx != no_aug_idx)
                    inp_seq, out_seq = np_grid_to_seq_translational_augment(inp, out, do_trans)
                    
                    inputs_list.append(inp_seq)
                    labels_list.append(out_seq)
                    total_examples += 1
                
                puzzle_indices.append(total_examples)
                puzzle_identifiers.append(pid)
                puzzle_idx += 1
            
            group_indices.append(puzzle_idx)
        
        # Save numpy arrays
        np.save(split_dir / 'all__inputs.npy', np.stack(inputs_list))
        np.save(split_dir / 'all__labels.npy', np.stack(labels_list))
        np.save(split_dir / 'all__puzzle_indices.npy', np.array(puzzle_indices, dtype=np.int32))
        np.save(split_dir / 'all__puzzle_identifiers.npy', np.array(puzzle_identifiers, dtype=np.int32))
        np.save(split_dir / 'all__group_indices.npy', np.array(group_indices, dtype=np.int32))
        
        # Metadata
        metadata = {
            'seq_len': ARC_SEQ_LEN,
            'vocab_size': ARC_VOCAB_SIZE,
            'pad_id': 0,
            'ignore_label_id': 0,
            'blank_identifier_id': 0,
            'num_puzzle_identifiers': next_puzzle_id,
            'total_groups': len(puzzle_groups),
            'total_puzzles': puzzle_idx,
            'total_examples': total_examples,
            'mean_puzzle_examples': total_examples / max(1, puzzle_idx),
        }
        
        with open(split_dir / 'dataset.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  {split_name}: {total_examples} examples from {puzzle_idx} puzzles")
    
    # Save ID mapping
    id_to_name = {v: k for k, v in puzzle_id_map.items()}
    with open(output_path / 'identifiers.json', 'w') as f:
        json.dump([id_to_name.get(i, '<blank>') for i in range(next_puzzle_id)], f)
    
    print(f"\n✓ Dataset saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build ARC dataset with HRM-style augmentations")
    parser.add_argument('--version', choices=['arc-1', 'arc-2'], default='arc-2',
                        help='ARC version (arc-1 or arc-2)')
    parser.add_argument('--output-dir', type=str, default='data/arc-2-aug-1000',
                        help='Output directory')
    parser.add_argument('--num-aug', type=int, default=1000,
                        help='Number of augmentations per puzzle (0 for minimal)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--raw-dir', type=str, default='data/raw-data',
                        help='Directory for raw ARC data')
    args = parser.parse_args()
    
    # Download raw data if needed
    raw_path = Path(args.raw_dir)
    
    if args.version == 'arc-2':
        repo_url = "https://github.com/arcprize/ARC-AGI-2"
        repo_name = "ARC-AGI-2"
    else:
        repo_url = "https://github.com/arcprize/ARC-AGI"
        repo_name = "ARC-AGI"
    
    repo_path = raw_path / repo_name
    
    if not repo_path.exists():
        print(f"Cloning {repo_url}...")
        raw_path.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--depth', '1', repo_url, str(repo_path)], check=True)
    
    # Build dataset
    build_arc_dataset(
        dataset_dirs=[str(repo_path / 'data')],
        output_dir=args.output_dir,
        num_aug=args.num_aug,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()

