#!/usr/bin/env python3
"""
PoT Sudoku Benchmark - Master-Level Sudoku Solver
==================================================

Replicates the HRM paper's Sudoku demo using PoT architecture.
Train a master-level Sudoku solver with 1000 extreme puzzles in ~10 hours.

Based on: https://github.com/sapientinc/HRM

Task: Given a 9x9 Sudoku puzzle with blanks (0), output the complete solution.
Input:  81 tokens (flattened 9x9), vocab 0-9 (0=blank)
Output: 81 tokens (complete solution), vocab 1-9

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
import argparse
import json
import csv
from tqdm import tqdm
from pathlib import Path

from src.pot.core.hrm_controller import HRMPointerController, HRMState
from src.pot.models.puzzle_embedding import PuzzleEmbedding
from src.pot.models.adaptive_halting import QHaltingController
from src.pot.models.hrm_layers import RMSNorm, SwiGLU


# ============================================================================
# Sudoku Dataset
# ============================================================================

class SudokuDataset(Dataset):
    """
    Sudoku dataset loader for HRM-format data.
    
    Supports two formats:
    1. HuggingFace CSV: sapientinc/sudoku-extreme
    2. Pre-built numpy: data/sudoku-extreme-1k-aug-1000/
    """
    
    def __init__(
        self, 
        data_dir: str, 
        split: str = 'train',
        augment: bool = False,
        num_aug: int = 0
    ):
        self.data_dir = Path(data_dir) / split
        self.augment = augment
        self.num_aug = num_aug
        
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
                # Each row is a unique puzzle (no augmentation loaded)
                self.puzzle_indices = np.arange(len(self.inputs))
        else:
            raise FileNotFoundError(
                f"Dataset not found at {self.data_dir}. "
                f"Run: python vendor/hrm/dataset/build_sudoku_dataset.py "
                f"--output-dir {data_dir}"
            )
        
        print(f"[{split}] Loaded {len(self.inputs)} Sudoku puzzles")
        print(f"  Input shape: {self.inputs.shape}")
        print(f"  Unique puzzles: {len(np.unique(self.puzzle_indices))}")
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        inp = torch.LongTensor(self.inputs[idx])  # (81,)
        label = torch.LongTensor(self.labels[idx])  # (81,)
        puzzle_id = torch.tensor(self.puzzle_indices[idx], dtype=torch.long)
        
        return {
            'input': inp,
            'label': label,
            'puzzle_id': puzzle_id,
        }


def download_sudoku_dataset(output_dir: str, subsample_size: int = 1000, num_aug: int = 1000):
    """Download and build Sudoku dataset from HuggingFace."""
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
            
            for puzzle_idx, (puzzle, solution, difficulty) in enumerate(tqdm(puzzles, desc=f"Processing {split}")):
                # Convert string to numpy array
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


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment Sudoku by applying validity-preserving transformations.
    
    Transformations:
    - Digit permutation (1-9 -> random mapping)
    - Row permutation within bands
    - Column permutation within stacks
    - Transpose
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


# ============================================================================
# PoH Sudoku Solver Model
# ============================================================================

class PoHSudokuSolver(nn.Module):
    """
    PoH-based Sudoku solver with HRM-style components.
    
    Architecture:
    - Puzzle embeddings for per-instance specialization
    - HRM pointer controller for hierarchical routing
    - Q-halting for adaptive computation
    - Post-norm transformer layers with SwiGLU
    """
    
    def __init__(
        self,
        vocab_size: int = 10,  # 0=blank, 1-9=digits
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.1,
        R: int = 8,  # Refinement iterations
        T: int = 4,  # HRM outer period
        num_puzzles: int = 1000,
        puzzle_emb_dim: int = 512,
        max_halting_steps: int = 16,
        latent_len: int = 16,
        latent_k: int = 3,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.R = R
        self.n_layers = n_layers
        self.seq_len = 81  # 9x9 Sudoku
        
        # Puzzle embeddings
        self.puzzle_emb = PuzzleEmbedding(num_puzzles, puzzle_emb_dim, init_std=0.02)
        self.puzzle_emb_len = (puzzle_emb_dim + d_model - 1) // d_model
        
        # Latent tokens (TRM-style)
        self.latent_len = latent_len
        self.latent_k = latent_k
        self.latent_init = nn.Parameter(torch.randn(1, latent_len, d_model) * 0.02)
        
        # Q-halting
        self.q_halt_controller = QHaltingController(d_model, max_steps=max_halting_steps)
        self.max_halting_steps = max_halting_steps
        
        # Input embedding
        self.input_embed = nn.Embedding(vocab_size, d_model)
        
        # Position embedding (seq + puzzle + latent)
        max_len = self.seq_len + self.puzzle_emb_len + latent_len
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.pre_norm = nn.LayerNorm(d_model)
        
        # HRM controller
        self.hrm_controller = HRMPointerController(
            d_model=d_model,
            n_heads=n_heads,
            T=T,
            dropout=dropout
        )
        
        # Transformer layers (Post-norm + SwiGLU)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            SwiGLU(d_model, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm1_layers = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.norm2_layers = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
        
        # Latent cross-attention
        self.latent_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.latent_ffn = SwiGLU(d_model, d_ff, dropout)
        self.latent_norm1 = RMSNorm(d_model)
        self.latent_norm2 = RMSNorm(d_model)
        self.latent_drop = nn.Dropout(dropout)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def _encode_once(self, x: torch.Tensor, hrm_state: Optional[HRMState] = None):
        """Single encoding pass with HRM routing."""
        B, T, D = x.shape
        
        # Get routing weights
        route_weights, hrm_state, _ = self.hrm_controller(x, state=hrm_state)
        
        # Apply transformer layers with head routing
        for attn, ffn, norm1, norm2, drop in zip(
            self.attn_layers, self.ffn_layers, 
            self.norm1_layers, self.norm2_layers, self.dropout_layers
        ):
            # Attention with head routing
            attn_out, _ = attn(x, x, x, need_weights=False)
            d_head = D // attn.num_heads
            attn_out_heads = attn_out.view(B, T, attn.num_heads, d_head)
            route_exp = route_weights.unsqueeze(1).unsqueeze(-1)
            attn_out_routed = (attn_out_heads * route_exp).view(B, T, D)
            x = norm1(x + drop(attn_out_routed))
            
            # FFN
            x = norm2(x + drop(ffn(x)))
        
        return x, hrm_state
    
    def forward(self, input_seq: torch.Tensor, puzzle_ids: torch.Tensor):
        """
        Forward pass with iterative refinement.
        
        Args:
            input_seq: (B, 81) - flattened Sudoku grid
            puzzle_ids: (B,) - puzzle identifiers
        
        Returns:
            logits: (B, 81, vocab_size)
            q_halt, q_continue: halting Q-values
            actual_steps: number of refinement steps
        """
        B = input_seq.size(0)
        device = input_seq.device
        
        # Puzzle embedding
        puzzle_emb = self.puzzle_emb(puzzle_ids)
        pad_size = self.puzzle_emb_len * self.d_model - puzzle_emb.size(-1)
        if pad_size > 0:
            puzzle_emb = F.pad(puzzle_emb, (0, pad_size))
        puzzle_emb = puzzle_emb.view(B, self.puzzle_emb_len, self.d_model)
        
        # Input embedding
        x_grid = self.input_embed(input_seq)
        x_grid_ref = x_grid  # Keep reference
        
        # Latent tokens
        latent_cur = self.latent_init.expand(B, -1, -1)
        
        # Initialize HRM state
        hrm_state = self.hrm_controller.init_state(B, device)
        actual_steps = self.max_halting_steps
        x_out = None
        
        # Iterative refinement
        for step in range(1, self.max_halting_steps + 1):
            # Inner latent updates
            for _ in range(self.latent_k):
                ctx = torch.cat([puzzle_emb, x_grid_ref], dim=1)
                lat_attn, _ = self.latent_attn(latent_cur, ctx, ctx, need_weights=False)
                latent_cur = self.latent_norm1(latent_cur + self.latent_drop(lat_attn))
                latent_cur = self.latent_norm2(latent_cur + self.latent_drop(self.latent_ffn(latent_cur)))
            
            # Build sequence
            x_step = torch.cat([puzzle_emb, latent_cur, x_grid_ref], dim=1)
            x_step = x_step + self.pos_embed[:, :x_step.size(1), :]
            x_step = self.pre_norm(x_step)
            
            # Encode
            x_out, hrm_state = self._encode_once(x_step, hrm_state)
            
            # Check halting
            q_halt, q_continue = self.q_halt_controller(x_out)
            should_halt = self.q_halt_controller.should_halt(q_halt, q_continue, step, self.training)
            
            if should_halt.all():
                actual_steps = step
                break
            
            # Detach for O(1) memory
            latent_cur = latent_cur.detach()
            x_out = x_out.detach()
            if hrm_state is not None:
                hrm_state = HRMState(
                    z_L=hrm_state.z_L.detach(),
                    z_H=hrm_state.z_H.detach(),
                    step=hrm_state.step
                )
        
        # Extract output (remove puzzle + latent tokens)
        x = x_out[:, self.puzzle_emb_len + self.latent_len:, :]
        logits = self.output_proj(x)
        
        return logits, q_halt, q_continue, actual_steps


# ============================================================================
# Baseline Model (Standard Transformer)
# ============================================================================

class BaselineSudokuSolver(nn.Module):
    """Standard Transformer baseline for comparison."""
    
    def __init__(
        self,
        vocab_size: int = 10,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.seq_len = 81
        
        self.input_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_seq: torch.Tensor, puzzle_ids: torch.Tensor = None):
        x = self.input_embed(input_seq) + self.pos_embed
        x = self.encoder(x)
        x = self.final_norm(x)
        logits = self.output_proj(x)
        return logits, None, None, 1


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, dataloader, optimizer, puzzle_optimizer, device, epoch, use_poh=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct_cells = 0
    total_cells = 0
    correct_grids = 0
    total_grids = 0
    total_steps = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        inp = batch['input'].to(device)
        label = batch['label'].to(device)
        puzzle_ids = batch['puzzle_id'].to(device)
        
        optimizer.zero_grad()
        if puzzle_optimizer:
            puzzle_optimizer.zero_grad()
        
        logits, q_halt, q_continue, steps = model(inp, puzzle_ids)
        
        # CE loss (ignore blank cells in input for loss? No - predict all)
        lm_loss = F.cross_entropy(
            logits.view(-1, model.vocab_size),
            label.view(-1)
        )
        
        # Q-halt loss (if PoH)
        if use_poh and q_halt is not None:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                is_correct = (preds == label).all(dim=1).float()
            
            q_halt_loss = F.binary_cross_entropy_with_logits(q_halt, is_correct)
            loss = lm_loss + 0.5 * q_halt_loss
        else:
            loss = lm_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if puzzle_optimizer:
            puzzle_optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct_cells += (preds == label).sum().item()
        total_cells += label.numel()
        correct_grids += (preds == label).all(dim=1).sum().item()
        total_grids += label.size(0)
        total_steps += steps
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cell_acc': f'{100*correct_cells/total_cells:.1f}%',
            'grid_acc': f'{100*correct_grids/total_grids:.1f}%',
        })
    
    return {
        'loss': total_loss / len(dataloader),
        'cell_acc': 100 * correct_cells / total_cells,
        'grid_acc': 100 * correct_grids / total_grids,
        'avg_steps': total_steps / len(dataloader),
    }


def evaluate(model, dataloader, device, use_poh=True):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct_cells = 0
    total_cells = 0
    correct_grids = 0
    total_grids = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inp = batch['input'].to(device)
            label = batch['label'].to(device)
            puzzle_ids = batch['puzzle_id'].to(device)
            
            logits, _, _, _ = model(inp, puzzle_ids)
            
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), label.view(-1))
            total_loss += loss.item()
            
            preds = logits.argmax(dim=-1)
            correct_cells += (preds == label).sum().item()
            total_cells += label.numel()
            correct_grids += (preds == label).all(dim=1).sum().item()
            total_grids += label.size(0)
    
    return {
        'loss': total_loss / len(dataloader),
        'cell_acc': 100 * correct_cells / total_cells,
        'grid_acc': 100 * correct_grids / total_grids,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='PoT Sudoku Benchmark')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='data/sudoku-extreme-1k-aug-1000',
                       help='Path to Sudoku dataset')
    parser.add_argument('--download', action='store_true',
                       help='Download and build dataset from HuggingFace')
    parser.add_argument('--subsample', type=int, default=1000,
                       help='Number of puzzles to use (for download)')
    parser.add_argument('--num-aug', type=int, default=1000,
                       help='Augmentations per puzzle (for download)')
    
    # Model
    parser.add_argument('--model', choices=['poh', 'baseline'], default='poh')
    parser.add_argument('--d-model', type=int, default=512)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--n-layers', type=int, default=2, help='Layers for PoH (baseline uses 6)')
    parser.add_argument('--d-ff', type=int, default=2048)
    parser.add_argument('--R', type=int, default=8, help='Refinement iterations')
    parser.add_argument('--T', type=int, default=4, help='HRM outer period')
    parser.add_argument('--max-halt', type=int, default=16, help='Max halting steps')
    
    # Training
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=384)
    parser.add_argument('--lr', type=float, default=7e-5)
    parser.add_argument('--puzzle-lr', type=float, default=7e-5)
    parser.add_argument('--weight-decay', type=float, default=1.0)
    parser.add_argument('--eval-interval', type=int, default=500)
    parser.add_argument('--patience', type=int, default=2000, help='Early stopping patience')
    
    # Output
    parser.add_argument('--output', type=str, default='experiments/results/sudoku_poh')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Download dataset if needed
    if args.download:
        download_sudoku_dataset(args.data_dir, args.subsample, args.num_aug)
    
    # Load data
    train_dataset = SudokuDataset(args.data_dir, 'train')
    test_dataset = SudokuDataset(args.data_dir, 'test')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Get number of unique puzzles
    num_puzzles = len(np.unique(train_dataset.puzzle_indices)) + 1000  # Buffer
    
    # Build model
    use_poh = args.model == 'poh'
    if use_poh:
        model = PoHSudokuSolver(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            R=args.R,
            T=args.T,
            num_puzzles=num_puzzles,
            max_halting_steps=args.max_halt,
        ).to(device)
    else:
        model = BaselineSudokuSolver(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=6,  # More layers for baseline
            d_ff=args.d_ff,
        ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {args.model.upper()}")
    print(f"Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
    
    # Optimizers (HRM style: separate for puzzle embeddings)
    if use_poh:
        puzzle_params = list(model.puzzle_emb.parameters())
        model_params = [p for p in model.parameters() if p not in set(puzzle_params)]
        
        optimizer = torch.optim.AdamW(model_params, lr=args.lr, weight_decay=args.weight_decay)
        puzzle_optimizer = torch.optim.AdamW(puzzle_params, lr=args.puzzle_lr, weight_decay=0.0)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        puzzle_optimizer = None
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Training {args.model.upper()} Sudoku Solver")
    print(f"{'='*60}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"LR: {args.lr}, Weight decay: {args.weight_decay}")
    if use_poh:
        print(f"R={args.R}, T={args.T}, max_halt={args.max_halt}")
    
    best_grid_acc = 0
    patience_counter = 0
    results = []
    
    os.makedirs(args.output, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, puzzle_optimizer, 
            device, epoch, use_poh=use_poh
        )
        
        # Evaluate periodically
        if epoch % args.eval_interval == 0 or epoch == 1:
            test_metrics = evaluate(model, test_loader, device, use_poh=use_poh)
            
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"  Train: Loss={train_metrics['loss']:.4f}, "
                  f"Cell={train_metrics['cell_acc']:.2f}%, Grid={train_metrics['grid_acc']:.2f}%")
            print(f"  Test:  Loss={test_metrics['loss']:.4f}, "
                  f"Cell={test_metrics['cell_acc']:.2f}%, Grid={test_metrics['grid_acc']:.2f}%")
            
            results.append({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_cell_acc': train_metrics['cell_acc'],
                'train_grid_acc': train_metrics['grid_acc'],
                'test_loss': test_metrics['loss'],
                'test_cell_acc': test_metrics['cell_acc'],
                'test_grid_acc': test_metrics['grid_acc'],
            })
            
            # Early stopping
            if test_metrics['grid_acc'] > best_grid_acc:
                best_grid_acc = test_metrics['grid_acc']
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'test_grid_acc': best_grid_acc,
                }, os.path.join(args.output, f'{args.model}_best.pt'))
                print(f"  ✓ New best: {best_grid_acc:.2f}%")
            else:
                patience_counter += args.eval_interval
            
            if patience_counter >= args.patience:
                print(f"\n⚠ Early stopping at epoch {epoch}")
                break
            
            # Check for near-perfect accuracy (HRM target)
            if train_metrics['grid_acc'] >= 99.5:
                print(f"\n🎉 Reached 99.5% training accuracy!")
    
    # Final results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Best Test Grid Accuracy: {best_grid_acc:.2f}%")
    print(f"HRM Paper Target: ~100%")
    
    # Save results
    with open(os.path.join(args.output, f'{args.model}_results.json'), 'w') as f:
        json.dump({
            'model': args.model,
            'parameters': param_count,
            'best_grid_acc': best_grid_acc,
            'config': vars(args),
            'history': results,
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {args.output}")


if __name__ == '__main__':
    main()

