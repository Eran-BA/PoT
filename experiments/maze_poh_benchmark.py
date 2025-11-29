#!/usr/bin/env python3
"""
PoT Maze Benchmark - Grid-to-Grid Maze Solver
==============================================

Benchmarks Hybrid PoT-HRM vs Baseline Transformer on 20x20 mazes.

Task: Given a maze grid (walls, empty, start, goal), predict the solution path.
Input:  N×N grid (flattened to N² tokens)
Output: N×N grid with path marked

Dataset: HRM paper's maze dataset from HuggingFace
- sapientinc/maze-20x20-hard (20×20 mazes)
- sapientinc/maze-30x30-hard (30×30 mazes)

Key HRM implementation details matched:
1. Embedding scaling by sqrt(d_model)
2. H_init/L_init as buffers (NOT learned)
3. Single shared puzzle embedding (id=0 for all)
4. No dropout in transformer blocks
5. Post-norm architecture with RMSNorm and SwiGLU
6. Only final iteration gets gradients

Usage:
    # Download and train on 20x20 mazes
    python experiments/maze_poh_benchmark.py --download --maze-size 20

    # Benchmark hybrid vs baseline
    python experiments/maze_poh_benchmark.py --model hybrid --maze-size 20
    python experiments/maze_poh_benchmark.py --model baseline --maze-size 20

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import sys
import os
import math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import argparse
import json
from tqdm import tqdm
from pathlib import Path

from src.pot.core.hrm_controller import HRMPointerController, HRMState
from src.pot.models import (
    ReasoningModule,
    HybridHRMBase,
    PuzzleEmbedding,
    RMSNorm,
    SwiGLU,
)


# ============================================================================
# Maze Dataset
# ============================================================================

# HRM maze character set: # (wall), space (empty), S (start), G (goal), o (path)
MAZE_CHARSET = " #SGoX"  # PAD=0, then each char gets an ID


class MazeDataset(Dataset):
    """
    Maze dataset loader for HRM-format data.
    
    Uses single shared puzzle embedding (id=0) like HRM.
    """
    
    def __init__(self, data_dir: str, split: str = 'train'):
        self.data_dir = Path(data_dir) / split
        self.split = split
        
        # Load numpy format
        inputs_path = self.data_dir / 'all__inputs.npy'
        labels_path = self.data_dir / 'all__labels.npy'
        
        if inputs_path.exists():
            self.inputs = np.load(inputs_path)
            self.labels = np.load(labels_path)
        else:
            raise FileNotFoundError(
                f"Dataset not found at {self.data_dir}. "
                f"Run with --download to fetch from HuggingFace."
            )
        
        # Infer grid size from sequence length
        self.seq_len = self.inputs.shape[1]
        self.grid_size = int(np.sqrt(self.seq_len))
        
        print(f"[{split}] Loaded {len(self.inputs)} mazes")
        print(f"  Grid size: {self.grid_size}×{self.grid_size}")
        print(f"  Sequence length: {self.seq_len}")
        print(f"  Input vocab range: {self.inputs.min()}-{self.inputs.max()}")
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        inp = torch.LongTensor(self.inputs[idx])
        label = torch.LongTensor(self.labels[idx])
        # HRM uses puzzle_id=0 for all puzzles (single shared embedding)
        puzzle_id = torch.tensor(0, dtype=torch.long)
        
        return {
            'input': inp,
            'label': label,
            'puzzle_id': puzzle_id,
        }


def download_maze_dataset(output_dir: str, maze_size: int = 20, aug: bool = True):
    """
    Download maze dataset from HuggingFace.
    
    Uses sapientinc/maze-{size}x{size}-hard-1k repos.
    """
    from huggingface_hub import hf_hub_download
    import csv
    
    output_path = Path(output_dir)
    
    # Choose repo based on maze size
    if maze_size == 20:
        source_repo = "sapientinc/maze-20x20-hard-1k"
    elif maze_size == 30:
        source_repo = "sapientinc/maze-30x30-hard-1k"
    else:
        raise ValueError(f"Unsupported maze size: {maze_size}. Use 20 or 30.")
    
    print(f"Downloading maze dataset from {source_repo}...")
    
    # HRM character set
    CHARSET = "# SGo"  # wall, space, start, goal, path
    char2id = np.zeros(256, dtype=np.uint8)
    char2id[np.array(list(map(ord, CHARSET)))] = np.arange(len(CHARSET)) + 1
    
    for split in ['train', 'test']:
        split_dir = output_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Download CSV
        print(f"  Downloading {split}...")
        csv_path = hf_hub_download(source_repo, f"{split}.csv", repo_type="dataset")
        
        # Parse CSV
        inputs = []
        labels = []
        
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in tqdm(reader, desc=f"  [{split}]"):
                if len(row) >= 4:
                    source, q, a, rating = row[:4]
                else:
                    continue
                
                grid_size = int(len(q) ** 0.5)
                inp = np.frombuffer(q.encode(), dtype=np.uint8).reshape(grid_size, grid_size)
                lab = np.frombuffer(a.encode(), dtype=np.uint8).reshape(grid_size, grid_size)
                
                inputs.append(inp)
                labels.append(lab)
        
        print(f"  [{split}] Loaded {len(inputs)} mazes from CSV")
        
        # Apply augmentation for training (8x dihedral group)
        if split == 'train' and aug:
            aug_inputs = []
            aug_labels = []
            for inp, lab in zip(inputs, labels):
                for k in range(8):
                    aug_inputs.append(dihedral_transform(inp, k))
                    aug_labels.append(dihedral_transform(lab, k))
            inputs = aug_inputs
            labels = aug_labels
            print(f"  [{split}] After 8x augmentation: {len(inputs)} mazes")
        
        # Convert to numpy with character mapping
        inputs_np = np.vstack([char2id[inp.flatten()] for inp in inputs])
        labels_np = np.vstack([char2id[lab.flatten()] for lab in labels])
        
        # Save
        np.save(split_dir / 'all__inputs.npy', inputs_np)
        np.save(split_dir / 'all__labels.npy', labels_np)
        
        # Save metadata
        metadata = {
            'seq_len': inputs_np.shape[1],
            'vocab_size': len(CHARSET) + 1,
            'grid_size': int(np.sqrt(inputs_np.shape[1])),
            'num_samples': len(inputs_np),
        }
        with open(split_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f)
    
    print(f"✓ Dataset saved to {output_path}")


def dihedral_transform(grid: np.ndarray, k: int) -> np.ndarray:
    """Apply one of 8 dihedral transformations (rotations + reflections)."""
    if k == 0:
        return grid
    elif k == 1:
        return np.rot90(grid, 1)
    elif k == 2:
        return np.rot90(grid, 2)
    elif k == 3:
        return np.rot90(grid, 3)
    elif k == 4:
        return np.fliplr(grid)
    elif k == 5:
        return np.fliplr(np.rot90(grid, 1))
    elif k == 6:
        return np.flipud(grid)
    elif k == 7:
        return np.flipud(np.rot90(grid, 1))
    return grid


# ============================================================================
# Hybrid PoT-HRM Maze Solver
# ============================================================================

class HybridMazeSolver(HybridHRMBase):
    """
    Hybrid PoT-HRM Maze solver.
    
    Extends HybridHRMBase with maze-specific embeddings and output.
    """
    
    def __init__(
        self,
        vocab_size: int = 7,  # PAD + 6 maze chars
        d_model: int = 512,
        n_heads: int = 8,
        H_layers: int = 2,
        L_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.0,
        H_cycles: int = 2,
        L_cycles: int = 8,
        T: int = 4,
        seq_len: int = 400,  # 20×20
        num_puzzles: int = 1,  # Single shared embedding
        puzzle_emb_dim: int = 512,
    ):
        # Initialize base class
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            H_layers=H_layers,
            L_layers=L_layers,
            d_ff=d_ff,
            seq_len=seq_len,
            H_cycles=H_cycles,
            L_cycles=L_cycles,
            dropout=dropout,
            T=T,
        )
        
        self.vocab_size = vocab_size
        embed_init_std = 1.0 / self.embed_scale
        
        # Maze-specific: Input embedding
        self.input_embed = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.input_embed.weight, mean=0, std=embed_init_std)
        
        # Maze-specific: Puzzle embedding (zero init like HRM)
        self.puzzle_emb = PuzzleEmbedding(num_puzzles, puzzle_emb_dim, init_std=0.0)
        self.puzzle_emb_proj = nn.Linear(puzzle_emb_dim, d_model) if puzzle_emb_dim != d_model else nn.Identity()
        
        # Maze-specific: Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * embed_init_std)
        
        # Maze-specific: Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_seq: torch.Tensor, puzzle_ids: torch.Tensor):
        """Forward pass for maze solving."""
        # Compute input embedding
        input_emb = self.input_embed(input_seq) + self.pos_embed[:, :input_seq.size(1), :]
        
        # Add puzzle embedding
        max_puzzle_id = self.puzzle_emb.num_puzzles - 1
        valid_mask = puzzle_ids <= max_puzzle_id
        clamped_ids = puzzle_ids.clamp(0, max_puzzle_id)
        puzzle_emb = self.puzzle_emb(clamped_ids)
        puzzle_emb = puzzle_emb * valid_mask.unsqueeze(-1).float()
        puzzle_emb = self.puzzle_emb_proj(puzzle_emb)
        input_emb = input_emb + puzzle_emb.unsqueeze(1)
        
        # Scale embeddings
        input_emb = self.embed_scale * input_emb
        
        # Run reasoning loop
        hidden, q_halt, q_continue, steps = self.reasoning_loop(input_emb)
        
        # Output projection
        logits = self.output_proj(hidden)
        
        return logits, q_halt, q_continue, steps


# ============================================================================
# Baseline Model (Standard Transformer)
# ============================================================================

class BaselineMazeSolver(nn.Module):
    """Standard Transformer baseline for comparison."""
    
    def __init__(
        self,
        vocab_size: int = 7,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        seq_len: int = 400,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Embeddings
        self.input_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        # Standard transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_seq: torch.Tensor, puzzle_ids: torch.Tensor = None):
        """Forward pass."""
        x = self.input_embed(input_seq) + self.pos_embed[:, :input_seq.size(1), :]
        x = self.encoder(x)
        x = self.final_norm(x)
        logits = self.output_proj(x)
        
        # Return dummy q values for API compatibility
        B = input_seq.size(0)
        device = input_seq.device
        return logits, torch.zeros(B, device=device), torch.zeros(B, device=device), 1


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, epoch, scheduler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct_cells = 0
    total_cells = 0
    correct_grids = 0
    total_grids = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        inp = batch['input'].to(device)
        label = batch['label'].to(device)
        puzzle_ids = batch['puzzle_id'].to(device)
        
        optimizer.zero_grad()
        
        logits, _, _, _ = model(inp, puzzle_ids)
        
        # Cross-entropy loss on all cells
        loss = F.cross_entropy(logits.view(-1, model.vocab_size), label.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct_cells += (preds == label).sum().item()
        total_cells += label.numel()
        correct_grids += (preds == label).all(dim=1).sum().item()
        total_grids += label.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cell_acc': f'{100*correct_cells/total_cells:.1f}%',
            'grid_acc': f'{100*correct_grids/total_grids:.1f}%',
        })
    
    return {
        'loss': total_loss / len(dataloader),
        'cell_acc': 100 * correct_cells / total_cells,
        'grid_acc': 100 * correct_grids / total_grids,
    }


def evaluate(model, dataloader, device):
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
    parser = argparse.ArgumentParser(description='PoT Maze Benchmark')
    
    # Data
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Path to maze dataset (auto-set based on maze-size)')
    parser.add_argument('--download', action='store_true',
                       help='Download dataset from HuggingFace')
    parser.add_argument('--maze-size', type=int, default=20, choices=[20, 30],
                       help='Maze size (20 or 30)')
    parser.add_argument('--no-aug', action='store_true',
                       help='Disable training data augmentation')
    
    # Model
    parser.add_argument('--model', choices=['hybrid', 'baseline'], default='hybrid')
    parser.add_argument('--d-model', type=int, default=512)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--d-ff', type=int, default=2048)
    # Hybrid model args
    parser.add_argument('--H-cycles', type=int, default=2, help='Outer iterations')
    parser.add_argument('--L-cycles', type=int, default=8, help='Inner iterations')
    parser.add_argument('--H-layers', type=int, default=2, help='H_level layers')
    parser.add_argument('--L-layers', type=int, default=2, help='L_level layers')
    parser.add_argument('--T', type=int, default=4, help='HRM period')
    # Baseline args
    parser.add_argument('--n-layers', type=int, default=6, help='Baseline layers')
    
    # Training
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=7e-5, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1.0)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--eval-interval', type=int, default=100)
    
    # Output
    parser.add_argument('--output', type=str, default='experiments/results/maze_poh')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set data directory based on maze size
    if args.data_dir is None:
        args.data_dir = f'data/maze-{args.maze_size}x{args.maze_size}-hard'
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Download dataset if needed
    if args.download:
        download_maze_dataset(args.data_dir, args.maze_size, aug=not args.no_aug)
    
    # Load data
    train_dataset = MazeDataset(args.data_dir, 'train')
    test_dataset = MazeDataset(args.data_dir, 'test')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    seq_len = train_dataset.seq_len
    
    # Build model
    if args.model == 'hybrid':
        model = HybridMazeSolver(
            vocab_size=7,  # PAD + 6 maze chars
            d_model=args.d_model,
            n_heads=args.n_heads,
            H_layers=args.H_layers,
            L_layers=args.L_layers,
            d_ff=args.d_ff,
            H_cycles=args.H_cycles,
            L_cycles=args.L_cycles,
            T=args.T,
            seq_len=seq_len,
        ).to(device)
        print(f"Hybrid model: H_cycles={args.H_cycles}, L_cycles={args.L_cycles}")
    else:
        model = BaselineMazeSolver(
            vocab_size=7,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            seq_len=seq_len,
        ).to(device)
        print(f"Baseline model: {args.n_layers} layers")
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {args.model.upper()}")
    print(f"Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
    print(f"Maze size: {args.maze_size}×{args.maze_size}")
    
    # Optimizer with cosine schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    total_steps = args.epochs * len(train_loader)
    
    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training
    print(f"\n{'='*60}")
    print(f"Training {args.model.upper()} Maze Solver")
    print(f"{'='*60}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"LR: {args.lr}, Weight decay: {args.weight_decay}")
    
    best_grid_acc = 0
    results = []
    os.makedirs(args.output, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, scheduler)
        
        # Evaluate periodically
        if epoch % args.eval_interval == 0 or epoch == 1:
            test_metrics = evaluate(model, test_loader, device)
            
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
            
            # Save best model
            if test_metrics['grid_acc'] > best_grid_acc:
                best_grid_acc = test_metrics['grid_acc']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'test_grid_acc': best_grid_acc,
                }, os.path.join(args.output, f'{args.model}_best.pt'))
                print(f"  ✓ New best: {best_grid_acc:.2f}%")
            
            # Check for near-perfect accuracy
            if train_metrics['grid_acc'] >= 99.5:
                print(f"\n🎉 Reached 99.5% training accuracy!")
    
    # Final results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Model: {args.model.upper()}")
    print(f"Best Test Grid Accuracy: {best_grid_acc:.2f}%")
    
    # Save results
    with open(os.path.join(args.output, f'{args.model}_results.json'), 'w') as f:
        json.dump({
            'model': args.model,
            'maze_size': args.maze_size,
            'params': param_count,
            'best_test_grid_acc': best_grid_acc,
            'config': vars(args),
            'history': results,
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {args.output}")


if __name__ == '__main__':
    main()

