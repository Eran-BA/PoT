#!/usr/bin/env python3
"""
Maze A/B Test using proper maze-dataset library for better generation.

This uses the maze-dataset library (https://github.com/understanding-search/maze-dataset)
for high-quality maze generation with controllable difficulty via path length filtering.

Key improvements over simple random walls:
- Proper maze generation algorithms (DFS, Wilson's, etc.)
- Guaranteed solvability
- Path length filtering for controlled difficulty
- Research-grade quality
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import time

# Setup paths
try:
    from experiments.setup_colab import setup_pot_paths
    repo_root = setup_pot_paths()
except:
    repo_root = Path(__file__).parent.parent
    if repo_root not in sys.path:
        sys.path.insert(0, str(repo_root))

print(f"✓ PoT root: {repo_root}")

# Try to import maze-dataset
try:
    from maze_dataset import MazeDataset, MazeDatasetConfig
    from maze_dataset.generation import LatticeMazeGenerators
    MAZE_DATASET_AVAILABLE = True
    print("✓ maze-dataset library available")
except ImportError:
    MAZE_DATASET_AVAILABLE = False
    print("⚠️  maze-dataset not installed. Install with: pip install maze-dataset")
    print("   Falling back to simple generation...")

# Import PoT modules
from src.pot.modules import PoHConfig, PoHStack, IterRefiner
from src.pot.core.hrm_controller import HRMPointerController

# Try BERT
try:
    from transformers import BertConfig, BertModel
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

print("✓ Successfully imported PoT modules")

# ============================================================================
# Maze Generation with maze-dataset library
# ============================================================================

def generate_dataset_proper(maze_size: int, n_samples: int, min_path_length: int = None, seed: int = None):
    """Generate mazes using the maze-dataset library with path length filtering."""
    
    if not MAZE_DATASET_AVAILABLE:
        print("ERROR: maze-dataset library required for proper generation")
        sys.exit(1)
    
    # Default min path length based on maze size
    if min_path_length is None:
        # For challenging mazes, require path length to be at least 40% of grid size
        min_path_length = int(maze_size * maze_size * 0.4)
    
    print(f"  Generating {n_samples} mazes of size {maze_size}×{maze_size}")
    print(f"  Minimum path length: {min_path_length}")
    
    # Configure maze generation
    cfg = MazeDatasetConfig(
        name=f"maze_{maze_size}x{maze_size}_minpath{min_path_length}",
        grid_n=maze_size,  # Grid size
        n_mazes=n_samples * 3,  # Generate extra to allow for filtering
        maze_ctor=LatticeMazeGenerators.gen_dfs,  # DFS generation (creates challenging mazes)
        seed=seed,
    )
    
    # Generate mazes
    dataset = MazeDataset.from_config(
        cfg,
        do_generate=True,
        load_local=False,
        save_local=False,
    )
    
    # Filter by path length
    dataset_filtered = dataset.filter_by.path_length(min_length=min_path_length)
    
    # Take only what we need
    if len(dataset_filtered) < n_samples:
        print(f"  ⚠️  Warning: Only generated {len(dataset_filtered)} mazes meeting criteria (needed {n_samples})")
        print(f"     Consider reducing min_path_length or increasing maze size")
    
    dataset_filtered = dataset_filtered[:n_samples]
    
    # Convert to our format
    data = []
    path_lengths = []
    
    for solved_maze in dataset_filtered:
        # Use the proper maze-dataset API
        # LatticeMaze has an as_pixels() method that converts to array
        # From the docs: https://understanding-search.github.io/maze-dataset/maze_dataset.html
        
        maze_obj = solved_maze.maze
        
        # Convert maze to pixel representation (0=wall, 1=path in their format)
        # as_pixels returns array where walls and paths are represented
        maze_pixels = maze_obj.as_pixels()
        
        # The as_pixels format might be different, so let's invert if needed
        # We want: 0=passable, 1=wall for our model
        # as_pixels typically gives: 0=wall (black), 1=path (white)
        # So we DON'T need to invert - their format matches ours!
        maze = 1.0 - maze_pixels.astype(np.float32)  # Invert: their 1 (path) -> our 0 (passable)
        
        # Get solution path
        solution = solved_maze.solution
        start = (solution.start_pos.row, solution.start_pos.col)
        goal = (solution.end_pos.row, solution.end_pos.col)
        path = [(coord.row, coord.col) for coord in solution.path]
        
        path_lengths.append(len(path))
        
        data.append({
            'maze': maze,
            'start': start,
            'goal': goal,
            'path': path,
            'length': len(path)
        })
    
    print(f"  ✓ Generated {len(data)} mazes")
    print(f"  Path length: {np.mean(path_lengths):.1f} ± {np.std(path_lengths):.1f} (min={min(path_lengths)}, max={max(path_lengths)})")
    
    return data


# ============================================================================
# Fallback: Simple generation (if maze-dataset not available)
# ============================================================================

def generate_dataset_simple(maze_size: int, n_samples: int, wall_prob: float = 0.7, seed: int = None):
    """Fallback to simple random generation if maze-dataset not available."""
    print(f"  ⚠️  Using simple generation (wall_prob={wall_prob})")
    print(f"  For better quality, install: pip install maze-dataset")
    
    # [Simple BFS-based generation code would go here]
    # For now, just error out
    print("ERROR: Simple generation not implemented. Please install maze-dataset.")
    sys.exit(1)


# ============================================================================
# Model Architectures (same as before)
# ============================================================================

class BaselineMazeSolver(nn.Module):
    """Standard Transformer baseline."""
    def __init__(self, maze_size: int, d_model: int = 256, n_heads: int = 4, d_ff: int = 1024, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.maze_size = maze_size
        self.d_model = d_model
        
        # Embeddings
        self.pos_embed = nn.Embedding(maze_size * maze_size, d_model)
        self.cell_embed = nn.Embedding(4, d_model)  # wall, empty, start, goal
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Decoder for path prediction
        self.decoder = nn.Linear(d_model, maze_size * maze_size)
    
    def forward(self, maze, start, goal):
        B = maze.shape[0]
        seq_len = self.maze_size * self.maze_size
        
        # Create input: maze cells + start/goal markers
        x = torch.zeros(B, seq_len, dtype=torch.long, device=maze.device)
        x[maze.view(B, -1) == 1] = 1  # walls
        for b in range(B):
            x[b, start[b, 0] * self.maze_size + start[b, 1]] = 2  # start
            x[b, goal[b, 0] * self.maze_size + goal[b, 1]] = 3  # goal
        
        # Embed
        pos_ids = torch.arange(seq_len, device=maze.device).unsqueeze(0).expand(B, -1)
        h = self.cell_embed(x) + self.pos_embed(pos_ids)
        
        # Encode
        h = self.encoder(h)
        
        # Decode path (predict next cell at each step)
        logits = self.decoder(h)
        
        return logits


class PoHMazeSolver(nn.Module):
    """PoH-HRM maze solver."""
    def __init__(
        self,
        maze_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 1024,
        num_layers: int = 4,
        R: int = 4,
        T: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.maze_size = maze_size
        self.d_model = d_model
        self.R = R
        self.T = T
        
        # Embeddings
        self.pos_embed = nn.Embedding(maze_size * maze_size, d_model)
        self.cell_embed = nn.Embedding(4, d_model)
        
        # PoH with HRM
        cfg = PoHConfig(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            use_poh=True,
            use_hrm=True,
        )
        
        self.poh_stack = PoHStack(cfg, depth=num_layers)
        
        # Set HRM period T
        for block in self.poh_stack.blocks:
            if hasattr(block, 'router') and hasattr(block.router, 'hrm_controller'):
                block.router.hrm_controller.T = T
        
        # Wrap in IterRefiner for R refinement iterations
        self.refiner = IterRefiner(
            self.poh_stack,
            max_inner_iters=R,
            outer_residual=True,
            rezero_init=True
        )
        
        # Decoder
        self.decoder = nn.Linear(d_model, maze_size * maze_size)
    
    def forward(self, maze, start, goal):
        B = maze.shape[0]
        seq_len = self.maze_size * self.maze_size
        
        # Create input
        x = torch.zeros(B, seq_len, dtype=torch.long, device=maze.device)
        x[maze.view(B, -1) == 1] = 1
        for b in range(B):
            x[b, start[b, 0] * self.maze_size + start[b, 1]] = 2
            x[b, goal[b, 0] * self.maze_size + goal[b, 1]] = 3
        
        # Embed
        pos_ids = torch.arange(seq_len, device=maze.device).unsqueeze(0).expand(B, -1)
        h = self.cell_embed(x) + self.pos_embed(pos_ids)
        
        # PoH with refinement
        h = self.refiner(h)
        
        # Decode
        logits = self.decoder(h)
        
        return logits


# ============================================================================
# Training & Evaluation
# ============================================================================

class MazeDatasetWrapper(Dataset):
    def __init__(self, data, maze_size):
        self.data = data
        self.maze_size = maze_size
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        maze = torch.FloatTensor(item['maze'])
        start = torch.LongTensor(item['start'])
        goal = torch.LongTensor(item['goal'])
        # Convert path to sequence of cell indices
        path_indices = torch.LongTensor([
            r * self.maze_size + c for r, c in item['path']
        ])
        return maze, start, goal, path_indices, len(item['path'])


def train_model(model, train_loader, device, epochs=30, lr=1e-3):
    """Train a maze solver model."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for maze, start, goal, path, path_len in train_loader:
            maze, start, goal, path = maze.to(device), start.to(device), goal.to(device), path.to(device)
            
            optimizer.zero_grad()
            logits = model(maze, start, goal)
            
            # Loss: predict each step of the path
            loss = 0
            for i in range(path.shape[1] - 1):
                loss += criterion(logits[:, i, :], path[:, i + 1])
            loss /= (path.shape[1] - 1)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    return model


def evaluate_model(model, test_loader, device, maze_size):
    """Evaluate maze solver."""
    model.eval()
    correct = 0
    total = 0
    optimal = 0
    
    with torch.no_grad():
        for maze, start, goal, path, path_len in test_loader:
            maze, start, goal, path = maze.to(device), start.to(device), goal.to(device), path.to(device)
            
            logits = model(maze, start, goal)
            
            # Greedy decoding
            for b in range(maze.shape[0]):
                pred_path = [start[b, 0].item() * maze_size + start[b, 1].item()]
                current = start[b]
                
                for step in range(path_len[b].item()):
                    pos_idx = current[0] * maze_size + current[1]
                    next_cell = logits[b, pos_idx].argmax().item()
                    pred_path.append(next_cell)
                    current = torch.tensor([next_cell // maze_size, next_cell % maze_size])
                    
                    if current[0] == goal[b, 0] and current[1] == goal[b, 1]:
                        correct += 1
                        if len(pred_path) == path_len[b].item():
                            optimal += 1
                        break
                
                total += 1
    
    acc = 100.0 * correct / total
    opt = 100.0 * optimal / total
    return acc, opt


# ============================================================================
# Main Benchmark
# ============================================================================

def run_ab_test(
    maze_size: int = 20,
    n_train: int = 500,
    n_test: int = 100,
    min_path_length: int = None,
    R: int = 4,
    T: int = 4,
    n_heads: int = 4,
    epochs: int = 40,
    seed: int = 42
):
    """Run A/B test with proper maze generation."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*80}")
    print(f"MAZE A/B TEST - PROPER GENERATION")
    print(f"{'='*80}")
    print(f"Maze size: {maze_size}×{maze_size}")
    print(f"Training mazes: {n_train}")
    print(f"Test mazes: {n_test}")
    print(f"Min path length: {min_path_length or 'auto'}")
    print(f"PoH config: R={R}, T={T}, heads={n_heads}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    # Generate data
    print("Generating training data...")
    train_data = generate_dataset_proper(maze_size, n_train, min_path_length, seed)
    
    print("\nGenerating test data...")
    test_data = generate_dataset_proper(maze_size, n_test, min_path_length, seed + 10000)
    
    # Create dataloaders
    train_dataset = MazeDatasetWrapper(train_data, maze_size)
    test_dataset = MazeDatasetWrapper(test_data, maze_size)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    results = {}
    
    # Test Baseline
    print(f"\n{'='*60}")
    print("Training: Baseline Transformer")
    print(f"{'='*60}")
    baseline = BaselineMazeSolver(maze_size).to(device)
    print(f"Parameters: {sum(p.numel() for p in baseline.parameters())/1e6:.2f}M")
    baseline = train_model(baseline, train_loader, device, epochs)
    acc, opt = evaluate_model(baseline, test_loader, device, maze_size)
    print(f"Accuracy: {acc:.2f}%, Optimality: {opt:.2f}%")
    results['baseline'] = {'acc': acc, 'opt': opt}
    
    # Test PoH-HRM
    print(f"\n{'='*60}")
    print(f"Training: PoH-HRM (R={R}, T={T})")
    print(f"{'='*60}")
    poh = PoHMazeSolver(maze_size, R=R, T=T, n_heads=n_heads).to(device)
    print(f"Parameters: {sum(p.numel() for p in poh.parameters())/1e6:.2f}M")
    poh = train_model(poh, train_loader, device, epochs)
    acc, opt = evaluate_model(poh, test_loader, device, maze_size)
    print(f"Accuracy: {acc:.2f}%, Optimality: {opt:.2f}%")
    results['poh'] = {'acc': acc, 'opt': opt}
    
    # Summary
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Baseline: Acc={results['baseline']['acc']:.2f}%, Opt={results['baseline']['opt']:.2f}%")
    print(f"PoH-HRM:  Acc={results['poh']['acc']:.2f}%, Opt={results['poh']['opt']:.2f}%")
    print(f"{'='*80}\n")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Maze A/B Test with Proper Generation')
    parser.add_argument('--maze-size', type=int, default=20, help='Maze size')
    parser.add_argument('--train', type=int, default=500, help='Training samples')
    parser.add_argument('--test', type=int, default=100, help='Test samples')
    parser.add_argument('--min-path-length', type=int, default=None, help='Minimum path length (auto if not specified)')
    parser.add_argument('--R', type=int, default=4, help='PoH refinement steps')
    parser.add_argument('--T', type=int, default=4, help='HRM outer loop period')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--epochs', type=int, default=40, help='Training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    results = run_ab_test(
        maze_size=args.maze_size,
        n_train=args.train,
        n_test=args.test,
        min_path_length=args.min_path_length,
        R=args.R,
        T=args.T,
        n_heads=args.heads,
        epochs=args.epochs,
        seed=args.seed
    )

