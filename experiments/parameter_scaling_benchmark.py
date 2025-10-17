#!/usr/bin/env python3
"""
Parameter Scaling Benchmark: Scale model size for both Baseline and PoT-HRM
Tests performance vs. parameter count on maze solving task.

Model sizes tested (approximate parameter counts):
- Tiny: ~1M params
- Small: ~3M params  
- Medium: ~10M params
- Large: ~30M params
- XL: ~100M params

For each size, we test:
1. Baseline Transformer (standard multi-head attention)
2. PoT-HRM (with R=4, T=4)

We scale by adjusting:
- d_model (embedding dimension)
- n_heads (number of attention heads)
- d_ff (feedforward dimension)
- depth (number of stacked transformer blocks)
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
from tqdm import tqdm
import json
import time
import argparse
from collections import defaultdict

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
    sys.exit(1)

# Import PoT modules
from src.pot.modules import PoHConfig, PoHStack, IterRefiner
from src.pot.core.hrm_controller import HRMPointerController, HRMState

print("✓ Successfully imported PoT modules")

# Check GPU (CUDA → MPS → CPU)
if torch.cuda.is_available():
    print(f"🚀 CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("🚀 Apple Silicon GPU (MPS) detected")
    print("   Using Metal Performance Shaders for acceleration")
else:
    print("⚠️  No GPU detected, using CPU")

# ============================================================================
# Model Size Configurations
# ============================================================================

MODEL_CONFIGS = {
    # Only testing Large and XL as requested
    'large': {
        'd_model': 768,
        'n_heads': 12,
        'd_ff': 3072,
        'depth': 6,
        'target_params': 30e6,
    },
    'xl': {
        'd_model': 1024,
        'n_heads': 16,
        'd_ff': 4096,
        'depth': 8,
        'target_params': 100e6,
    },
}

# ============================================================================
# Maze Generation (from maze_ab_proper_generation.py)
# ============================================================================

def generate_dataset_proper(maze_size: int, n_samples: int, min_path_length: int = None, seed: int = None):
    """Generate mazes using the maze-dataset library with path length filtering."""
    
    if min_path_length is None:
        min_path_length = int(maze_size * maze_size * 0.4)
    
    print(f"  Generating {n_samples} mazes of size {maze_size}×{maze_size}")
    print(f"  Minimum path length: {min_path_length}")
    
    cfg = MazeDatasetConfig(
        name=f"maze_{maze_size}x{maze_size}_minpath{min_path_length}",
        grid_n=maze_size,
        n_mazes=n_samples * 3,
        maze_ctor=LatticeMazeGenerators.gen_dfs,
        seed=seed,
    )
    
    dataset = MazeDataset.from_config(
        cfg,
        do_generate=True,
        load_local=False,
        save_local=False,
    )
    
    dataset_filtered = dataset.filter_by.path_length(min_length=min_path_length)
    
    if len(dataset_filtered) < n_samples:
        print(f"  ⚠️  Warning: Only generated {len(dataset_filtered)} mazes meeting criteria (requested {n_samples})")
        print(f"      Using all {len(dataset_filtered)} available mazes")
        n_samples = len(dataset_filtered)
    else:
        dataset_filtered = dataset_filtered[:n_samples]
    
    data = []
    path_lengths = []
    
    for solved_maze in dataset_filtered:
        maze_obj = solved_maze.maze
        
        # Build grid representation: 0 = passable, 1 = wall
        # Use connection_list to determine passable cells
        grid = np.ones((maze_size, maze_size), dtype=np.float32)  # Start with all walls
        
        # Get nodes (passable cells) from the maze
        nodes = maze_obj.get_nodes()
        for node in nodes:
            if isinstance(node, np.ndarray):
                row, col = int(node[0]), int(node[1])
            else:
                row, col = node.row, node.col
            grid[row, col] = 0.0  # Mark as passable
        
        maze = grid
        
        if hasattr(maze_obj, 'start_pos'):
            start = (maze_obj.start_pos.row, maze_obj.start_pos.col)
            goal = (maze_obj.end_pos.row, maze_obj.end_pos.col)
        else:
            solution_array = solved_maze.solution
            if solution_array.ndim == 2:
                start = tuple(solution_array[0])
                goal = tuple(solution_array[-1])
            else:
                print(f"  ⚠️  Skipping maze with unclear start/goal structure")
                continue
        
        solution_array = solved_maze.solution
        if solution_array.ndim == 2:
            path = [tuple(coord) for coord in solution_array]
        else:
            print(f"  ⚠️  Skipping maze with unclear solution structure")
            continue
        
        path_lengths.append(len(path))
        data.append({
            'maze': maze,
            'start': start,
            'goal': goal,
            'path': path,
            'length': len(path)
        })
    
    print(f"  ✓ Generated {len(data)} mazes")
    if path_lengths:
        print(f"  Path length: {np.mean(path_lengths):.1f} ± {np.std(path_lengths):.1f} (min={min(path_lengths)}, max={max(path_lengths)})")
    
    return data


class MazeDatasetWrapper(Dataset):
    def __init__(self, data, maze_size):
        self.data = data
        self.maze_size = maze_size
        self.max_path_len = max(len(item['path']) for item in data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        maze = torch.FloatTensor(item['maze'])
        start = torch.LongTensor(item['start'])
        goal = torch.LongTensor(item['goal'])
        
        path_indices = [r * self.maze_size + c for r, c in item['path']]
        path_len = len(path_indices)
        
        path_indices_padded = path_indices + [-1] * (self.max_path_len - path_len)
        path_indices_tensor = torch.LongTensor(path_indices_padded)
        
        return maze, start, goal, path_indices_tensor, path_len

# ============================================================================
# Model Definitions
# ============================================================================

class BaselineMazeSolver(nn.Module):
    """Standard Transformer baseline for maze solving."""
    
    def __init__(
        self,
        maze_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 1024,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_maze_encoder: bool = False,
    ):
        super().__init__()
        self.maze_size = maze_size
        self.d_model = d_model
        self.use_maze_encoder = use_maze_encoder
        
        # Embeddings
        self.cell_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Embedding(maze_size * maze_size, d_model)
        self.start_embed = nn.Embedding(2, d_model)
        self.goal_embed = nn.Embedding(2, d_model)
        
        if self.use_maze_encoder:
            self.maze_cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4))
            )
            self.maze_proj = nn.Linear(32 * 4 * 4, d_model)
        
        # Transformer layers (stacked)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output decoder
        self.decoder = nn.Linear(d_model, maze_size * maze_size)
    
    def forward(self, maze, start, goal):
        B = maze.shape[0]
        N = self.maze_size * self.maze_size
        
        # Flatten maze
        maze_flat = maze.view(B, N, 1)
        
        # Embeddings
        h = self.cell_embed(maze_flat)
        
        pos = torch.arange(N, device=maze.device).unsqueeze(0).expand(B, -1)
        h = h + self.pos_embed(pos)
        
        if getattr(self, 'use_maze_encoder', False):
            g = maze.unsqueeze(1)
            g = self.maze_cnn(g)
            g = torch.flatten(g, start_dim=1)
            g = self.maze_proj(g)
            h = h + g.unsqueeze(1)
        
        start_idx = start[:, 0] * self.maze_size + start[:, 1]
        goal_idx = goal[:, 0] * self.maze_size + goal[:, 1]
        
        start_marker = torch.zeros(B, N, device=maze.device, dtype=torch.long)
        start_marker.scatter_(1, start_idx.unsqueeze(1), 1)
        h = h + self.start_embed(start_marker)
        
        goal_marker = torch.zeros(B, N, device=maze.device, dtype=torch.long)
        goal_marker.scatter_(1, goal_idx.unsqueeze(1), 1)
        h = h + self.goal_embed(goal_marker)
        
        # Transformer
        h = self.transformer(h)
        
        # Decode
        logits = self.decoder(h)
        return logits


class StatefulHRMRouter(nn.Module):
    """Wrapper to make HRMPointerController compatible with PoHBlock's HeadRouter interface."""
    
    def __init__(self, hrm_controller: HRMPointerController, n_heads: int):
        super().__init__()
        self.hrm = hrm_controller
        self.n_heads = n_heads
        self.state = None
    
    def forward(self, x_ctrl: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_ctrl: [B, T, d_model]
        Returns:
            route_logits: [B, T, H] tensor
        """
        B, T, d_model = x_ctrl.shape
        
        if self.state is None:
            device = x_ctrl.device
            self.state = HRMState(
                z_L=torch.zeros(B, self.hrm.d_ctrl, device=device),
                z_H=torch.zeros(B, self.hrm.d_ctrl, device=device),
                step=torch.zeros(B, dtype=torch.long, device=device),
            )
        
        x_ctrl_mean = x_ctrl.mean(dim=1)
        alphas, new_state, _ = self.hrm(x_ctrl_mean, self.state)
        
        self.state = HRMState(
            z_L=new_state.z_L.detach(),
            z_H=new_state.z_H.detach(),
            step=new_state.step.detach(),
        )
        
        alphas_expanded = alphas.unsqueeze(1).expand(B, T, self.n_heads)
        route_logits = torch.log(alphas_expanded + 1e-8)
        
        return route_logits


class PoHMazeSolver(nn.Module):
    """PoH-HRM model for maze solving with iterative refinement."""
    
    def __init__(
        self,
        maze_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 1024,
        depth: int = 4,
        R: int = 4,
        T: int = 4,
        dropout: float = 0.1,
        use_maze_encoder: bool = False,
    ):
        super().__init__()
        self.maze_size = maze_size
        self.d_model = d_model
        self.R = R
        self.T = T
        self.use_maze_encoder = use_maze_encoder
        
        # Embeddings (same as baseline)
        self.cell_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Embedding(maze_size * maze_size, d_model)
        self.start_embed = nn.Embedding(2, d_model)
        self.goal_embed = nn.Embedding(2, d_model)
        
        if self.use_maze_encoder:
            self.maze_cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4))
            )
            self.maze_proj = nn.Linear(32 * 4 * 4, d_model)
        
        # PoH Stack with HRM controller
        cfg = PoHConfig(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
        )
        
        stack = PoHStack(cfg, depth=depth)
        
        # Replace routers with HRM controllers
        for block in stack.blocks:
            if hasattr(block, 'router'):
                hrm = HRMPointerController(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ctrl=d_model // 2,
                    T=T,
                )
                block.router = StatefulHRMRouter(hrm, n_heads)
        
        # Iterative refinement wrapper
        self.refiner = IterRefiner(
            stack,
            max_inner_iters=R,
            outer_residual=True,
            rezero_init=True
        )
        
        # Output decoder
        self.decoder = nn.Linear(d_model, maze_size * maze_size)
    
    def forward(self, maze, start, goal):
        B = maze.shape[0]
        N = self.maze_size * self.maze_size
        
        # Flatten maze and create embeddings (same as baseline)
        maze_flat = maze.view(B, N, 1)
        h = self.cell_embed(maze_flat)
        
        pos = torch.arange(N, device=maze.device).unsqueeze(0).expand(B, -1)
        h = h + self.pos_embed(pos)
        
        start_idx = start[:, 0] * self.maze_size + start[:, 1]
        goal_idx = goal[:, 0] * self.maze_size + goal[:, 1]
        
        start_marker = torch.zeros(B, N, device=maze.device, dtype=torch.long)
        start_marker.scatter_(1, start_idx.unsqueeze(1), 1)
        h = h + self.start_embed(start_marker)
        
        goal_marker = torch.zeros(B, N, device=maze.device, dtype=torch.long)
        goal_marker.scatter_(1, goal_idx.unsqueeze(1), 1)
        h = h + self.goal_embed(goal_marker)
        
        # PoH iterative refinement
        h, _ = self.refiner(h)
        
        # Decode
        logits = self.decoder(h)
        return logits

    def forward_with_stats(self, maze, start, goal):
        B = maze.shape[0]
        N = self.maze_size * self.maze_size
        maze_flat = maze.view(B, N, 1)
        h = self.cell_embed(maze_flat)
        pos = torch.arange(N, device=maze.device).unsqueeze(0).expand(B, -1)
        h = h + self.pos_embed(pos)
        if getattr(self, 'use_maze_encoder', False):
            g = maze.unsqueeze(1)
            g = self.maze_cnn(g)
            g = torch.flatten(g, start_dim=1)
            g = self.maze_proj(g)
            h = h + g.unsqueeze(1)
        start_idx = start[:, 0] * self.maze_size + start[:, 1]
        goal_idx = goal[:, 0] * self.maze_size + goal[:, 1]
        start_marker = torch.zeros(B, N, device=maze.device, dtype=torch.long)
        start_marker.scatter_(1, start_idx.unsqueeze(1), 1)
        h = h + self.start_embed(start_marker)
        goal_marker = torch.zeros(B, N, device=maze.device, dtype=torch.long)
        goal_marker.scatter_(1, goal_idx.unsqueeze(1), 1)
        h = h + self.goal_embed(goal_marker)
        h, stats = self.refiner(h, return_inner_stats=True)
        logits = self.decoder(h)
        return logits, stats

# ============================================================================
# Training and Evaluation
# ============================================================================

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(
    model,
    train_loader,
    device,
    epochs=30,
    lr=1e-3,
    label_smoothing: float = 0.0,
    warmup_steps: int = 1000,
    multi_horizon: int = 1,
    validity_mask: bool = False,
    route_ent_weight: float = 0.0,
    ent_anneal: bool = False,
    maze_size: int = None,
):
    """Train a model on maze solving with advanced options."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=label_smoothing)
    steps_per_epoch = len(train_loader)
    total_steps = max(1, epochs * steps_per_epoch)
    def lr_at(step):
        if step < warmup_steps:
            return lr * step / max(1, warmup_steps)
        rem = max(1, total_steps - warmup_steps)
        prog = (step - warmup_steps) / rem
        min_lr = 0.1 * lr
        return min_lr + (lr - min_lr) * 0.5 * (1 + np.cos(np.pi * prog))
    global_step = 0
    
    for epoch in range(epochs):
        total_loss = 0
        for maze, start, goal, path, path_len in train_loader:
            cur_lr = lr_at(global_step)
            for g in optimizer.param_groups:
                g['lr'] = cur_lr
            maze = maze.to(device)
            start = start.to(device)
            goal = goal.to(device)
            path = path.to(device)
            
            optimizer.zero_grad()
            
            if route_ent_weight > 0.0 and hasattr(model, 'forward_with_stats'):
                logits, stats = model.forward_with_stats(maze, start, goal)
            else:
                logits = model(maze, start, goal)
                stats = None
            
            B, N, V = logits.shape
            max_len = path.shape[1]
            loss = 0.0
            count = 0
            K = max(1, multi_horizon)
            for i in range(max_len - 1):
                for k in range(1, K + 1):
                    if i + k >= max_len:
                        break
                    mask = (path[:, i] != -1) & (path[:, i + k] != -1)
                    if not mask.any():
                        continue
                    curr_pos = path[mask, i]
                    target_pos = path[mask, i + k]
                    curr_logits = logits[mask].gather(1, curr_pos.unsqueeze(1).unsqueeze(2).expand(-1, 1, V)).squeeze(1)
                    if validity_mask and k == 1 and maze_size is not None:
                        masked_logits = torch.full_like(curr_logits, fill_value=-1e9)
                        for bi, cur_idx in enumerate(curr_pos):
                            r = (cur_idx // maze_size).item(); c = (cur_idx % maze_size).item()
                            allowed = []
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < maze_size and 0 <= nc < maze_size and maze[mask][bi, nr, nc] < 0.5:
                                    allowed.append(nr * maze_size + nc)
                            if not allowed:
                                pass_cells = (maze[mask][bi].view(-1) < 0.5).nonzero(as_tuple=False).view(-1)
                                masked_logits[bi, pass_cells] = curr_logits[bi, pass_cells]
                            else:
                                masked_logits[bi, allowed] = curr_logits[bi, allowed]
                        use_logits = masked_logits
                    else:
                        use_logits = curr_logits
                    loss = loss + criterion(use_logits, target_pos)
                    count += 1
            
            if stats is not None and route_ent_weight > 0.0:
                ent_vals = []
                for s in stats:
                    if 'route_entropy_mean' in s:
                        ent_vals.append(float(s['route_entropy_mean']))
                if ent_vals:
                    ent = torch.tensor(np.mean(ent_vals), dtype=torch.float32, device=device)
                    w = route_ent_weight
                    if ent_anneal:
                        w = w * max(0.0, 1.0 - global_step / float(total_steps))
                    loss = loss + w * ent
            
            if count > 0:
                loss = loss / count
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            global_step += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, LR: {cur_lr:.2e}")
    
    return model


def evaluate_model(model, test_loader, device, maze_size):
    """Evaluate model on maze solving."""
    model.eval()
    correct = 0
    optimal = 0
    total = 0
    
    with torch.no_grad():
        for maze, start, goal, path, path_len in test_loader:
            maze = maze.to(device)
            start = start.to(device)
            goal = goal.to(device)
            
            B = maze.shape[0]
            
            for b in range(B):
                # Greedy decoding with valid move constraints
                current = start[b].cpu().numpy()
                target = goal[b].cpu().numpy()
                maze_grid = maze[b].cpu().numpy()
                
                visited = set()
                predicted_path = [tuple(current)]
                visited.add(tuple(current))
                
                max_steps = maze_size * maze_size
                for step in range(max_steps):
                    if tuple(current) == tuple(target):
                        break
                    
                    # Get logits for current position
                    maze_input = maze[b:b+1].to(device)
                    start_input = torch.LongTensor([current]).to(device)
                    goal_input = goal[b:b+1].to(device)
                    
                    logits = model(maze_input, start_input, goal_input)
                    curr_idx = current[0] * maze_size + current[1]
                    probs = torch.softmax(logits[0, curr_idx], dim=-1).cpu().numpy()
                    
                    # Get valid neighbors
                    valid_moves = []
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = current[0] + dr, current[1] + dc
                        if 0 <= nr < maze_size and 0 <= nc < maze_size:
                            if maze_grid[nr, nc] < 0.5:  # Passable
                                if (nr, nc) not in visited:
                                    valid_moves.append((nr, nc, probs[nr * maze_size + nc]))
                    
                    if not valid_moves:
                        # Deadend - allow revisiting
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                            nr, nc = current[0] + dr, current[1] + dc
                            if 0 <= nr < maze_size and 0 <= nc < maze_size:
                                if maze_grid[nr, nc] < 0.5:
                                    valid_moves.append((nr, nc, probs[nr * maze_size + nc]))
                        
                        if not valid_moves:
                            break
                    
                    # Choose best valid move
                    valid_moves.sort(key=lambda x: x[2], reverse=True)
                    current = np.array([valid_moves[0][0], valid_moves[0][1]])
                    predicted_path.append(tuple(current))
                    visited.add(tuple(current))
                
                # Check if reached goal
                if tuple(current) == tuple(target):
                    correct += 1
                    
                    # Check optimality (within 5% of shortest path)
                    true_path_len = path_len[b].item()
                    pred_path_len = len(predicted_path)
                    if pred_path_len <= true_path_len * 1.05:
                        optimal += 1
                
                total += 1
    
    accuracy = 100 * correct / total if total > 0 else 0
    optimality = 100 * optimal / total if total > 0 else 0
    
    return accuracy, optimality


# ============================================================================
# Main Scaling Benchmark
# ============================================================================

def run_scaling_benchmark(
    maze_size=16,
    n_train=1000,
    n_test=100,
    min_path_length=None,
    epochs=50,
    R=4,
    T=4,
    seed=42,
    output_dir='experiments/results/parameter_scaling'
):
    """Run parameter scaling benchmark."""
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print("\n" + "="*80)
    print("PARAMETER SCALING BENCHMARK")
    print("="*80)
    print(f"Maze size: {maze_size}×{maze_size}")
    print(f"Training samples: {n_train}")
    print(f"Test samples: {n_test}")
    print(f"Epochs: {epochs}")
    print(f"PoH config: R={R}, T={T}")
    print(f"Device: {device}")
    print("="*80)
    
    # Generate data once
    print("\nGenerating training data...")
    train_data = generate_dataset_proper(maze_size, n_train, min_path_length, seed)
    
    print("\nGenerating test data...")
    test_data = generate_dataset_proper(maze_size, n_test, min_path_length, seed+10000)
    
    train_dataset = MazeDatasetWrapper(train_data, maze_size)
    test_dataset = MazeDatasetWrapper(test_data, maze_size)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Results storage
    results = []
    
    # Test each model size
    for size_name, config in MODEL_CONFIGS.items():
        print(f"\n{'='*80}")
        print(f"Testing {size_name.upper()} models (~{config['target_params']/1e6:.0f}M params)")
        print(f"{'='*80}")
        print(f"Config: d_model={config['d_model']}, n_heads={config['n_heads']}, "
              f"d_ff={config['d_ff']}, depth={config['depth']}")
        
        # Test Baseline
        print(f"\n{'-'*80}")
        print(f"Training: Baseline Transformer ({size_name})")
        print(f"{'-'*80}")
        
        baseline = BaselineMazeSolver(
            maze_size=maze_size,
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            d_ff=config['d_ff'],
            num_layers=config['depth'],
            use_maze_encoder=True,
        ).to(device)
        
        baseline_params = count_parameters(baseline)
        print(f"Parameters: {baseline_params/1e6:.2f}M")
        
        baseline = train_model(baseline, train_loader, device, epochs=epochs)
        baseline_acc, baseline_opt = evaluate_model(baseline, test_loader, device, maze_size)
        
        print(f"Accuracy: {baseline_acc:.2f}%, Optimality: {baseline_opt:.2f}%")
        
        # Test PoH-HRM with parameter parity
        # PoH has overhead from HRM controllers, so we reduce depth to match baseline params
        print(f"\n{'-'*80}")
        print(f"Training: PoH-HRM ({size_name}, R={R}, T={T})")
        print(f"{'-'*80}")
        
        # Try different depths to match baseline parameter count
        best_poh = None
        best_poh_params = float('inf')
        best_depth = config['depth']
        
        for trial_depth in range(config['depth'], 0, -1):
            trial_poh = PoHMazeSolver(
                maze_size=maze_size,
                d_model=config['d_model'],
                n_heads=config['n_heads'],
                d_ff=config['d_ff'],
                depth=trial_depth,
                R=R,
                T=T,
                use_maze_encoder=True,
            )
            trial_params = count_parameters(trial_poh)
            
            # Accept if within 10% of baseline or fewer params
            if trial_params <= baseline_params * 1.1:
                best_poh = trial_poh
                best_poh_params = trial_params
                best_depth = trial_depth
                break
            
            # Keep searching for closer match
            if abs(trial_params - baseline_params) < abs(best_poh_params - baseline_params):
                best_poh = trial_poh
                best_poh_params = trial_params
                best_depth = trial_depth
        
        poh = best_poh.to(device)
        poh_params = best_poh_params
        
        param_ratio = (poh_params / baseline_params) * 100
        print(f"Parameters: {poh_params/1e6:.2f}M (depth={best_depth}, {param_ratio:.1f}% of baseline)")
        
        if poh_params > baseline_params:
            print(f"⚠️  Warning: PoH has {(poh_params/baseline_params - 1)*100:.1f}% more parameters than baseline")
        
        poh = train_model(poh, train_loader, device, epochs=epochs)
        poh_acc, poh_opt = evaluate_model(poh, test_loader, device, maze_size)
        
        print(f"Accuracy: {poh_acc:.2f}%, Optimality: {poh_opt:.2f}%")
        
        # Store results
        results.append({
            'size': size_name,
            'd_model': config['d_model'],
            'n_heads': config['n_heads'],
            'd_ff': config['d_ff'],
            'baseline_depth': config['depth'],
            'poh_depth': best_depth,
            'baseline_params': baseline_params,
            'baseline_acc': baseline_acc,
            'baseline_opt': baseline_opt,
            'poh_params': poh_params,
            'poh_acc': poh_acc,
            'poh_opt': poh_opt,
            'param_ratio': param_ratio,
            'poh_advantage_acc': poh_acc - baseline_acc,
            'poh_advantage_opt': poh_opt - baseline_opt,
        })
        
        # Cleanup
        del baseline, poh
        torch.cuda.empty_cache()
    
    # Print summary
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Size':<10} {'Params':<12} {'Baseline':<20} {'PoH-HRM':<20} {'Advantage':<20}")
    print(f"{'':<10} {'':<12} {'Acc% / Opt%':<20} {'Acc% / Opt%':<20} {'Acc% / Opt%':<20}")
    print(f"{'-'*80}")
    
    for r in results:
        print(f"{r['size']:<10} "
              f"{r['baseline_params']/1e6:>5.1f}M / {r['poh_params']/1e6:>4.1f}M "
              f"{r['baseline_acc']:>5.1f} / {r['baseline_opt']:>5.1f}    "
              f"{r['poh_acc']:>5.1f} / {r['poh_opt']:>5.1f}    "
              f"{r['poh_advantage_acc']:>+5.1f} / {r['poh_advantage_opt']:>+5.1f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'scaling_results_maze{maze_size}.json')
    
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'maze_size': maze_size,
                'n_train': n_train,
                'n_test': n_test,
                'epochs': epochs,
                'R': R,
                'T': T,
                'seed': seed,
            },
            'results': results,
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    return results


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Scaling Benchmark')
    parser.add_argument('--maze-size', type=int, default=16, help='Maze size (default: 16)')
    parser.add_argument('--train', type=int, default=1000, help='Training samples (default: 1000)')
    parser.add_argument('--test', type=int, default=100, help='Test samples (default: 100)')
    parser.add_argument('--min-path', type=int, default=None, help='Minimum path length filter')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs (default: 50)')
    parser.add_argument('--R', type=int, default=4, help='PoH refinement iterations (default: 4)')
    parser.add_argument('--T', type=int, default=4, help='HRM outer loop period (default: 4)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, default='experiments/results/parameter_scaling',
                        help='Output directory (default: experiments/results/parameter_scaling)')
    
    args = parser.parse_args()
    
    results = run_scaling_benchmark(
        maze_size=args.maze_size,
        n_train=args.train,
        n_test=args.test,
        min_path_length=args.min_path,
        epochs=args.epochs,
        R=args.R,
        T=args.T,
        seed=args.seed,
        output_dir=args.output,
    )

