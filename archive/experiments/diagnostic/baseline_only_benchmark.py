#!/usr/bin/env python3
"""
Baseline-only benchmark: Train Large and XL baseline transformers on maze solving.
Focus on ensuring convergence before comparing to PoH.
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import json

# Setup paths
repo_root = Path(__file__).parent.parent
if repo_root not in sys.path:
    sys.path.insert(0, str(repo_root))

print(f"✓ PoT root: {repo_root}")

# Import maze-dataset
try:
    from maze_dataset import MazeDataset, MazeDatasetConfig
    from maze_dataset.generation import LatticeMazeGenerators
    print("✓ maze-dataset library available")
except ImportError:
    print("✗ maze-dataset not installed. Install with: pip install maze-dataset")
    sys.exit(1)

print("✓ Successfully imported modules")

# Check GPU
if torch.cuda.is_available():
    print(f"🚀 CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("🚀 Apple Silicon GPU (MPS) detected")
    print("   Using Metal Performance Shaders for acceleration")
else:
    print("⚠️  No GPU detected, using CPU")

# ============================================================================
# Model Configurations (Large and XL only)
# ============================================================================

MODEL_CONFIGS = {
    'large': {
        'd_model': 768,
        'n_heads': 12,
        'd_ff': 3072,
        'depth': 6,
    },
    'xl': {
        'd_model': 1024,
        'n_heads': 16,
        'd_ff': 4096,
        'depth': 8,
    },
}

# ============================================================================
# Maze Generation
# ============================================================================

def generate_dataset_proper(maze_size: int, n_samples: int, min_path_length: int = None, seed: int = None):
    """Generate mazes using the maze-dataset library."""
    
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
    
    dataset = MazeDataset.from_config(cfg, do_generate=True, load_local=False, save_local=False)
    dataset_filtered = dataset.filter_by.path_length(min_length=min_path_length)
    
    if len(dataset_filtered) < n_samples:
        print(f"  ⚠️  Warning: Only generated {len(dataset_filtered)} mazes meeting criteria (requested {n_samples})")
        n_samples = len(dataset_filtered)
    else:
        dataset_filtered = dataset_filtered[:n_samples]
    
    data = []
    path_lengths = []
    
    for solved_maze in dataset_filtered:
        maze_obj = solved_maze.maze
        
        # Build grid representation
        grid = np.ones((maze_size, maze_size), dtype=np.float32)
        nodes = maze_obj.get_nodes()
        for node in nodes:
            if isinstance(node, np.ndarray):
                row, col = int(node[0]), int(node[1])
            else:
                row, col = node.row, node.col
            grid[row, col] = 0.0
        
        if hasattr(maze_obj, 'start_pos'):
            start = (maze_obj.start_pos.row, maze_obj.start_pos.col)
            goal = (maze_obj.end_pos.row, maze_obj.end_pos.col)
        else:
            solution_array = solved_maze.solution
            if solution_array.ndim == 2:
                start = tuple(solution_array[0])
                goal = tuple(solution_array[-1])
            else:
                continue
        
        solution_array = solved_maze.solution
        if solution_array.ndim == 2:
            path = [tuple(coord) for coord in solution_array]
        else:
            continue
        
        path_lengths.append(len(path))
        data.append({
            'maze': grid,
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
# Model Definition
# ============================================================================

class BaselineMazeSolver(nn.Module):
    """Standard Transformer baseline for maze solving."""
    
    def __init__(self, maze_size: int, d_model: int = 256, n_heads: int = 4, 
                 d_ff: int = 1024, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.maze_size = maze_size
        self.d_model = d_model
        
        # Embeddings
        self.cell_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Embedding(maze_size * maze_size, d_model)
        self.start_embed = nn.Embedding(2, d_model)
        self.goal_embed = nn.Embedding(2, d_model)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output
        self.decoder = nn.Linear(d_model, maze_size * maze_size)
    
    def forward(self, maze, start, goal):
        B = maze.shape[0]
        N = self.maze_size * self.maze_size
        
        # Embeddings
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
        
        # Transformer
        h = self.transformer(h)
        
        # Decode
        logits = self.decoder(h)
        return logits

# ============================================================================
# Training and Evaluation
# ============================================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model_with_monitoring(model, train_loader, device, epochs=50, lr=1e-3):
    """Train with detailed convergence monitoring."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    epoch_losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        for maze, start, goal, path, path_len in train_loader:
            maze = maze.to(device)
            start = start.to(device)
            goal = goal.to(device)
            path = path.to(device)
            
            optimizer.zero_grad()
            logits = model(maze, start, goal)
            
            # Compute loss
            B, N, V = logits.shape
            max_len = path.shape[1]
            
            loss = 0
            count = 0
            for i in range(max_len - 1):
                mask = (path[:, i] != -1) & (path[:, i + 1] != -1)
                if mask.any():
                    curr_pos = path[mask, i]
                    next_pos = path[mask, i + 1]
                    curr_logits = logits[mask].gather(1, curr_pos.unsqueeze(1).unsqueeze(2).expand(-1, 1, V)).squeeze(1)
                    loss += criterion(curr_logits, next_pos)
                    count += 1
            
            if count > 0:
                loss = loss / count
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        epoch_losses.append(avg_loss)
        
        # Print every 5 epochs or last epoch
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # Check convergence
            if len(epoch_losses) >= 10:
                recent_std = np.std(epoch_losses[-10:])
                if recent_std < 0.01:
                    print(f"  ✓ Converged! Loss stable (std={recent_std:.4f})")
    
    return model, epoch_losses


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
                    
                    maze_input = maze[b:b+1].to(device)
                    start_input = torch.LongTensor([current]).to(device)
                    goal_input = goal[b:b+1].to(device)
                    
                    logits = model(maze_input, start_input, goal_input)
                    curr_idx = current[0] * maze_size + current[1]
                    probs = torch.softmax(logits[0, curr_idx], dim=-1).cpu().numpy()
                    
                    valid_moves = []
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = current[0] + dr, current[1] + dc
                        if 0 <= nr < maze_size and 0 <= nc < maze_size:
                            if maze_grid[nr, nc] < 0.5:
                                if (nr, nc) not in visited:
                                    valid_moves.append((nr, nc, probs[nr * maze_size + nc]))
                    
                    if not valid_moves:
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                            nr, nc = current[0] + dr, current[1] + dc
                            if 0 <= nr < maze_size and 0 <= nc < maze_size:
                                if maze_grid[nr, nc] < 0.5:
                                    valid_moves.append((nr, nc, probs[nr * maze_size + nc]))
                        if not valid_moves:
                            break
                    
                    valid_moves.sort(key=lambda x: x[2], reverse=True)
                    current = np.array([valid_moves[0][0], valid_moves[0][1]])
                    predicted_path.append(tuple(current))
                    visited.add(tuple(current))
                
                if tuple(current) == tuple(target):
                    correct += 1
                    true_path_len = path_len[b].item()
                    pred_path_len = len(predicted_path)
                    if pred_path_len <= true_path_len * 1.05:
                        optimal += 1
                
                total += 1
    
    accuracy = 100 * correct / total if total > 0 else 0
    optimality = 100 * optimal / total if total > 0 else 0
    
    return accuracy, optimality

# ============================================================================
# Main Benchmark
# ============================================================================

def run_baseline_benchmark(
    maze_size=16,
    n_train=1000,
    n_test=100,
    min_path_length=None,
    epochs=50,
    seed=42,
    output_dir='experiments/results/baseline_only'
):
    """Run baseline-only benchmark."""
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print("\n" + "="*80)
    print("BASELINE-ONLY BENCHMARK")
    print("="*80)
    print(f"Maze size: {maze_size}×{maze_size}")
    print(f"Training samples: {n_train}")
    print(f"Test samples: {n_test}")
    print(f"Epochs: {epochs}")
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
    
    results = []
    
    # Test each model size
    for size_name, config in MODEL_CONFIGS.items():
        print(f"\n{'='*80}")
        print(f"Training: Baseline Transformer ({size_name.upper()})")
        print(f"{'='*80}")
        print(f"Config: d_model={config['d_model']}, n_heads={config['n_heads']}, "
              f"d_ff={config['d_ff']}, depth={config['depth']}")
        
        baseline = BaselineMazeSolver(
            maze_size=maze_size,
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            d_ff=config['d_ff'],
            num_layers=config['depth'],
        ).to(device)
        
        baseline_params = count_parameters(baseline)
        print(f"Parameters: {baseline_params/1e6:.2f}M")
        
        print("\nTraining...")
        baseline, epoch_losses = train_model_with_monitoring(baseline, train_loader, device, epochs=epochs)
        
        print("\nEvaluating...")
        baseline_acc, baseline_opt = evaluate_model(baseline, test_loader, device, maze_size)
        
        print(f"\n{'='*80}")
        print(f"RESULTS: {size_name.upper()}")
        print(f"{'='*80}")
        print(f"Accuracy: {baseline_acc:.2f}%")
        print(f"Optimality: {baseline_opt:.2f}%")
        print(f"Final loss: {epoch_losses[-1]:.4f}")
        print(f"{'='*80}")
        
        results.append({
            'size': size_name,
            'params': baseline_params,
            'accuracy': baseline_acc,
            'optimality': baseline_opt,
            'final_loss': epoch_losses[-1],
            'epoch_losses': epoch_losses,
        })
        
        del baseline
        torch.cuda.empty_cache()
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'baseline_results_maze{maze_size}.json')
    
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'maze_size': maze_size,
                'n_train': n_train,
                'n_test': n_test,
                'epochs': epochs,
                'seed': seed,
            },
            'results': results,
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline-only Benchmark')
    parser.add_argument('--maze-size', type=int, default=16)
    parser.add_argument('--train', type=int, default=1000)
    parser.add_argument('--test', type=int, default=100)
    parser.add_argument('--min-path', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='experiments/results/baseline_only')
    
    args = parser.parse_args()
    
    run_baseline_benchmark(
        maze_size=args.maze_size,
        n_train=args.train,
        n_test=args.test,
        min_path_length=args.min_path,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output,
    )

