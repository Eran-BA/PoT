"""
Maze Solving A/B Test: PoH-HRM vs Baseline

Tests the ability of models to find paths through mazes of varying complexity.
The task requires sequential decision-making and multi-step planning - ideal
for hierarchical reasoning.

Author: Eran Ben Artzy
Date: October 2025
"""

import os
import sys
import time
import csv
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pot.core.hrm_controller import HRMPointerController, HRMState


# ========== Maze Generation ==========

def generate_maze(size: int, seed: int = None) -> np.ndarray:
    """Generate a maze using DFS algorithm.
    
    Returns:
        maze: (size, size) array where 0=wall, 1=path, 2=start, 3=goal
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize with walls
    maze = np.zeros((size, size), dtype=np.int32)
    
    # Start from a random position
    start_x, start_y = 1, 1
    maze[start_x, start_y] = 1
    
    # DFS stack
    stack = [(start_x, start_y)]
    visited = set([(start_x, start_y)])
    
    directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
    
    while stack:
        x, y = stack[-1]
        
        # Get unvisited neighbors
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 < nx < size-1 and 0 < ny < size-1 and (nx, ny) not in visited:
                neighbors.append((nx, ny, dx, dy))
        
        if neighbors:
            # Choose random neighbor
            nx, ny, dx, dy = neighbors[np.random.randint(len(neighbors))]
            
            # Carve path
            maze[x + dx//2, y + dy//2] = 1
            maze[nx, ny] = 1
            
            visited.add((nx, ny))
            stack.append((nx, ny))
        else:
            stack.pop()
    
    # Set start and goal
    maze[1, 1] = 2  # Start
    maze[size-2, size-2] = 3  # Goal
    
    return maze


def solve_maze_bfs(maze: np.ndarray) -> List[Tuple[int, int]]:
    """Find shortest path using BFS."""
    from collections import deque
    
    size = maze.shape[0]
    start = (1, 1)
    goal = (size-2, size-2)
    
    queue = deque([(start, [start])])
    visited = {start}
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while queue:
        (x, y), path = queue.popleft()
        
        if (x, y) == goal:
            return path
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < size and 0 <= ny < size and 
                maze[nx, ny] != 0 and (nx, ny) not in visited):
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))
    
    return []  # No path found


def generate_maze_dataset(num_samples: int, maze_size: int, seed: int = 42):
    """Generate dataset of mazes with solutions."""
    np.random.seed(seed)
    
    data = []
    for i in range(num_samples):
        maze = generate_maze(maze_size, seed=seed + i)
        solution = solve_maze_bfs(maze)
        
        if len(solution) > 0:
            data.append({
                'maze': maze,
                'solution': solution,
                'length': len(solution)
            })
    
    return data


# ========== Models ==========

class PoHBlock(nn.Module):
    """PoH transformer block with HRM controller."""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, max_iters: int, T: int = 4
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_iters = max_iters
        self.T = T

        # HRM controller for routing
        self.controller = HRMPointerController(
            d_model=d_model,
            n_heads=n_heads,
            T=T,
            topk=None,
            temperature_init=2.0,
            temperature_min=0.7,
            entropy_reg=1e-3,
            use_layernorm=True,
            dropout=0.1
        )

        # Per-head attention
        self.q_proj = nn.ModuleList(
            [nn.Linear(d_model, d_model // n_heads) for _ in range(n_heads)]
        )
        self.k_proj = nn.ModuleList(
            [nn.Linear(d_model, d_model // n_heads) for _ in range(n_heads)]
        )
        self.v_proj = nn.ModuleList(
            [nn.Linear(d_model, d_model // n_heads) for _ in range(n_heads)]
        )
        self.out_proj = nn.Linear(d_model, d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), 
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, z):
        B = z.size(0)
        device = z.device
        
        # Initialize HRM state
        hrm_state = self.controller.init_state(B, device)
        
        for iter_idx in range(self.max_iters):
            # HRM controller routing
            alphas, hrm_state, aux = self.controller(
                x=z,
                state=hrm_state,
                return_aux=True
            )
            
            # Per-head attention
            head_outs = []
            for h_idx in range(self.n_heads):
                q = self.q_proj[h_idx](z)
                k = self.k_proj[h_idx](z)
                v = self.v_proj[h_idx](z)

                scores = torch.einsum("btd,bsd->bts", q, k) / (
                    (self.d_model // self.n_heads) ** 0.5
                )
                attn = F.softmax(scores, dim=-1)
                out = torch.einsum("bts,bsd->btd", attn, v)
                head_outs.append(out)

            # Concat heads and project
            head_outs_concat = torch.cat(head_outs, dim=-1)
            attn_out = self.out_proj(head_outs_concat)

            # Residual + norm
            z = self.ln1(z + attn_out)
            z_refined = z

            # FFN
            z = self.ln2(z + self.ffn(z))

        return z_refined


class MazeSolver(nn.Module):
    """Sequence-to-sequence maze solver."""
    
    def __init__(
        self,
        maze_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 1024,
        max_inner_iters: int = 3,
        T: int = 4,
        use_poh: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        self.maze_size = maze_size
        self.d_model = d_model
        self.use_poh = use_poh
        
        # Encode maze (flatten 2D maze to sequence)
        self.cell_embed = nn.Embedding(4, d_model)  # 0=wall, 1=path, 2=start, 3=goal
        self.pos_embed = nn.Embedding(maze_size * maze_size, d_model)
        
        # Encoder
        if use_poh:
            self.encoder = PoHBlock(d_model, n_heads, d_ff, max_inner_iters, T=T)
        else:
            self.encoder_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.encoder_ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model)
            )
            self.encoder_ln1 = nn.LayerNorm(d_model)
            self.encoder_ln2 = nn.LayerNorm(d_model)
        
        # Decoder for path prediction
        self.step_embed = nn.Embedding(maze_size * maze_size + 2, d_model)  # +2 for SOS/EOS
        self.decoder_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.decoder_cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.decoder_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.decoder_ln1 = nn.LayerNorm(d_model)
        self.decoder_ln2 = nn.LayerNorm(d_model)
        self.decoder_ln3 = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, maze_size * maze_size + 1)  # +1 for EOS token
        
    def encode_maze(self, maze):
        """Encode maze as sequence.
        
        Args:
            maze: (B, H, W) tensor
        Returns:
            encoded: (B, H*W, d_model) tensor
        """
        B, H, W = maze.shape
        
        # Flatten maze
        maze_flat = maze.view(B, H * W)  # (B, H*W)
        
        # Embed cells
        cell_emb = self.cell_embed(maze_flat)  # (B, H*W, d_model)
        
        # Add position embeddings
        positions = torch.arange(H * W, device=maze.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embed(positions)
        
        x = cell_emb + pos_emb
        
        # Encode
        if self.use_poh:
            encoded = self.encoder(x)
        else:
            # Standard transformer encoder
            attn_out, _ = self.encoder_attn(x, x, x)
            x = self.encoder_ln1(x + attn_out)
            ffn_out = self.encoder_ffn(x)
            encoded = self.encoder_ln2(x + ffn_out)
        
        return encoded
    
    def decode_step(self, encoded, prev_steps, step_idx):
        """Decode one step of the path."""
        # Embed previous steps
        step_emb = self.step_embed(prev_steps)  # (B, seq_len, d_model)
        
        # Self-attention
        # Create causal mask
        seq_len = step_emb.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=step_emb.device), diagonal=1).bool()
        
        attn_out, _ = self.decoder_attn(step_emb, step_emb, step_emb, attn_mask=causal_mask)
        step_emb = self.decoder_ln1(step_emb + attn_out)
        
        # Cross-attention to encoded maze
        cross_out, _ = self.decoder_cross_attn(step_emb, encoded, encoded)
        step_emb = self.decoder_ln2(step_emb + cross_out)
        
        # FFN
        ffn_out = self.decoder_ffn(step_emb)
        step_emb = self.decoder_ln3(step_emb + ffn_out)
        
        # Project to next position logits
        logits = self.output_proj(step_emb[:, -1, :])  # (B, H*W+1)
        
        return logits
    
    def forward(self, maze, target_path=None, max_steps=100):
        """
        Args:
            maze: (B, H, W) tensor
            target_path: (B, max_path_len) tensor of position indices (optional, for training)
        """
        B = maze.size(0)
        device = maze.device
        
        # Encode maze
        encoded = self.encode_maze(maze)
        
        if self.training and target_path is not None:
            # Teacher forcing during training
            # Prepend SOS token
            SOS_TOKEN = self.maze_size * self.maze_size
            sos = torch.full((B, 1), SOS_TOKEN, device=device, dtype=torch.long)
            decoder_input = torch.cat([sos, target_path[:, :-1]], dim=1)
            
            # Decode all steps at once
            logits_list = []
            for t in range(decoder_input.size(1)):
                logits = self.decode_step(encoded, decoder_input[:, :t+1], t)
                logits_list.append(logits)
            
            logits = torch.stack(logits_list, dim=1)  # (B, seq_len, vocab_size)
            return logits
        else:
            # Autoregressive generation during inference
            SOS_TOKEN = self.maze_size * self.maze_size
            EOS_TOKEN = self.maze_size * self.maze_size
            
            generated = torch.full((B, 1), SOS_TOKEN, device=device, dtype=torch.long)
            
            for _ in range(max_steps):
                logits = self.decode_step(encoded, generated, generated.size(1) - 1)
                next_token = logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if all sequences have EOS
                if (next_token == EOS_TOKEN).all():
                    break
            
            return generated[:, 1:]  # Remove SOS token


# ========== Training & Evaluation ==========

def path_to_indices(path: List[Tuple[int, int]], maze_size: int) -> List[int]:
    """Convert path coordinates to flat indices."""
    return [x * maze_size + y for x, y in path]


def evaluate_paths(pred_paths, true_paths, maze_size):
    """Evaluate predicted paths."""
    correct = 0
    total = len(pred_paths)
    path_accuracies = []
    
    for pred, true in zip(pred_paths, true_paths):
        # Convert to indices
        true_indices = set(path_to_indices(true, maze_size))
        pred_indices = set(pred.cpu().tolist())
        
        # Check if reached goal
        goal_idx = (maze_size - 2) * maze_size + (maze_size - 2)
        reached_goal = goal_idx in pred_indices
        
        # Path overlap
        overlap = len(true_indices & pred_indices) / max(len(true_indices), 1)
        path_accuracies.append(overlap)
        
        if reached_goal and overlap > 0.8:
            correct += 1
    
    return {
        'success_rate': correct / total if total > 0 else 0,
        'avg_overlap': np.mean(path_accuracies) if path_accuracies else 0,
    }


def train_and_evaluate(
    model, model_name, train_data, test_data,
    epochs=100, lr=1e-3, device='cpu', maze_size=11
):
    """Train and evaluate a maze solver."""
    print(f"\nTraining {model_name}...")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    start_time = time.time()
    best_success = 0
    
    pbar = tqdm(range(epochs), desc=model_name)
    for epoch in pbar:
        model.train()
        total_loss = 0
        
        # Training
        for batch_idx in range(0, len(train_data), 32):
            batch = train_data[batch_idx:batch_idx + 32]
            
            # Prepare batch
            mazes = torch.tensor(np.stack([d['maze'] for d in batch]), device=device)
            
            # Prepare target paths (padded)
            max_path_len = max(len(d['solution']) for d in batch)
            EOS_TOKEN = maze_size * maze_size
            
            target_paths = []
            for d in batch:
                path_indices = path_to_indices(d['solution'], maze_size)
                # Pad with EOS
                padded = path_indices + [EOS_TOKEN] * (max_path_len - len(path_indices))
                target_paths.append(padded)
            
            target_paths = torch.tensor(target_paths, device=device)
            
            # Forward
            logits = model(mazes, target_paths)
            
            # Loss
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_paths.reshape(-1))
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / max(len(train_data) // 32, 1)
        
        # Evaluation every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                pred_paths = []
                true_paths = []
                
                for batch_idx in range(0, min(len(test_data), 100), 16):
                    batch = test_data[batch_idx:batch_idx + 16]
                    mazes = torch.tensor(np.stack([d['maze'] for d in batch]), device=device)
                    
                    # Generate paths
                    preds = model(mazes, max_steps=maze_size * 2)
                    
                    pred_paths.extend(preds)
                    true_paths.extend([d['solution'] for d in batch])
                
                metrics = evaluate_paths(pred_paths, true_paths, maze_size)
                success = metrics['success_rate']
                overlap = metrics['avg_overlap']
                
                if success > best_success:
                    best_success = success
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.3f}',
                    'success': f'{success:.2%}',
                    'overlap': f'{overlap:.2%}'
                })
        else:
            pbar.set_postfix({'loss': f'{avg_loss:.3f}'})
    
    training_time = (time.time() - start_time) / 60
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        pred_paths = []
        true_paths = []
        
        for batch_idx in range(0, len(test_data), 16):
            batch = test_data[batch_idx:batch_idx + 16]
            mazes = torch.tensor(np.stack([d['maze'] for d in batch]), device=device)
            
            preds = model(mazes, max_steps=maze_size * 2)
            pred_paths.extend(preds)
            true_paths.extend([d['solution'] for d in batch])
        
        final_metrics = evaluate_paths(pred_paths, true_paths, maze_size)
    
    return {
        'best_success': best_success,
        'final_success': final_metrics['success_rate'],
        'final_overlap': final_metrics['avg_overlap'],
        'time_min': training_time,
        'params_M': sum(p.numel() for p in model.parameters()) / 1e6,
    }


# ========== Main A/B Test ==========

def run_ab_test(maze_size, train_samples=1000, test_samples=200, R=4, T=4, n_heads=4,
                epochs=100, seed=42):
    """Run A/B test for a specific maze size."""
    
    print(f"\n{'='*80}")
    print(f"A/B Test: Maze Size {maze_size}x{maze_size}")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Maze size: {maze_size}x{maze_size}")
    print(f"  Training samples: {train_samples:,}")
    print(f"  Test samples: {test_samples:,}")
    print(f"  PoH n_heads: {n_heads}")
    print(f"  PoH R (refinement steps): {R}")
    print(f"  PoH T (HRM outer loop period): {T}")
    print(f"  Epochs: {epochs}")
    print(f"  Seed: {seed}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    # Generate data
    print(f"\nGenerating mazes...")
    train_data = generate_maze_dataset(train_samples, maze_size, seed=seed)
    test_data = generate_maze_dataset(test_samples, maze_size, seed=seed+10000)
    
    print(f"  Train: {len(train_data)} mazes")
    print(f"  Test: {len(test_data)} mazes")
    print(f"  Avg path length: {np.mean([d['length'] for d in train_data]):.1f}")
    
    # Build models
    print(f"\nBuilding models...")
    
    baseline = MazeSolver(
        maze_size=maze_size,
        d_model=256,
        n_heads=n_heads,
        d_ff=1024,
        max_inner_iters=1,
        T=1,
        use_poh=False
    ).to(device)
    
    poh = MazeSolver(
        maze_size=maze_size,
        d_model=256,
        n_heads=n_heads,
        d_ff=1024,
        max_inner_iters=R,
        T=T,
        use_poh=True
    ).to(device)
    
    # Train baseline
    baseline_results = train_and_evaluate(
        baseline, "Baseline", train_data, test_data,
        epochs=epochs, device=device, maze_size=maze_size
    )
    
    # Train PoH
    poh_results = train_and_evaluate(
        poh, f"PoH-HRM (R={R}, T={T})", train_data, test_data,
        epochs=epochs, device=device, maze_size=maze_size
    )
    
    # Results
    print(f"\n{'='*80}")
    print("📊 RESULTS")
    print(f"{'='*80}")
    
    print(f"\n📚 Baseline (Standard Transformer)")
    print(f"  Parameters: {baseline_results['params_M']:.2f}M")
    print(f"  Best Success Rate: {baseline_results['best_success']:.2%}")
    print(f"  Final Success Rate: {baseline_results['final_success']:.2%}")
    print(f"  Final Path Overlap: {baseline_results['final_overlap']:.2%}")
    print(f"  Training time: {baseline_results['time_min']:.2f} min")
    
    print(f"\n🔬 PoH with HRM (R={R}, T={T}, n_heads={n_heads})")
    print(f"  Parameters: {poh_results['params_M']:.2f}M")
    print(f"  Best Success Rate: {poh_results['best_success']:.2%}")
    print(f"  Final Success Rate: {poh_results['final_success']:.2%}")
    print(f"  Final Path Overlap: {poh_results['final_overlap']:.2%}")
    print(f"  Training time: {poh_results['time_min']:.2f} min")
    
    # Comparison
    delta_success = poh_results['final_success'] - baseline_results['final_success']
    delta_pct = (delta_success / baseline_results['final_success'] * 100) if baseline_results['final_success'] > 0 else 0
    
    print(f"\n{'='*80}")
    print("📈 COMPARISON")
    print(f"{'='*80}")
    print(f"Success Rate delta: {delta_success:+.2%} ({delta_pct:+.1f}%)")
    
    if delta_success > 0.05:
        winner = "PoH-HRM"
        print(f"🏆 Winner: {winner} by {delta_success:.2%}")
    elif delta_success < -0.05:
        winner = "Baseline"
        print(f"🏆 Winner: {winner} by {-delta_success:.2%}")
    else:
        winner = "Tie"
        print(f"⚖️  TIE (difference < 5%)")
    
    return {
        'maze_size': maze_size,
        'baseline': baseline_results,
        'poh': poh_results,
        'delta_success': delta_success,
        'winner': winner
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Maze Solving A/B Test')
    parser.add_argument('--sizes', type=int, nargs='+', default=[11, 15],
                        help='Maze sizes to test (default: 11 15)')
    parser.add_argument('--R', type=int, default=4, help='PoH refinement steps (default: 4)')
    parser.add_argument('--T', type=int, default=4, help='HRM outer loop period (default: 4)')
    parser.add_argument('--n-heads', type=int, default=4, help='Number of heads (default: 4)')
    parser.add_argument('--train-samples', type=int, default=1000, help='Training samples')
    parser.add_argument('--test-samples', type=int, default=200, help='Test samples')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save-dir', type=str, default='experiments/results/maze_ab',
                        help='Save directory')
    
    args = parser.parse_args()
    
    print("="*80)
    print("MAZE SOLVING A/B TEST")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  n_heads = {args.n_heads}")
    print(f"  R = {args.R} (refinement steps)")
    print(f"  T = {args.T} (HRM outer loop period)")
    print(f"\nTesting on maze sizes: {args.sizes}")
    print("="*80)
    
    # Run tests for each size
    all_results = []
    
    for maze_size in args.sizes:
        result = run_ab_test(
            maze_size,
            train_samples=args.train_samples,
            test_samples=args.test_samples,
            R=args.R,
            T=args.T,
            n_heads=args.n_heads,
            epochs=args.epochs,
            seed=args.seed
        )
        all_results.append(result)
    
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    results_file = os.path.join(args.save_dir, f"maze_ab_R{args.R}_T{args.T}_nheads{args.n_heads}.csv")
    
    with open(results_file, 'w', newline='') as f:
        fieldnames = [
            'timestamp', 'maze_size', 'model', 'R', 'T', 'n_heads',
            'best_success', 'final_success', 'final_overlap',
            'time_min', 'params_M', 'delta_success', 'winner'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for result in all_results:
            # Baseline row
            writer.writerow({
                'timestamp': timestamp,
                'maze_size': result['maze_size'],
                'model': 'Baseline',
                'R': '-',
                'T': '-',
                'n_heads': args.n_heads,
                'best_success': f"{result['baseline']['best_success']:.4f}",
                'final_success': f"{result['baseline']['final_success']:.4f}",
                'final_overlap': f"{result['baseline']['final_overlap']:.4f}",
                'time_min': f"{result['baseline']['time_min']:.2f}",
                'params_M': f"{result['baseline']['params_M']:.2f}",
                'delta_success': "0.0000",
                'winner': ''
            })
            
            # PoH row
            writer.writerow({
                'timestamp': timestamp,
                'maze_size': result['maze_size'],
                'model': 'PoH-HRM',
                'R': args.R,
                'T': args.T,
                'n_heads': args.n_heads,
                'best_success': f"{result['poh']['best_success']:.4f}",
                'final_success': f"{result['poh']['final_success']:.4f}",
                'final_overlap': f"{result['poh']['final_overlap']:.4f}",
                'time_min': f"{result['poh']['time_min']:.2f}",
                'params_M': f"{result['poh']['params_M']:.2f}",
                'delta_success': f"{result['delta_success']:+.4f}",
                'winner': result['winner']
            })
    
    # Summary
    print(f"\n\n{'='*80}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Size':<10} {'Baseline':<15} {'PoH-HRM':<15} {'Delta':<15} {'Winner':<10}")
    print("-"*80)
    for result in all_results:
        print(f"{result['maze_size']:<10} "
              f"{result['baseline']['final_success']:<15.2%} "
              f"{result['poh']['final_success']:<15.2%} "
              f"{result['delta_success']:+<15.2%} "
              f"{result['winner']:<10}")
    
    print(f"\n✅ Results saved to: {results_file}")
    print("="*80)


if __name__ == "__main__":
    main()

