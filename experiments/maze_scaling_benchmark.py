#!/usr/bin/env python3
"""
Maze Size Scaling Benchmark
============================

Following the HRM paper methodology, this script benchmarks model performance
across increasing maze sizes (8x8 to 30x30) to test hierarchical reasoning
capabilities at scale.

Based on: Hierarchical Reasoning Model (HRM) maze benchmarks
Reference: arXiv 2506.21734
"""

import sys
import os

# Setup paths for PoT imports (must be first!)
try:
    # Try importing the setup helper
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    from setup_colab import setup_pot_paths
    pot_root = setup_pot_paths()
    print(f"✓ PoT root: {pot_root}")
except Exception as e:
    # Fallback: simple path setup
    cwd = os.getcwd()
    if os.path.exists(os.path.join(cwd, 'src', 'pot')):
        sys.path.insert(0, cwd)
    elif os.path.exists(os.path.join(os.path.dirname(cwd), 'src', 'pot')):
        sys.path.insert(0, os.path.dirname(cwd))
    else:
        print(f"ERROR: {e}")
        sys.exit(1)

# Now safe to import other modules
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from collections import deque
import time
import json
from pathlib import Path

# Try to import transformers for BERT, fallback if not available
try:
    from transformers import BertModel, BertConfig
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("Warning: transformers library not available. BERT baseline will be skipped.")

# Import PoT modules (correct paths!)
from src.pot.core.hrm_controller import HRMPointerController, HRMState
print("✓ Successfully imported PoT modules")


# ============================================================================
# PoH Block with HRM Controller
# ============================================================================

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
                attn = torch.nn.functional.softmax(scores, dim=-1)
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


# ============================================================================
# Maze Generation (BFS for optimal paths)
# ============================================================================

def generate_maze(size, wall_prob=0.3, seed=None):
    """Generate a random maze with guaranteed solution."""
    if seed is not None:
        np.random.seed(seed)
    
    # Create maze
    maze = np.random.rand(size, size) < wall_prob
    
    # Set start and goal
    start = (0, 0)
    goal = (size-1, size-1)
    maze[start] = False
    maze[goal] = False
    
    # Ensure there's a path using BFS
    visited = set()
    queue = deque([start])
    visited.add(start)
    parent = {start: None}
    
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < size and 0 <= ny < size:
                if not maze[nx, ny] and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    parent[(nx, ny)] = current
                    queue.append((nx, ny))
    
    # If no path exists, carve one
    if goal not in parent:
        current = start
        while current != goal:
            next_pos = (
                current[0] + (1 if current[0] < goal[0] else 0),
                current[1] + (1 if current[1] < goal[1] else 0)
            )
            maze[next_pos] = False
            parent[next_pos] = current
            current = next_pos
    
    # Extract optimal path
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parent.get(current)
    path.reverse()
    
    return maze, path, start, goal


def maze_to_tensor(maze, start, goal):
    """Convert maze to tensor encoding."""
    # 0=wall, 1=path, 2=start, 3=goal
    tensor = np.zeros_like(maze, dtype=np.int64)
    tensor[maze] = 0  # walls
    tensor[~maze] = 1  # paths
    tensor[start] = 2
    tensor[goal] = 3
    return tensor


# ============================================================================
# Models
# ============================================================================

class MazeSolver(nn.Module):
    """Transformer-based maze solver with optional PoH."""
    
    def __init__(
        self,
        maze_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 1024,
        max_inner_iters: int = 1,
        T: int = 1,
        use_poh: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.maze_size = maze_size
        self.d_model = d_model
        self.use_poh = use_poh
        
        # Maze encoding
        self.cell_embed = nn.Embedding(4, d_model)  # 0=wall, 1=path, 2=start, 3=goal
        self.pos_embed = nn.Embedding(maze_size * maze_size, d_model)
        
        # Encoder
        if use_poh:
            self.encoder = PoHBlock(d_model, n_heads, d_ff, max_inner_iters, T=T)
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Path decoder (autoregressive)
        self.step_embed = nn.Embedding(maze_size * maze_size + 2, d_model)  # +2 for SOS/EOS
        self.decoder_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.decoder_cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.decoder_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.out_proj = nn.Linear(d_model, maze_size * maze_size + 1)  # +1 for EOS
        
        self.sos_token = maze_size * maze_size
        self.eos_token = maze_size * maze_size
    
    def encode_maze(self, maze):
        """Encode maze into latent representation."""
        B, H, W = maze.shape
        
        # Flatten maze
        flat_maze = maze.view(B, -1)
        
        # Embeddings
        x = self.cell_embed(flat_maze)
        pos_ids = torch.arange(H * W, device=maze.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embed(pos_ids)
        
        # Encode
        if self.use_poh:
            encoded = self.encoder(x)
        else:
            encoded = self.encoder(x)
        
        return encoded
    
    def decode_step(self, step_tokens, encoded, mask=None):
        """Single decoder step."""
        step_emb = self.step_embed(step_tokens)
        
        # Self-attention
        attn_out, _ = self.decoder_attn(step_emb, step_emb, step_emb, attn_mask=mask)
        step_emb = step_emb + attn_out
        
        # Cross-attention to encoded maze
        cross_out, _ = self.decoder_cross_attn(step_emb, encoded, encoded)
        step_emb = step_emb + cross_out
        
        # FFN
        ffn_out = self.decoder_ffn(step_emb)
        step_emb = step_emb + ffn_out
        
        # Output projection
        logits = self.out_proj(step_emb)
        return logits
    
    def forward(self, maze, path=None, teacher_forcing_ratio=0.5):
        """Forward pass with optional teacher forcing."""
        B = maze.size(0)
        device = maze.device
        
        # Encode maze
        encoded = self.encode_maze(maze)
        
        if path is None:
            # Inference mode
            return self.generate_path(encoded, max_len=self.maze_size * self.maze_size)
        
        # Training mode
        max_len = path.size(1)
        sos = torch.full((B, 1), self.sos_token, dtype=torch.long, device=device)
        decoder_input = torch.cat([sos, path[:, :-1]], dim=1)
        
        # Create causal mask
        mask = torch.triu(torch.ones(max_len, max_len, device=device), diagonal=1).bool()
        mask = mask.masked_fill(mask, float('-inf'))
        
        # Decode
        logits = self.decode_step(decoder_input, encoded, mask)
        
        return logits
    
    def generate_path(self, encoded, max_len=100):
        """Generate path autoregressively."""
        B = encoded.size(0)
        device = encoded.device
        
        generated = torch.full((B, 1), self.sos_token, dtype=torch.long, device=device)
        
        for _ in range(max_len):
            logits = self.decode_step(generated, encoded)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences generated EOS
            if (next_token == self.eos_token).all():
                break
        
        return generated[:, 1:]  # Remove SOS token


class BERTMazeSolver(nn.Module):
    """BERT-based maze solver for comparison."""
    
    def __init__(
        self,
        maze_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 1024,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if not BERT_AVAILABLE:
            raise ImportError("transformers library required for BERT baseline")
        
        self.maze_size = maze_size
        self.d_model = d_model
        
        # Embedding layers
        self.cell_embed = nn.Embedding(4, d_model)
        self.pos_embed = nn.Embedding(maze_size * maze_size, d_model)
        
        # BERT encoder (with extended position embeddings for large mazes)
        max_seq_len = max(1024, maze_size * maze_size + 10)  # Support large mazes
        
        bert_config = BertConfig(
            hidden_size=d_model,
            num_hidden_layers=num_layers,
            num_attention_heads=n_heads,
            intermediate_size=d_ff,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=max_seq_len,  # Fix for large mazes
        )
        self.bert = BertModel(bert_config)
        
        # Decoder
        self.step_embed = nn.Embedding(maze_size * maze_size + 2, d_model)
        self.decoder_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.decoder_cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.decoder_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.out_proj = nn.Linear(d_model, maze_size * maze_size + 1)
        
        self.sos_token = maze_size * maze_size
        self.eos_token = maze_size * maze_size
    
    def encode_maze(self, maze):
        """Encode maze using BERT."""
        B, H, W = maze.shape
        flat_maze = maze.view(B, -1)
        
        x = self.cell_embed(flat_maze)
        pos_ids = torch.arange(H * W, device=maze.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embed(pos_ids)
        
        bert_output = self.bert(inputs_embeds=x)
        encoded = bert_output.last_hidden_state
        return encoded
    
    def decode_step(self, step_tokens, encoded, mask=None):
        """Single decoder step."""
        step_emb = self.step_embed(step_tokens)
        
        attn_out, _ = self.decoder_attn(step_emb, step_emb, step_emb, attn_mask=mask)
        step_emb = step_emb + attn_out
        
        cross_out, _ = self.decoder_cross_attn(step_emb, encoded, encoded)
        step_emb = step_emb + cross_out
        
        ffn_out = self.decoder_ffn(step_emb)
        step_emb = step_emb + ffn_out
        
        logits = self.out_proj(step_emb)
        return logits
    
    def forward(self, maze, path=None):
        """Forward pass."""
        B = maze.size(0)
        device = maze.device
        
        encoded = self.encode_maze(maze)
        
        if path is None:
            return self.generate_path(encoded, max_len=self.maze_size * self.maze_size)
        
        max_len = path.size(1)
        sos = torch.full((B, 1), self.sos_token, dtype=torch.long, device=device)
        decoder_input = torch.cat([sos, path[:, :-1]], dim=1)
        
        mask = torch.triu(torch.ones(max_len, max_len, device=device), diagonal=1).bool()
        mask = mask.masked_fill(mask, float('-inf'))
        
        logits = self.decode_step(decoder_input, encoded, mask)
        return logits
    
    def generate_path(self, encoded, max_len=100):
        """Generate path autoregressively."""
        B = encoded.size(0)
        device = encoded.device
        
        generated = torch.full((B, 1), self.sos_token, dtype=torch.long, device=device)
        
        for _ in range(max_len):
            logits = self.decode_step(generated, encoded)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            if (next_token == self.eos_token).all():
                break
        
        return generated[:, 1:]


# ============================================================================
# Training & Evaluation
# ============================================================================

def generate_dataset(maze_size, n_samples, wall_prob=0.3, seed=None):
    """Generate maze dataset."""
    if seed is not None:
        np.random.seed(seed)
    
    data = []
    for i in range(n_samples):
        maze, path, start, goal = generate_maze(maze_size, wall_prob, seed=seed+i if seed else None)
        maze_tensor = maze_to_tensor(maze, start, goal)
        
        # Convert path to linear indices
        path_indices = [p[0] * maze_size + p[1] for p in path]
        
        data.append({
            'maze': maze_tensor,
            'path': path_indices,
            'length': len(path)
        })
    
    return data


def train_epoch(model, data, optimizer, device, maze_size):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for item in data:
        maze = torch.tensor(item['maze'], dtype=torch.long, device=device).unsqueeze(0)
        path = torch.tensor(item['path'] + [maze_size * maze_size], dtype=torch.long, device=device).unsqueeze(0)
        
        optimizer.zero_grad()
        logits = model(maze, path)
        
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            path.view(-1),
            ignore_index=-1
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data)


def evaluate(model, data, device, maze_size):
    """Evaluate model on dataset."""
    model.eval()
    correct = 0
    optimal = 0
    
    with torch.no_grad():
        for item in data:
            maze = torch.tensor(item['maze'], dtype=torch.long, device=device).unsqueeze(0)
            true_path = item['path']
            
            pred_path = model(maze, None)
            pred_path = pred_path[0].cpu().tolist()
            
            # Remove EOS token
            if maze_size * maze_size in pred_path:
                eos_idx = pred_path.index(maze_size * maze_size)
                pred_path = pred_path[:eos_idx]
            
            # Check if path reaches goal
            if pred_path and pred_path[-1] == true_path[-1]:
                correct += 1
                
                # Check if path is optimal length
                if len(pred_path) == len(true_path):
                    optimal += 1
    
    accuracy = correct / len(data)
    optimality = optimal / len(data)
    
    return accuracy, optimality


def train_and_evaluate(model, model_name, train_data, test_data, epochs, device, maze_size):
    """Train and evaluate a model."""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params / 1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_accuracy = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_data, optimizer, device, maze_size)
        test_acc, test_opt = evaluate(model, test_data, device, maze_size)
        scheduler.step()
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Acc: {test_acc:.2%} | Opt: {test_opt:.2%}")
    
    training_time = (time.time() - start_time) / 60
    
    final_acc, final_opt = evaluate(model, test_data, device, maze_size)
    
    print(f"\nFinal Performance:")
    print(f"  Accuracy: {final_acc:.2%}")
    print(f"  Optimality: {final_opt:.2%}")
    print(f"  Training time: {training_time:.2f} min")
    
    return {
        'params_M': params / 1e6,
        'best_accuracy': best_accuracy,
        'final_accuracy': final_acc,
        'final_optimality': final_opt,
        'time_min': training_time
    }


# ============================================================================
# Scaling Benchmark
# ============================================================================

def run_scaling_benchmark(
    maze_sizes=[8, 12, 16, 20, 24, 30],
    n_train=1000,
    n_test=200,
    R=4,
    T=4,
    n_heads=4,
    epochs=50,
    seed=42
):
    """Run scaling benchmark across maze sizes."""
    
    print(f"\n{'='*80}")
    print(f"MAZE SCALING BENCHMARK (HRM Paper Protocol)")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Maze sizes: {maze_sizes}")
    print(f"  Training mazes per size: {n_train}")
    print(f"  Test mazes per size: {n_test}")
    print(f"  PoH R (refinement steps): {R}")
    print(f"  PoH T (HRM outer loop period): {T}")
    print(f"  PoH n_heads: {n_heads}")
    print(f"  Epochs per size: {epochs}")
    print(f"  Seed: {seed}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    results = {
        'maze_sizes': maze_sizes,
        'baseline': {'accuracy': [], 'optimality': [], 'params': []},
        'poh': {'accuracy': [], 'optimality': [], 'params': []},
        'config': {
            'n_train': n_train,
            'n_test': n_test,
            'R': R,
            'T': T,
            'n_heads': n_heads,
            'epochs': epochs,
            'seed': seed
        }
    }
    
    if BERT_AVAILABLE:
        results['bert'] = {'accuracy': [], 'optimality': [], 'params': []}
    
    for maze_size in maze_sizes:
        print(f"\n{'='*80}")
        print(f"MAZE SIZE: {maze_size}x{maze_size}")
        print(f"{'='*80}")
        
        # Generate data
        print(f"\nGenerating data...")
        # Harder mazes: increased wall probability for better differentiation
        train_data = generate_dataset(maze_size, n_train, wall_prob=0.45, seed=seed)
        test_data = generate_dataset(maze_size, n_test, wall_prob=0.45, seed=seed+10000)
        
        print(f"  Train: {len(train_data)} mazes")
        print(f"  Test: {len(test_data)} mazes")
        print(f"  Avg optimal path length: {np.mean([d['length'] for d in train_data]):.1f}")
        
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
        
        baseline_params = sum(p.numel() for p in baseline.parameters())
        print(f"  Baseline parameters: {baseline_params / 1e6:.2f}M")
        
        # Create PoH-HRM first to get target parameter count
        poh = MazeSolver(
            maze_size=maze_size,
            d_model=256,
            n_heads=n_heads,
            d_ff=1024,
            max_inner_iters=R,
            T=T,
            use_poh=True
        ).to(device)
        
        poh_params = sum(p.numel() for p in poh.parameters())
        print(f"  PoH-HRM parameters: {poh_params / 1e6:.2f}M")
        
        # BERT (if available) - dynamically adjust layers for parameter parity with PoH
        bert = None
        if BERT_AVAILABLE:
            # Skip BERT if maze is too large (>1024 tokens is risky even with extended embeddings)
            if maze_size * maze_size > 1024:
                print(f"  ⚠️  Skipping BERT (maze too large: {maze_size}×{maze_size}={maze_size*maze_size} tokens > 1024)")
            else:
                try:
                    # Start with fewer layers for larger mazes
                    # Heuristic: larger mazes need fewer BERT layers to match PoH params
                    if maze_size <= 12:
                        bert_layers = 3
                    elif maze_size <= 20:
                        bert_layers = 2
                    else:
                        bert_layers = 1
                    
                    # Try to match PoH parameters by adjusting BERT config
                    for attempt_layers in [bert_layers, max(1, bert_layers-1), 1]:
                        if attempt_layers < 1:
                            break
                        
                        bert_test = BERTMazeSolver(
                            maze_size=maze_size,
                            d_model=256,
                            n_heads=n_heads,
                            d_ff=1024,
                            num_layers=attempt_layers
                        ).to(device)
                        
                        bert_params = sum(p.numel() for p in bert_test.parameters())
                        parity_pct = abs(bert_params - poh_params) / poh_params * 100
                        
                        if parity_pct < 20:  # Accept if within 20%
                            bert = bert_test
                            print(f"  BERT parameters: {bert_params / 1e6:.2f}M (layers={attempt_layers}, parity={parity_pct:.1f}%)")
                            break
                        else:
                            del bert_test
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    if bert is None:
                        print(f"  Warning: Could not achieve parameter parity (<20%) for BERT at maze size {maze_size}")
                        # Use smallest BERT anyway for comparison
                        bert = BERTMazeSolver(
                            maze_size=maze_size,
                            d_model=256,
                            n_heads=n_heads,
                            d_ff=1024,
                            num_layers=1
                        ).to(device)
                        bert_params = sum(p.numel() for p in bert.parameters())
                        parity_pct = abs(bert_params - poh_params) / poh_params * 100
                        print(f"  BERT parameters: {bert_params / 1e6:.2f}M (layers=1, parity={parity_pct:.1f}%) [best effort]")
                        
                except Exception as e:
                    print(f"  Warning: Could not create BERT model: {e}")
                    bert = None
        
        # Train and evaluate
        baseline_results = train_and_evaluate(
            baseline, "Baseline (Transformer)", train_data, test_data,
            epochs=epochs, device=device, maze_size=maze_size
        )
        
        results['baseline']['accuracy'].append(baseline_results['final_accuracy'])
        results['baseline']['optimality'].append(baseline_results['final_optimality'])
        results['baseline']['params'].append(baseline_results['params_M'])
        
        if bert is not None:
            bert_results = train_and_evaluate(
                bert, "BERT", train_data, test_data,
                epochs=epochs, device=device, maze_size=maze_size
            )
            
            results['bert']['accuracy'].append(bert_results['final_accuracy'])
            results['bert']['optimality'].append(bert_results['final_optimality'])
            results['bert']['params'].append(bert_results['params_M'])
        
        poh_results = train_and_evaluate(
            poh, f"PoH-HRM (R={R}, T={T})", train_data, test_data,
            epochs=epochs, device=device, maze_size=maze_size
        )
        
        results['poh']['accuracy'].append(poh_results['final_accuracy'])
        results['poh']['optimality'].append(poh_results['final_optimality'])
        results['poh']['params'].append(poh_results['params_M'])
    
    return results


def plot_results(results, save_path='experiments/results/maze_scaling.png'):
    """Plot scaling results."""
    maze_sizes = results['maze_sizes']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1.plot(maze_sizes, results['baseline']['accuracy'], 'o-', label='Baseline', linewidth=2, markersize=8)
    if 'bert' in results:
        ax1.plot(maze_sizes, results['bert']['accuracy'], 's-', label='BERT', linewidth=2, markersize=8)
    ax1.plot(maze_sizes, results['poh']['accuracy'], '^-', label='PoH-HRM', linewidth=2, markersize=8)
    ax1.set_xlabel('Maze Size', fontsize=12)
    ax1.set_ylabel('Path Finding Accuracy', fontsize=12)
    ax1.set_title('Accuracy vs Maze Size', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Optimality plot
    ax2.plot(maze_sizes, results['baseline']['optimality'], 'o-', label='Baseline', linewidth=2, markersize=8)
    if 'bert' in results:
        ax2.plot(maze_sizes, results['bert']['optimality'], 's-', label='BERT', linewidth=2, markersize=8)
    ax2.plot(maze_sizes, results['poh']['optimality'], '^-', label='PoH-HRM', linewidth=2, markersize=8)
    ax2.set_xlabel('Maze Size', fontsize=12)
    ax2.set_ylabel('Optimal Path Rate', fontsize=12)
    ax2.set_title('Path Optimality vs Maze Size', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {save_path}")
    
    plt.close()


def print_summary(results):
    """Print results summary."""
    print(f"\n{'='*80}")
    print("📊 SCALING BENCHMARK RESULTS")
    print(f"{'='*80}")
    
    maze_sizes = results['maze_sizes']
    
    print(f"\n{'Size':<10} {'Baseline':>15} {'BERT':>15} {'PoH-HRM':>15}")
    print(f"{'-'*10} {'-'*15} {'-'*15} {'-'*15}")
    
    for i, size in enumerate(maze_sizes):
        baseline_acc = results['baseline']['accuracy'][i]
        poh_acc = results['poh']['accuracy'][i]
        
        line = f"{size}x{size:<7} {baseline_acc:>14.2%}"
        
        if 'bert' in results:
            bert_acc = results['bert']['accuracy'][i]
            line += f" {bert_acc:>14.2%}"
        else:
            line += f" {'N/A':>15}"
        
        line += f" {poh_acc:>14.2%}"
        
        # Mark winner
        if 'bert' in results:
            winner_acc = max(baseline_acc, bert_acc, poh_acc)
        else:
            winner_acc = max(baseline_acc, poh_acc)
        
        if poh_acc == winner_acc:
            line += " 🏆"
        
        print(line)
    
    # Average performance
    print(f"\n{'Average':<10} {np.mean(results['baseline']['accuracy']):>14.2%}", end='')
    if 'bert' in results:
        print(f" {np.mean(results['bert']['accuracy']):>14.2%}", end='')
    else:
        print(f" {'N/A':>15}", end='')
    print(f" {np.mean(results['poh']['accuracy']):>14.2%}")
    
    # Performance at largest maze
    print(f"\n🎯 Performance on {maze_sizes[-1]}x{maze_sizes[-1]} maze (most challenging):")
    print(f"   Baseline: {results['baseline']['accuracy'][-1]:.2%}")
    if 'bert' in results:
        print(f"   BERT: {results['bert']['accuracy'][-1]:.2%}")
    print(f"   PoH-HRM: {results['poh']['accuracy'][-1]:.2%}")
    
    delta = results['poh']['accuracy'][-1] - results['baseline']['accuracy'][-1]
    print(f"   PoH-HRM advantage: {delta:+.2%}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Maze Scaling Benchmark')
    parser.add_argument('--maze-sizes', type=int, nargs='+', default=[8, 12, 16, 20, 24, 30],
                        help='Maze sizes to test')
    parser.add_argument('--train', type=int, default=1000, help='Training mazes per size')
    parser.add_argument('--test', type=int, default=200, help='Test mazes per size')
    parser.add_argument('--R', type=int, default=4, help='PoH refinement steps')
    parser.add_argument('--T', type=int, default=4, help='HRM outer loop period')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs per size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='experiments/results/maze_scaling',
                        help='Output path (without extension)')
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_scaling_benchmark(
        maze_sizes=args.maze_sizes,
        n_train=args.train,
        n_test=args.test,
        R=args.R,
        T=args.T,
        n_heads=args.heads,
        epochs=args.epochs,
        seed=args.seed
    )
    
    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(f'{args.output}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {args.output}.json")
    
    # Plot
    plot_results(results, save_path=f'{args.output}.png')
    
    # Print summary
    print_summary(results)


if __name__ == '__main__':
    main()

