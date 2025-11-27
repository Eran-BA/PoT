#!/usr/bin/env python3
"""
Grid2Grid A/B Test - Simple Transformation Tasks

Compares PoH-HRM vs Baseline Transformer on simple grid transformation tasks.
These tasks are similar to HRM paper benchmarks and should achieve high grid accuracy.

Task Types:
- Rotate: Rotate grid 90/180/270 degrees
- Flip: Mirror horizontally/vertically
- Translate: Shift pattern by N cells
- Fill: Fill a region with a color
- Scale: Upscale pattern 2x

This benchmark provides meaningful grid-level accuracy (80-95%) unlike full ARC.

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
from typing import List, Tuple, Dict, Optional

# Setup paths
try:
    from experiments.setup_colab import setup_pot_paths
    repo_root = setup_pot_paths()
except:
    repo_root = Path(__file__).parent.parent
    if repo_root not in sys.path:
        sys.path.insert(0, str(repo_root))

print(f"✓ PoT root: {repo_root}")

# Import PoT modules
from src.pot.modules import PoHConfig, PoHStack, IterRefiner
from src.pot.core.hrm_controller import HRMPointerController, HRMState

print("✓ Successfully imported PoT modules")

# ============================================================================
# Constants
# ============================================================================

GRID_SIZE = 12  # Fixed grid size for simplicity
NUM_COLORS = 10  # Colors 0-9
VOCAB_SIZE = NUM_COLORS + 2  # 0=PAD, 1=EOS, 2-11=colors


# ============================================================================
# Task Generators
# ============================================================================

def generate_random_pattern(size: int = 6, num_colors: int = 3, density: float = 0.4) -> np.ndarray:
    """Generate a random colored pattern."""
    pattern = np.zeros((size, size), dtype=np.int64)
    colors = np.random.choice(range(1, num_colors + 1), size=num_colors, replace=False)
    
    for i in range(size):
        for j in range(size):
            if np.random.random() < density:
                pattern[i, j] = np.random.choice(colors)
    
    return pattern


def pad_to_grid(arr: np.ndarray, grid_size: int = GRID_SIZE) -> np.ndarray:
    """Pad array to fixed grid size."""
    h, w = arr.shape
    padded = np.zeros((grid_size, grid_size), dtype=np.int64)
    padded[:h, :w] = arr
    return padded


# --- Task: Rotate ---
def generate_rotate_task(angle: int = 90) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a rotation task (90, 180, or 270 degrees)."""
    size = np.random.randint(4, 8)
    pattern = generate_random_pattern(size)
    
    k = angle // 90  # Number of 90-degree rotations
    rotated = np.rot90(pattern, k=-k)  # Clockwise rotation
    
    return pad_to_grid(pattern), pad_to_grid(rotated)


# --- Task: Flip ---
def generate_flip_task(axis: str = 'horizontal') -> Tuple[np.ndarray, np.ndarray]:
    """Generate a flip task (horizontal or vertical)."""
    size = np.random.randint(4, 8)
    pattern = generate_random_pattern(size)
    
    if axis == 'horizontal':
        flipped = np.fliplr(pattern)
    else:
        flipped = np.flipud(pattern)
    
    return pad_to_grid(pattern), pad_to_grid(flipped)


# --- Task: Translate ---
def generate_translate_task(dx: int = None, dy: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a translation task."""
    size = np.random.randint(3, 6)
    pattern = generate_random_pattern(size, density=0.5)
    
    if dx is None:
        dx = np.random.randint(-2, 3)
    if dy is None:
        dy = np.random.randint(-2, 3)
    
    output = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64)
    
    for i in range(size):
        for j in range(size):
            ni, nj = i + dy, j + dx
            if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE:
                output[ni, nj] = pattern[i, j]
    
    return pad_to_grid(pattern), output


# --- Task: Fill ---
def generate_fill_task() -> Tuple[np.ndarray, np.ndarray]:
    """Generate a fill task - fill enclosed region with a color."""
    size = 8
    input_grid = np.zeros((size, size), dtype=np.int64)
    
    # Create a border/frame
    border_color = np.random.randint(1, 5)
    fill_color = np.random.randint(5, 10)
    
    # Random rectangle border
    x1 = np.random.randint(1, 3)
    y1 = np.random.randint(1, 3)
    x2 = np.random.randint(5, 7)
    y2 = np.random.randint(5, 7)
    
    # Draw border
    input_grid[y1, x1:x2+1] = border_color
    input_grid[y2, x1:x2+1] = border_color
    input_grid[y1:y2+1, x1] = border_color
    input_grid[y1:y2+1, x2] = border_color
    
    # Output: fill inside
    output_grid = input_grid.copy()
    output_grid[y1+1:y2, x1+1:x2] = fill_color
    
    return pad_to_grid(input_grid), pad_to_grid(output_grid)


# --- Task: Scale ---
def generate_scale_task(factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a scale-up task."""
    size = np.random.randint(3, 5)
    pattern = generate_random_pattern(size, num_colors=4, density=0.6)
    
    # Scale up
    scaled = np.repeat(np.repeat(pattern, factor, axis=0), factor, axis=1)
    
    return pad_to_grid(pattern), pad_to_grid(scaled)


# --- Task: Color Replace ---
def generate_color_replace_task() -> Tuple[np.ndarray, np.ndarray]:
    """Generate a color replacement task."""
    size = np.random.randint(5, 8)
    pattern = generate_random_pattern(size, num_colors=4, density=0.5)
    
    # Pick a color to replace
    unique_colors = np.unique(pattern)
    unique_colors = unique_colors[unique_colors > 0]
    
    if len(unique_colors) >= 2:
        old_color = np.random.choice(unique_colors)
        new_color = np.random.randint(6, 10)
        output = pattern.copy()
        output[pattern == old_color] = new_color
    else:
        output = pattern.copy()
    
    return pad_to_grid(pattern), pad_to_grid(output)


# ============================================================================
# Dataset
# ============================================================================

class Grid2GridDataset(Dataset):
    """Dataset of simple grid transformation tasks."""
    
    TASK_TYPES = ['rotate90', 'rotate180', 'rotate270', 'flip_h', 'flip_v', 
                  'translate', 'fill', 'scale', 'color_replace']
    
    def __init__(self, n_samples: int = 1000, task_types: List[str] = None, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        
        self.task_types = task_types or self.TASK_TYPES
        self.examples = []
        
        samples_per_task = n_samples // len(self.task_types)
        
        for task_type in self.task_types:
            for _ in range(samples_per_task):
                input_grid, output_grid = self._generate_task(task_type)
                
                # Convert to token format: colors 0-9 -> tokens 2-11, 0=PAD
                input_tokens = input_grid + 2
                input_tokens[input_grid == 0] = 0  # Keep PAD as 0
                
                output_tokens = output_grid + 2
                output_tokens[output_grid == 0] = 0
                
                self.examples.append({
                    'task_type': task_type,
                    'input': torch.LongTensor(input_tokens),
                    'output': torch.LongTensor(output_tokens)
                })
        
        np.random.shuffle(self.examples)
        print(f"  Generated {len(self.examples)} examples across {len(self.task_types)} task types")
    
    def _generate_task(self, task_type: str) -> Tuple[np.ndarray, np.ndarray]:
        if task_type == 'rotate90':
            return generate_rotate_task(90)
        elif task_type == 'rotate180':
            return generate_rotate_task(180)
        elif task_type == 'rotate270':
            return generate_rotate_task(270)
        elif task_type == 'flip_h':
            return generate_flip_task('horizontal')
        elif task_type == 'flip_v':
            return generate_flip_task('vertical')
        elif task_type == 'translate':
            return generate_translate_task()
        elif task_type == 'fill':
            return generate_fill_task()
        elif task_type == 'scale':
            return generate_scale_task()
        elif task_type == 'color_replace':
            return generate_color_replace_task()
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        return ex['input'], ex['output']


# ============================================================================
# HRM Router Wrapper
# ============================================================================

class StatefulHRMRouter(nn.Module):
    """Wrapper for HRMPointerController compatible with HeadRouter interface."""
    
    def __init__(self, d_model: int, n_heads: int, T: int = 4):
        super().__init__()
        self.hrm = HRMPointerController(d_model=d_model, n_heads=n_heads, T=T)
        self.state = None
        self.d_model = d_model
        self.n_heads = n_heads
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        
        if self.state is None or self.state.z_L.shape[0] != B:
            self.state = self.hrm.init_state(B, x.device)
        
        alphas, new_state, aux = self.hrm(x, state=self.state, return_aux=True)
        
        self.state = HRMState(
            z_L=new_state.z_L.detach(),
            z_H=new_state.z_H.detach(),
            step=new_state.step.detach() if isinstance(new_state.step, torch.Tensor) else new_state.step
        )
        
        alphas_expanded = alphas.unsqueeze(1).expand(B, T, self.n_heads)
        route_logits = torch.log(alphas_expanded + 1e-8)
        
        return route_logits
    
    def reset_state(self):
        self.state = None


# ============================================================================
# Model Architectures
# ============================================================================

class BaselineGrid2Grid(nn.Module):
    """Standard Transformer for Grid2Grid tasks."""
    
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = None,  # Default: 4 * d_model
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        if d_ff is None:
            d_ff = 4 * d_model
        super().__init__()
        self.d_model = d_model
        self.seq_len = GRID_SIZE * GRID_SIZE
        
        # Token embedding
        self.token_embed = nn.Embedding(VOCAB_SIZE, d_model)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, d_model) * 0.02)
        
        # Transformer encoder with Pre-LN
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, 
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Final norm and output
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, VOCAB_SIZE)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Flatten grid to sequence
        x_flat = x.view(B, -1)
        
        # Embed
        h = self.token_embed(x_flat) + self.pos_embed
        
        # Encode
        h = self.encoder(h)
        h = self.final_norm(h)
        
        # Output
        logits = self.output(h)
        logits = logits.view(B, GRID_SIZE, GRID_SIZE, VOCAB_SIZE)
        
        return logits


class PoHGrid2Grid(nn.Module):
    """PoH-HRM for Grid2Grid tasks."""
    
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = None,  # Default: 2 * d_model (smaller for PoH since iterative)
        num_layers: int = 2,
        R: int = 4,
        T: int = 4,
        dropout: float = 0.1
    ):
        if d_ff is None:
            d_ff = 2 * d_model
        super().__init__()
        self.d_model = d_model
        self.seq_len = GRID_SIZE * GRID_SIZE
        self.R = R
        self.T = T
        
        # Token embedding
        self.token_embed = nn.Embedding(VOCAB_SIZE, d_model)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, d_model) * 0.02)
        
        # PoH stack
        cfg = PoHConfig(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            pos_encoding="none",
        )
        
        self.poh_stack = PoHStack(cfg, depth=num_layers)
        
        # Replace router with HRM
        for block in self.poh_stack.blocks:
            if hasattr(block, 'router'):
                block.router = StatefulHRMRouter(d_model, n_heads, T)
        
        # Iterative refinement
        self.refiner = IterRefiner(
            self.poh_stack,
            max_inner_iters=R,
            outer_residual=True,
            rezero_init=True
        )
        
        # Final norm and output
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, VOCAB_SIZE)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Flatten grid to sequence
        x_flat = x.view(B, -1)
        
        # Embed
        h = self.token_embed(x_flat) + self.pos_embed
        
        # PoH with refinement
        h, _ = self.refiner(h)
        h = self.final_norm(h)
        
        # Output
        logits = self.output(h)
        logits = logits.view(B, GRID_SIZE, GRID_SIZE, VOCAB_SIZE)
        
        return logits


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_model(model, train_loader, device, epochs=50, lr=1e-3, warmup_epochs=5, model_name="Model"):
    """Train a Grid2Grid solver with warmup + cosine annealing."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Warmup + cosine annealing
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for input_grid, output_grid in train_loader:
            input_grid = input_grid.to(device)
            output_grid = output_grid.to(device)
            
            optimizer.zero_grad()
            
            logits = model(input_grid)
            logits_flat = logits.view(-1, VOCAB_SIZE)
            targets_flat = output_grid.view(-1)
            
            loss = criterion(logits_flat, targets_flat)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / max(1, n_batches)
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
    
    return model


def evaluate_model(model, test_loader, device):
    """Evaluate Grid2Grid solver."""
    model.eval()
    
    correct_pixels = 0
    total_pixels = 0
    correct_grids = 0
    total_grids = 0
    
    with torch.no_grad():
        for input_grid, output_grid in test_loader:
            input_grid = input_grid.to(device)
            output_grid = output_grid.to(device)
            
            logits = model(input_grid)
            preds = logits.argmax(dim=-1)
            
            # Pixel accuracy (excluding PAD)
            mask = output_grid != 0
            correct_pixels += ((preds == output_grid) & mask).sum().item()
            total_pixels += mask.sum().item()
            
            # Grid accuracy
            for i in range(preds.shape[0]):
                grid_mask = mask[i]
                if grid_mask.sum() > 0 and (preds[i][grid_mask] == output_grid[i][grid_mask]).all():
                    correct_grids += 1
                total_grids += 1
    
    pixel_acc = 100.0 * correct_pixels / max(1, total_pixels)
    grid_acc = 100.0 * correct_grids / max(1, total_grids)
    
    return pixel_acc, grid_acc


# ============================================================================
# Main Benchmark
# ============================================================================

def run_grid2grid_benchmark(
    n_train: int = 2000,
    n_test: int = 500,
    epochs: int = 50,
    R: int = 4,
    T: int = 4,
    n_heads: int = 4,
    d_model: int = 128,
    d_ff: int = None,
    baseline_layers: int = 4,
    poh_layers: int = 2,
    batch_size: int = 32,
    seed: int = 42,
    baseline_lr: float = 3e-4,
    poh_lr: float = 1e-3
):
    """Run Grid2Grid A/B test."""
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*80}")
    print("GRID2GRID A/B TEST - SIMPLE TRANSFORMATIONS")
    print(f"{'='*80}")
    print(f"Training samples: {n_train}")
    print(f"Test samples: {n_test}")
    print(f"Epochs: {epochs}")
    print(f"PoH config: R={R}, T={T}, heads={n_heads}")
    print(f"Baseline LR: {baseline_lr}, PoH LR: {poh_lr}")
    print(f"{'='*80}\n")
    
    # Generate data
    print("Generating training data...")
    train_dataset = Grid2GridDataset(n_samples=n_train, seed=seed)
    
    print("\nGenerating test data...")
    test_dataset = Grid2GridDataset(n_samples=n_test, seed=seed + 10000)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    results = {}
    
    # Baseline
    print(f"\n{'='*60}")
    print("Training: Baseline Transformer")
    print(f"{'='*60}")
    
    baseline = BaselineGrid2Grid(
        d_model=d_model, 
        n_heads=n_heads, 
        d_ff=d_ff,
        num_layers=baseline_layers
    ).to(device)
    n_params = sum(p.numel() for p in baseline.parameters())
    print(f"Parameters: {n_params/1e6:.2f}M")
    print(f"Config: d_model={d_model}, d_ff={d_ff or 4*d_model}, layers={baseline_layers}")
    print(f"Using LR: {baseline_lr} with warmup + cosine schedule")
    
    baseline_params = n_params
    baseline = train_model(baseline, train_loader, device, epochs, lr=baseline_lr,
                          warmup_epochs=max(5, epochs//10), model_name="Baseline")
    pixel_acc, grid_acc = evaluate_model(baseline, test_loader, device)
    print(f"Pixel Accuracy: {pixel_acc:.2f}%, Grid Accuracy: {grid_acc:.2f}%")
    results['baseline'] = {'pixel_acc': pixel_acc, 'grid_acc': grid_acc, 'params': baseline_params}
    
    # PoH-HRM
    print(f"\n{'='*60}")
    print(f"Training: PoH-HRM (R={R}, T={T})")
    print(f"{'='*60}")
    
    poh = PoHGrid2Grid(
        d_model=d_model, 
        n_heads=n_heads, 
        d_ff=d_ff,
        num_layers=poh_layers,
        R=R, 
        T=T
    ).to(device)
    n_params = sum(p.numel() for p in poh.parameters())
    poh_params = n_params
    print(f"Parameters: {n_params/1e6:.2f}M")
    print(f"Config: d_model={d_model}, d_ff={d_ff or 2*d_model}, layers={poh_layers}, R={R}, T={T}")
    print(f"Using LR: {poh_lr} with warmup + cosine schedule")
    
    poh = train_model(poh, train_loader, device, epochs, lr=poh_lr,
                     warmup_epochs=max(5, epochs//10), model_name="PoH-HRM")
    pixel_acc, grid_acc = evaluate_model(poh, test_loader, device)
    print(f"Pixel Accuracy: {pixel_acc:.2f}%, Grid Accuracy: {grid_acc:.2f}%")
    results['poh'] = {'pixel_acc': pixel_acc, 'grid_acc': grid_acc, 'params': poh_params}
    
    # Summary
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Baseline ({baseline_params/1e6:.2f}M): Pixel={results['baseline']['pixel_acc']:.2f}%, Grid={results['baseline']['grid_acc']:.2f}%")
    print(f"PoH-HRM  ({poh_params/1e6:.2f}M): Pixel={results['poh']['pixel_acc']:.2f}%, Grid={results['poh']['grid_acc']:.2f}%")
    print(f"{'='*80}\n")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Grid2Grid A/B Test')
    parser.add_argument('--train', type=int, default=2000, help='Training samples')
    parser.add_argument('--test', type=int, default=500, help='Test samples')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--R', type=int, default=4, help='PoH refinement steps')
    parser.add_argument('--T', type=int, default=4, help='HRM outer loop period')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
    parser.add_argument('--d-ff', type=int, default=None, help='FFN dimension (default: 4*d_model for baseline, 2*d_model for PoH)')
    parser.add_argument('--baseline-layers', type=int, default=4, help='Number of layers for baseline')
    parser.add_argument('--poh-layers', type=int, default=2, help='Number of layers for PoH')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--baseline-lr', type=float, default=3e-4, help='Learning rate for baseline')
    parser.add_argument('--poh-lr', type=float, default=1e-3, help='Learning rate for PoH')
    
    # Preset for 14M parameter models
    parser.add_argument('--14m', dest='use_14m', action='store_true', 
                       help='Use ~14M parameter preset (d_model=512, d_ff=2048, baseline_layers=4, poh_layers=3)')
    
    args = parser.parse_args()
    
    # Apply 14M preset if requested
    if args.use_14m:
        args.d_model = 512
        args.d_ff = 2048
        args.baseline_layers = 4
        args.poh_layers = 3
        args.heads = 8
        print("🔥 Using 14M parameter preset!")
    
    results = run_grid2grid_benchmark(
        n_train=args.train,
        n_test=args.test,
        epochs=args.epochs,
        R=args.R,
        T=args.T,
        n_heads=args.heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        baseline_layers=args.baseline_layers,
        poh_layers=args.poh_layers,
        batch_size=args.batch_size,
        seed=args.seed,
        baseline_lr=args.baseline_lr,
        poh_lr=args.poh_lr
    )

