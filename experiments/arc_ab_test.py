#!/usr/bin/env python3
"""
ARC (Abstraction and Reasoning Corpus) A/B Test

Compares PoH-HRM vs Baseline Transformer on ARC-style visual reasoning tasks.
Based on HRM's ARC benchmark: https://github.com/sapientinc/HRM

ARC tasks require:
- Pattern recognition across examples
- Abstract rule inference
- Generalization to unseen test cases

This is a harder benchmark than mazes - requires true abstract reasoning.

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
import json
import hashlib
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
# ARC Constants (matching HRM)
# ============================================================================

ARC_GRID_SIZE = 30  # Max grid size
ARC_VOCAB_SIZE = 12  # PAD=0, EOS=1, colors 0-9 -> 2-11
ARC_SEQ_LEN = ARC_GRID_SIZE * ARC_GRID_SIZE  # 900 tokens


# ============================================================================
# ARC Dataset Utilities
# ============================================================================

def download_arc_dataset(data_dir: str = "data/arc"):
    """Download ARC-AGI dataset from HuggingFace or GitHub."""
    import urllib.request
    import zipfile
    
    os.makedirs(data_dir, exist_ok=True)
    
    arc_url = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"
    zip_path = os.path.join(data_dir, "arc.zip")
    
    if not os.path.exists(os.path.join(data_dir, "ARC-AGI-master")):
        print("Downloading ARC-AGI dataset...")
        urllib.request.urlretrieve(arc_url, zip_path)
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(data_dir)
        
        os.remove(zip_path)
        print("✓ ARC dataset ready")
    else:
        print("✓ ARC dataset already exists")
    
    return os.path.join(data_dir, "ARC-AGI-master", "data")


def load_arc_task(filepath: str) -> Dict:
    """Load a single ARC task from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def grid_to_tensor(grid: List[List[int]], pad_to: int = ARC_GRID_SIZE) -> torch.Tensor:
    """Convert ARC grid to padded tensor.
    
    Maps colors 0-9 to tokens 2-11 (0=PAD, 1=EOS).
    """
    arr = np.array(grid, dtype=np.int64)
    h, w = arr.shape
    
    # Map colors: 0-9 -> 2-11
    arr = arr + 2
    
    # Pad to max size
    padded = np.zeros((pad_to, pad_to), dtype=np.int64)
    padded[:h, :w] = arr
    
    # Add EOS markers
    if h < pad_to:
        padded[h, :w] = 1  # EOS row
    if w < pad_to:
        padded[:h, w] = 1  # EOS column
    
    return torch.LongTensor(padded)


def tensor_to_grid(tensor: torch.Tensor) -> List[List[int]]:
    """Convert tensor back to ARC grid."""
    arr = tensor.cpu().numpy()
    
    # Find EOS boundaries
    h, w = arr.shape
    for i in range(h):
        if arr[i, 0] == 1:  # EOS marker
            h = i
            break
    for j in range(w):
        if arr[0, j] == 1:
            w = j
            break
    
    # Map tokens back: 2-11 -> 0-9
    grid = arr[:h, :w] - 2
    grid = np.clip(grid, 0, 9)
    
    return grid.tolist()


class ARCDataset(Dataset):
    """ARC dataset for training/evaluation."""
    
    def __init__(self, data_dir: str, split: str = "training", max_tasks: int = None):
        """
        Args:
            data_dir: Path to ARC data directory
            split: "training" or "evaluation"
            max_tasks: Limit number of tasks (for quick testing)
        """
        self.data_dir = data_dir
        self.split = split
        
        # Load all tasks
        split_dir = os.path.join(data_dir, split)
        task_files = sorted([f for f in os.listdir(split_dir) if f.endswith('.json')])
        
        if max_tasks:
            task_files = task_files[:max_tasks]
        
        self.examples = []
        self.task_ids = []
        
        for task_file in tqdm(task_files, desc=f"Loading {split}"):
            task_path = os.path.join(split_dir, task_file)
            task = load_arc_task(task_path)
            task_id = task_file.replace('.json', '')
            
            # Each task has train examples and test examples
            # For training: use train examples as input-output pairs
            # For evaluation: use train examples as context, predict test output
            
            for i, example in enumerate(task.get('train', [])):
                input_grid = grid_to_tensor(example['input'])
                output_grid = grid_to_tensor(example['output'])
                
                self.examples.append({
                    'task_id': task_id,
                    'example_idx': i,
                    'input': input_grid,
                    'output': output_grid,
                    'is_test': False
                })
                self.task_ids.append(task_id)
            
            # Include test examples for evaluation
            for i, example in enumerate(task.get('test', [])):
                input_grid = grid_to_tensor(example['input'])
                output_grid = grid_to_tensor(example['output'])
                
                self.examples.append({
                    'task_id': task_id,
                    'example_idx': i,
                    'input': input_grid,
                    'output': output_grid,
                    'is_test': True
                })
        
        print(f"  Loaded {len(self.examples)} examples from {len(set(self.task_ids))} tasks")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        return ex['input'], ex['output'], ex['is_test']


# ============================================================================
# HRM Router Wrapper (same as maze benchmark)
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

class BaselineARCSolver(nn.Module):
    """Standard Transformer for ARC tasks."""
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        num_layers: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # Token embedding (vocab size = 12)
        self.token_embed = nn.Embedding(ARC_VOCAB_SIZE, d_model)
        
        # 2D positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, ARC_SEQ_LEN, d_model) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output = nn.Linear(d_model, ARC_VOCAB_SIZE)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input grid [B, H, W] with tokens 0-11
        
        Returns:
            logits: [B, H, W, vocab_size]
        """
        B = x.shape[0]
        
        # Flatten grid to sequence
        x_flat = x.view(B, -1)  # [B, seq_len]
        
        # Embed
        h = self.token_embed(x_flat) + self.pos_embed  # [B, seq_len, d_model]
        
        # Encode
        h = self.encoder(h)
        
        # Project to vocab
        logits = self.output(h)  # [B, seq_len, vocab_size]
        
        # Reshape back to grid
        logits = logits.view(B, ARC_GRID_SIZE, ARC_GRID_SIZE, ARC_VOCAB_SIZE)
        
        return logits


class PoHARCSolver(nn.Module):
    """PoH-HRM for ARC tasks."""
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        num_layers: int = 4,
        R: int = 4,
        T: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.R = R
        self.T = T
        
        # Token embedding
        self.token_embed = nn.Embedding(ARC_VOCAB_SIZE, d_model)
        
        # 2D positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, ARC_SEQ_LEN, d_model) * 0.02)
        
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
        
        # Output projection
        self.output = nn.Linear(d_model, ARC_VOCAB_SIZE)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Flatten and embed
        x_flat = x.view(B, -1)
        h = self.token_embed(x_flat) + self.pos_embed
        
        # PoH with refinement
        h, _ = self.refiner(h)
        
        # Project to vocab
        logits = self.output(h)
        logits = logits.view(B, ARC_GRID_SIZE, ARC_GRID_SIZE, ARC_VOCAB_SIZE)
        
        return logits


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_model(model, train_loader, device, epochs=50, lr=1e-3):
    """Train an ARC solver."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore PAD
    
    for epoch in range(epochs):
        total_loss = 0
        for input_grid, output_grid, is_test in train_loader:
            input_grid = input_grid.to(device)
            output_grid = output_grid.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            logits = model(input_grid)  # [B, H, W, vocab]
            
            # Compute loss
            logits_flat = logits.view(-1, ARC_VOCAB_SIZE)
            targets_flat = output_grid.view(-1)
            
            loss = criterion(logits_flat, targets_flat)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model


def evaluate_model(model, test_loader, device):
    """Evaluate ARC solver."""
    model.eval()
    
    correct_pixels = 0
    total_pixels = 0
    correct_grids = 0
    total_grids = 0
    
    with torch.no_grad():
        for input_grid, output_grid, is_test in test_loader:
            input_grid = input_grid.to(device)
            output_grid = output_grid.to(device)
            
            # Predict
            logits = model(input_grid)
            preds = logits.argmax(dim=-1)  # [B, H, W]
            
            # Compute accuracy (excluding PAD tokens)
            mask = output_grid != 0  # Non-PAD positions
            
            correct_pixels += ((preds == output_grid) & mask).sum().item()
            total_pixels += mask.sum().item()
            
            # Grid-level accuracy
            for i in range(preds.shape[0]):
                grid_mask = mask[i]
                if (preds[i][grid_mask] == output_grid[i][grid_mask]).all():
                    correct_grids += 1
                total_grids += 1
    
    pixel_acc = 100.0 * correct_pixels / total_pixels if total_pixels > 0 else 0
    grid_acc = 100.0 * correct_grids / total_grids if total_grids > 0 else 0
    
    return pixel_acc, grid_acc


# ============================================================================
# Main Benchmark
# ============================================================================

def run_arc_benchmark(
    max_tasks: int = 100,
    epochs: int = 50,
    R: int = 4,
    T: int = 4,
    n_heads: int = 8,
    d_model: int = 256,
    batch_size: int = 16,
    seed: int = 42
):
    """Run ARC A/B test."""
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*80}")
    print("ARC A/B TEST - ABSTRACT REASONING")
    print(f"{'='*80}")
    print(f"Max tasks: {max_tasks}")
    print(f"Epochs: {epochs}")
    print(f"PoH config: R={R}, T={T}, heads={n_heads}")
    print(f"{'='*80}\n")
    
    # Download and load data
    data_dir = download_arc_dataset()
    
    print("\nLoading training data...")
    train_dataset = ARCDataset(data_dir, "training", max_tasks=max_tasks)
    
    print("\nLoading evaluation data...")
    eval_dataset = ARCDataset(data_dir, "evaluation", max_tasks=max_tasks // 4)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    results = {}
    
    # Baseline
    print(f"\n{'='*60}")
    print("Training: Baseline Transformer")
    print(f"{'='*60}")
    
    baseline = BaselineARCSolver(d_model=d_model, n_heads=n_heads).to(device)
    n_params = sum(p.numel() for p in baseline.parameters())
    print(f"Parameters: {n_params/1e6:.2f}M")
    
    baseline = train_model(baseline, train_loader, device, epochs)
    pixel_acc, grid_acc = evaluate_model(baseline, eval_loader, device)
    print(f"Pixel Accuracy: {pixel_acc:.2f}%, Grid Accuracy: {grid_acc:.2f}%")
    results['baseline'] = {'pixel_acc': pixel_acc, 'grid_acc': grid_acc}
    
    # PoH-HRM
    print(f"\n{'='*60}")
    print(f"Training: PoH-HRM (R={R}, T={T})")
    print(f"{'='*60}")
    
    poh = PoHARCSolver(d_model=d_model, n_heads=n_heads, R=R, T=T).to(device)
    n_params = sum(p.numel() for p in poh.parameters())
    print(f"Parameters: {n_params/1e6:.2f}M")
    
    poh = train_model(poh, train_loader, device, epochs)
    pixel_acc, grid_acc = evaluate_model(poh, eval_loader, device)
    print(f"Pixel Accuracy: {pixel_acc:.2f}%, Grid Accuracy: {grid_acc:.2f}%")
    results['poh'] = {'pixel_acc': pixel_acc, 'grid_acc': grid_acc}
    
    # Summary
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Baseline: Pixel={results['baseline']['pixel_acc']:.2f}%, Grid={results['baseline']['grid_acc']:.2f}%")
    print(f"PoH-HRM:  Pixel={results['poh']['pixel_acc']:.2f}%, Grid={results['poh']['grid_acc']:.2f}%")
    print(f"{'='*80}\n")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ARC A/B Test')
    parser.add_argument('--max-tasks', type=int, default=100, help='Max tasks to load')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--R', type=int, default=4, help='PoH refinement steps')
    parser.add_argument('--T', type=int, default=4, help='HRM outer loop period')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d-model', type=int, default=256, help='Model dimension')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    results = run_arc_benchmark(
        max_tasks=args.max_tasks,
        epochs=args.epochs,
        R=args.R,
        T=args.T,
        n_heads=args.heads,
        d_model=args.d_model,
        batch_size=args.batch_size,
        seed=args.seed
    )

