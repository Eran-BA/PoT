#!/usr/bin/env python3
"""
Controller Comparison Script

Compares different depth controller types on Sudoku:
- gru (default HRM-style)
- lstm 
- xlstm (exponential gating)
- mingru (simplified)
- transformer (causal depth attention)

Measures:
- Training speed (steps/sec)
- Memory usage (peak GPU memory)
- Accuracy (cell accuracy on validation)
- Parameter count

Usage:
    python scripts/compare_controllers.py --epochs 3 --batch-size 64

Author: Eran Ben Artzy
Year: 2025
"""

import argparse
import time
import gc
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pot.models.sudoku_solver import HybridPoHHRMSolver
from src.pot.core import CONTROLLER_TYPES, get_controller_info


@dataclass
class ControllerResult:
    """Results for a single controller type."""
    controller_type: str
    param_count: int
    train_loss: float
    val_accuracy: float
    steps_per_sec: float
    peak_memory_mb: float
    time_seconds: float


def create_synthetic_sudoku_data(n_samples: int = 1000, device: str = "cpu"):
    """Create synthetic Sudoku-like data for testing."""
    # Random inputs (0 = empty, 1-9 = filled)
    inputs = torch.randint(0, 10, (n_samples, 81))
    # Random targets (1-9)
    targets = torch.randint(1, 10, (n_samples, 81))
    # Puzzle IDs
    puzzle_ids = torch.zeros(n_samples, dtype=torch.long)
    
    return inputs.to(device), targets.to(device), puzzle_ids.to(device)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_peak_memory_mb() -> float:
    """Get peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
) -> tuple:
    """Train for one epoch, return loss and steps/sec."""
    model.train()
    total_loss = 0.0
    total_steps = 0
    
    start_time = time.time()
    
    for inputs, targets, puzzle_ids in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        puzzle_ids = puzzle_ids.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs, puzzle_ids)
        logits = outputs[0]  # [B, 81, 10]
        
        # Compute loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, 10),
            targets.view(-1),
            ignore_index=0,  # Ignore padding
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_steps += 1
    
    elapsed = time.time() - start_time
    steps_per_sec = total_steps / elapsed if elapsed > 0 else 0
    avg_loss = total_loss / total_steps if total_steps > 0 else 0
    
    return avg_loss, steps_per_sec, elapsed


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: str) -> float:
    """Evaluate model, return cell accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    for inputs, targets, puzzle_ids in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        puzzle_ids = puzzle_ids.to(device)
        
        outputs = model(inputs, puzzle_ids)
        logits = outputs[0]
        preds = logits.argmax(dim=-1)
        
        # Only count non-zero targets (filled cells)
        mask = targets > 0
        correct += ((preds == targets) & mask).sum().item()
        total += mask.sum().item()
    
    return correct / total if total > 0 else 0.0


def test_controller(
    controller_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int = 3,
    d_model: int = 256,
    n_heads: int = 8,
) -> ControllerResult:
    """Test a single controller type."""
    print(f"\n{'='*60}")
    info = get_controller_info(controller_type)
    print(f"Testing: {info['name']}")
    print(f"Description: {info['description']}")
    print(f"{'='*60}")
    
    # Clear memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Create model
    controller_kwargs = {}
    if controller_type == "transformer":
        controller_kwargs = {"max_depth": 32, "n_ctrl_layers": 1, "n_ctrl_heads": 4}
    
    model = HybridPoHHRMSolver(
        d_model=d_model,
        n_heads=n_heads,
        H_layers=1,
        L_layers=1,
        H_cycles=1,
        L_cycles=4,
        controller_type=controller_type,
        controller_kwargs=controller_kwargs,
    ).to(device)
    
    param_count = count_parameters(model)
    print(f"Parameters: {param_count:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training
    total_time = 0
    total_steps_per_sec = 0
    final_loss = 0
    
    for epoch in range(epochs):
        loss, sps, elapsed = train_one_epoch(model, train_loader, optimizer, device)
        total_time += elapsed
        total_steps_per_sec += sps
        final_loss = loss
        print(f"  Epoch {epoch+1}/{epochs}: loss={loss:.4f}, speed={sps:.1f} steps/sec")
    
    avg_steps_per_sec = total_steps_per_sec / epochs
    
    # Evaluation
    val_acc = evaluate(model, val_loader, device)
    print(f"  Validation accuracy: {val_acc*100:.2f}%")
    
    # Memory
    peak_mem = get_peak_memory_mb()
    
    # Cleanup
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return ControllerResult(
        controller_type=controller_type,
        param_count=param_count,
        train_loss=final_loss,
        val_accuracy=val_acc,
        steps_per_sec=avg_steps_per_sec,
        peak_memory_mb=peak_mem,
        time_seconds=total_time,
    )


def print_comparison_table(results: List[ControllerResult]):
    """Print a comparison table."""
    print("\n" + "="*80)
    print("CONTROLLER COMPARISON RESULTS")
    print("="*80)
    
    # Header
    print(f"{'Controller':<12} {'Params':>10} {'Loss':>8} {'Acc%':>8} {'Steps/s':>10} {'Memory MB':>10} {'Time':>8}")
    print("-"*80)
    
    # Sort by accuracy (descending)
    sorted_results = sorted(results, key=lambda x: x.val_accuracy, reverse=True)
    
    for r in sorted_results:
        print(f"{r.controller_type:<12} {r.param_count:>10,} {r.train_loss:>8.4f} "
              f"{r.val_accuracy*100:>7.2f}% {r.steps_per_sec:>10.1f} "
              f"{r.peak_memory_mb:>10.1f} {r.time_seconds:>7.1f}s")
    
    print("="*80)
    
    # Winner
    best = sorted_results[0]
    fastest = max(results, key=lambda x: x.steps_per_sec)
    smallest = min(results, key=lambda x: x.param_count)
    
    print(f"\n🏆 Best accuracy: {best.controller_type} ({best.val_accuracy*100:.2f}%)")
    print(f"⚡ Fastest: {fastest.controller_type} ({fastest.steps_per_sec:.1f} steps/sec)")
    print(f"📦 Smallest: {smallest.controller_type} ({smallest.param_count:,} params)")


def main():
    parser = argparse.ArgumentParser(description="Compare depth controller types")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs per controller")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--train-samples", type=int, default=1000, help="Training samples")
    parser.add_argument("--val-samples", type=int, default=200, help="Validation samples")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--controllers", type=str, nargs="+", default=None,
                       help=f"Controllers to test (default: all). Options: {CONTROLLER_TYPES}")
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Data
    print("Creating synthetic Sudoku data...")
    train_inputs, train_targets, train_ids = create_synthetic_sudoku_data(args.train_samples, device)
    val_inputs, val_targets, val_ids = create_synthetic_sudoku_data(args.val_samples, device)
    
    train_dataset = TensorDataset(train_inputs, train_targets, train_ids)
    val_dataset = TensorDataset(val_inputs, val_targets, val_ids)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Controllers to test
    controllers = args.controllers or CONTROLLER_TYPES
    print(f"Testing controllers: {controllers}")
    
    # Run tests
    results = []
    for ctrl_type in controllers:
        try:
            result = test_controller(
                controller_type=ctrl_type,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=args.epochs,
                d_model=args.d_model,
                n_heads=args.n_heads,
            )
            results.append(result)
        except Exception as e:
            print(f"Error testing {ctrl_type}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print comparison
    if results:
        print_comparison_table(results)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
