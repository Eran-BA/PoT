#!/usr/bin/env python3
"""Quick test: PoH with O(1) memory (last iteration only backprop)"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

# Import from the benchmark script
from experiments.run_12x12_8m_benchmark import (
    PoH, generate_dataset, MazeDS, count_params, evaluate
)

def quick_test():
    print("="*80)
    print("QUICK TEST: PoH with O(1) Memory (Last Iteration Only)")
    print("="*80)
    
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Device: {device}")
    
    # Generate tiny dataset
    print("\nGenerating data...")
    size = 12
    min_path = int(size * size * 0.35)
    train_data = generate_dataset(size, 30, min_path, 42)
    test_data = generate_dataset(size, 10, min_path, 10042)
    
    train_loader = DataLoader(MazeDS(train_data, size), batch_size=8, shuffle=True)
    test_loader = DataLoader(MazeDS(test_data, size), batch_size=4, shuffle=False)
    
    # Create PoH model
    print("\nCreating PoH model...")
    poh = PoH(
        size=size,
        d_model=256,
        n_heads=4,
        d_ff=1024,
        depth=3,
        R=3,
        T=3
    ).to(device)
    
    print(f"Parameters: {count_params(poh) / 1e6:.2f}M")
    
    # Test forward pass with O(1) mode
    print("\nTesting O(1) forward pass...")
    poh.train()
    
    for maze, start, goal, path, _ in train_loader:
        maze = maze.to(device)
        start = start.to(device)
        goal = goal.to(device)
        path = path.to(device)
        
        # Test with last_iter_only=True
        try:
            logits = poh(maze, start, goal, last_iter_only=True)
            print(f"  ✓ O(1) forward pass successful! Output shape: {logits.shape}")
        except Exception as e:
            print(f"  ✗ O(1) forward pass failed: {e}")
            return False
        
        # Test backward
        try:
            # Simple loss: predict next cell from current cell
            B = maze.size(0)
            V = logits.size(-1)  # vocab size (all cells)
            mask = (path[:, 0] != -1) & (path[:, 1] != -1)
            if mask.any():
                curr = path[mask, 0]  # current positions
                target = path[mask, 1]  # next positions
                # Gather logits for current positions
                curr_logits = logits[mask].gather(1, curr.unsqueeze(1).unsqueeze(2).expand(-1, 1, V)).squeeze(1)
                loss = nn.CrossEntropyLoss()(curr_logits, target)
                loss.backward()
                print(f"  ✓ O(1) backward pass successful! Loss: {loss.item():.4f}")
            else:
                print(f"  ⚠️  No valid path positions to compute loss")
        except Exception as e:
            print(f"  ✗ O(1) backward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        break  # Just test one batch
    
    # Quick training test (3 epochs)
    print("\nQuick training test (3 epochs with O(1) memory)...")
    optimizer = torch.optim.AdamW(poh.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    for epoch in range(3):
        poh.train()
        total_loss = 0.0
        n_batches = 0
        
        for maze, start, goal, path, _ in train_loader:
            maze = maze.to(device)
            start = start.to(device)
            goal = goal.to(device)
            path = path.to(device)
            
            optimizer.zero_grad()
            
            # Forward with O(1) memory
            logits = poh(maze, start, goal, last_iter_only=True)
            
            # Simple loss: predict next cell from current cell
            V = logits.size(-1)
            mask = (path[:, 0] != -1) & (path[:, 1] != -1)
            if mask.any():
                curr = path[mask, 0]
                target = path[mask, 1]
                curr_logits = logits[mask].gather(1, curr.unsqueeze(1).unsqueeze(2).expand(-1, 1, V)).squeeze(1)
                loss = criterion(curr_logits, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
        
        avg_loss = total_loss / max(1, n_batches)
        print(f"  Epoch {epoch+1}/3, Loss: {avg_loss:.4f}")
    
    # Evaluate
    print("\nEvaluating...")
    acc, opt = evaluate(poh, test_loader, device, size)
    print(f"  Accuracy: {acc:.2f}%, Optimality: {opt:.2f}%")
    
    print("\n" + "="*80)
    print("✓ O(1) MEMORY MODE WORKS!")
    print("="*80)
    return True

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)

