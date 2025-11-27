"""
Quick local sanity test for PoT on 30x30 mazes (no HRM comparison)
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from experiments.maze_scaling_benchmark import (
    MazeSolver, generate_dataset, evaluate
)

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Local PoT 30x30] Device: {device}")
    
    # Generate small dataset
    print("\n[1/3] Generating 30x30 mazes...")
    train_data = generate_dataset(maze_size=30, n_samples=30, seed=42)
    test_data = generate_dataset(maze_size=30, n_samples=10, seed=43)
    print(f"  Train: {len(train_data)} mazes, Test: {len(test_data)} mazes")
    
    # Create PoT model with HRM-style features
    print("\n[2/3] Creating PoT model...")
    model = MazeSolver(
        maze_size=30,
        d_model=256,
        n_heads=8,
        d_ff=1024,
        max_inner_iters=4,
        T=4,
        use_poh=True,
        dropout=0.1
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  PoT parameters: {param_count:,}")
    
    # Train
    print("\n[3/3] Training (5 epochs)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    for epoch in range(5):
        model.train()
        total_loss = 0
        for item in train_data:
            maze = torch.tensor(item['maze'], dtype=torch.long, device=device).unsqueeze(0)
            path = torch.tensor(item['path'] + [30 * 30], dtype=torch.long, device=device).unsqueeze(0)
            
            logits = model(maze, path[:, :-1])
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                path[:, 1:].reshape(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_data)
        print(f"  Epoch {epoch+1}/5 - Loss: {avg_loss:.4f}")
    
    # Evaluate
    print("\n[Evaluation]")
    model.eval()
    acc, opt = evaluate(model, test_data, device, maze_size=30)
    print(f"  Accuracy: {acc:.1f}%, Optimality: {opt:.1f}%")
    print("\n✓ Local sanity check complete")

if __name__ == "__main__":
    main()

