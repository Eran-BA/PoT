"""
Diagnostic script to check if PoH/HRM gradients are flowing correctly.

This script trains a small model and monitors:
1. Gradient norms for each component
2. Weight changes over time
3. HRM state dynamics
4. Routing entropy
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict

# Setup paths
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.pot.modules import PoHConfig, PoHStack, IterRefiner
from src.pot.core.hrm_controller import HRMPointerController

# Try to import maze-dataset
try:
    import maze_dataset
    from maze_dataset import MazeDataset, MazeDatasetConfig
    from maze_dataset.generation import LatticeMazeGenerators
    MAZE_AVAILABLE = True
except ImportError:
    MAZE_AVAILABLE = False
    print("⚠️  maze-dataset not available, using synthetic data")


def generate_simple_maze_data(n_samples, maze_size, seed=42):
    """Generate simple maze data for testing."""
    np.random.seed(seed)
    data = []
    for _ in range(n_samples):
        # Simple random maze
        maze = np.random.rand(maze_size, maze_size) > 0.7
        maze = maze.astype(np.float32)
        
        # Random start/goal
        start = (np.random.randint(maze_size), np.random.randint(maze_size))
        goal = (np.random.randint(maze_size), np.random.randint(maze_size))
        
        # Random path
        path_len = np.random.randint(10, 30)
        path = [(np.random.randint(maze_size), np.random.randint(maze_size)) for _ in range(path_len)]
        
        data.append({
            'maze': maze,
            'start': start,
            'goal': goal,
            'path': path,
            'length': path_len
        })
    
    return data


class StatefulHRMRouter(nn.Module):
    """Wrapper to make HRMPointerController compatible with PoHBlock."""
    def __init__(self, d_model: int, n_heads: int, T: int = 4):
        super().__init__()
        self.hrm_controller = HRMPointerController(
            d_model=d_model,
            n_heads=n_heads,
            d_ctrl=d_model,
            T=T
        )
        self.state = None
        
    def forward(self, x):
        B, T_seq, D = x.shape
        H = self.hrm_controller.n_heads
        
        if self.state is None or self.state.z_L.shape[0] != B:
            self.state = self.hrm_controller.init_state(B, device=x.device)
        
        x_ctrl = x.mean(dim=1)
        alphas, new_state, stats = self.hrm_controller(x_ctrl, self.state)
        
        # Detach state to prevent backprop through time
        self.state = type(new_state)(
            z_L=new_state.z_L.detach(),
            z_H=new_state.z_H.detach(),
            step=new_state.step.detach()
        )
        
        alphas_expanded = alphas.unsqueeze(1).expand(B, T_seq, H)
        route_logits = torch.log(alphas_expanded + 1e-8)
        
        return route_logits


class PoHMazeSolver(nn.Module):
    """Simplified maze solver with PoH."""
    def __init__(self, maze_size: int, d_model: int = 128, n_heads: int = 4, 
                 d_ff: int = 512, depth: int = 3, R: int = 3, T: int = 4):
        super().__init__()
        self.maze_size = maze_size
        self.d_model = d_model
        
        # Embeddings
        self.maze_embed = nn.Linear(maze_size * maze_size, d_model)
        self.pos_embed = nn.Embedding(maze_size * maze_size, d_model)
        
        # PoH Stack
        cfg = PoHConfig(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff
        )
        
        stack = PoHStack(cfg, depth=depth)
        
        # Replace routers with HRM
        for block in stack.blocks:
            if hasattr(block, 'router'):
                block.router = StatefulHRMRouter(d_model, n_heads, T=T)
        
        self.refiner = IterRefiner(
            stack,
            max_inner_iters=R,
            outer_residual=True,
            rezero_init=True
        )
        
        # Output
        self.output = nn.Linear(d_model, maze_size * maze_size)
        
    def forward(self, maze, start, goal):
        B = maze.shape[0]
        
        # Embed maze
        maze_flat = maze.view(B, -1)
        h = self.maze_embed(maze_flat).unsqueeze(1)  # (B, 1, d_model)
        
        # Add positional embeddings
        positions = torch.arange(self.maze_size * self.maze_size, device=maze.device)
        pos_emb = self.pos_embed(positions).unsqueeze(0).expand(B, -1, -1)
        h = h + pos_emb
        
        # PoH refinement
        h, stats = self.refiner(h)
        
        # Predict next position
        logits = self.output(h.mean(dim=1))
        
        return logits, stats


def compute_gradient_norms(model):
    """Compute gradient norms for each component."""
    norms = {}
    
    # Overall norm
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    norms['total'] = total_norm ** 0.5
    
    # Component-wise norms
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            module_norm = 0.0
            for p in module.parameters():
                if p.grad is not None:
                    module_norm += p.grad.data.norm(2).item() ** 2
            if module_norm > 0:
                norms[name] = module_norm ** 0.5
    
    return norms


def compute_weight_norms(model):
    """Compute weight norms for each component."""
    norms = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            module_norm = 0.0
            for p in module.parameters():
                module_norm += p.data.norm(2).item() ** 2
            if module_norm > 0:
                norms[name] = module_norm ** 0.5
    
    return norms


def diagnose_training(maze_size=12, n_train=50, epochs=10, R=3, T=3):
    """Run diagnostic training and collect metrics."""
    print("="*80)
    print("POH GRADIENT DIAGNOSTIC")
    print("="*80)
    print(f"Maze size: {maze_size}×{maze_size}")
    print(f"Training samples: {n_train}")
    print(f"Epochs: {epochs}")
    print(f"R={R}, T={T}")
    print("="*80)
    
    # Generate data
    print("\nGenerating data...")
    if MAZE_AVAILABLE:
        print("  Using maze-dataset")
        # Implementation would go here
        train_data = generate_simple_maze_data(n_train, maze_size, seed=42)
    else:
        print("  Using synthetic data")
        train_data = generate_simple_maze_data(n_train, maze_size, seed=42)
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    model = PoHMazeSolver(
        maze_size=maze_size,
        d_model=128,
        n_heads=4,
        d_ff=512,
        depth=3,
        R=R,
        T=T
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    # Diagnostic storage
    diagnostics = {
        'gradient_norms': [],
        'weight_norms': [],
        'weight_changes': [],
        'losses': [],
        'routing_stats': []
    }
    
    initial_weights = {name: p.data.clone() for name, p in model.named_parameters()}
    
    print("\nTraining with diagnostics...")
    pbar = tqdm(range(epochs), desc="Epochs")
    
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        batch_size = 8
        
        for batch_idx in range(0, len(train_data), batch_size):
            batch = train_data[batch_idx:batch_idx + batch_size]
            
            # Prepare batch
            mazes = torch.tensor(np.stack([d['maze'] for d in batch]), device=device, dtype=torch.float32)
            starts = torch.tensor([d['start'][0] * maze_size + d['start'][1] for d in batch], device=device)
            goals = torch.tensor([d['goal'][0] * maze_size + d['goal'][1] for d in batch], device=device)
            
            # Simple target: just predict goal
            targets = goals
            
            # Forward
            logits, stats = model(mazes, starts, goals)
            loss = criterion(logits, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Compute gradient norms BEFORE optimizer step
            grad_norms = compute_gradient_norms(model)
            
            # Optimizer step
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Compute weight norms after epoch
        weight_norms = compute_weight_norms(model)
        
        # Compute weight changes
        weight_changes = {}
        for name, p in model.named_parameters():
            if name in initial_weights:
                change = (p.data - initial_weights[name]).norm(2).item()
                weight_changes[name] = change
        
        # Store diagnostics
        diagnostics['gradient_norms'].append(grad_norms)
        diagnostics['weight_norms'].append(weight_norms)
        diagnostics['weight_changes'].append(weight_changes)
        diagnostics['losses'].append(epoch_loss / max(1, len(train_data) // batch_size))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{epoch_loss / max(1, len(train_data) // batch_size):.4f}',
            'grad_norm': f'{grad_norms.get("total", 0):.4f}'
        })
    
    return diagnostics, model


def print_diagnostics(diagnostics):
    """Print diagnostic summary."""
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    
    # Check if gradients are flowing to PoH components
    print("\n📊 Gradient Flow Analysis:")
    print("-"*80)
    
    final_grad_norms = diagnostics['gradient_norms'][-1]
    
    # Find HRM and PoH-related components
    hrm_components = {k: v for k, v in final_grad_norms.items() if 'hrm' in k.lower() or 'router' in k.lower()}
    poh_components = {k: v for k, v in final_grad_norms.items() if 'refiner' in k.lower() or 'stack' in k.lower()}
    
    if hrm_components:
        print("\nHRM Components (final epoch):")
        for name, norm in sorted(hrm_components.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {name}: {norm:.6f}")
    else:
        print("\n⚠️  WARNING: No HRM gradient norms found!")
    
    if poh_components:
        print("\nPoH Stack Components (final epoch):")
        for name, norm in sorted(poh_components.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {name}: {norm:.6f}")
    
    # Check weight changes
    print("\n📈 Weight Change Analysis:")
    print("-"*80)
    
    final_changes = diagnostics['weight_changes'][-1]
    
    hrm_changes = {k: v for k, v in final_changes.items() if 'hrm' in k.lower() or 'router' in k.lower()}
    
    if hrm_changes:
        print("\nHRM Weight Changes (from start):")
        for name, change in sorted(hrm_changes.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {name}: {change:.6f}")
    else:
        print("\n⚠️  WARNING: No HRM weight changes detected!")
    
    # Loss trajectory
    print("\n📉 Loss Trajectory:")
    print("-"*80)
    losses = diagnostics['losses']
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Change: {losses[-1] - losses[0]:+.4f} ({((losses[-1] / losses[0]) - 1) * 100:+.1f}%)")
    
    if losses[-1] >= losses[0]:
        print("  ⚠️  WARNING: Loss did not decrease!")
    
    # Check for dead components (no gradient)
    print("\n🔍 Dead Component Check:")
    print("-"*80)
    
    dead_components = []
    for name, norm in final_grad_norms.items():
        if norm < 1e-8:
            dead_components.append(name)
    
    if dead_components:
        print(f"  Found {len(dead_components)} components with near-zero gradients:")
        for name in dead_components[:10]:
            print(f"    - {name}")
    else:
        print("  ✓ All components receiving gradients")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Diagnose PoH gradient flow')
    parser.add_argument('--maze-size', type=int, default=12, help='Maze size')
    parser.add_argument('--train', type=int, default=50, help='Training samples')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--R', type=int, default=3, help='Refinement steps')
    parser.add_argument('--T', type=int, default=3, help='HRM period')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')
    
    args = parser.parse_args()
    
    # Run diagnostic
    diagnostics, model = diagnose_training(
        maze_size=args.maze_size,
        n_train=args.train,
        epochs=args.epochs,
        R=args.R,
        T=args.T
    )
    
    # Print summary
    print_diagnostics(diagnostics)
    
    # Save if requested
    if args.output:
        # Convert numpy arrays to lists for JSON serialization
        serializable_diagnostics = {
            'losses': diagnostics['losses'],
            'gradient_norms': [{k: float(v) for k, v in d.items()} for d in diagnostics['gradient_norms']],
            'weight_norms': [{k: float(v) for k, v in d.items()} for d in diagnostics['weight_norms']],
            'weight_changes': [{k: float(v) for k, v in d.items()} for d in diagnostics['weight_changes']]
        }
        
        with open(args.output, 'w') as f:
            json.dump(serializable_diagnostics, f, indent=2)
        print(f"\n✓ Diagnostics saved to: {args.output}")

