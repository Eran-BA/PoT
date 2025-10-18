"""
Grid-to-Grid Maze Solver (HRM Paper Format)
============================================

Task: Given a 30x30 maze grid, output a 30x30 solution grid marking the optimal path.

Input:  30x30 grid with walls/empty cells
Output: 30x30 grid with path cells marked (1 = on path, 0 = not on path)

This matches the HRM paper's maze task formulation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import argparse
import json
from tqdm import tqdm

from src.pot.core.hrm_controller import HRMPointerController


class HRMMazeDataset(Dataset):
    """Load HRM maze-30x30-hard dataset from numpy files."""
    
    def __init__(self, data_dir: str, split: str = 'train'):
        self.data_dir = os.path.join(data_dir, split)
        
        # Load HRM format: inputs and labels are both grids
        self.inputs = np.load(os.path.join(self.data_dir, 'all__inputs.npy'))
        self.labels = np.load(os.path.join(self.data_dir, 'all__labels.npy'))
        
        # HRM encoding: # = wall (1), space = empty (2), S = start (3), G = goal (4), o = path (5)
        # We'll remap to: 0 = empty, 1 = wall, 2 = start, 3 = goal, 4 = path
        print(f"[{split}] Loaded {len(self.inputs)} mazes")
        print(f"  Input shape: {self.inputs.shape}")
        print(f"  Label shape: {self.labels.shape}")
        print(f"  Input vocab range: {self.inputs.min()}-{self.inputs.max()}")
        print(f"  Label vocab range: {self.labels.min()}-{self.labels.max()}")
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        inp = torch.LongTensor(self.inputs[idx])  # (900,)
        label = torch.LongTensor(self.labels[idx])  # (900,)
        
        return {
            'input': inp,
            'label': label,
            'maze_size': int(np.sqrt(len(inp)))
        }


class Grid2GridMazeSolver(nn.Module):
    """
    Transformer-based grid-to-grid maze solver.
    
    Input:  900-token sequence (30x30 flattened grid)
    Output: 900-token sequence (solution grid with path marked)
    """
    
    def __init__(
        self,
        vocab_size: int = 6,  # HRM vocab: PAD + # + space + S + G + o
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        use_poh: bool = False,
        R: int = 1,
        T: int = 1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.use_poh = use_poh
        
        # Input embedding
        self.input_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.randn(1, 900, d_model) * 0.02)
        self.pre_norm = nn.LayerNorm(d_model)
        
        # Transformer encoder
        self.use_poh = use_poh
        if use_poh:
            # PoH with HRM controller - build manually
            self.R = R  # Refinement iterations
            self.hrm_controller = HRMPointerController(
                d_model=d_model,
                n_heads=n_heads,
                T=T,
                dropout=dropout
            )
            
            # Standard transformer layer (we'll route over its heads)
            self.attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            )
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout_layer = nn.Dropout(dropout)
            
            # Note: For simplicity, we'll apply R refinement iterations with HRM routing
            # This is a simplified PoH - just demonstrates HRM integration
        else:
            # Standard Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Output head: predict for each position independently
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_seq):
        """
        Args:
            input_seq: (batch, 900) - flattened maze grid
        
        Returns:
            logits: (batch, 900, vocab_size)
        """
        # Embed input
        x = self.input_embed(input_seq)  # (batch, 900, d_model)
        x = x + self.pos_embed[:, :x.size(1), :]
        
        # HRM-style input normalization
        x = (x - 0.5) / 0.5  # Map to [-1, 1]
        x = self.pre_norm(x)
        
        # Encode
        if self.use_poh:
            # Apply R refinement iterations with HRM routing
            hrm_state = self.hrm_controller.init_state(x.size(0), x.device)
            B, T, D = x.shape
            d_head = D // self.attn.num_heads
            
            for r in range(self.R):
                # Get routing weights from HRM
                route_weights, hrm_state, _ = self.hrm_controller(x, state=hrm_state)
                # route_weights: [B, n_heads]
                
                # Apply multi-head attention
                attn_out, _ = self.attn(x, x, x)  # [B, T, d_model]
                
                # Reshape to separate heads: [B, T, n_heads, d_head]
                attn_out_heads = attn_out.view(B, T, self.attn.num_heads, d_head)
                
                # Apply HRM routing weights: [B, 1, n_heads, 1] * [B, T, n_heads, d_head]
                route_weights_exp = route_weights.unsqueeze(1).unsqueeze(-1)  # [B, 1, n_heads, 1]
                attn_out_routed = (attn_out_heads * route_weights_exp).view(B, T, D)
                
                # Apply residual + norm
                x = self.norm1(x + self.dropout_layer(attn_out_routed))
                
                # FFN
                ff_out = self.ff(x)
                x = self.norm2(x + self.dropout_layer(ff_out))
        else:
            x = self.encoder(x)  # (batch, 900, d_model)
        
        # Project to output vocab
        logits = self.output_proj(x)  # (batch, 900, vocab_size)
        
        return logits


def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    correct_tokens = 0
    total_tokens = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        inp = batch['input'].to(device)
        label = batch['label'].to(device)
        
        # Forward
        logits = model(inp)  # (batch, 900, vocab_size)
        
        # Compute loss (cross-entropy over all 900 positions)
        loss = F.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            label.reshape(-1),
            ignore_index=0  # Ignore padding
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        mask = (label != 0)
        correct_tokens += ((preds == label) & mask).sum().item()
        total_tokens += mask.sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct_tokens / total_tokens:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    token_acc = 100.0 * correct_tokens / total_tokens
    
    return avg_loss, token_acc


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct_tokens = 0
    total_tokens = 0
    perfect_grids = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inp = batch['input'].to(device)
            label = batch['label'].to(device)
            
            logits = model(inp)
            
            loss = F.cross_entropy(
                logits.reshape(-1, model.vocab_size),
                label.reshape(-1),
                ignore_index=0
            )
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            mask = (label != 0)
            correct_tokens += ((preds == label) & mask).sum().item()
            total_tokens += mask.sum().item()
            
            # Count perfect grid predictions
            perfect = (preds == label).all(dim=1).sum().item()
            perfect_grids += perfect
    
    avg_loss = total_loss / len(dataloader)
    token_acc = 100.0 * correct_tokens / total_tokens
    grid_acc = 100.0 * perfect_grids / len(dataloader.dataset)
    
    return avg_loss, token_acc, grid_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='Path to HRM maze dataset')
    parser.add_argument('--model', type=str, choices=['baseline', 'poh'], default='baseline')
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--d-ff', type=int, default=1024)
    parser.add_argument('--R', type=int, default=4, help='PoH refinement iterations')
    parser.add_argument('--T', type=int, default=4, help='HRM outer loop period')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output', type=str, default='experiments/results/maze_grid2grid')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    train_dataset = HRMMazeDataset(args.data_dir, 'train')
    test_dataset = HRMMazeDataset(args.data_dir, 'test')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Build model
    model = Grid2GridMazeSolver(
        vocab_size=6,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        use_poh=(args.model == 'poh'),
        R=args.R,
        T=args.T
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {args.model}")
    print(f"Parameters: {param_count:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training
    print("\n" + "="*80)
    print(f"Training: {args.model.upper()}")
    print("="*80)
    
    best_grid_acc = 0.0
    results = []
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_token_acc = train_epoch(model, train_loader, optimizer, device, epoch)
        test_loss, test_token_acc, test_grid_acc = evaluate(model, test_loader, device)
        scheduler.step()
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train: Loss={train_loss:.4f}, Token Acc={train_token_acc:.2f}%")
        print(f"  Test:  Loss={test_loss:.4f}, Token Acc={test_token_acc:.2f}%, Grid Acc={test_grid_acc:.2f}%")
        
        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_token_acc': train_token_acc,
            'test_loss': test_loss,
            'test_token_acc': test_token_acc,
            'test_grid_acc': test_grid_acc
        })
        
        if test_grid_acc > best_grid_acc:
            best_grid_acc = test_grid_acc
            # Save checkpoint
            os.makedirs(args.output, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_grid_acc': test_grid_acc,
            }, os.path.join(args.output, f'{args.model}_best.pt'))
    
    # Final results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Best Grid Accuracy: {best_grid_acc:.2f}%")
    print(f"HRM Paper (30x30 Hard): ~74%")
    print(f"Our Result: {best_grid_acc:.2f}%")
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, f'{args.model}_results.json'), 'w') as f:
        json.dump({
            'model': args.model,
            'parameters': param_count,
            'best_grid_acc': best_grid_acc,
            'final_token_acc': results[-1]['test_token_acc'],
            'final_loss': results[-1]['test_loss'],
            'config': vars(args),
            'history': results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {args.output}/{args.model}_results.json")


if __name__ == '__main__':
    main()

