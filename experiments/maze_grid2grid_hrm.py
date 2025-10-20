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
from src.pot.models.puzzle_embedding import PuzzleEmbedding
from src.pot.models.adaptive_halting import QHaltingController
from src.pot.models.hrm_layers import RMSNorm, SwiGLU, PostNormTransformerLayer


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
            'puzzle_id': torch.tensor(idx, dtype=torch.long),  # NEW: unique puzzle ID
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
        T: int = 1,
        num_puzzles: int = 1000,  # NEW: number of unique puzzles
        puzzle_emb_dim: int = 256,  # NEW: puzzle embedding dimension
        max_halting_steps: int = 16,  # NEW: max adaptive computation steps
        latent_len: int = 16,  # NEW: number of latent tokens (TRM-style recursion)
        latent_k: int = 3  # NEW: inner latent updates per outer step
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.use_poh = use_poh
        
        # NEW: Puzzle embeddings (prepended to input)
        # Use 0.02 std for better gradient flow (not zero!)
        self.puzzle_emb = PuzzleEmbedding(num_puzzles, puzzle_emb_dim, init_std=0.02)
        self.puzzle_emb_len = (puzzle_emb_dim + d_model - 1) // d_model  # Ceil div
        
        # NEW: TRM-style latent recursion parameters
        self.latent_len = int(latent_len)
        self.latent_k = int(latent_k)
        # Learnable initial latent tokens shared across batch
        self.latent_init = nn.Parameter(torch.randn(1, self.latent_len, d_model) * 0.02)
        
        # NEW: Q-halting controller
        self.q_halt_controller = QHaltingController(d_model, max_steps=max_halting_steps)
        self.max_halting_steps = max_halting_steps
        
        # Input embedding
        self.input_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # Update positional embedding to include puzzle + latent positions
        self.pos_embed = nn.Parameter(
            torch.randn(1, 900 + self.puzzle_emb_len + self.latent_len, d_model) * 0.02
        )
        self.pre_norm = nn.LayerNorm(d_model)
        
        # Transformer encoder
        self.use_poh = use_poh
        if use_poh:
            # PoH with HRM controller + HRM architecture (Post-norm + SwiGLU + RMSNorm)
            self.R = R  # Refinement iterations
            self.n_layers = n_layers
            self.hrm_controller = HRMPointerController(
                d_model=d_model,
                n_heads=n_heads,
                T=T,
                dropout=dropout
            )

            # Stack HRM-style transformer layers (Post-norm + SwiGLU + RMSNorm)
            self.attn_layers = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=n_heads,
                    dropout=dropout,
                    batch_first=True
                ) for _ in range(n_layers)
            ])
            self.ffn_layers = nn.ModuleList([
                SwiGLU(d_model, d_ff, dropout) for _ in range(n_layers)
            ])
            self.norm1_layers = nn.ModuleList([
                RMSNorm(d_model) for _ in range(n_layers)
            ])
            self.norm2_layers = nn.ModuleList([
                RMSNorm(d_model) for _ in range(n_layers)
            ])
            self.dropout_layers = nn.ModuleList([
                nn.Dropout(dropout) for _ in range(n_layers)
            ])

            # Post-norm: norm AFTER residual (HRM architecture)
            self.use_post_norm = True
            
            # NEW: Latent updater (TRM-style): cross-attn from latent -> context (puzzle+grid)
            self.latent_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            )
            self.latent_ffn = SwiGLU(d_model, d_ff, dropout)
            self.latent_norm1 = RMSNorm(d_model)
            self.latent_norm2 = RMSNorm(d_model)
            self.latent_drop = nn.Dropout(dropout)
        else:
            # Standard Transformer encoder (Pre-norm, ReLU, LayerNorm)
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
    
    def _encode_once(self, x, hrm_state=None):
        """Single encoding pass (for adaptive computation loop)."""
        if self.use_poh:
            # Apply one refinement iteration with HRM routing
            B, T, D = x.shape
            
            # Get routing weights from HRM
            route_weights, hrm_state, _ = self.hrm_controller(x, state=hrm_state)
            # route_weights: [B, n_heads]
            
            # Apply stacked HRM-style layers
            for attn, ffn, norm1, norm2, drop in zip(self.attn_layers, self.ffn_layers, self.norm1_layers, self.norm2_layers, self.dropout_layers):
                # Attention
                attn_out, _ = attn(x, x, x, need_weights=False)
                d_head = D // attn.num_heads
                attn_out_heads = attn_out.view(B, T, attn.num_heads, d_head)
                route_weights_exp = route_weights.unsqueeze(1).unsqueeze(-1)
                attn_out_routed = (attn_out_heads * route_weights_exp).view(B, T, D)
                x = x + drop(attn_out_routed)
                x = norm1(x)

                # FFN
                ffn_out = ffn(x)
                x = x + drop(ffn_out)
                x = norm2(x)
            
            return x, hrm_state
        else:
            # Standard transformer (no HRM routing)
            x = self.encoder(x)
            return x, None
    
    def forward(self, input_seq, puzzle_ids):
        """
        Args:
            input_seq: (batch, 900) - flattened maze grid
            puzzle_ids: (batch,) - unique puzzle identifiers
        
        Returns:
            logits: (batch, 900, vocab_size)
            q_halt: (batch,) - Q-value for halting
            q_continue: (batch,) - Q-value for continuing
            actual_steps: int - number of computation steps taken
        """
        B = input_seq.size(0)
        
        # Embed puzzle
        puzzle_emb = self.puzzle_emb(puzzle_ids)  # [B, puzzle_emb_dim]
        
        # Reshape to [B, puzzle_emb_len, d_model]
        pad_size = self.puzzle_emb_len * self.d_model - puzzle_emb.size(-1)
        if pad_size > 0:
            puzzle_emb = F.pad(puzzle_emb, (0, pad_size))
        puzzle_emb = puzzle_emb.view(B, self.puzzle_emb_len, self.d_model)
        
        # Embed input tokens
        x_grid = self.input_embed(input_seq)  # [B, 900, d_model]
        
        # Prepare latent tokens
        latent = self.latent_init.expand(B, -1, -1)  # [B, latent_len, d_model]
        
        # Keep original input embeddings as constant reference (TRM-style)
        x_grid_ref = x_grid  # [B, 900, d_model]
        latent_cur = latent  # current latent tokens
        
        # Adaptive computation loop with Q-halting
        device = x_grid_ref.device
        hrm_state = self.hrm_controller.init_state(B, device) if self.use_poh else None
        actual_steps = self.max_halting_steps
        x_out = None
        
        if self.use_poh:
            # PoH with adaptive halting: refine R times
            for step in range(1, self.max_halting_steps + 1):
                # TRM-style inner latent updates (K steps)
                for _ in range(self.latent_k):
                    # Indices
                    pL = self.puzzle_emb_len
                    zL = self.latent_len
                    # Context is constant reference: puzzle_emb + original grid embeddings
                    ctx = torch.cat([puzzle_emb, x_grid_ref], dim=1)  # [B, pL+900, d_model]
                    # Cross-attn: latent queries over context
                    lat_attn, _ = self.latent_attn(latent_cur, ctx, ctx, need_weights=False)
                    latent_cur = latent_cur + self.latent_drop(lat_attn)
                    latent_cur = self.latent_norm1(latent_cur)
                    lat_ffn = self.latent_ffn(latent_cur)
                    latent_cur = latent_cur + self.latent_drop(lat_ffn)
                    latent_cur = self.latent_norm2(latent_cur)
                
                # Reconstruct sequence with constant inputs + updated latent
                x_step = torch.cat([puzzle_emb, latent_cur, x_grid_ref], dim=1)  # [B, pL+zL+900, d]
                x_step = x_step + self.pos_embed[:, :x_step.size(1), :]
                x_step = self.pre_norm(x_step)
                
                x_out, hrm_state = self._encode_once(x_step, hrm_state)
                
                # Check if should halt
                q_halt, q_continue = self.q_halt_controller(x_out)
                
                # Decide whether to stop (batch-wise decision)
                should_halt = self.q_halt_controller.should_halt(
                    q_halt, q_continue, step, self.training
                )
                
                # If all sequences should halt, stop early
                if should_halt.all():
                    actual_steps = step
                    break
                
                # O(1) grads across outer steps: detach state and latents before next iteration
                latent_cur = latent_cur.detach()
                x_out = x_out.detach()
                if hrm_state is not None:
                    # Detach common tensor fields if present
                    if hasattr(hrm_state, "z_L") and torch.is_tensor(hrm_state.z_L):
                        hrm_state.z_L = hrm_state.z_L.detach()
                    if hasattr(hrm_state, "z_H") and torch.is_tensor(hrm_state.z_H):
                        hrm_state.z_H = hrm_state.z_H.detach()
                    if hasattr(hrm_state, "step") and torch.is_tensor(hrm_state.step):
                        hrm_state.step = hrm_state.step.detach()
        else:
            # Standard transformer (no adaptive halting)
            # Build once with constant inputs
            x_step = torch.cat([puzzle_emb, latent_cur, x_grid_ref], dim=1)
            x_step = x_step + self.pos_embed[:, :x_step.size(1), :]
            x_step = self.pre_norm(x_step)
            x_out, _ = self._encode_once(x_step, None)
            q_halt, q_continue = self.q_halt_controller(x_out)
            actual_steps = 1
        
        # Remove puzzle and latent positions from output
        if x_out is None:
            # Fallback to using last constructed x_step if needed
            x_out = x_step
        x = x_out[:, self.puzzle_emb_len + self.latent_len:, :]  # [B, 900, d_model]
        
        # Project to output vocab
        logits = self.output_proj(x)  # [B, 900, vocab_size]
        
        return logits, q_halt, q_continue, actual_steps


def train_epoch(model, dataloader, optimizer, puzzle_optimizer, device, epoch):
    """Train for one epoch with Q-learning losses."""
    model.train()
    total_loss = 0
    total_lm_loss = 0
    total_q_halt_loss = 0
    total_q_continue_loss = 0
    correct_tokens = 0
    total_tokens = 0
    total_steps = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        inp = batch['input'].to(device)
        label = batch['label'].to(device)
        puzzle_ids = batch['puzzle_id'].to(device)  # NEW: puzzle identifiers
        
        # Forward with Q-halting
        logits, q_halt, q_continue, steps = model(inp, puzzle_ids)
        
        # Main LM loss (cross-entropy over all 900 positions)
        lm_loss = F.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            label.reshape(-1),
            ignore_index=0  # Ignore padding
        )
        
        # Compute correctness (for Q-halt target)
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            mask = (label != 0)
            is_correct = ((preds == label) | ~mask).all(dim=1).float()  # [B]
        
        # Q-halt loss: predict sequence correctness
        q_halt_loss = F.binary_cross_entropy_with_logits(
            q_halt, is_correct, reduction='mean'
        )
        
        # Q-continue loss: bootstrap from next step's Q-values
        # Train q_continue to predict future reward (Q-learning)
        q_continue_loss = torch.tensor(0.0, device=device)
        if model.training:
            # Run one more forward pass to get next-step Q-values
            # This teaches q_continue to predict: "if I keep going, what will happen?"
            with torch.no_grad():
                _, next_q_halt, next_q_continue, _ = model(inp, puzzle_ids)
                # Bootstrap target: best Q-value at next step (Bellman equation)
                # Use sigmoid to convert logits to [0,1] probabilities
                target_q = torch.sigmoid(torch.maximum(next_q_halt, next_q_continue))
            
            # Train q_continue to match the bootstrap target
            q_continue_loss = F.binary_cross_entropy_with_logits(
                q_continue, target_q, reduction='mean'
            )
        
        # Total loss (HRM uses 0.5 weight on Q losses)
        loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
        
        # Backward (two optimizers!)
        optimizer.zero_grad()
        puzzle_optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        puzzle_optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        total_lm_loss += lm_loss.item()
        total_q_halt_loss += q_halt_loss.item()
        total_q_continue_loss += q_continue_loss.item() if isinstance(q_continue_loss, torch.Tensor) else 0
        correct_tokens += ((preds == label) & mask).sum().item()
        total_tokens += mask.sum().item()
        total_steps += steps
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lm': f'{lm_loss.item():.4f}',
            'q_h': f'{q_halt_loss.item():.4f}',
            'acc': f'{100.0 * correct_tokens / total_tokens:.2f}%',
            'steps': f'{steps}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_lm_loss = total_lm_loss / len(dataloader)
    avg_q_halt_loss = total_q_halt_loss / len(dataloader)
    avg_q_continue_loss = total_q_continue_loss / len(dataloader)
    token_acc = 100.0 * correct_tokens / total_tokens
    avg_steps = total_steps / num_batches
    
    return avg_loss, token_acc, avg_lm_loss, avg_q_halt_loss, avg_q_continue_loss, avg_steps


def evaluate(model, dataloader, device):
    """Evaluate model with Q-values tracking."""
    model.eval()
    total_loss = 0
    correct_tokens = 0
    total_tokens = 0
    perfect_grids = 0
    total_steps = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inp = batch['input'].to(device)
            label = batch['label'].to(device)
            puzzle_ids = batch['puzzle_id'].to(device)
            
            logits, q_halt, q_continue, steps = model(inp, puzzle_ids)
            
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
            total_steps += steps
            num_batches += 1
            
            # Count perfect grid predictions
            perfect = ((preds == label) | ~mask).all(dim=1).sum().item()
            perfect_grids += perfect
    
    avg_loss = total_loss / len(dataloader)
    token_acc = 100.0 * correct_tokens / total_tokens
    grid_acc = 100.0 * perfect_grids / len(dataloader.dataset)
    avg_steps = total_steps / num_batches
    
    return avg_loss, token_acc, grid_acc, avg_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='Path to HRM maze dataset')
    parser.add_argument('--model', type=str, choices=['baseline', 'poh'], default='baseline')
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--d-ff', type=int, default=1024)
    parser.add_argument('--R', type=int, default=4, help='PoH refinement iterations (deprecated, now uses adaptive halting)')
    parser.add_argument('--T', type=int, default=4, help='HRM outer loop period')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-epochs', type=int, default=5000, help='Maximum epochs (early stopping may end sooner)')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=1e-4, help='Main optimizer learning rate (HRM: 1e-4)')
    parser.add_argument('--puzzle-emb-lr', type=float, default=1e-4, help='Puzzle embedding learning rate (HRM: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1.0, help='Weight decay (HRM: 1.0)')
    parser.add_argument('--num-puzzles', type=int, default=1000, help='Number of unique puzzles')
    parser.add_argument('--puzzle-emb-dim', type=int, default=256, help='Puzzle embedding dimension')
    parser.add_argument('--max-halting-steps', type=int, default=16, help='Max adaptive computation steps')
    parser.add_argument('--latent-len', type=int, default=16, help='TRM latent tokens length')
    parser.add_argument('--latent-k', type=int, default=3, help='Inner latent updates per outer step')
    parser.add_argument('--output', type=str, default='experiments/results/maze_grid2grid')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Device selection with optional override via environment variable
    force = os.getenv("FORCE_DEVICE")
    if force in {"cpu", "cuda", "mps"}:
        device = torch.device(force)
    else:
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
        T=args.T,
        num_puzzles=args.num_puzzles,
        puzzle_emb_dim=args.puzzle_emb_dim,
        max_halting_steps=args.max_halting_steps,
        latent_len=args.latent_len,
        latent_k=args.latent_k
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    puzzle_param_count = sum(p.numel() for p in model.puzzle_emb.parameters() if p.requires_grad)
    print(f"\nModel: {args.model}")
    print(f"Parameters: {param_count:,} (puzzle emb: {puzzle_param_count:,})")
    print(f"Config: T={args.T}, max_halting_steps={args.max_halting_steps}, latent_len={args.latent_len}, latent_k={args.latent_k}")
    
    # Dual optimizers (HRM approach: separate puzzle embedding optimizer)
    puzzle_params = list(model.puzzle_emb.parameters())
    model_params = [p for p in model.parameters() if p not in set(puzzle_params)]
    
    # Main model: strong weight decay for generalization
    optimizer = torch.optim.AdamW(model_params, lr=args.lr, weight_decay=args.weight_decay)
    
    # Puzzle embeddings: NO weight decay (they should memorize individual mazes!)
    # Weight decay would push embeddings back to zero, preventing specialization
    puzzle_optimizer = torch.optim.AdamW(puzzle_params, lr=args.puzzle_emb_lr, weight_decay=0.0)
    
    # Cosine annealing schedulers
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    puzzle_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(puzzle_optimizer, T_max=args.max_epochs)
    
    # Training with early stopping
    print("\n" + "="*80)
    print(f"Training: {args.model.upper()} with Q-Halting + Puzzle Embeddings")
    print("="*80)
    print(f"Early stopping: patience={args.patience}, max_epochs={args.max_epochs}")
    
    best_grid_acc = 0.0
    patience_counter = 0
    results = []
    
    for epoch in range(1, args.max_epochs + 1):
        train_loss, train_token_acc, train_lm_loss, train_q_halt_loss, train_q_continue_loss, train_steps = \
            train_epoch(model, train_loader, optimizer, puzzle_optimizer, device, epoch)
        test_loss, test_token_acc, test_grid_acc, test_steps = \
            evaluate(model, test_loader, device)
        
        scheduler.step()
        puzzle_scheduler.step()
        
        print(f"\nEpoch {epoch}/{args.max_epochs}")
        print(f"  Train: Loss={train_loss:.4f}, LM={train_lm_loss:.4f}, Q_halt={train_q_halt_loss:.4f}, "
              f"Q_cont={train_q_continue_loss:.4f}, Token Acc={train_token_acc:.2f}%, Avg Steps={train_steps:.1f}")
        print(f"  Test:  Loss={test_loss:.4f}, Token Acc={test_token_acc:.2f}%, "
              f"Grid Acc={test_grid_acc:.2f}%, Avg Steps={test_steps:.1f}")
        
        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_lm_loss': train_lm_loss,
            'train_q_halt_loss': train_q_halt_loss,
            'train_q_continue_loss': train_q_continue_loss,
            'train_token_acc': train_token_acc,
            'train_steps': train_steps,
            'test_loss': test_loss,
            'test_token_acc': test_token_acc,
            'test_grid_acc': test_grid_acc,
            'test_steps': test_steps
        })
        
        # Early stopping logic
        if test_grid_acc > best_grid_acc:
            best_grid_acc = test_grid_acc
            patience_counter = 0
            # Save checkpoint
            os.makedirs(args.output, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'puzzle_optimizer_state_dict': puzzle_optimizer.state_dict(),
                'test_grid_acc': test_grid_acc,
                'test_token_acc': test_token_acc,
            }, os.path.join(args.output, f'{args.model}_best.pt'))
            print(f"  ✓ New best grid acc: {best_grid_acc:.2f}% (saved checkpoint)")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{args.patience}")
        
        # Early stopping check
        if patience_counter >= args.patience:
            print(f"\n⚠ Early stopping triggered at epoch {epoch} (no improvement for {args.patience} epochs)")
            break
    
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

