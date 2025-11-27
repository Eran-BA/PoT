#!/usr/bin/env python3
"""
Sorting A/B Test with Optimal R and T Parameters

Tests PoH with optimal hyperparameters (R=4, T=4, n_heads=2) against baseline
on sorting tasks using the proper pointer decoder architecture.

Based on:
- Optimal n_heads=2 from head search
- Optimal R=4, T=4 from R/T grid search
- Working architecture from examples/synthetic/sort_pointer_fixed.py

Author: Eran Ben Artzy
Date: October 2025
"""

import os
import sys
import time
import csv
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pot.core.hrm_controller import HRMPointerController, HRMState


# ========== Data Generation ==========

def generate_sort_data(num_samples: int, array_len: int, seed: int = 42):
    """Generate random sorting data."""
    np.random.seed(seed)
    arrays = []
    targets = []
    
    for _ in range(num_samples):
        # Random values in [-100, 100]
        values = np.random.randint(-100, 100, size=array_len)
        
        # Target = argsort (indices that would sort the array)
        perm = np.argsort(values, kind='stable')
        
        arrays.append(values.reshape(-1, 1).astype(np.float32))
        targets.append(perm.astype(np.int64))
    
    return (
        torch.from_numpy(np.array(arrays)),
        torch.from_numpy(np.array(targets))
    )


# ========== Models ==========

class PoHBlock(nn.Module):
    """Single PoH transformer block with HRM controller and iterative refinement."""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, max_iters: int, T: int = 4
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_iters = max_iters
        self.T = T

        # HRM controller for routing (with T parameter)
        self.controller = HRMPointerController(
            d_model=d_model,
            n_heads=n_heads,
            T=T,
            topk=None,  # Soft routing
            temperature_init=2.0,
            temperature_min=0.7,
            entropy_reg=1e-3,
            use_layernorm=True,
            dropout=0.0
        )

        # Per-head attention projections
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
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, z):
        """
        Iteratively refine latent z using HRM controller.
        Returns the refined latent (pre-FFN).
        """
        B = z.size(0)
        device = z.device
        
        # Initialize HRM state
        hrm_state = self.controller.init_state(B, device)
        
        for iter_idx in range(self.max_iters):
            # HRM controller routing (updates f_L every step, f_H every T steps)
            alphas, hrm_state, aux = self.controller(
                x=z,  # [B, T, d_model]
                state=hrm_state,
                return_aux=True
            )
            
            # alphas: [B, n_heads] - routing weights over heads

            # Per-head attention
            head_outs = []
            for h_idx in range(self.n_heads):
                q = self.q_proj[h_idx](z)  # [B, T, d_model//n_heads]
                k = self.k_proj[h_idx](z)
                v = self.v_proj[h_idx](z)

                scores = torch.einsum("btd,bsd->bts", q, k) / (
                    (self.d_model // self.n_heads) ** 0.5
                )
                attn = F.softmax(scores, dim=-1)
                out = torch.einsum("bts,bsd->btd", attn, v)
                head_outs.append(out)

            # Weight heads by HRM routing weights
            # alphas: [B, n_heads], expand to [B, 1, n_heads, 1]
            # head_outs: list of [B, T, d_head]
            head_outs_stacked = torch.stack(head_outs, dim=2)  # [B, T, n_heads, d_head]
            alphas_expanded = alphas.unsqueeze(1).unsqueeze(-1)  # [B, 1, n_heads, 1]
            
            # Weighted sum of heads
            weighted_heads = (head_outs_stacked * alphas_expanded).sum(dim=2)  # [B, T, d_head]
            
            # Project back to d_model
            # Since we have n_heads outputs, concat them
            head_outs_concat = torch.cat(head_outs, dim=-1)  # [B, T, d_model]
            attn_out = self.out_proj(head_outs_concat)

            # Residual + norm
            z = self.ln1(z + attn_out)
            z_refined = z  # Store latent before FFN

            # FFN (only applied at the end, not used for next iteration's input)
            z = self.ln2(z + self.ffn(z))

        return z_refined  # Return pre-FFN latent


class PointerDecoderSort(nn.Module):
    """Pointer decoder with optional PoH routing."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 2,
        d_ff: int = 256,
        max_inner_iters: int = 3,
        T: int = 4,
        use_poh: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_poh = use_poh

        # Encoder: embed input values + positions
        self.value_embed = nn.Sequential(
            nn.Linear(1, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, d_model)
        )
        self.pos_embed = nn.Embedding(100, d_model)

        # Encoder transformer
        self.encoder_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.encoder_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.encoder_ln1 = nn.LayerNorm(d_model)
        self.encoder_ln2 = nn.LayerNorm(d_model)

        # Decoder: rank embeddings
        self.rank_embed = nn.Embedding(100, d_model)

        # PoH block for iterative refinement (if enabled)
        if use_poh:
            self.poh_block = PoHBlock(d_model, n_heads, d_ff, max_inner_iters, T=T)

        # Pointer mechanism
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)

    def encode(self, x):
        """Encode input array values."""
        B, N, _ = x.shape

        # Embed values + positions
        positions = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.value_embed(x.squeeze(-1).unsqueeze(-1)) + self.pos_embed(positions)

        # Encoder transformer
        attn_out, _ = self.encoder_attn(h, h, h)
        h = self.encoder_ln1(h + attn_out)
        h = self.encoder_ln2(h + self.encoder_ffn(h))

        return h

    def forward(self, x, targets=None):
        """
        x: [B, N, 1] input array values
        targets: [B, N] ground truth indices (for training)

        Returns:
        - logits: [B, N, N] pointer logits for each decoding step
        - loss: scalar (if targets provided)
        """
        B, N, _ = x.shape
        enc = self.encode(x)  # [B, N, d_model]

        # Apply PoH refinement if enabled
        if self.use_poh:
            enc = self.poh_block(enc)

        # Prepare keys from encoder
        keys = self.key_proj(enc)  # [B, N, d_model]

        # Autoregressive decoding
        logits_all = []

        for rank in range(N):
            # Query: current rank embedding
            rank_ids = torch.full((B,), rank, dtype=torch.long, device=x.device)
            query = self.query_proj(self.rank_embed(rank_ids))  # [B, d_model]

            # Compute pointer logits
            scores = torch.einsum("bd,bnd->bn", query, keys)  # [B, N]

            # Mask already-selected positions
            if rank > 0:
                # Create mask for previously selected positions
                mask = torch.zeros(B, N, device=x.device)
                for r in range(rank):
                    if targets is not None:
                        prev_idx = targets[:, r]
                    else:
                        prev_idx = torch.argmax(logits_all[r], dim=-1)
                    mask.scatter_(1, prev_idx.unsqueeze(1), float('-inf'))
                scores = scores + mask

            logits_all.append(scores)

        # Stack all logits: [B, N, N]
        logits = torch.stack(logits_all, dim=1)

        if targets is not None:
            # Compute cross-entropy loss
            loss = F.cross_entropy(
                logits.reshape(B * N, N),
                targets.reshape(B * N)
            )
            return logits, loss
        else:
            return logits, None

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ========== Training ==========

def train_epoch(model, data, optimizer, batch_size, device):
    """Train for one epoch."""
    model.train()
    arrays, targets = data
    num_samples = len(arrays)
    
    total_loss = 0.0
    num_batches = 0
    
    indices = torch.randperm(num_samples)
    
    for i in range(0, num_samples, batch_size):
        batch_idx = indices[i:i+batch_size]
        batch_arrays = arrays[batch_idx].to(device)
        batch_targets = targets[batch_idx].to(device)
        
        # Forward
        _, loss = model(batch_arrays, targets=batch_targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(model, data, batch_size, device):
    """Evaluate model on data."""
    model.eval()
    arrays, targets = data
    num_samples = len(arrays)
    
    total_correct = 0
    total_elements = 0
    perfect_sorts = 0
    kendall_taus = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_arrays = arrays[i:i+batch_size].to(device)
            batch_targets = targets[i:i+batch_size].to(device)
            
            logits, _ = model(batch_arrays, targets=None)
            preds = logits.argmax(dim=-1)
            
            # Accuracy
            correct = (preds == batch_targets).sum().item()
            total_correct += correct
            total_elements += batch_targets.numel()
            
            # Perfect sorts
            perfect = (preds == batch_targets).all(dim=1).sum().item()
            perfect_sorts += perfect
            
            # Kendall's tau
            for j in range(len(batch_targets)):
                tau, _ = kendalltau(
                    preds[j].cpu().numpy(),
                    batch_targets[j].cpu().numpy()
                )
                kendall_taus.append(tau if not np.isnan(tau) else 0.0)
    
    return {
        'accuracy': total_correct / total_elements,
        'perfect': perfect_sorts / num_samples,
        'kendall_tau': np.mean(kendall_taus)
    }


def train_and_evaluate(
    model,
    model_name,
    train_data,
    test_data,
    epochs=100,
    batch_size=32,
    lr=1e-3,
    device='cpu'
):
    """Train and evaluate a model."""
    print(f"\nTraining {model_name}...")
    print(f"  Parameters: {model.count_parameters()/1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    start_time = time.time()
    best_tau = -1.0
    final_metrics = None
    
    pbar = tqdm(range(1, epochs + 1), desc=model_name)
    
    for epoch in pbar:
        # Train
        train_loss = train_epoch(model, train_data, optimizer, batch_size, device)
        
        # Evaluate every 10 epochs
        if epoch % 10 == 0 or epoch == epochs:
            test_metrics = evaluate(model, test_data, batch_size, device)
            
            if test_metrics['kendall_tau'] > best_tau:
                best_tau = test_metrics['kendall_tau']
            
            final_metrics = test_metrics
            
            pbar.set_postfix({
                'loss': f"{train_loss:.3f}",
                'tau': f"{test_metrics['kendall_tau']:.3f}",
                'perfect': f"{test_metrics['perfect']:.2f}"
            })
    
    pbar.close()
    
    time_minutes = (time.time() - start_time) / 60
    
    return {
        'best_tau': best_tau,
        'final_tau': final_metrics['kendall_tau'],
        'final_accuracy': final_metrics['accuracy'],
        'final_perfect': final_metrics['perfect'],
        'time_min': time_minutes,
        'params_M': model.count_parameters() / 1e6
    }


# ========== Main A/B Test ==========

def run_ab_test(array_len, train_samples=2000, test_samples=500, R=4, T=4, n_heads=2,
                epochs=100, seed=42):
    """Run A/B test for a specific array length."""
    
    print(f"\n{'='*80}")
    print(f"A/B Test: Array Length {array_len}")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Array length: {array_len}")
    print(f"  Training samples: {train_samples:,}")
    print(f"  Test samples: {test_samples:,}")
    print(f"  PoH n_heads: {n_heads} (optimal from head search)")
    print(f"  PoH R (refinement steps): {R} (optimal from R/T search)")
    print(f"  PoH T (HRM outer loop period): {T} (optimal from R/T search)")
    print(f"  Epochs: {epochs}")
    print(f"  Seed: {seed}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    # Generate data
    print(f"\nGenerating data...")
    train_data = generate_sort_data(train_samples, array_len, seed=seed)
    test_data = generate_sort_data(test_samples, array_len, seed=seed+1)
    
    # Build models
    print(f"\nBuilding models...")
    
    baseline = PointerDecoderSort(
        d_model=128,
        n_heads=n_heads,
        d_ff=256,
        max_inner_iters=1,
        T=1,
        use_poh=False
    ).to(device)
    
    poh = PointerDecoderSort(
        d_model=128,
        n_heads=n_heads,
        d_ff=256,
        max_inner_iters=R,
        T=T,  # HRM outer loop period
        use_poh=True
    ).to(device)
    
    # Train baseline
    baseline_results = train_and_evaluate(
        baseline, "Baseline", train_data, test_data,
        epochs=epochs, device=device
    )
    
    # Train PoH
    poh_results = train_and_evaluate(
        poh, f"PoH (R={R}, T={T}, n_heads={n_heads})", train_data, test_data,
        epochs=epochs, device=device
    )
    
    # Results
    print(f"\n{'='*80}")
    print("📊 RESULTS")
    print(f"{'='*80}")
    
    print(f"\n📚 Baseline (Standard Pointer Decoder)")
    print(f"  Parameters: {baseline_results['params_M']:.2f}M")
    print(f"  Best Kendall's τ: {baseline_results['best_tau']:.4f}")
    print(f"  Final Kendall's τ: {baseline_results['final_tau']:.4f}")
    print(f"  Final accuracy: {baseline_results['final_accuracy']:.4f}")
    print(f"  Final perfect sorts: {baseline_results['final_perfect']:.2%}")
    print(f"  Training time: {baseline_results['time_min']:.2f} min")
    
    print(f"\n🔬 PoH with HRM (R={R}, T={T}, n_heads={n_heads})")
    print(f"  Parameters: {poh_results['params_M']:.2f}M")
    print(f"  Best Kendall's τ: {poh_results['best_tau']:.4f}")
    print(f"  Final Kendall's τ: {poh_results['final_tau']:.4f}")
    print(f"  Final accuracy: {poh_results['final_accuracy']:.4f}")
    print(f"  Final perfect sorts: {poh_results['final_perfect']:.2%}")
    print(f"  Training time: {poh_results['time_min']:.2f} min")
    
    # Comparison
    delta_tau = poh_results['final_tau'] - baseline_results['final_tau']
    delta_pct = (delta_tau / baseline_results['final_tau']) * 100 if baseline_results['final_tau'] > 0 else 0
    
    print(f"\n{'='*80}")
    print("📈 COMPARISON")
    print(f"{'='*80}")
    print(f"Kendall's τ delta: {delta_tau:+.4f} ({delta_pct:+.2f}%)")
    
    if delta_tau > 0.01:
        print(f"✅ PoH WINS by {delta_tau:.4f} ({delta_pct:+.1f}%)")
        winner = "PoH"
    elif delta_tau < -0.01:
        print(f"❌ Baseline WINS by {-delta_tau:.4f} ({-delta_pct:+.1f}%)")
        winner = "Baseline"
    else:
        print(f"⚖️  TIE (difference < 0.01)")
        winner = "Tie"
    
    return {
        'array_len': array_len,
        'baseline': baseline_results,
        'poh': poh_results,
        'delta_tau': delta_tau,
        'delta_pct': delta_pct,
        'winner': winner
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Sorting A/B Test with Optimal Parameters')
    parser.add_argument('--lengths', type=int, nargs='+', default=[12, 16, 20],
                        help='Array lengths to test (default: 12 16 20)')
    parser.add_argument('--R', type=int, default=4, help='PoH refinement steps (default: 4)')
    parser.add_argument('--T', type=int, default=4, help='HRM outer loop period (default: 4)')
    parser.add_argument('--n-heads', type=int, default=2, help='Number of heads (default: 2)')
    parser.add_argument('--train-samples', type=int, default=2000, help='Training samples')
    parser.add_argument('--test-samples', type=int, default=500, help='Test samples')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save-dir', type=str, default='experiments/results/sorting_ab',
                        help='Save directory')
    
    args = parser.parse_args()
    
    print("="*80)
    print("SORTING A/B TEST WITH OPTIMAL PARAMETERS")
    print("="*80)
    print(f"\nOptimal parameters from grid search:")
    print(f"  n_heads = {args.n_heads} (from head search)")
    print(f"  R = {args.R} (refinement steps from R/T search)")
    print(f"  T = {args.T} (HRM outer loop period from R/T search)")
    print(f"\nTesting on array lengths: {args.lengths}")
    print("="*80)
    
    # Run tests for each length
    all_results = []
    
    for array_len in args.lengths:
        result = run_ab_test(
            array_len,
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
    results_file = os.path.join(args.save_dir, f"ab_results_R{args.R}_T{args.T}_nheads{args.n_heads}.csv")
    
    with open(results_file, 'w', newline='') as f:
        fieldnames = [
            'timestamp', 'array_len', 'model', 'R', 'T', 'n_heads',
            'best_tau', 'final_tau', 'final_accuracy', 'final_perfect',
            'time_min', 'params_M', 'delta_tau', 'winner'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for result in all_results:
            # Baseline row
            writer.writerow({
                'timestamp': timestamp,
                'array_len': result['array_len'],
                'model': 'Baseline',
                'R': '-',
                'T': '-',
                'n_heads': args.n_heads,
                'best_tau': f"{result['baseline']['best_tau']:.4f}",
                'final_tau': f"{result['baseline']['final_tau']:.4f}",
                'final_accuracy': f"{result['baseline']['final_accuracy']:.4f}",
                'final_perfect': f"{result['baseline']['final_perfect']:.4f}",
                'time_min': f"{result['baseline']['time_min']:.2f}",
                'params_M': f"{result['baseline']['params_M']:.2f}",
                'delta_tau': "0.0000",
                'winner': ''
            })
            
            # PoH row
            writer.writerow({
                'timestamp': timestamp,
                'array_len': result['array_len'],
                'model': 'PoH-HRM',
                'R': args.R,
                'T': args.T,
                'n_heads': args.n_heads,
                'best_tau': f"{result['poh']['best_tau']:.4f}",
                'final_tau': f"{result['poh']['final_tau']:.4f}",
                'final_accuracy': f"{result['poh']['final_accuracy']:.4f}",
                'final_perfect': f"{result['poh']['final_perfect']:.4f}",
                'time_min': f"{result['poh']['time_min']:.2f}",
                'params_M': f"{result['poh']['params_M']:.2f}",
                'delta_tau': f"{result['delta_tau']:+.4f}",
                'winner': result['winner']
            })
    
    # Summary
    print(f"\n\n{'='*80}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Length':<10} {'Baseline τ':<15} {'PoH τ':<15} {'Delta':<15} {'Winner':<15}")
    print("-"*80)
    
    for result in all_results:
        print(f"{result['array_len']:<10} "
              f"{result['baseline']['final_tau']:.4f}         "
              f"{result['poh']['final_tau']:.4f}         "
              f"{result['delta_tau']:+.4f}         "
              f"{result['winner']}")
    
    print(f"\n✅ Results saved to: {results_file}")
    print("="*80)


if __name__ == "__main__":
    main()
