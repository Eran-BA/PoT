#!/usr/bin/env python3
"""
NLI A/B Test: HybridPoHHRM vs Baseline Transformer
===================================================

Fair comparison between HybridPoHHRM and a standard Transformer baseline
on Natural Language Inference (SNLI/MultiNLI).

Both models use approximately the same number of parameters.

Usage:
    python experiments/nli_ab_test.py --dataset snli --max-samples 10000 --epochs 5

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import sys
import os
import math
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.pot.models.hybrid_nli import HybridPoHHRMForNLI
from src.pot.tasks.nli_dataset import (
    SNLIDataset, 
    MultiNLIDataset, 
    collate_nli_batch,
    SimpleTokenizer,
)


# ============================================================================
# Baseline Transformer for NLI
# ============================================================================

class BaselineTransformerForNLI(nn.Module):
    """
    Standard Transformer baseline for NLI.
    
    Uses PyTorch TransformerEncoder with similar parameter count
    to HybridPoHHRM for fair comparison.
    """
    
    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        num_labels: int = 3,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.segment_emb = nn.Embedding(2, d_model)
        
        self.emb_norm = nn.LayerNorm(d_model)
        self.emb_dropout = nn.Dropout(dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classification head
        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )
        self.classifier_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_labels)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        B, T = input_ids.size()
        device = input_ids.device
        
        # Embeddings
        x = self.token_emb(input_ids)
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        x = x + self.pos_emb(positions)
        if token_type_ids is not None:
            x = x + self.segment_emb(token_type_ids)
        
        x = self.emb_norm(x)
        x = self.emb_dropout(x)
        
        # Transformer encoding
        if attention_mask is not None:
            # Convert to transformer mask format (True = ignore)
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Classification
        cls_output = x[:, 0, :]
        pooled = self.pooler(cls_output)
        pooled = self.classifier_dropout(pooled)
        logits = self.classifier(pooled)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, scheduler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask=attention_mask)
        loss = F.cross_entropy(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': 100 * correct / total,
    }


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(logits, labels)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': 100 * correct / total,
    }


def train_model(model, train_loader, val_loader, optimizer, scheduler, device, epochs, model_name):
    """Train a model and return results."""
    results = []
    best_acc = 0
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, scheduler)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch}/{epochs} | "
              f"Train: {train_metrics['accuracy']:.1f}% | "
              f"Val: {val_metrics['accuracy']:.1f}%")
        
        results.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
        })
        
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
    
    elapsed = time.time() - start_time
    
    return {
        'model': model_name,
        'best_val_acc': best_acc,
        'final_train_acc': train_metrics['accuracy'],
        'final_val_acc': val_metrics['accuracy'],
        'training_time': elapsed,
        'history': results,
    }


def main():
    parser = argparse.ArgumentParser(description='NLI A/B Test: HybridPoHHRM vs Baseline')
    
    # Data
    parser.add_argument('--dataset', choices=['snli', 'mnli'], default='snli')
    parser.add_argument('--max-samples', type=int, default=10000)
    parser.add_argument('--max-length', type=int, default=128)
    
    # Model (shared)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--d-ff', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # HybridPoHHRM specific
    parser.add_argument('--H-cycles', type=int, default=2)
    parser.add_argument('--L-cycles', type=int, default=4)
    parser.add_argument('--H-layers', type=int, default=2)
    parser.add_argument('--L-layers', type=int, default=2)
    
    # Baseline specific
    parser.add_argument('--baseline-layers', type=int, default=4,
                       help='Number of layers for baseline (adjusted for param parity)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-steps', type=int, default=200)
    
    # Output
    parser.add_argument('--output', type=str, default='experiments/results/nli_ab_test')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print(f"\nLoading {args.dataset.upper()} dataset...")
    tokenizer = SimpleTokenizer()
    
    def collate_fn(examples):
        return collate_nli_batch(examples, tokenizer, max_length=args.max_length)
    
    if args.dataset == 'snli':
        train_dataset = SNLIDataset('train', max_samples=args.max_samples)
        val_dataset = SNLIDataset('validation', max_samples=args.max_samples // 5)
    else:
        train_dataset = MultiNLIDataset('train', max_samples=args.max_samples)
        val_dataset = MultiNLIDataset('validation_matched', max_samples=args.max_samples // 5)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_fn
    )
    
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    
    # ========================================================================
    # Model A: HybridPoHHRM
    # ========================================================================
    print("\n" + "="*60)
    print("MODEL A: HybridPoHHRM")
    print("="*60)
    
    model_a = HybridPoHHRMForNLI(
        vocab_size=30522,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        H_cycles=args.H_cycles,
        L_cycles=args.L_cycles,
        H_layers=args.H_layers,
        L_layers=args.L_layers,
        max_seq_len=args.max_length,
        dropout=args.dropout,
    ).to(device)
    
    params_a = model_a.count_parameters()
    print(f"Parameters: {params_a:,}")
    print(f"Reasoning steps: {args.H_cycles * args.L_cycles}")
    
    optimizer_a = torch.optim.AdamW(model_a.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    
    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler_a = torch.optim.lr_scheduler.LambdaLR(optimizer_a, lr_lambda)
    
    results_a = train_model(
        model_a, train_loader, val_loader, optimizer_a, scheduler_a,
        device, args.epochs, "HybridPoHHRM"
    )
    results_a['parameters'] = params_a
    
    # ========================================================================
    # Model B: Baseline Transformer
    # ========================================================================
    print("\n" + "="*60)
    print("MODEL B: Baseline Transformer")
    print("="*60)
    
    model_b = BaselineTransformerForNLI(
        vocab_size=30522,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.baseline_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_length,
        dropout=args.dropout,
    ).to(device)
    
    params_b = model_b.count_parameters()
    print(f"Parameters: {params_b:,}")
    print(f"Layers: {args.baseline_layers}")
    
    optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler_b = torch.optim.lr_scheduler.LambdaLR(optimizer_b, lr_lambda)
    
    results_b = train_model(
        model_b, train_loader, val_loader, optimizer_b, scheduler_b,
        device, args.epochs, "Baseline Transformer"
    )
    results_b['parameters'] = params_b
    
    # ========================================================================
    # Final Comparison
    # ========================================================================
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    print(f"\n{'Model':<25} {'Params':>12} {'Best Val':>10} {'Time':>10}")
    print("-" * 60)
    print(f"{'HybridPoHHRM':<25} {params_a:>12,} {results_a['best_val_acc']:>9.2f}% {results_a['training_time']:>9.1f}s")
    print(f"{'Baseline Transformer':<25} {params_b:>12,} {results_b['best_val_acc']:>9.2f}% {results_b['training_time']:>9.1f}s")
    
    diff = results_a['best_val_acc'] - results_b['best_val_acc']
    winner = "HybridPoHHRM" if diff > 0 else "Baseline"
    print(f"\n{'🏆 Winner:':<25} {winner} ({'+' if diff > 0 else ''}{diff:.2f}%)")
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    
    with open(os.path.join(args.output, 'ab_test_results.json'), 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'max_samples': args.max_samples,
            'epochs': args.epochs,
            'config': vars(args),
            'hybrid': results_a,
            'baseline': results_b,
            'winner': winner,
            'difference': diff,
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {args.output}")


if __name__ == '__main__':
    main()

