#!/usr/bin/env python3
"""
HybridPoHHRM NLI Benchmark
==========================

Natural Language Inference using the HybridPoHHRM architecture.
Tests two-timescale reasoning on premise-hypothesis classification.

Task: Given premise and hypothesis, classify as:
  - 0: Entailment
  - 1: Neutral  
  - 2: Contradiction

Usage:
    python experiments/nli_hybrid_benchmark.py --dataset snli --max-samples 10000
    python experiments/nli_hybrid_benchmark.py --dataset mnli --epochs 10

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import numpy as np
import torch
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


def train_epoch(model, dataloader, optimizer, device, epoch, scheduler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
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
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.1f}%',
        })
    
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
    
    # Per-class metrics
    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(logits, labels)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Per-class accuracy
            for c in range(3):
                mask = labels == c
                class_total[c] += mask.sum().item()
                class_correct[c] += ((preds == labels) & mask).sum().item()
    
    class_acc = [100 * c / t if t > 0 else 0 for c, t in zip(class_correct, class_total)]
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': 100 * correct / total,
        'entailment_acc': class_acc[0],
        'neutral_acc': class_acc[1],
        'contradiction_acc': class_acc[2],
    }


def main():
    parser = argparse.ArgumentParser(description='HybridPoHHRM NLI Benchmark')
    
    # Data
    parser.add_argument('--dataset', choices=['snli', 'mnli'], default='snli')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Limit dataset size for quick experiments')
    parser.add_argument('--max-length', type=int, default=128)
    
    # Model
    parser.add_argument('--d-model', type=int, default=512)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--d-ff', type=int, default=2048)
    parser.add_argument('--H-cycles', type=int, default=2)
    parser.add_argument('--L-cycles', type=int, default=8)
    parser.add_argument('--H-layers', type=int, default=2)
    parser.add_argument('--L-layers', type=int, default=2)
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-steps', type=int, default=500)
    
    # Output
    parser.add_argument('--output', type=str, default='experiments/results/nli_hybrid')
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
        val_dataset = SNLIDataset('validation', max_samples=args.max_samples // 10 if args.max_samples else None)
    else:
        train_dataset = MultiNLIDataset('train', max_samples=args.max_samples)
        val_dataset = MultiNLIDataset('validation_matched', max_samples=args.max_samples // 10 if args.max_samples else None)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn
    )
    
    # Build model
    model = HybridPoHHRMForNLI(
        vocab_size=30522,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        H_cycles=args.H_cycles,
        L_cycles=args.L_cycles,
        H_layers=args.H_layers,
        L_layers=args.L_layers,
        max_seq_len=args.max_length,
        T=args.T,
        dropout=args.dropout,
    ).to(device)
    
    param_count = model.count_parameters()
    print(f"\nModel: HybridPoHHRM for NLI")
    print(f"Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
    print(f"H_cycles={args.H_cycles}, L_cycles={args.L_cycles}")
    print(f"H_layers={args.H_layers}, L_layers={args.L_layers}")
    print(f"Total reasoning steps: {args.H_cycles * args.L_cycles}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    
    # LR scheduler with warmup
    total_steps = args.epochs * len(train_loader)
    
    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"\n{'='*60}")
    print(f"Training HybridPoHHRM on {args.dataset.upper()}")
    print(f"{'='*60}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"LR: {args.lr}, Weight decay: {args.weight_decay}")
    print(f"Warmup: {args.warmup_steps} steps, Total: {total_steps} steps")
    
    best_acc = 0
    results = []
    
    os.makedirs(args.output, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, scheduler
        )
        
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.2f}%")
        print(f"  Val:   Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.2f}%")
        print(f"         Ent={val_metrics['entailment_acc']:.1f}%, "
              f"Neu={val_metrics['neutral_acc']:.1f}%, "
              f"Con={val_metrics['contradiction_acc']:.1f}%")
        
        results.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
        })
        
        # Save best model
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': best_acc,
            }, os.path.join(args.output, 'hybrid_best.pt'))
            print(f"  ✓ New best: {best_acc:.2f}%")
    
    # Final results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    
    # Save results
    with open(os.path.join(args.output, 'hybrid_results.json'), 'w') as f:
        json.dump({
            'model': 'HybridPoHHRM',
            'dataset': args.dataset,
            'parameters': param_count,
            'best_val_acc': best_acc,
            'config': vars(args),
            'history': results,
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {args.output}")


if __name__ == '__main__':
    main()

