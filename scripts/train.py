#!/usr/bin/env python3
"""
Main training script for A/B comparison: Baseline vs Pointer-over-Heads.

Usage:
    python scripts/train.py --data_source conllu --conllu_dir data/ --epochs 5
    
Examples:
    # Basic training with dummy data
    python scripts/train.py --data_source dummy --epochs 2
    
    # Real data training
    python scripts/train.py --data_source conllu --conllu_dir data/ --epochs 10
    
    # TRM mode
    python scripts/train.py --trm_mode --trm_supervision_steps 2 --trm_inner_updates 1

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import argparse
import torch
from transformers import AutoTokenizer

from src.models import BaselineParser, PoHParser
from src.data.loaders import get_dataset, build_label_vocab
from src.training.trainer import Trainer
from src.training.schedulers import get_linear_schedule_with_warmup


def main():
    ap = argparse.ArgumentParser(description="Train and compare Baseline vs PoH parsers")
    
    # Data
    ap.add_argument("--data_source", type=str, default="dummy", choices=["hf", "conllu", "dummy"])
    ap.add_argument("--conllu_dir", type=str, default=None)
    
    # Model architecture
    ap.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=2048)
    
    # PoH-specific
    ap.add_argument("--halting_mode", type=str, default="fixed", choices=["fixed", "entropy", "halting"])
    ap.add_argument("--max_inner_iters", type=int, default=1)
    ap.add_argument("--routing_topk", type=int, default=0, help="0=soft, >0=hard top-k")
    ap.add_argument("--combination", type=str, default="mask_concat", choices=["mask_concat", "mixture"])
    ap.add_argument("--ent_threshold", type=float, default=0.8)
    
    # Iterative refinement
    ap.add_argument("--deep_supervision", action="store_true")
    ap.add_argument("--act_halting", action="store_true")
    ap.add_argument("--ponder_coef", type=float, default=1e-3)
    ap.add_argument("--ramp_strength", type=float, default=1.0)
    ap.add_argument("--grad_mode", type=str, default="full", choices=["full", "last"])
    
    # TRM mode
    ap.add_argument("--trm_mode", action="store_true")
    ap.add_argument("--trm_supervision_steps", type=int, default=2)
    ap.add_argument("--trm_inner_updates", type=int, default=None)
    ap.add_argument("--trm_ramp_strength", type=float, default=1.0)
    
    # Training
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    
    # Evaluation
    ap.add_argument("--ignore_punct", action="store_true")
    ap.add_argument("--emit_conllu", action="store_true")
    
    # Logging
    ap.add_argument("--log_csv", type=str, default=None)
    
    args = ap.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print(f"\n{'='*80}")
    print(f"Loading data from {args.data_source}...")
    print(f"{'='*80}\n")
    
    train_data = get_dataset(args.data_source, "train", args.conllu_dir)
    dev_data = get_dataset(args.data_source, "validation", args.conllu_dir)
    
    print(f"Train: {len(train_data)} examples")
    print(f"Dev: {len(dev_data)} examples")
    
    # Build label vocabulary
    label_vocab = build_label_vocab(train_data + dev_data)
    n_labels = len(label_vocab)
    print(f"Labels: {n_labels} unique dependency relations")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create models
    print(f"\n{'='*80}")
    print("Creating models...")
    print(f"{'='*80}\n")
    
    baseline = BaselineParser(
        enc_name=args.model_name,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_labels=n_labels,
        use_labels=True
    ).to(device)
    
    poh = PoHParser(
        enc_name=args.model_name,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        halting_mode=args.halting_mode,
        max_inner_iters=args.max_inner_iters,
        routing_topk=args.routing_topk,
        combination=args.combination,
        ent_threshold=args.ent_threshold,
        n_labels=n_labels,
        use_labels=True,
        deep_supervision=args.deep_supervision,
        act_halting=args.act_halting,
        ponder_coef=args.ponder_coef,
        ramp_strength=args.ramp_strength,
        grad_mode=args.grad_mode
    ).to(device)
    
    # Count parameters
    baseline_params = sum(p.numel() for p in baseline.parameters())
    poh_params = sum(p.numel() for p in poh.parameters())
    print(f"Baseline parameters: {baseline_params:,}")
    print(f"PoH parameters: {poh_params:,}")
    
    # Create trainers
    baseline_trainer = Trainer(baseline, tokenizer, device, label_vocab)
    poh_trainer = Trainer(poh, tokenizer, device, label_vocab)
    
    # Calculate training steps for scheduler
    steps_per_epoch = len(train_data) // args.batch_size
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(args.warmup_ratio * total_steps)
    
    # Training loop
    print(f"\n{'='*80}")
    print("Training...")
    print(f"{'='*80}\n")
    
    results = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 80)
        
        # Train baseline
        print("\n[Baseline]")
        train_metrics_b = baseline_trainer.train_epoch(
            train_data,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        eval_metrics_b = baseline_trainer.eval_epoch(dev_data, batch_size=args.batch_size)
        print(f"  Train Loss: {train_metrics_b['loss']:.4f}")
        print(f"  Dev UAS: {eval_metrics_b['uas']:.4f}, LAS: {eval_metrics_b['las']:.4f}")
        
        # Train PoH
        print("\n[PoH]")
        train_metrics_p = poh_trainer.train_epoch(
            train_data,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            args=args
        )
        eval_metrics_p = poh_trainer.eval_epoch(
            dev_data,
            batch_size=args.batch_size,
            args=args
        )
        print(f"  Train Loss: {train_metrics_p['loss']:.4f}")
        print(f"  Dev UAS: {eval_metrics_p['uas']:.4f}, LAS: {eval_metrics_p['las']:.4f}")
        if eval_metrics_p.get('mean_inner_iters', 0) > 0:
            print(f"  Mean Inner Iters: {eval_metrics_p['mean_inner_iters']:.2f}")
        
        # Log results
        results.append({
            "epoch": epoch + 1,
            "baseline_dev_uas": eval_metrics_b["uas"],
            "baseline_dev_las": eval_metrics_b["las"],
            "poh_dev_uas": eval_metrics_p["uas"],
            "poh_dev_las": eval_metrics_p["las"],
            "poh_mean_inner_iters": eval_metrics_p.get("mean_inner_iters", 0.0)
        })
    
    # Final summary
    print(f"\n{'='*80}")
    print("Final Results")
    print(f"{'='*80}\n")
    
    best_baseline = max(results, key=lambda x: x["baseline_dev_uas"])
    best_poh = max(results, key=lambda x: x["poh_dev_uas"])
    
    print(f"Best Baseline - Epoch {best_baseline['epoch']}: "
          f"UAS={best_baseline['baseline_dev_uas']:.4f}, LAS={best_baseline['baseline_dev_las']:.4f}")
    print(f"Best PoH - Epoch {best_poh['epoch']}: "
          f"UAS={best_poh['poh_dev_uas']:.4f}, LAS={best_poh['poh_dev_las']:.4f}")
    
    # CSV logging
    if args.log_csv:
        import csv
        with open(args.log_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {args.log_csv}")


if __name__ == "__main__":
    main()

