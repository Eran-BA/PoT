#!/usr/bin/env python3
"""
Simple PoH parser training script (single model, no A/B comparison).

Usage:
    python scripts/train_simple.py --epochs 5 --batch_size 32

Examples:
    # Quick test with dummy data
    python scripts/train_simple.py --data_source dummy --epochs 2

    # Real data training
    python scripts/train_simple.py --data_source conllu --conllu_dir data/ --epochs 10

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import argparse
import torch
from transformers import AutoTokenizer

from src.models import PoHParser
from src.data.loaders import get_dataset, build_label_vocab
from src.training.trainer import Trainer
from src.training.schedulers import get_linear_schedule_with_warmup  # noqa: F401 (reserved)
from src.utils.helpers import seed_everything, load_yaml_config
from src.utils.logger import get_env_info


def main():
    """Main training function for single PoH parser."""
    ap = argparse.ArgumentParser(description="Train PoH dependency parser")
    ap.add_argument("--config", type=str, default=None, help="Optional YAML config file")

    # Data
    ap.add_argument("--data_source", type=str, default="dummy", choices=["hf", "conllu", "dummy"])
    ap.add_argument("--conllu_dir", type=str, default=None)

    # Model
    ap.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=2048)

    # PoH configuration
    ap.add_argument(
        "--halting_mode", type=str, default="fixed", choices=["fixed", "entropy", "halting"]
    )
    ap.add_argument("--max_inner_iters", type=int, default=1)
    ap.add_argument("--routing_topk", type=int, default=0, help="0=soft routing, >0=hard top-k")
    ap.add_argument(
        "--combination", type=str, default="mask_concat", choices=["mask_concat", "mixture"]
    )
    ap.add_argument("--ent_threshold", type=float, default=0.8)

    # Training
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--compile", action="store_true", help="Enable torch.compile for speed (PyTorch>=2.0)"
    )

    # Logging
    ap.add_argument("--log_csv", type=str, default=None)
    ap.add_argument("--emit_conllu", action="store_true")

    args = ap.parse_args()

    # Load YAML config and override defaults if provided
    if args.config:
        cfg = load_yaml_config(args.config)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    # Set seed for reproducibility (full determinism)
    seed_everything(args.seed)

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

    # Create PoH parser
    print(f"\n{'='*80}")
    print("Creating PoH parser...")
    print(f"{'='*80}\n")

    parser = PoHParser(
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
    ).to(device)

    # Optional torch.compile for speed (PyTorch 2.0+)
    if args.compile:
        try:
            parser = torch.compile(parser)  # type: ignore[attr-defined]
            print("Enabled torch.compile")
        except Exception as e:
            print(f"Warning: torch.compile not available/failed: {e}")

    # Count parameters
    total_params = sum(p.numel() for p in parser.parameters())
    print(f"Total parameters: {total_params:,}")

    # Create trainer
    trainer = Trainer(parser, tokenizer, device, label_vocab)

    # Training loop
    print(f"\n{'='*80}")
    print("Training...")
    print(f"{'='*80}\n")

    best_uas = 0.0
    best_epoch = 0
    results = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 80)

        # Train
        train_metrics = trainer.train_epoch(
            train_data,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            use_amp=True,
            grad_accum_steps=1,
        )

        # Evaluate
        dev_metrics = trainer.eval_epoch(
            dev_data,
            batch_size=args.batch_size,
            emit_conllu=args.emit_conllu,
            conllu_path=f"predictions_epoch{epoch+1}.conllu" if args.emit_conllu else None,
        )

        # Print results
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Dev UAS: {dev_metrics['uas']:.4f}, LAS: {dev_metrics['las']:.4f}")
        print(f"  Time: {dev_metrics['time']:.1f}s")

        if dev_metrics.get("mean_inner_iters", 0) > 0:
            print(f"  Mean Inner Iterations: {dev_metrics['mean_inner_iters']:.2f}")

        # Track best model
        if dev_metrics["uas"] > best_uas:
            best_uas = dev_metrics["uas"]
            best_epoch = epoch + 1
            print("  ✓ New best UAS!")
            try:
                torch.save(parser.state_dict(), "best_poh.pt")
                print("  Saved best checkpoint to best_poh.pt")
            except Exception as e:
                print(f"  Warning: failed to save checkpoint: {e}")

        # Log results
        results.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "dev_uas": dev_metrics["uas"],
                "dev_las": dev_metrics["las"],
                "mean_inner_iters": dev_metrics.get("mean_inner_iters", 0.0),
                "time": dev_metrics["time"],
            }
        )

    # Final summary
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}\n")
    print(f"Best model: Epoch {best_epoch} with UAS = {best_uas:.4f}")

    # CSV logging
    if args.log_csv:
        import csv

        # augment with env info on first row
        env = get_env_info()
        results[0].update({f"env_{k}": v for k, v in env.items()})
        with open(args.log_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to {args.log_csv}")


if __name__ == "__main__":
    main()
