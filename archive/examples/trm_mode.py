#!/usr/bin/env python3
"""
TRM Mode Example - Training with TRM-style outer supervision.

This example demonstrates the TRM (Tiny Recursive Model) training mode,
which applies supervision at multiple outer steps with inner refinement loops.

Reference: "Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871)

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import argparse
import torch
from transformers import AutoTokenizer

from src.models import PoHParser
from src.data.loaders import get_dataset, build_label_vocab


class SimpleArgs:
    """Simple args container for TRM configuration."""

    def __init__(self, trm_supervision_steps, trm_inner_updates, trm_ramp_strength):
        self.trm_supervision_steps = trm_supervision_steps
        self.trm_inner_updates = trm_inner_updates
        self.trm_ramp_strength = trm_ramp_strength
        self.grad_mode = "full"


def main():
    """TRM-style training example."""
    ap = argparse.ArgumentParser(description="TRM Mode Training Example")
    ap.add_argument("--data_source", type=str, default="dummy", choices=["conllu", "dummy"])
    ap.add_argument("--conllu_dir", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--trm_supervision_steps", type=int, default=2)
    ap.add_argument("--trm_inner_updates", type=int, default=1)
    ap.add_argument("--trm_ramp_strength", type=float, default=1.0)
    args = ap.parse_args()

    print("=" * 80)
    print("TRM Mode Training Example")
    print("=" * 80)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load data
    print(f"Loading {args.data_source} data...")
    train_data = get_dataset(args.data_source, "train", args.conllu_dir)
    dev_data = get_dataset(args.data_source, "validation", args.conllu_dir)

    # Build label vocab
    label_vocab = build_label_vocab(train_data + dev_data)
    n_labels = len(label_vocab)
    print(f"Dataset: {len(train_data)} train, {len(dev_data)} dev")
    print(f"Labels: {n_labels} unique relations")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Create PoH parser
    print("\nCreating PoH parser for TRM mode...")
    parser = PoHParser(
        enc_name="distilbert-base-uncased",
        d_model=768,
        n_heads=8,
        d_ff=2048,
        max_inner_iters=args.trm_inner_updates,  # Inner updates per step
        routing_topk=0,
        halting_mode="fixed",
        n_labels=n_labels,
        use_labels=True,
    ).to(device)

    print("TRM Configuration:")
    print(f"  Supervision Steps: {args.trm_supervision_steps}")
    print(f"  Inner Updates: {args.trm_inner_updates}")
    print(f"  Ramp Strength: {args.trm_ramp_strength}")

    # Create TRM args
    trm_args = SimpleArgs(
        trm_supervision_steps=args.trm_supervision_steps,
        trm_inner_updates=args.trm_inner_updates,
        trm_ramp_strength=args.trm_ramp_strength,
    )

    # Optimizer with differentiated learning rates
    encoder_params = list(parser.encoder.parameters())
    controller_params = list(parser.block.controller.parameters())
    other_params = [
        p for n, p in parser.named_parameters() if "encoder" not in n and "controller" not in n
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.lr},
            {"params": controller_params, "lr": args.lr * 20},  # 20x for controller
            {"params": other_params, "lr": args.lr * 2},
        ],
        weight_decay=0.01,
    )

    # Training loop
    print("\n" + "=" * 80)
    print("Training with TRM Mode")
    print("=" * 80)

    from src.data.collate import collate_batch

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        # Training
        parser.train()
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(train_data), args.batch_size):
            batch = train_data[i : i + args.batch_size]

            # Collate
            enc, word_ids, heads, labels = collate_batch(batch, tokenizer, device, label_vocab)

            # Forward with TRM mode
            loss, metrics = parser.forward_trm(enc, word_ids, heads, labels, args=trm_args)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parser.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"  Train Loss: {avg_loss:.4f}")

        # Evaluation (use standard forward, not TRM)
        parser.eval()
        total_uas = 0.0
        total_las = 0.0
        total_tokens = 0

        with torch.no_grad():
            for i in range(0, len(dev_data), args.batch_size):
                batch = dev_data[i : i + args.batch_size]

                # Collate
                enc, word_ids, heads, labels = collate_batch(batch, tokenizer, device, label_vocab)

                # Standard forward for evaluation
                _, metrics = parser(enc, word_ids, heads, labels)

                total_uas += metrics["uas"] * metrics["tokens"]
                total_las += metrics["las"] * metrics["tokens"]
                total_tokens += metrics["tokens"]

        dev_uas = total_uas / total_tokens
        dev_las = total_las / total_tokens

        print(f"  Dev UAS: {dev_uas:.4f}")
        print(f"  Dev LAS: {dev_las:.4f}")

    # Summary
    print("\n" + "=" * 80)
    print("TRM Training Complete!")
    print("=" * 80)
    print(f"Final Dev UAS: {dev_uas:.4f}")
    print(f"Final Dev LAS: {dev_las:.4f}")

    print("\nTRM Mode Details:")
    print(f"  - Applied supervision at {args.trm_supervision_steps} outer steps")
    print(f"  - Each step performed {args.trm_inner_updates} inner updates")
    print(f"  - Deep supervision ramp: {args.trm_ramp_strength}")
    print("\nThis allows the model to recursively refine its predictions")
    print("with intermediate supervision, encouraging progressive improvement.")


if __name__ == "__main__":
    main()
