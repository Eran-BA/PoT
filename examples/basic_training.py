#!/usr/bin/env python3
"""
Basic Training Example - Minimal code to train a PoH parser.

This example shows the simplest way to train a Pointer-over-Heads
dependency parser with dummy data.

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import torch
from transformers import AutoTokenizer

from src.models import PoHParser
from src.data.loaders import create_dummy_dataset, build_label_vocab
from src.training.trainer import Trainer


def main():
    """Basic training example."""
    print("=" * 80)
    print("Basic PoH Parser Training Example")
    print("=" * 80)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Create dummy data (for quick testing)
    print("Creating dummy data...")
    train_data = create_dummy_dataset(n_samples=100)
    dev_data = create_dummy_dataset(n_samples=50)

    # Build label vocabulary
    label_vocab = build_label_vocab(train_data + dev_data)
    n_labels = len(label_vocab)
    print(f"Dataset: {len(train_data)} train, {len(dev_data)} dev")
    print(f"Labels: {n_labels} unique dependency relations")

    # Create PoH parser with optimal configuration
    print("\nCreating PoH parser...")
    parser = PoHParser(
        enc_name="distilbert-base-uncased",
        d_model=768,
        n_heads=8,
        d_ff=2048,
        max_inner_iters=1,  # 1 iteration is optimal!
        routing_topk=0,  # Soft routing performs best
        halting_mode="fixed",  # Simple fixed iterations
        n_labels=n_labels,
        use_labels=True,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in parser.parameters())
    trainable_params = sum(p.numel() for p in parser.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create trainer
    print("\nCreating trainer...")
    trainer = Trainer(parser, tokenizer, device, label_vocab)

    # Training loop
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)

    num_epochs = 3
    batch_size = 16
    learning_rate = 3e-5

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        # Train
        train_metrics = trainer.train_epoch(train_data, batch_size=batch_size, lr=learning_rate)

        # Evaluate
        dev_metrics = trainer.eval_epoch(dev_data, batch_size=batch_size)

        # Print results
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Dev UAS: {dev_metrics['uas']:.4f}")
        print(f"  Dev LAS: {dev_metrics['las']:.4f}")
        print(f"  Time: {dev_metrics['time']:.1f}s")

    # Final summary
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Final Dev UAS: {dev_metrics['uas']:.4f}")
    print(f"Final Dev LAS: {dev_metrics['las']:.4f}")

    # Save model (optional)
    # torch.save(parser.state_dict(), "poh_parser.pt")
    # print("\nModel saved to poh_parser.pt")


if __name__ == "__main__":
    main()
