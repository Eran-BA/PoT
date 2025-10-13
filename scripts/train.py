"""
Unified training entry point for all PoT tasks.

Usage:
    python scripts/train.py --task sorting --config experiments/configs/sorting/len12.yaml
    python scripts/train.py --task dependency --config experiments/configs/parsing/ud_en.yaml

Author: Eran Ben Artzy
Year: 2025
"""

import argparse
import yaml
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pot.tasks import TaskAdapter, SortingTask, DependencyParsingTask


TASK_REGISTRY = {
    'sorting': SortingTask,
    'dependency': DependencyParsingTask,
}


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    task: TaskAdapter,
    config: dict,
    scaler: GradScaler = None,
    device: str = 'cuda'
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_metrics = {}

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        optimizer.zero_grad()

        # Forward pass (with AMP if enabled)
        if scaler is not None:
            with autocast():
                output = model(batch)
                loss = task.compute_loss(output, batch, config)
        else:
            output = model(batch)
            loss = task.compute_loss(output, batch, config)

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        # Compute metrics
        with torch.no_grad():
            metrics = task.compute_metrics(output, batch, config)
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v

    # Average
    avg_loss = total_loss / len(dataloader)
    avg_metrics = {k: v / len(dataloader) for k, v in total_metrics.items()}

    return avg_loss, avg_metrics


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    task: TaskAdapter,
    config: dict,
    device: str = 'cuda'
):
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_metrics = {}

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        output = model(batch)
        loss = task.compute_loss(output, batch, config)
        total_loss += loss.item()

        metrics = task.compute_metrics(output, batch, config)
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v

    avg_loss = total_loss / len(dataloader)
    avg_metrics = {k: v / len(dataloader) for k, v in total_metrics.items()}

    return avg_loss, avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Train PoT model on any task')
    parser.add_argument('--task', type=str, required=True,
                        choices=list(TASK_REGISTRY.keys()),
                        help='Task name')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='experiments/results',
                        help='Output directory')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config: {args.config}")
    print(json.dumps(config, indent=2))

    # Initialize task
    task_cls = TASK_REGISTRY[args.task]
    task = task_cls()

    # Prepare data
    print(f"\nPreparing {args.task} data...")
    train_ds, val_ds, test_ds = task.prepare_data(config)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.get('batch_size', 64),
        shuffle=True,
        collate_fn=task.collate_fn,
        num_workers=config.get('num_workers', 0)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.get('batch_size', 64),
        shuffle=False,
        collate_fn=task.collate_fn,
        num_workers=config.get('num_workers', 0)
    )

    # Build model
    print("\nBuilding model...")
    model = task.build_model(config)
    model = model.to(args.device)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.get('lr', 1e-4),
        weight_decay=config.get('weight_decay', 0.01)
    )

    # AMP scaler
    scaler = GradScaler() if args.device == 'cuda' and config.get('use_amp', False) else None

    # Training loop
    print(f"\nTraining for {config.get('epochs', 10)} epochs...")
    best_val_metric = -float('inf')

    for epoch in range(config.get('epochs', 10)):
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, task, config, scaler, args.device
        )

        # Validate
        val_loss, val_metrics = eval_epoch(
            model, val_loader, task, config, args.device
        )

        print(f"\nEpoch {epoch + 1}/{config.get('epochs', 10)}:")
        print(f"  Train loss: {train_loss:.4f} | {train_metrics}")
        print(f"  Val loss:   {val_loss:.4f} | {val_metrics}")

        # Save best model (based on first metric)
        val_metric = list(val_metrics.values())[0]
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            # Save checkpoint
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_dir / f'{args.task}_best.pt')
            print(f"  ✅ Saved best model (val_metric={val_metric:.4f})")

    print("\n✅ Training complete!")


if __name__ == '__main__':
    main()
