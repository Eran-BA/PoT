#!/usr/bin/env python3
"""
PPO Blocksworld Benchmark.

Trains a PoT Blocksworld solver using PPO with contrastive good/bad trajectory
learning. Compares against supervised baseline.

Key features:
- Good trajectories: C(n+1,2) sub-trajectory augmentation, reward +1
- Bad trajectories: No augmentation (full trajectories), reward -1
- PPO with clipped objective and GAE
- Actor: PoT model outputting all block positions at once
- Critic: Value network for advantage estimation

Usage:
    # Download data and generate trajectories
    python experiments/blocksworld_ppo_benchmark.py --download --generate-trajectories \\
        --fd-path /path/to/fast-downward.py
    
    # Train with PPO
    python experiments/blocksworld_ppo_benchmark.py --epochs 100 --R 4 --good-bad-ratio 1.0
    
    # Compare with supervised baseline
    python experiments/blocksworld_ppo_benchmark.py --mode supervised --epochs 100

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from torch.utils.data import DataLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='PPO Blocksworld Benchmark',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/blocksworld',
                       help='Directory for blocksworld data')
    parser.add_argument('--download', action='store_true',
                       help='Download dataset from HuggingFace')
    parser.add_argument('--generate-trajectories', action='store_true',
                       help='Generate trajectories with FastDownward')
    parser.add_argument('--fd-path', type=str, default='fast-downward.py',
                       help='Path to fast-downward.py')
    parser.add_argument('--fd-timeout', type=int, default=30,
                       help='FastDownward timeout per problem (seconds)')
    parser.add_argument('--max-blocks', type=int, default=6,
                       help='Maximum number of blocks')
    parser.add_argument('--max-plan-length', type=int, default=None,
                       help='Maximum plan length for training samples')
    
    # Training mode
    parser.add_argument('--mode', type=str, default='ppo',
                       choices=['ppo', 'supervised'],
                       help='Training mode: ppo or supervised')
    
    # PPO arguments
    parser.add_argument('--good-bad-ratio', type=float, default=1.0,
                       help='Ratio of bad to good trajectories')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable C(n+1,2) sub-trajectory augmentation for good trajectories')
    parser.add_argument('--clip-epsilon', type=float, default=0.2,
                       help='PPO clip epsilon')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                       help='Entropy coefficient')
    parser.add_argument('--value-coef', type=float, default=0.5,
                       help='Value loss coefficient')
    parser.add_argument('--ppo-epochs', type=int, default=4,
                       help='PPO epochs per batch')
    parser.add_argument('--target-kl', type=float, default=0.01,
                       help='Target KL for early stopping')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, default='simple',
                       choices=['simple', 'hybrid'],
                       help='Model type: simple (SimplePoT) or hybrid (HybridPoT with H/L cycles)')
    parser.add_argument('--R', type=int, default=4,
                       help='Number of refinement iterations (for simple model)')
    parser.add_argument('--H-cycles', type=int, default=2,
                       help='H_level (slow) cycles per ACT step (for hybrid model)')
    parser.add_argument('--L-cycles', type=int, default=8,
                       help='L_level (fast) cycles per H_cycle (for hybrid model)')
    parser.add_argument('--T', type=int, default=4,
                       help='HRM period for pointer controller (for hybrid model)')
    parser.add_argument('--halt-max-steps', type=int, default=1,
                       help='Max halting steps for ACT (for hybrid model)')
    parser.add_argument('--d-model', type=int, default=128,
                       help='Model hidden dimension')
    parser.add_argument('--n-heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--n-layers', type=int, default=2,
                       help='Number of transformer layers (H_layers and L_layers for hybrid)')
    parser.add_argument('--d-ff', type=int, default=512,
                       help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--controller-type', type=str, default='transformer',
                       help='Controller type for PoT')
    parser.add_argument('--share-embeddings', action='store_true',
                       help='Share embeddings between actor and critic')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr-actor', type=float, default=3e-4,
                       help='Actor learning rate')
    parser.add_argument('--lr-critic', type=float, default=1e-3,
                       help='Critic learning rate')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (for supervised mode)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--eval-interval', type=int, default=5,
                       help='Evaluation interval (epochs)')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Checkpoint save interval (epochs)')
    
    # Output
    parser.add_argument('--output-dir', type=str, 
                       default='experiments/results/blocksworld_ppo',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def setup_device():
    """Setup compute device."""
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Device: {device}")
    return device


def download_and_prepare_data(args):
    """Download and prepare Blocksworld dataset."""
    from src.data.blocksworld import download_blocksworld_dataset
    
    download_blocksworld_dataset(
        args.data_dir,
        max_blocks=args.max_blocks,
        generate_trajectories=args.generate_trajectories,
        fd_path=args.fd_path,
        fd_timeout=args.fd_timeout,
    )


def create_ppo_dataloaders(args, device):
    """Create dataloaders for PPO training."""
    from src.data.blocksworld import BlocksworldPPODataset
    
    augment_good = not args.no_augmentation
    
    train_dataset = BlocksworldPPODataset(
        args.data_dir,
        split='train',
        max_blocks=args.max_blocks,
        good_bad_ratio=args.good_bad_ratio,
        max_plan_length=args.max_plan_length,
        seed=args.seed,
        augment_good=augment_good,
    )
    
    val_dataset = BlocksworldPPODataset(
        args.data_dir,
        split='val',
        max_blocks=args.max_blocks,
        good_bad_ratio=0.0,  # No bad trajectories in validation
        max_plan_length=args.max_plan_length,
        seed=args.seed,
        augment_good=augment_good,
    )
    
    test_dataset = BlocksworldPPODataset(
        args.data_dir,
        split='test',
        max_blocks=args.max_blocks,
        good_bad_ratio=0.0,  # No bad trajectories in test
        max_plan_length=args.max_plan_length,
        seed=args.seed,
        augment_good=augment_good,
    )
    
    # Print stats
    print("\nDataset Statistics:")
    aug_str = "NO augmentation" if args.no_augmentation else "with C(n+1,2) augmentation"
    train_stats = train_dataset.get_stats()
    print(f"  Train: {train_stats['total']} samples "
          f"(good: {train_stats['good']}, bad: {train_stats['bad']}) [{aug_str}]")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    return train_loader, val_loader, test_loader, train_dataset


def create_supervised_dataloaders(args, device):
    """Create dataloaders for supervised training."""
    from src.data.blocksworld import BlocksworldDataset
    
    train_dataset = BlocksworldDataset(
        args.data_dir,
        split='train',
        max_blocks=args.max_blocks,
        mode='goal',
        max_plan_length=args.max_plan_length,
        augment=True,  # Sub-trajectory augmentation
    )
    
    val_dataset = BlocksworldDataset(
        args.data_dir,
        split='val',
        max_blocks=args.max_blocks,
        mode='goal',
        max_plan_length=args.max_plan_length,
        augment=False,
    )
    
    test_dataset = BlocksworldDataset(
        args.data_dir,
        split='test',
        max_blocks=args.max_blocks,
        mode='goal',
        max_plan_length=args.max_plan_length,
        augment=False,
    )
    
    print(f"\nDataset sizes: train={len(train_dataset)}, "
          f"val={len(val_dataset)}, test={len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    return train_loader, val_loader, test_loader, train_dataset


def create_actor_critic_model(args, device):
    """Create actor-critic model for PPO."""
    from src.pot.models.blocksworld_critic import create_actor_critic
    
    model = create_actor_critic(
        num_blocks=args.max_blocks,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        R=args.R,
        controller_type=args.controller_type,
        share_embeddings=args.share_embeddings,
        model_type=args.model_type,
        H_cycles=args.H_cycles,
        L_cycles=args.L_cycles,
        T=args.T,
        halt_max_steps=args.halt_max_steps,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    actor_params = sum(p.numel() for p in model.actor.parameters())
    critic_params = sum(p.numel() for p in model.critic.parameters())
    
    print(f"\nModel: Actor-Critic for PPO ({args.model_type})")
    print(f"  Actor params: {actor_params:,} ({actor_params/1e6:.2f}M)")
    print(f"  Critic params: {critic_params:,} ({critic_params/1e6:.2f}M)")
    print(f"  Total params: {total_params:,} ({total_params/1e6:.2f}M)")
    if args.model_type == 'hybrid':
        print(f"  H_cycles={args.H_cycles}, L_cycles={args.L_cycles}, T={args.T}")
    else:
        print(f"  R={args.R}, d_model={args.d_model}")
    
    return model.to(device)


def create_supervised_model(args, device):
    """Create model for supervised training."""
    if args.model_type == 'hybrid':
        from src.pot.models.blocksworld_solver import HybridPoTBlocksworldSolver
        
        model = HybridPoTBlocksworldSolver(
            num_blocks=args.max_blocks,
            d_model=args.d_model,
            n_heads=args.n_heads,
            H_layers=args.n_layers,
            L_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            H_cycles=args.H_cycles,
            L_cycles=args.L_cycles,
            T=args.T,
            halt_max_steps=args.halt_max_steps,
            controller_type=args.controller_type,
            goal_conditioned=True,
        )
        model_name = "HybridPoTBlocksworldSolver"
        config_str = f"H_cycles={args.H_cycles}, L_cycles={args.L_cycles}, T={args.T}"
    else:
        from src.pot.models.blocksworld_solver import SimplePoTBlocksworldSolver
        
        model = SimplePoTBlocksworldSolver(
            num_blocks=args.max_blocks,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            R=args.R,
            controller_type=args.controller_type,
            goal_conditioned=True,
        )
        model_name = "SimplePoTBlocksworldSolver"
        config_str = f"R={args.R}, d_model={args.d_model}"
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {model_name}")
    print(f"  Total params: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  {config_str}")
    
    return model.to(device)


def train_ppo(args, model, train_loader, val_loader, device, train_dataset):
    """Train using PPO."""
    from src.training.ppo_blocksworld import BlocksworldPPOTrainer, PPOConfig
    
    config = PPOConfig(
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=min(args.batch_size, 64),
        target_kl=args.target_kl,
    )
    
    trainer = BlocksworldPPOTrainer(
        actor_critic=model,
        config=config,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        device=device,
    )
    
    # Training loop
    best_val_acc = 0.0
    history = []
    
    print("\n" + "=" * 60)
    print("PPO Training")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        
        # Shuffle for next epoch
        train_dataset.on_epoch_end()
        
        # Evaluate periodically
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            val_metrics = trainer.evaluate(val_loader)
            
            print(f"\nEpoch {epoch}/{args.epochs}:")
            print(f"  Train - Policy Loss: {train_metrics['policy_loss']:.4f}, "
                  f"Value Loss: {train_metrics['value_loss']:.4f}, "
                  f"Entropy: {train_metrics['entropy']:.4f}")
            print(f"  Train - Slot Acc: {train_metrics['slot_accuracy']*100:.2f}%, "
                  f"Exact Match: {train_metrics['exact_match']*100:.2f}%")
            print(f"  Train - Mean Reward: {train_metrics['mean_reward']:.3f} "
                  f"(good: {train_metrics['good_reward']:.3f}, "
                  f"bad: {train_metrics['bad_reward']:.3f})")
            print(f"  Val   - Slot Acc: {val_metrics['slot_accuracy']*100:.2f}%, "
                  f"Exact Match: {val_metrics['exact_match']*100:.2f}%")
            
            if val_metrics['slot_accuracy'] > best_val_acc:
                best_val_acc = val_metrics['slot_accuracy']
                save_checkpoint(model, args.output_dir, 'best_model.pt')
                print(f"  [NEW BEST] Saved checkpoint")
            
            history.append({
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics,
            })
        else:
            print(f"\rEpoch {epoch}/{args.epochs} - "
                  f"Loss: {train_metrics['total_loss']:.4f}, "
                  f"Slot Acc: {train_metrics['slot_accuracy']*100:.2f}%", 
                  end='')
    
    print()
    return trainer, history, best_val_acc


def train_supervised(args, model, train_loader, val_loader, device, train_dataset):
    """Train using supervised learning."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    history = []
    
    print("\n" + "=" * 60)
    print("Supervised Training")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for batch in train_loader:
            states = batch['init_state'].to(device)
            goals = batch['goal_state'].to(device)
            
            optimizer.zero_grad()
            
            logits, _ = model(states, goals)
            
            # Loss
            B, N, V = logits.shape
            loss = criterion(logits.view(-1, V), goals.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Accuracy
            preds = logits.argmax(dim=-1)
            epoch_correct += (preds == goals).sum().item()
            epoch_total += goals.numel()
        
        train_loss = epoch_loss / len(train_loader)
        train_acc = epoch_correct / epoch_total
        
        # Shuffle for next epoch
        train_dataset.on_epoch_end()
        
        # Evaluate
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            val_loss, val_acc, val_exact = evaluate_supervised(
                model, val_loader, device
            )
            
            print(f"\nEpoch {epoch}/{args.epochs}:")
            print(f"  Train - Loss: {train_loss:.4f}, Slot Acc: {train_acc*100:.2f}%")
            print(f"  Val   - Loss: {val_loss:.4f}, Slot Acc: {val_acc*100:.2f}%, "
                  f"Exact: {val_exact*100:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(model, args.output_dir, 'best_model.pt')
                print(f"  [NEW BEST] Saved checkpoint")
            
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_exact': val_exact,
            })
        else:
            print(f"\rEpoch {epoch}/{args.epochs} - "
                  f"Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%", end='')
    
    print()
    return model, history, best_val_acc


@torch.no_grad()
def evaluate_supervised(model, dataloader, device):
    """Evaluate supervised model."""
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_exact = 0
    total_samples = 0
    total_positions = 0
    
    criterion = nn.CrossEntropyLoss()
    
    for batch in dataloader:
        states = batch['init_state'].to(device)
        goals = batch['goal_state'].to(device)
        
        logits, _ = model(states, goals)
        
        B, N, V = logits.shape
        loss = criterion(logits.view(-1, V), goals.view(-1))
        total_loss += loss.item()
        
        preds = logits.argmax(dim=-1)
        total_correct += (preds == goals).sum().item()
        total_exact += (preds == goals).all(dim=1).sum().item()
        total_samples += B
        total_positions += goals.numel()
    
    return (
        total_loss / len(dataloader),
        total_correct / total_positions,
        total_exact / total_samples,
    )


def save_checkpoint(model, output_dir, filename):
    """Save model checkpoint."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path / filename)


def save_results(args, history, best_val_acc, test_metrics):
    """Save training results."""
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        'args': vars(args),
        'history': history,
        'best_val_acc': best_val_acc,
        'test_metrics': test_metrics,
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_path / 'results.json'}")


def main():
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Setup
    device = setup_device()
    
    # Download data if requested
    if args.download:
        download_and_prepare_data(args)
    
    # Create dataloaders based on mode
    if args.mode == 'ppo':
        train_loader, val_loader, test_loader, train_dataset = create_ppo_dataloaders(
            args, device
        )
        model = create_actor_critic_model(args, device)
        
        trainer, history, best_val_acc = train_ppo(
            args, model, train_loader, val_loader, device, train_dataset
        )
        
        # Final test evaluation
        print("\n" + "=" * 60)
        print("FINAL TEST EVALUATION")
        print("=" * 60)
        
        # Load best model
        best_path = Path(args.output_dir) / 'best_model.pt'
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device))
        
        test_metrics = trainer.evaluate(test_loader)
        
    else:  # supervised
        train_loader, val_loader, test_loader, train_dataset = create_supervised_dataloaders(
            args, device
        )
        model = create_supervised_model(args, device)
        
        model, history, best_val_acc = train_supervised(
            args, model, train_loader, val_loader, device, train_dataset
        )
        
        # Final test evaluation
        print("\n" + "=" * 60)
        print("FINAL TEST EVALUATION")
        print("=" * 60)
        
        # Load best model
        best_path = Path(args.output_dir) / 'best_model.pt'
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device))
        
        test_loss, test_acc, test_exact = evaluate_supervised(model, test_loader, device)
        test_metrics = {
            'loss': test_loss,
            'slot_accuracy': test_acc,
            'exact_match': test_exact,
        }
    
    print(f"\nTest Results:")
    print(f"  Slot Accuracy: {test_metrics['slot_accuracy']*100:.2f}%")
    print(f"  Exact Match: {test_metrics['exact_match']*100:.2f}%")
    
    # Save results
    save_results(args, history, best_val_acc, test_metrics)
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()

