#!/usr/bin/env python3
"""
Sudoku HPO - Hyperparameter Optimization with Optuna, Ray, and W&B
===================================================================

Runs parallel hyperparameter search across multiple GPUs using:
- Optuna for HPO (TPE sampler, MedianPruner)
- Ray Tune for parallel trial execution (1 trial per GPU)
- Weights & Biases for logging

Usage:
    # Run HPO with 4 parallel trials on 4 GPUs
    python experiments/sudoku_hpo.py --n-trials 50 --epochs-per-trial 500
    
    # Resume a study
    python experiments/sudoku_hpo.py --study-name my_study --resume
    
    # Quick test
    python experiments/sudoku_hpo.py --n-trials 4 --epochs-per-trial 50 --eval-interval 10

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import sys
import os
import math
import argparse
from typing import Dict, Any, Optional
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Optuna
import optuna
from optuna.trial import Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Ray
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune import Trainable
from ray.train import Checkpoint
from ray.air.integrations.wandb import WandbLoggerCallback
import tempfile

# W&B
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Install with: pip install wandb")

# Project imports - only download_sudoku_dataset at top level
# Other imports happen inside train_trial for Ray workers
from src.data import download_sudoku_dataset


# ============================================================================
# Search Space Definition
# ============================================================================

def get_search_space(trial: Trial) -> Dict[str, Any]:
    """Define the hyperparameter search space."""
    return {
        # Learning rate (log scale)
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        
        # Weight decay (HRM uses 0.1)
        "weight_decay": trial.suggest_float("weight_decay", 0.05, 0.5),
        
        # Puzzle embedding learning rate multiplier
        "puzzle_lr_multiplier": trial.suggest_float("puzzle_lr_multiplier", 10.0, 200.0),
        
        # Puzzle weight decay
        "puzzle_weight_decay": trial.suggest_float("puzzle_weight_decay", 0.01, 2.0),
        
        # Architecture: H and L cycles
        "H_cycles": 2,  # Fixed at 2
        "L_cycles": trial.suggest_categorical("L_cycles", [4, 8]),
        
        # ACT halting
        "halt_max_steps": trial.suggest_categorical("halt_max_steps", [2, 3, 4]),
        "halt_exploration": trial.suggest_float("halt_exploration", 0.05, 0.15),
        
        # Regularization
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        
        # Optimizer settings
        "beta2": trial.suggest_float("beta2", 0.9, 0.999),
        
        # Warmup
        "warmup_steps": trial.suggest_int("warmup_steps", 500, 4000, step=500),
        
        # Training mode
        "async_batch": trial.suggest_categorical("async_batch", [True, False]),
    }


def get_ray_search_space() -> Dict[str, Any]:
    """Define search space for Ray Tune format."""
    return {
        "lr": tune.loguniform(1e-5, 1e-3),
        "weight_decay": tune.uniform(0.05, 0.5),
        "puzzle_lr_multiplier": tune.uniform(10.0, 200.0),
        "puzzle_weight_decay": tune.uniform(0.01, 2.0),
        "H_cycles": 2,  # Fixed at 2
        "L_cycles": tune.choice([4, 8]),
        "halt_max_steps": tune.choice([2, 3, 4]),
        "halt_exploration": tune.uniform(0.05, 0.15),
        "dropout": tune.uniform(0.0, 0.3),
        "beta2": tune.uniform(0.9, 0.999),
        "warmup_steps": tune.choice([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]),
        "async_batch": tune.choice([True, False]),
    }


# ============================================================================
# Training Function
# ============================================================================

class SudokuTrainable(Trainable):
    """Ray Tune Trainable for Sudoku HPO."""
    
    def setup(self, config: Dict[str, Any]):
        """Initialize model, data, and optimizers."""
        self.config = config
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load data
        data_dir = config.get("data_dir", "data/sudoku-extreme-10k-aug-100")
        batch_size = config.get("batch_size", 128)
        
        self.train_dataset = SudokuDataset(data_dir, 'train')
        self.val_dataset = SudokuDataset(data_dir, 'val')
        
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True,
        )
        
        # Model
        self.model = HybridPoHHRMSolver(
            d_model=config.get("d_model", 512),
            n_heads=config.get("n_heads", 8),
            H_layers=config.get("H_layers", 2),
            L_layers=config.get("L_layers", 2),
            d_ff=config.get("d_ff", 2048),
            dropout=config["dropout"],
            H_cycles=config["H_cycles"],
            L_cycles=config["L_cycles"],
            T=config.get("T", 4),
            num_puzzles=1,
            hrm_grad_style=config.get("hrm_grad_style", True),
            halt_max_steps=config.get("halt_max_steps", 4),
            halt_exploration_prob=config.get("halt_exploration", 0.1),
        ).to(self.device)
        
        # Optimizers
        puzzle_lr = config["lr"] * config["puzzle_lr_multiplier"]
        betas = (0.9, config["beta2"])
        
        puzzle_params = list(self.model.puzzle_emb.parameters())
        model_params = [p for p in self.model.parameters() if p not in set(puzzle_params)]
        
        self.optimizer = torch.optim.AdamW(
            model_params, 
            lr=config["lr"], 
            weight_decay=config["weight_decay"],
            betas=betas,
        )
        self.puzzle_optimizer = torch.optim.AdamW(
            puzzle_params, 
            lr=puzzle_lr, 
            weight_decay=config["puzzle_weight_decay"],
            betas=betas,
        )
        
        # LR Scheduler
        total_steps = config.get("epochs_per_trial", 500) * len(self.train_loader)
        warmup_steps = config["warmup_steps"]
        
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.puzzle_scheduler = torch.optim.lr_scheduler.LambdaLR(self.puzzle_optimizer, lr_lambda)
        
        self.epoch = 0
        self.best_grid_acc = 0.0
    
    def step(self) -> Dict[str, float]:
        """Run one epoch of training."""
        self.epoch += 1
        
        # Train
        train_metrics = train_epoch(
            self.model, 
            self.train_loader, 
            self.optimizer, 
            self.puzzle_optimizer,
            self.device, 
            self.epoch, 
            use_poh=True,
            scheduler=self.scheduler,
            puzzle_scheduler=self.puzzle_scheduler,
            constraint_weight=0.0,
        )
        
        # Resample augmentations
        self.train_dataset.on_epoch_end()
        
        # Evaluate
        eval_interval = self.config.get("eval_interval", 50)
        if self.epoch % eval_interval == 0 or self.epoch == 1:
            val_metrics = evaluate(self.model, self.val_loader, self.device, use_poh=True)
            
            if val_metrics["grid_acc"] > self.best_grid_acc:
                self.best_grid_acc = val_metrics["grid_acc"]
            
            return {
                "train_loss": train_metrics["loss"],
                "train_cell_acc": train_metrics["cell_acc"],
                "train_grid_acc": train_metrics["grid_acc"],
                "val_loss": val_metrics["loss"],
                "val_cell_acc": val_metrics["cell_acc"],
                "val_grid_acc": val_metrics["grid_acc"],
                "best_grid_acc": self.best_grid_acc,
                "epoch": self.epoch,
            }
        else:
            return {
                "train_loss": train_metrics["loss"],
                "train_cell_acc": train_metrics["cell_acc"],
                "train_grid_acc": train_metrics["grid_acc"],
                "best_grid_acc": self.best_grid_acc,
                "epoch": self.epoch,
            }
    
    def save_checkpoint(self, checkpoint_dir: str) -> str:
        """Save model checkpoint."""
        path = os.path.join(checkpoint_dir, "checkpoint.pt")
        torch.save({
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_grid_acc": self.best_grid_acc,
        }, path)
        return path
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load model checkpoint."""
        path = os.path.join(checkpoint_dir, "checkpoint.pt")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_grid_acc = checkpoint["best_grid_acc"]


def train_trial(config: Dict[str, Any]) -> None:
    """
    Training function for a single trial (function-based API).
    
    This is used when running with OptunaSearch directly.
    """
    # Import everything inside the function for Ray workers
    import os
    import sys
    import math
    import tempfile
    
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from ray import tune
    from ray.tune import Checkpoint
    
    # Add project root to path
    project_root = config.get("project_root")
    if project_root and project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Now import project modules
    from src.pot.models import HybridPoHHRMSolver
    from src.training import train_epoch, train_epoch_async, evaluate
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get data from Ray object store (shared by main process)
    import ray
    train_dataset = ray.get(config["train_data_ref"])
    val_dataset = ray.get(config["val_data_ref"])
    batch_size = config.get("batch_size", 128)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Model
    model = HybridPoHHRMSolver(
        d_model=config.get("d_model", 512),
        n_heads=config.get("n_heads", 8),
        H_layers=config.get("H_layers", 2),
        L_layers=config.get("L_layers", 2),
        d_ff=config.get("d_ff", 2048),
        dropout=config["dropout"],
        H_cycles=config["H_cycles"],
        L_cycles=config["L_cycles"],
        T=config.get("T", 4),
        num_puzzles=1,
        hrm_grad_style=config.get("hrm_grad_style", True),
        halt_max_steps=config.get("halt_max_steps", 4),
        halt_exploration_prob=config.get("halt_exploration", 0.1),
    ).to(device)
    
    # Optimizers
    puzzle_lr = config["lr"] * config["puzzle_lr_multiplier"]
    betas = (0.9, config["beta2"])
    
    puzzle_params = list(model.puzzle_emb.parameters())
    model_params = [p for p in model.parameters() if p not in set(puzzle_params)]
    
    optimizer = torch.optim.AdamW(
        model_params, lr=config["lr"], weight_decay=config["weight_decay"], betas=betas
    )
    puzzle_optimizer = torch.optim.AdamW(
        puzzle_params, lr=puzzle_lr, weight_decay=config["puzzle_weight_decay"], betas=betas
    )
    
    # LR Scheduler
    epochs_per_trial = config.get("epochs_per_trial", 500)
    total_steps = epochs_per_trial * len(train_loader)
    warmup_steps = config["warmup_steps"]
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    puzzle_scheduler = torch.optim.lr_scheduler.LambdaLR(puzzle_optimizer, lr_lambda)
    
    best_grid_acc = 0.0
    eval_interval = config.get("eval_interval", 50)
    use_async = config.get("async_batch", False)
    checkpoint_interval = config.get("checkpoint_interval", eval_interval)
    
    for epoch in range(1, epochs_per_trial + 1):
        # Train (async or regular)
        if use_async:
            train_metrics = train_epoch_async(
                model, train_loader, optimizer, puzzle_optimizer,
                device, epoch, use_poh=True,
                scheduler=scheduler, puzzle_scheduler=puzzle_scheduler,
            )
        else:
            train_metrics = train_epoch(
                model, train_loader, optimizer, puzzle_optimizer,
                device, epoch, use_poh=True,
                scheduler=scheduler, puzzle_scheduler=puzzle_scheduler,
                constraint_weight=0.0,
            )
        
        train_dataset.on_epoch_end()
        
        # Evaluate and save checkpoint at eval intervals
        if epoch % eval_interval == 0 or epoch == 1:
            val_metrics = evaluate(model, val_loader, device, use_poh=True)
            
            if val_metrics["grid_acc"] > best_grid_acc:
                best_grid_acc = val_metrics["grid_acc"]
            
            metrics = {
                "train_loss": train_metrics["loss"],
                "train_cell_acc": train_metrics["cell_acc"],
                "train_grid_acc": train_metrics["grid_acc"],
                "val_loss": val_metrics["loss"],
                "val_cell_acc": val_metrics["cell_acc"],
                "val_grid_acc": val_metrics["grid_acc"],
                "best_grid_acc": best_grid_acc,
                "epoch": epoch,
            }
            
            # Save checkpoint at checkpoint intervals
            if epoch % checkpoint_interval == 0:
                with tempfile.TemporaryDirectory() as tmpdir:
                    checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_grid_acc": best_grid_acc,
                    }, checkpoint_path)
                    tune.report(**metrics, checkpoint=Checkpoint.from_directory(tmpdir))
            else:
                tune.report(**metrics)
        else:
            tune.report(
                train_loss=train_metrics["loss"],
                train_cell_acc=train_metrics["cell_acc"],
                train_grid_acc=train_metrics["grid_acc"],
                best_grid_acc=best_grid_acc,
                epoch=epoch,
            )


# ============================================================================
# Main HPO Runner
# ============================================================================

def run_hpo(args):
    """Run hyperparameter optimization."""
    
    # Get project root first
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Convert data_dir to absolute path
    abs_data_dir = os.path.join(project_root, args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir
    abs_data_dir = os.path.abspath(abs_data_dir)
    
    # Download dataset to absolute path if needed (BEFORE Ray starts)
    if args.download:
        download_sudoku_dataset(abs_data_dir, args.subsample, args.num_aug)
    
    # Verify data exists before starting Ray
    train_path = os.path.join(abs_data_dir, 'train')
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Data not found at {train_path}. Run with --download to fetch from HuggingFace."
        )
    print(f"✓ Data verified at: {abs_data_dir}")
    
    # Load data BEFORE Ray starts (on host that has filesystem access)
    print("Loading data into memory...")
    from src.data import SudokuDataset
    train_dataset = SudokuDataset(abs_data_dir, 'train')
    val_dataset = SudokuDataset(abs_data_dir, 'val')
    print(f"✓ Loaded {len(train_dataset)} train, {len(val_dataset)} val puzzles")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_gpus=args.num_gpus)
    
    # Put datasets in Ray object store for workers to access
    print("Sharing data via Ray object store...")
    train_data_ref = ray.put(train_dataset)
    val_data_ref = ray.put(val_dataset)
    print("✓ Data shared with workers")
    
    print(f"\n{'='*60}")
    print("Sudoku HPO - Hyperparameter Optimization")
    print(f"{'='*60}")
    print(f"Trials: {args.n_trials}")
    print(f"Epochs per trial: {args.epochs_per_trial}")
    print(f"Parallel trials: {args.num_gpus}")
    print(f"Study name: {args.study_name}")
    
    # Search space
    search_space = get_ray_search_space()
    
    # Add fixed config
    search_space.update({
        "project_root": project_root,
        "train_data_ref": train_data_ref,  # Ray object store reference
        "val_data_ref": val_data_ref,      # Ray object store reference
        "batch_size": args.batch_size,
        "epochs_per_trial": args.epochs_per_trial,
        "eval_interval": args.eval_interval,
        "checkpoint_interval": args.eval_interval,  # Save checkpoint at eval intervals
        "d_model": 512,
        "n_heads": 8,
        "H_layers": 2,
        "L_layers": 2,
        "d_ff": 2048,
        "T": 4,
        "hrm_grad_style": True,
    })
    
    # Override async_batch if specified via CLI (otherwise search both)
    if args.async_batch:
        search_space["async_batch"] = True
        print(f"Async batching: FORCED ON")
    else:
        print(f"Async batching: SEARCHING [True, False]")
    
    # Optuna search
    optuna_search = OptunaSearch(
        metric="best_grid_acc",
        mode="max",
        sampler=TPESampler(seed=args.seed),
    )
    
    # ASHA scheduler for early stopping
    scheduler = ASHAScheduler(
        time_attr="epoch",
        metric="best_grid_acc",
        mode="max",
        max_t=args.epochs_per_trial,
        grace_period=args.grace_period,
        reduction_factor=2,
    )
    
    # Callbacks
    callbacks = []
    if HAS_WANDB and args.wandb_project:
        callbacks.append(
            WandbLoggerCallback(
                project=args.wandb_project,
                group=args.study_name,
                log_config=True,
            )
        )
    
    # Run HPO
    tuner = tune.Tuner(
        tune.with_resources(
            train_trial,
            resources={"gpu": 1, "cpu": 4},
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            scheduler=scheduler,
            num_samples=args.n_trials,
            max_concurrent_trials=args.num_gpus,
        ),
        run_config=ray.air.RunConfig(
            name=args.study_name,
            storage_path=args.output_dir,
            callbacks=callbacks,
            stop={"epoch": args.epochs_per_trial},
            checkpoint_config=ray.air.CheckpointConfig(
                num_to_keep=2,  # Keep 2 most recent checkpoints per trial
            ),
        ),
    )
    
    results = tuner.fit()
    
    # Print results
    print(f"\n{'='*60}")
    print("HPO Results")
    print(f"{'='*60}")
    
    best_result = results.get_best_result(metric="best_grid_acc", mode="max")
    print(f"\nBest trial:")
    print(f"  Grid Accuracy: {best_result.metrics['best_grid_acc']:.2f}%")
    print(f"  Config:")
    for key, value in best_result.config.items():
        if key not in ["data_dir", "batch_size", "epochs_per_trial", "eval_interval",
                       "d_model", "n_heads", "H_layers", "L_layers", "d_ff", "T",
                       "hrm_grad_style", "halt_max_steps"]:
            print(f"    {key}: {value}")
    
    # Save best config
    import json
    best_config_path = os.path.join(args.output_dir, f"{args.study_name}_best_config.json")
    with open(best_config_path, "w") as f:
        json.dump({
            "best_grid_acc": best_result.metrics["best_grid_acc"],
            "config": {k: v for k, v in best_result.config.items() 
                      if not k.startswith("_") and k not in ["data_dir"]},
        }, f, indent=2)
    print(f"\nBest config saved to: {best_config_path}")
    
    # Upload best checkpoint to W&B
    if HAS_WANDB and args.wandb_project:
        try:
            # Find the best checkpoint
            best_checkpoint_dir = best_result.checkpoint
            if best_checkpoint_dir:
                checkpoint_path = best_checkpoint_dir.to_directory()
                
                # Initialize W&B run for artifact upload
                run = wandb.init(
                    project=args.wandb_project,
                    name=f"{args.study_name}_best_model",
                    job_type="model-upload",
                )
                
                # Create and log artifact
                artifact = wandb.Artifact(
                    name=f"sudoku-hpo-best-{args.study_name}",
                    type="model",
                    description=f"Best model from HPO study {args.study_name}",
                    metadata={
                        "best_grid_acc": best_result.metrics["best_grid_acc"],
                        "config": {k: v for k, v in best_result.config.items() 
                                  if not k.startswith("_") and k not in ["data_dir"]},
                    }
                )
                artifact.add_dir(checkpoint_path)
                run.log_artifact(artifact)
                wandb.finish()
                
                print(f"✓ Best model uploaded to W&B: {args.wandb_project}/{args.study_name}_best_model")
        except Exception as e:
            print(f"Warning: Could not upload checkpoint to W&B: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Sudoku HPO with Optuna + Ray + W&B')
    
    # HPO settings
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of HPO trials')
    parser.add_argument('--epochs-per-trial', type=int, default=500,
                       help='Epochs per trial')
    parser.add_argument('--grace-period', type=int, default=100,
                       help='Minimum epochs before pruning')
    parser.add_argument('--num-gpus', type=int, default=4,
                       help='Number of GPUs (parallel trials)')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='data/sudoku-extreme-10k-aug-100')
    parser.add_argument('--download', action='store_true',
                       help='Download dataset')
    parser.add_argument('--subsample', type=int, default=10000)
    parser.add_argument('--num-aug', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    
    # Eval
    parser.add_argument('--eval-interval', type=int, default=50,
                       help='Evaluate every N epochs')
    
    # Training mode
    parser.add_argument('--async-batch', action='store_true',
                       help='Use async batching (HRM-style)')
    
    # Study
    parser.add_argument('--study-name', type=str, 
                       default=f'sudoku_hpo_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: <workspace>/experiments/hpo_results)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume existing study')
    parser.add_argument('--seed', type=int, default=42)
    
    # W&B
    parser.add_argument('--wandb-project', type=str, default='sudoku-hpo',
                       help='W&B project name (empty to disable)')
    
    args = parser.parse_args()
    
    # Set default output dir to absolute path
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.join(script_dir, 'hpo_results')
    else:
        args.output_dir = os.path.abspath(args.output_dir)
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Run HPO
    run_hpo(args)


if __name__ == '__main__':
    main()

