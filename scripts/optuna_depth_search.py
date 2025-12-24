#!/usr/bin/env python3
"""
Optuna Depth Configuration Search for Swin Sudoku Solver

Uses TPE (Tree-structured Parzen Estimator) to find optimal depth configurations:
- h_cycles: Outer slow-thinking loops (most important!)
- l_cycles: Inner fast-thinking loops
- halt_max_steps: ACT pondering steps

Evaluates per-sample (batch_size=1) with torch.no_grad() to allow maximum depth.

Usage:
    python scripts/optuna_depth_search.py \
        --checkpoint "wandb:entity/project/artifact:version" \
        --n-trials 50 \
        --device cuda

Author: Eran Ben Artzy
"""

import argparse
import os
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("Please install optuna: pip install optuna")
    exit(1)

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pot.models.sudoku_solver import HybridPoHHRMSolver


# =============================================================================
# Data Loading (minimal, no augmentation for eval)
# =============================================================================

class SudokuDataset(Dataset):
    """Simple Sudoku dataset for evaluation."""
    
    def __init__(self, inputs, solutions):
        self.inputs = inputs
        self.solutions = solutions
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.long),
            torch.tensor(self.solutions[idx], dtype=torch.long),
            torch.tensor(0, dtype=torch.long),  # puzzle_id placeholder
        )


def load_sudoku_data(data_dir: str):
    """Load Sudoku validation data."""
    data_path = Path(data_dir)
    val_inputs = np.load(data_path / "val" / "all__inputs.npy")
    val_labels = np.load(data_path / "val" / "all__labels.npy")
    return val_inputs, val_labels


# =============================================================================
# Evaluation Function
# =============================================================================

@torch.no_grad()
def evaluate_config(
    model: nn.Module,
    val_inputs: np.ndarray,
    val_labels: np.ndarray,
    h_cycles: int,
    l_cycles: int,
    halt_max_steps: int,
    device: str,
    max_samples: int = None,
) -> dict:
    """
    Evaluate model with specific depth configuration.
    
    Modifies model's cycle parameters temporarily for evaluation.
    Uses batch_size=1 to allow maximum depth.
    """
    # Store original config
    orig_h_cycles = model.H_cycles
    orig_l_cycles = model.L_cycles
    orig_halt_max = model.halt_max_steps
    
    # Set new config
    model.H_cycles = h_cycles
    model.L_cycles = l_cycles
    model.halt_max_steps = halt_max_steps
    
    # Also update the L_level and H_level modules if they have these attributes
    if hasattr(model, 'L_level'):
        model.L_level.H_cycles = h_cycles if hasattr(model.L_level, 'H_cycles') else None
    if hasattr(model, 'H_level'):
        model.H_level.L_cycles = l_cycles if hasattr(model.H_level, 'L_cycles') else None
    
    model.eval()
    
    total_correct = 0
    total_cells = 0
    total_grids_correct = 0
    total_grids = 0
    
    n_samples = len(val_inputs) if max_samples is None else min(max_samples, len(val_inputs))
    
    start_time = time.time()
    
    for i in tqdm(range(n_samples), desc=f"H={h_cycles}, L={l_cycles}, halt={halt_max_steps}", leave=False):
        inp = torch.tensor(val_inputs[i:i+1], dtype=torch.long, device=device)
        target = torch.tensor(val_labels[i:i+1], dtype=torch.long, device=device)
        puzzle_id = torch.zeros(1, dtype=torch.long, device=device)
        
        # Forward pass
        outputs = model(inp, puzzle_id)
        logits = outputs[0]  # [1, 81, 10]
        
        preds = logits.argmax(dim=-1)
        
        total_correct += (preds == target).sum().item()
        total_cells += target.numel()
        total_grids_correct += (preds == target).all(dim=1).sum().item()
        total_grids += 1
    
    elapsed = time.time() - start_time
    
    # Restore original config
    model.H_cycles = orig_h_cycles
    model.L_cycles = orig_l_cycles
    model.halt_max_steps = orig_halt_max
    
    return {
        'cell_acc': total_correct / total_cells,
        'grid_acc': total_grids_correct / total_grids,
        'total_steps': h_cycles * l_cycles * halt_max_steps,
        'time_per_sample': elapsed / n_samples,
        'n_samples': n_samples,
    }


# =============================================================================
# Optuna Objective
# =============================================================================

def create_objective(
    model: nn.Module,
    val_inputs: np.ndarray,
    val_labels: np.ndarray,
    device: str,
    max_samples: int,
    max_depth: int,
):
    """Create Optuna objective function."""
    
    def objective(trial: optuna.Trial) -> float:
        # Sample from search space using ×2 scaling (powers of 2)
        # Constrained to keep total steps < 3000
        # h_cycles: 2, 4, 8, 16, 32 (most important!)
        h_cycles = trial.suggest_categorical("h_cycles", [2, 4, 8, 16, 32])
        # l_cycles: 6, 12 (capped at 12)
        l_cycles = trial.suggest_categorical("l_cycles", [6, 12])
        # halt_max_steps: 4, 8, 16, 32 (×2 from 4)
        halt_max_steps = trial.suggest_categorical("halt_max_steps", [4, 8, 16, 32])
        
        total_steps = h_cycles * l_cycles * halt_max_steps
        
        # Skip if exceeds max_depth
        if total_steps > max_depth:
            raise optuna.TrialPruned(f"Total steps {total_steps} > max_depth {max_depth}")
        
        # Evaluate
        try:
            results = evaluate_config(
                model, val_inputs, val_labels,
                h_cycles, l_cycles, halt_max_steps,
                device, max_samples,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                raise optuna.TrialPruned(f"OOM with {total_steps} steps")
            raise
        
        # Log metrics
        trial.set_user_attr("cell_acc", results['cell_acc'])
        trial.set_user_attr("total_steps", results['total_steps'])
        trial.set_user_attr("time_per_sample", results['time_per_sample'])
        
        return results['grid_acc']
    
    return objective


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Optuna depth search for Swin Sudoku")
    
    # Data & Model
    parser.add_argument("--data-dir", type=str, default="data/sudoku-extreme-10k-aug-100")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path or wandb:entity/project/artifact:version")
    
    # Model architecture (must match checkpoint)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--h-layers", type=int, default=2)
    parser.add_argument("--l-layers", type=int, default=2)
    parser.add_argument("--d-ctrl", type=int, default=256)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--n-stages", type=int, default=2)
    parser.add_argument("--T", type=int, default=4)
    
    # Search settings
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--max-samples", type=int, default=100, help="Max val samples per trial (for speed)")
    parser.add_argument("--max-depth", type=int, default=3000, help="Max total steps to try")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda")
    
    # Output
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="optuna_results")
    
    # W&B
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--project", type=str, default="sudoku-depth-search", help="W&B project name")
    
    args = parser.parse_args()
    
    # Device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Load data
    print(f"\nLoading data from {args.data_dir}")
    val_inputs, val_labels = load_sudoku_data(args.data_dir)
    print(f"  Val samples: {len(val_inputs)}")
    
    # Create model with placeholder config (will be overridden during trials)
    print("\nCreating model...")
    swin_kwargs = {
        "d_ctrl": args.d_ctrl,
        "window_size": args.window_size,
        "n_stages": args.n_stages,
        "max_depth": args.max_depth,  # Set high for exploration
        "token_conditioned": True,
        "depth_skip": True,
    }
    
    model = HybridPoHHRMSolver(
        d_model=args.d_model,
        n_heads=args.n_heads,
        H_layers=args.h_layers,
        L_layers=args.l_layers,
        d_ff=args.d_ff,
        dropout=0.0,
        H_cycles=2,  # Placeholder
        L_cycles=6,  # Placeholder
        T=args.T,
        halt_max_steps=5,  # Placeholder
        controller_type="swin",
        controller_kwargs=swin_kwargs,
    ).to(device)
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize W&B if enabled
    wandb_callback = None
    if args.wandb:
        import wandb
        from optuna.integration.wandb import WeightsAndBiasesCallback
        
        study_name = args.study_name or f"depth_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.project,
            name=study_name,
            config={
                "n_trials": args.n_trials,
                "max_samples": args.max_samples,
                "max_depth": args.max_depth,
                "checkpoint": args.checkpoint,
            }
        )
        wandb_callback = WeightsAndBiasesCallback(
            metric_name="grid_acc",
            wandb_kwargs={"project": args.project},
            as_multirun=True,
        )
    
    # Load checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path.startswith("wandb:"):
        import wandb
        artifact_ref = checkpoint_path[6:]
        print(f"\nDownloading checkpoint from W&B: {artifact_ref}")
        if not args.wandb:
            wandb.init(mode="disabled")
        artifact = wandb.use_artifact(artifact_ref, type="model")
        artifact_dir = artifact.download()
        pt_files = list(Path(artifact_dir).glob("*.pt"))
        if pt_files:
            checkpoint_path = str(pt_files[0])
        else:
            raise FileNotFoundError("No .pt file in artifact")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle size mismatches (max_depth might differ)
    ckpt_state = checkpoint['model_state_dict']
    model_state = model.state_dict()
    
    for key in list(ckpt_state.keys()):
        if key in model_state:
            if ckpt_state[key].shape != model_state[key].shape:
                if 'depth_pos' in key:
                    ckpt_depth = ckpt_state[key].shape[0]
                    model_depth = model_state[key].shape[0]
                    if model_depth > ckpt_depth:
                        new_param = torch.zeros_like(model_state[key])
                        new_param[:ckpt_depth] = ckpt_state[key]
                        nn.init.normal_(new_param[ckpt_depth:], std=0.02)
                        ckpt_state[key] = new_param
                        print(f"  Resized {key}: {ckpt_depth} -> {model_depth}")
    
    model.load_state_dict(ckpt_state)
    print(f"✓ Loaded checkpoint")
    
    # Create Optuna study
    study_name = args.study_name or f"depth_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\n{'='*60}")
    print(f"Starting Optuna Depth Search: {study_name}")
    print(f"  Trials: {args.n_trials}")
    print(f"  Max samples per trial: {args.max_samples}")
    print(f"  Max depth: {args.max_depth}")
    print(f"{'='*60}\n")
    
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=TPESampler(seed=42),
    )
    
    objective = create_objective(
        model, val_inputs, val_labels,
        device, args.max_samples, args.max_depth,
    )
    
    callbacks = [wandb_callback] if wandb_callback else []
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True, callbacks=callbacks)
    
    # Results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    
    print(f"\nBest trial:")
    print(f"  h_cycles: {study.best_params['h_cycles']}")
    print(f"  l_cycles: {study.best_params['l_cycles']}")
    print(f"  halt_max_steps: {study.best_params['halt_max_steps']}")
    total = study.best_params['h_cycles'] * study.best_params['l_cycles'] * study.best_params['halt_max_steps']
    print(f"  Total steps: {total}")
    print(f"  Grid accuracy: {100*study.best_value:.2f}%")
    
    # Top 10 trials
    print(f"\nTop 10 configurations:")
    print("-" * 80)
    print(f"{'Rank':<6} {'H':<4} {'L':<4} {'halt':<6} {'Steps':<8} {'Grid%':<8} {'Cell%':<8} {'Time/sample':<12}")
    print("-" * 80)
    
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)
    for i, trial in enumerate(trials_sorted[:10]):
        if trial.value is None:
            continue
        h = trial.params['h_cycles']
        l = trial.params['l_cycles']
        halt = trial.params['halt_max_steps']
        steps = h * l * halt
        grid = trial.value * 100
        cell = trial.user_attrs.get('cell_acc', 0) * 100
        time_ps = trial.user_attrs.get('time_per_sample', 0)
        print(f"{i+1:<6} {h:<4} {l:<4} {halt:<6} {steps:<8} {grid:<8.2f} {cell:<8.2f} {time_ps:<12.3f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"{study_name}_results.txt"
    with open(results_file, "w") as f:
        f.write(f"Best config:\n")
        f.write(f"  h_cycles: {study.best_params['h_cycles']}\n")
        f.write(f"  l_cycles: {study.best_params['l_cycles']}\n")
        f.write(f"  halt_max_steps: {study.best_params['halt_max_steps']}\n")
        f.write(f"  Total steps: {total}\n")
        f.write(f"  Grid accuracy: {100*study.best_value:.2f}%\n\n")
        
        f.write("All trials:\n")
        for trial in trials_sorted:
            if trial.value is None:
                continue
            h = trial.params['h_cycles']
            l = trial.params['l_cycles']
            halt = trial.params['halt_max_steps']
            steps = h * l * halt
            f.write(f"  H={h}, L={l}, halt={halt}, steps={steps}, grid={trial.value*100:.2f}%\n")
    
    print(f"\nResults saved to {results_file}")
    
    # Log final results to W&B
    if args.wandb:
        import wandb
        wandb.run.summary["best_h_cycles"] = study.best_params['h_cycles']
        wandb.run.summary["best_l_cycles"] = study.best_params['l_cycles']
        wandb.run.summary["best_halt_max_steps"] = study.best_params['halt_max_steps']
        wandb.run.summary["best_total_steps"] = total
        wandb.run.summary["best_grid_acc"] = study.best_value
        
        # Log results table
        table = wandb.Table(columns=["rank", "h_cycles", "l_cycles", "halt_max", "total_steps", "grid_acc", "cell_acc"])
        for i, trial in enumerate(trials_sorted[:20]):
            if trial.value is None:
                continue
            h = trial.params['h_cycles']
            l = trial.params['l_cycles']
            halt = trial.params['halt_max_steps']
            steps = h * l * halt
            cell = trial.user_attrs.get('cell_acc', 0)
            table.add_data(i+1, h, l, halt, steps, trial.value, cell)
        wandb.log({"top_configs": table})
        
        wandb.finish()
        print("✓ Results logged to W&B")


if __name__ == "__main__":
    main()

