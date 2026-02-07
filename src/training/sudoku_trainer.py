"""
Sudoku Training Utilities.

Training and evaluation functions for Sudoku solvers.
Includes debugging utilities for analyzing model behavior.

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Dict, Any, Optional
from tqdm import tqdm



def _is_main_process() -> bool:
    """Check if this is the main process (rank 0 or non-distributed)."""
    rank = int(os.environ.get('RANK', 0))
    return rank == 0


def _is_distributed() -> bool:
    """Check if running in distributed mode."""
    return dist.is_available() and dist.is_initialized()


def _reduce_metrics(metrics_dict: Dict[str, float], device: torch.device) -> Dict[str, float]:
    """
    All-reduce metric values across ranks so every rank has the global sum.
    
    Expects raw counts/sums (not averages). Returns reduced sums.
    """
    if not _is_distributed():
        return metrics_dict
    
    keys = sorted(metrics_dict.keys())
    values = torch.tensor([metrics_dict[k] for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {k: v.item() for k, v in zip(keys, values)}


class InfiniteDataLoader:
    """
    Wrapper around DataLoader that cycles infinitely.
    
    Used for async batching where we need to replace halted samples
    with new ones from the dataset continuously.
    """
    
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.total_samples = 0
    
    def get_next(self, num_needed: int):
        """
        Get the next batch, cycling if needed.
        
        Args:
            num_needed: Number of samples needed (for tracking only)
            
        Returns:
            Next batch from dataloader
        """
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        
        self.total_samples += batch['input'].size(0)
        return batch
    
    def reset(self):
        """Reset the iterator and sample counter."""
        self.iterator = iter(self.dataloader)
        self.total_samples = 0


def train_epoch_async(
    model: nn.Module,
    dataloader,
    optimizer,
    puzzle_optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    epoch: int,
    use_poh: bool = True,
    debug: bool = False,
    scheduler: Optional[Any] = None,
    puzzle_scheduler: Optional[Any] = None,
    samples_per_epoch: Optional[int] = None,
    track_halt_histogram: bool = False,
    grad_clip: float = 1.0,
) -> Dict[str, Any]:
    """
    Train for one epoch using HRM-style async batching.
    
    Each forward call runs one ACT step. Samples that halt are immediately
    replaced with new puzzles, maximizing GPU utilization.
    
    Args:
        model: The Sudoku solver model (must be HybridPoHHRMSolver with async support)
        dataloader: Training data loader
        optimizer: Main optimizer
        puzzle_optimizer: Optimizer for puzzle embeddings (optional)
        device: Device to train on
        epoch: Current epoch number
        use_poh: Whether model uses PoH/puzzle embeddings
        debug: Enable debug logging
        scheduler: Learning rate scheduler
        puzzle_scheduler: LR scheduler for puzzle optimizer
        samples_per_epoch: Total samples to process per epoch (default: len(dataloader) * batch_size)
        track_halt_histogram: If True, track which halt step each sample finished at
        grad_clip: Maximum gradient norm (0 to disable)
        
    Returns:
        Dictionary with loss, cell_acc, grid_acc, avg_steps, and optionally halt_histogram
    """
    model.train()
    
    # Get underlying model (handles DDP wrapping)
    base_model = model.module if hasattr(model, 'module') else model
    
    # Determine batch size from first batch
    first_batch = next(iter(dataloader))
    batch_size = first_batch['input'].size(0)
    
    # Default samples per epoch = one pass through data
    if samples_per_epoch is None:
        samples_per_epoch = len(dataloader) * batch_size
    
    # Initialize infinite data loader
    inf_loader = InfiniteDataLoader(dataloader)
    
    # Initialize carry state (all halted, so first batch loads data)
    carry = base_model.create_async_carry(batch_size, device)
    
    # Metrics tracking
    total_loss = 0
    correct_cells = 0
    total_cells = 0
    correct_grids = 0
    total_grids = 0
    total_steps = 0
    completed_samples = 0
    forward_calls = 0
    
    # Halt histogram tracking: {step: {'halted': count, 'solved': count}}
    halt_histogram = {} if track_halt_histogram else None
    
    pbar = tqdm(total=samples_per_epoch, desc=f"Epoch {epoch} (async)", 
                 disable=not _is_main_process())
    
    while completed_samples < samples_per_epoch:
        # Get new batch for replacing halted samples
        new_batch = inf_loader.get_next(carry.halted.sum().item())
        
        optimizer.zero_grad()
        if puzzle_optimizer:
            puzzle_optimizer.zero_grad()
        
        # Record which samples were halted before this step (these just completed)
        was_halted = carry.halted.clone()
        
        # Forward: one ACT step (use base_model for async methods)
        new_carry, outputs = base_model.async_forward(carry, new_batch)
        
        logits = outputs['logits']
        labels = outputs['labels']
        q_halt = outputs['q_halt_logits']
        q_continue = outputs['q_continue_logits']
        target_q_continue = outputs.get('target_q_continue')
        
        # Compute loss for ALL samples (gradient flows for all)
        # Note: HRM also computes loss for all, the async nature is just about data replacement
        lm_loss = F.cross_entropy(
            logits.view(-1, base_model.vocab_size),
            labels.view(-1)
        )
        
        # Q-halt loss
        if use_poh and q_halt is not None:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                is_correct = (preds == labels).all(dim=1).float()
            
            q_halt_loss = F.binary_cross_entropy_with_logits(q_halt, is_correct)
            loss = lm_loss + 0.5 * q_halt_loss
            
            # ACT Q-learning loss (if target available)
            if target_q_continue is not None:
                q_continue_loss = F.mse_loss(torch.sigmoid(q_continue), target_q_continue)
                loss = loss + 0.5 * q_continue_loss
        else:
            loss = lm_loss
        
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if puzzle_optimizer:
            puzzle_optimizer.step()
        
        # Detach carry to prevent gradient accumulation across steps
        new_carry = new_carry.detach()
        
        # Step schedulers
        if scheduler:
            scheduler.step()
        if puzzle_scheduler:
            puzzle_scheduler.step()
        
        # Count newly halted samples (these are completing now)
        newly_halted = new_carry.halted & ~was_halted
        # Also count samples that were already halted (just got replaced and will complete eventually)
        # For simplicity, count halted samples as completed
        num_completed = new_carry.halted.sum().item()
        
        # Accumulate metrics for ALL samples (they all contribute to loss)
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct_cells += (preds == labels).sum().item()
        total_cells += labels.numel()
        grid_correct = (preds == labels).all(dim=1)  # [B] bool
        correct_grids += grid_correct.sum().item()
        total_grids += labels.size(0)
        total_steps += 1  # Each forward is one ACT step
        forward_calls += 1
        
        # Track halt histogram for newly halted samples
        if track_halt_histogram and newly_halted.any():
            # Get step count for each sample from carry
            steps_tensor = new_carry.steps  # [B] int32
            
            for i in range(newly_halted.size(0)):
                if newly_halted[i]:
                    step = steps_tensor[i].item()
                    solved = grid_correct[i].item()
                    
                    if step not in halt_histogram:
                        halt_histogram[step] = {'halted': 0, 'solved': 0}
                    halt_histogram[step]['halted'] += 1
                    if solved:
                        halt_histogram[step]['solved'] += 1
        
        # Track completed samples (halted samples that will be replaced next step)
        completed_samples += num_completed
        pbar.update(num_completed)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cell_acc': f'{100*correct_cells/total_cells:.1f}%',
            'grid_acc': f'{100*correct_grids/total_grids:.1f}%',
        })
        
        # Update carry for next step
        carry = new_carry
    
    pbar.close()
    
    # Run debug at end of epoch if enabled
    if debug:
        run_debug(model, dataloader, optimizer, device, epoch)
    
    # All-reduce metrics across ranks for DDP
    reduced = _reduce_metrics({
        'total_loss': total_loss,
        'correct_cells': float(correct_cells),
        'total_cells': float(total_cells),
        'correct_grids': float(correct_grids),
        'total_grids': float(total_grids),
        'total_steps': float(total_steps),
        'forward_calls': float(forward_calls),
    }, device)
    
    result = {
        'loss': reduced['total_loss'] / max(1, reduced['forward_calls']),
        'cell_acc': 100 * reduced['correct_cells'] / max(1, reduced['total_cells']),
        'grid_acc': 100 * reduced['correct_grids'] / max(1, reduced['total_grids']),
        'avg_steps': reduced['total_steps'] / max(1, reduced['forward_calls']),
        'forward_calls': int(reduced['forward_calls']),
    }
    
    if track_halt_histogram and halt_histogram:
        result['halt_histogram'] = halt_histogram
        
        # Print halt histogram summary
        if _is_main_process():
            print("\n📊 Train Halt Step Histogram:")
            print(f"{'Step':<6} {'Halted':<10} {'Solved':<10} {'Solve Rate':<12}")
            print("-" * 40)
            for step in sorted(halt_histogram.keys()):
                data = halt_histogram[step]
                rate = 100 * data['solved'] / data['halted'] if data['halted'] > 0 else 0
                print(f"{step:<6} {data['halted']:<10} {data['solved']:<10} {rate:<10.2f}%")
            print()
    
    return result


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    puzzle_optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    epoch: int,
    use_poh: bool = True,
    debug: bool = False,
    scheduler: Optional[Any] = None,
    puzzle_scheduler: Optional[Any] = None,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: The Sudoku solver model
        dataloader: Training data loader
        optimizer: Main optimizer
        puzzle_optimizer: Optimizer for puzzle embeddings (optional)
        device: Device to train on
        epoch: Current epoch number
        use_poh: Whether model uses PoH/puzzle embeddings
        debug: Enable debug logging
        scheduler: Learning rate scheduler
        puzzle_scheduler: LR scheduler for puzzle optimizer
        grad_clip: Maximum gradient norm (0 to disable)
        
    Returns:
        Dictionary with loss, cell_acc, grid_acc, avg_steps
    """
    model.train()
    
    # Get underlying model (handles DDP wrapping)
    base_model = model.module if hasattr(model, 'module') else model
    
    total_loss = 0
    correct_cells = 0
    total_cells = 0
    correct_grids = 0
    total_grids = 0
    total_steps = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not _is_main_process())
    for batch in pbar:
        inp = batch['input'].to(device)
        label = batch['label'].to(device)
        puzzle_ids = batch['puzzle_id'].to(device)
        
        optimizer.zero_grad()
        if puzzle_optimizer:
            puzzle_optimizer.zero_grad()
        
        # Handle variable output length (ACT returns 5 values, non-ACT returns 4)
        model_out = model(inp, puzzle_ids)
        if len(model_out) == 5:
            logits, q_halt, q_continue, steps, target_q_continue = model_out
        else:
            logits, q_halt, q_continue, steps = model_out
            target_q_continue = None
        
        # CE loss on ALL cells (HRM-style)
        lm_loss = F.cross_entropy(
            logits.view(-1, base_model.vocab_size),
            label.view(-1)
        )
        
        # Q-halt loss (if PoH)
        if use_poh and q_halt is not None:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                is_correct = (preds == label).all(dim=1).float()
            
            q_halt_loss = F.binary_cross_entropy_with_logits(q_halt, is_correct)
            loss = lm_loss + 0.5 * q_halt_loss
            
            # ACT Q-learning loss (if target available)
            if target_q_continue is not None:
                q_continue_loss = F.mse_loss(torch.sigmoid(q_continue), target_q_continue)
                loss = loss + 0.5 * q_continue_loss
        else:
            loss = lm_loss
        
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if puzzle_optimizer:
            puzzle_optimizer.step()
        
        # Step schedulers
        if scheduler:
            scheduler.step()
        if puzzle_scheduler:
            puzzle_scheduler.step()
        
        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct_cells += (preds == label).sum().item()
        total_cells += label.numel()
        correct_grids += (preds == label).all(dim=1).sum().item()
        total_grids += label.size(0)
        total_steps += steps
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cell_acc': f'{100*correct_cells/total_cells:.1f}%',
            'grid_acc': f'{100*correct_grids/total_grids:.1f}%',
        })
    
    # Run debug at end of epoch if enabled
    if debug:
        run_debug(model, dataloader, optimizer, device, epoch)
    
    # All-reduce metrics across ranks for DDP
    reduced = _reduce_metrics({
        'total_loss': total_loss,
        'correct_cells': float(correct_cells),
        'total_cells': float(total_cells),
        'correct_grids': float(correct_grids),
        'total_grids': float(total_grids),
        'total_steps': float(total_steps),
        'num_batches': float(len(dataloader)),
    }, device)
    
    return {
        'loss': reduced['total_loss'] / max(1, reduced['num_batches']),
        'cell_acc': 100 * reduced['correct_cells'] / max(1, reduced['total_cells']),
        'grid_acc': 100 * reduced['correct_grids'] / max(1, reduced['total_grids']),
        'avg_steps': reduced['total_steps'] / max(1, reduced['num_batches']),
    }


def evaluate(
    model: nn.Module,
    dataloader,
    device: torch.device,
    use_poh: bool = True,
    track_halt_histogram: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate model.
    
    Args:
        model: The Sudoku solver model
        dataloader: Test data loader
        device: Device to evaluate on
        use_poh: Whether model uses PoH/puzzle embeddings
        track_halt_histogram: If True, track which halt step each puzzle finished at
                              and whether it was solved correctly
        
    Returns:
        Dictionary with loss, cell_acc, grid_acc, and optionally halt_histogram
    """
    model.eval()
    
    # Get underlying model (handles DDP wrapping)
    base_model = model.module if hasattr(model, 'module') else model
    
    total_loss = 0
    correct_cells = 0
    total_cells = 0
    correct_grids = 0
    total_grids = 0
    
    # Halt histogram tracking: {step: {'halted': count, 'solved': count}}
    halt_histogram = {} if track_halt_histogram else None
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=not _is_main_process()):
            inp = batch['input'].to(device)
            label = batch['label'].to(device)
            puzzle_ids = batch['puzzle_id'].to(device)
            
            # Handle variable output length (ACT returns 5 values, non-ACT returns 4)
            model_out = model(inp, puzzle_ids)
            logits = model_out[0]
            
            # Extract steps if available (index 3 in model output)
            steps_tensor = None
            if track_halt_histogram and len(model_out) > 3:
                steps_output = model_out[3]
                # steps can be a tensor [B] or a single int
                if isinstance(steps_output, torch.Tensor):
                    steps_tensor = steps_output
                elif isinstance(steps_output, int):
                    # If single int, all samples halted at same step
                    steps_tensor = torch.full((inp.size(0),), steps_output, dtype=torch.int32, device=device)
            
            loss = F.cross_entropy(logits.view(-1, base_model.vocab_size), label.view(-1))
            total_loss += loss.item()
            
            preds = logits.argmax(dim=-1)
            correct_cells += (preds == label).sum().item()
            total_cells += label.numel()
            
            # Per-sample solve status
            grid_correct = (preds == label).all(dim=1)  # [B] bool
            correct_grids += grid_correct.sum().item()
            total_grids += label.size(0)
            
            # Update halt histogram
            if track_halt_histogram and steps_tensor is not None:
                for i in range(inp.size(0)):
                    step = steps_tensor[i].item()
                    solved = grid_correct[i].item()
                    
                    if step not in halt_histogram:
                        halt_histogram[step] = {'halted': 0, 'solved': 0}
                    halt_histogram[step]['halted'] += 1
                    if solved:
                        halt_histogram[step]['solved'] += 1
    
    # All-reduce metrics across ranks for DDP
    reduced = _reduce_metrics({
        'total_loss': total_loss,
        'correct_cells': float(correct_cells),
        'total_cells': float(total_cells),
        'correct_grids': float(correct_grids),
        'total_grids': float(total_grids),
        'num_batches': float(len(dataloader)),
    }, device)
    
    result = {
        'loss': reduced['total_loss'] / max(1, reduced['num_batches']),
        'cell_acc': 100 * reduced['correct_cells'] / max(1, reduced['total_cells']),
        'grid_acc': 100 * reduced['correct_grids'] / max(1, reduced['total_grids']),
    }
    
    if track_halt_histogram and halt_histogram:
        result['halt_histogram'] = halt_histogram
        
        # Print halt histogram summary
        if _is_main_process():
            print("\n📊 Halt Step Histogram:")
            print(f"{'Step':<6} {'Halted':<10} {'Solved':<10} {'Solve Rate':<12}")
            print("-" * 40)
            for step in sorted(halt_histogram.keys()):
                data = halt_histogram[step]
                rate = 100 * data['solved'] / data['halted'] if data['halted'] > 0 else 0
                print(f"{step:<6} {data['halted']:<10} {data['solved']:<10} {rate:<10.2f}%")
            print()
    
    return result


def log_halt_histogram_to_wandb(halt_histogram: Dict[int, Dict[str, int]], prefix: str = "") -> Dict[str, Any]:
    """
    Prepare halt histogram data for W&B logging.
    
    Args:
        halt_histogram: Dict from evaluate() with {step: {'halted': N, 'solved': M}}
        prefix: Optional prefix for metric names (e.g., "val_" or "test_")
        
    Returns:
        Dict ready for wandb.log() containing:
        - {prefix}halt_steps: list of all halt steps for histogram
        - {prefix}halt_step_{N}_count: count of puzzles halted at step N
        - {prefix}halt_step_{N}_solved: count solved at step N  
        - {prefix}halt_step_{N}_rate: solve rate at step N
    """
    if not halt_histogram:
        return {}
    
    log_dict = {}
    all_steps = []
    
    for step in sorted(halt_histogram.keys()):
        data = halt_histogram[step]
        halted = data['halted']
        solved = data['solved']
        rate = 100 * solved / halted if halted > 0 else 0
        
        # Add to list for histogram
        all_steps.extend([step] * halted)
        
        # Individual metrics per step
        log_dict[f"{prefix}halt_step_{step}_count"] = halted
        log_dict[f"{prefix}halt_step_{step}_solved"] = solved
        log_dict[f"{prefix}halt_step_{step}_rate"] = rate
    
    # Store raw list for W&B histogram
    log_dict[f"{prefix}halt_steps_raw"] = all_steps
    
    # Summary stats
    total_halted = sum(d['halted'] for d in halt_histogram.values())
    total_solved = sum(d['solved'] for d in halt_histogram.values())
    log_dict[f"{prefix}halt_total_puzzles"] = total_halted
    log_dict[f"{prefix}halt_total_solved"] = total_solved
    
    return log_dict


# ============================================================================
# Debugging Functions
# ============================================================================

def debug_gradients(model: nn.Module) -> None:
    """Log gradient norms per layer."""
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    
    # Summarize by component
    components = {}
    for name, norm in grad_norms.items():
        component = name.split('.')[0]
        if component not in components:
            components[component] = []
        components[component].append(norm)
    
    print("\n📊 Gradient Norms:")
    for comp, norms in sorted(components.items()):
        avg = sum(norms) / len(norms)
        max_norm = max(norms)
        print(f"  {comp}: avg={avg:.2e}, max={max_norm:.2e}")
    
    # Check for zero/vanishing gradients
    zero_grads = [n for n, v in grad_norms.items() if v < 1e-10]
    if zero_grads:
        print(f"  ⚠️ Zero gradients in: {zero_grads[:5]}...")


def debug_activations(model: nn.Module, x: torch.Tensor, puzzle_ids: torch.Tensor) -> None:
    """Check activation statistics through the model."""
    model.eval()
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'max': output.abs().max().item(),
                    'nan': torch.isnan(output).any().item(),
                    'inf': torch.isinf(output).any().item(),
                }
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.LayerNorm, nn.MultiheadAttention)):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    with torch.no_grad():
        model(x, puzzle_ids)
    
    for h in hooks:
        h.remove()
    
    print("\n📈 Activation Statistics:")
    problems = []
    for name, stats in sorted(activations.items())[:10]:
        status = "✓" if not stats['nan'] and not stats['inf'] and stats['max'] < 100 else "⚠️"
        print(f"  {status} {name}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, max={stats['max']:.3f}")
        if stats['nan'] or stats['inf']:
            problems.append(f"{name}: NaN={stats['nan']}, Inf={stats['inf']}")
    
    if problems:
        print(f"  🚨 PROBLEMS: {problems}")
    
    model.train()


def debug_predictions(model: nn.Module, batch: Dict, device: torch.device, num_samples: int = 2) -> None:
    """Visualize sample predictions vs ground truth."""
    model.eval()
    inp = batch['input'][:num_samples].to(device)
    label = batch['label'][:num_samples].to(device)
    puzzle_ids = batch['puzzle_id'][:num_samples].to(device)
    
    with torch.no_grad():
        model_out = model(inp, puzzle_ids)
        logits = model_out[0]
        steps = model_out[3]
        preds = logits.argmax(dim=-1)
    
    print(f"\n🧩 Sample Predictions (steps={steps}):")
    for i in range(num_samples):
        inp_grid = inp[i].view(9, 9).cpu().numpy()
        pred_grid = preds[i].view(9, 9).cpu().numpy()
        label_grid = label[i].view(9, 9).cpu().numpy()
        
        # Count errors
        blank_mask = inp_grid == 0
        errors = (pred_grid != label_grid) & blank_mask
        total_blanks = blank_mask.sum()
        correct_blanks = total_blanks - errors.sum()
        
        # Check given cells
        given_mask = inp_grid > 0
        given_correct = (pred_grid == label_grid) & given_mask
        
        print(f"\n  Puzzle {i+1}: {correct_blanks}/{total_blanks} blanks correct, "
              f"{given_correct.sum()}/{given_mask.sum()} given correct")
        print("  Input:      Prediction:  Label:")
        for row in range(9):
            inp_row = ''.join(str(x) if x > 0 else '.' for x in inp_grid[row])
            pred_row = ''.join(str(x) for x in pred_grid[row])
            label_row = ''.join(str(x) for x in label_grid[row])
            err_row = ''.join('X' if pred_grid[row][c] != label_grid[row][c] and inp_grid[row][c] == 0 
                            else ' ' for c in range(9))
            print(f"  {inp_row}    {pred_row}    {label_row}    {err_row}")
    
    # Output distribution analysis
    print(f"\n📊 Output Distribution:")
    pred_counts = torch.bincount(preds.view(-1), minlength=10).cpu().numpy()
    print(f"  Digit counts: {dict(enumerate(pred_counts))}")
    print(f"  Most common: {pred_counts.argmax()} ({pred_counts.max()}/{pred_counts.sum()} = "
          f"{100*pred_counts.max()/pred_counts.sum():.1f}%)")
    
    model.train()


def run_debug(model: nn.Module, dataloader, optimizer, device: torch.device, epoch: int) -> None:
    """Run all debug checks."""
    print(f"\n{'='*60}")
    print(f"🔍 DEBUG REPORT - Epoch {epoch}")
    print(f"{'='*60}")
    
    batch = next(iter(dataloader))
    inp = batch['input'].to(device)
    puzzle_ids = batch['puzzle_id'].to(device)
    
    debug_gradients(model)
    debug_activations(model, inp, puzzle_ids)
    debug_predictions(model, batch, device)
    
    print(f"{'='*60}\n")

