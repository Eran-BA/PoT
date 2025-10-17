#!/usr/bin/env python3
"""
Huge (~500M params) baseline vs PoH-HRM benchmark on maze solving.

Defaults target ~520M parameters for the baseline using:
- d_model=1344, n_heads=32, d_ff=5376, depth=24

PoH-HRM will be auto-adjusted (by depth) to have ≤ baseline params (≤110% tolerance),
favoring fewer parameters when feasible.

Device selection: CUDA → MPS (Apple Silicon) → CPU.

Tip: Start with fewer epochs (e.g., 5–10) to validate runtime/memory.
"""

import os
import sys
from pathlib import Path
import argparse
import json
import math
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ------------------ Path setup ------------------
try:
    from experiments.setup_colab import setup_pot_paths
    repo_root = setup_pot_paths()
except Exception:
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

print(f"✓ PoT root: {repo_root}")


# ------------------ Maze dataset ------------------
try:
    from maze_dataset import MazeDataset, MazeDatasetConfig
    from maze_dataset.generation import LatticeMazeGenerators
    print("✓ maze-dataset library available")
except ImportError:
    print("✗ maze-dataset not installed. Install with: pip install maze-dataset")
    sys.exit(1)


# ------------------ PoT modules ------------------
from src.pot.modules import PoHConfig, PoHStack, IterRefiner
from src.pot.core.hrm_controller import HRMPointerController, HRMState

print("✓ Successfully imported PoT modules")


# ------------------ Device selection ------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🚀 CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Apple Silicon GPU (MPS) detected")
    print("   Using Metal Performance Shaders for acceleration")
else:
    device = torch.device("cpu")
    print("⚠️  No GPU detected, using CPU")


# ------------------ Default HUGE config (~520M) ------------------
HUGE_CONFIG = {
    "d_model": 1344,
    "n_heads": 32,     # 1344 / 32 = 42
    "d_ff": 5376,      # 4 * d_model
    "depth": 24,
}


# ------------------ Maze generation ------------------
def generate_dataset_proper(maze_size: int, n_samples: int, min_path_length: Optional[int], seed: int):
    if min_path_length is None:
        min_path_length = int(maze_size * maze_size * 0.4)

    print(f"  Generating {n_samples} mazes of size {maze_size}×{maze_size}")
    print(f"  Minimum path length: {min_path_length}")

    cfg = MazeDatasetConfig(
        name=f"maze_{maze_size}x{maze_size}_minpath{min_path_length}",
        grid_n=maze_size,
        n_mazes=n_samples * 3,
        maze_ctor=LatticeMazeGenerators.gen_dfs,
        seed=seed,
    )

    dataset = MazeDataset.from_config(cfg, do_generate=True, load_local=False, save_local=False)
    dataset_filtered = dataset.filter_by.path_length(min_length=min_path_length)

    if len(dataset_filtered) < n_samples:
        print(f"  ⚠️  Warning: Only generated {len(dataset_filtered)} mazes meeting criteria (requested {n_samples})")
        n_samples = len(dataset_filtered)
    else:
        dataset_filtered = dataset_filtered[:n_samples]

    data = []
    path_lengths = []
    for solved_maze in dataset_filtered:
        maze_obj = solved_maze.maze

        # Build simple grid: 0 = passable (node), 1 = wall (non-node)
        grid = np.ones((maze_size, maze_size), dtype=np.float32)
        nodes = maze_obj.get_nodes()
        for node in nodes:
            if isinstance(node, np.ndarray):
                r, c = int(node[0]), int(node[1])
            else:
                r, c = node.row, node.col
            grid[r, c] = 0.0

        # Start/goal
        if hasattr(maze_obj, "start_pos"):
            start = (maze_obj.start_pos.row, maze_obj.start_pos.col)
            goal = (maze_obj.end_pos.row, maze_obj.end_pos.col)
        else:
            sol = solved_maze.solution
            if sol.ndim == 2:
                start = tuple(sol[0])
                goal = tuple(sol[-1])
            else:
                continue

        # Path
        sol = solved_maze.solution
        if sol.ndim == 2:
            path = [tuple(coord) for coord in sol]
        else:
            continue

        path_lengths.append(len(path))
        data.append({
            "maze": grid,
            "start": start,
            "goal": goal,
            "path": path,
            "length": len(path),
        })

    print(f"  ✓ Generated {len(data)} mazes")
    if path_lengths:
        print(
            f"  Path length: {np.mean(path_lengths):.1f} ± {np.std(path_lengths):.1f} "
            f"(min={min(path_lengths)}, max={max(path_lengths)})"
        )
    return data


class MazeDatasetWrapper(Dataset):
    def __init__(self, data, maze_size):
        self.data = data
        self.maze_size = maze_size
        self.max_path_len = max(len(item["path"]) for item in data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        maze = torch.FloatTensor(item["maze"])  # [H, W]
        start = torch.LongTensor(item["start"])  # [2]
        goal = torch.LongTensor(item["goal"])    # [2]

        path_indices = [r * self.maze_size + c for r, c in item["path"]]
        path_len = len(path_indices)
        path_padded = path_indices + [-1] * (self.max_path_len - path_len)
        path_t = torch.LongTensor(path_padded)
        return maze, start, goal, path_t, path_len


# ------------------ Models ------------------
class BaselineMazeSolver(nn.Module):
    def __init__(self, maze_size: int, d_model: int, n_heads: int, d_ff: int, depth: int, dropout: float = 0.1):
        super().__init__()
        self.maze_size = maze_size
        self.d_model = d_model

        self.cell_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Embedding(maze_size * maze_size, d_model)
        self.start_embed = nn.Embedding(2, d_model)
        self.goal_embed = nn.Embedding(2, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.decoder = nn.Linear(d_model, maze_size * maze_size)

    def forward(self, maze, start, goal):
        B = maze.size(0)
        N = self.maze_size * self.maze_size
        x = self.cell_embed(maze.view(B, N, 1))
        pos = torch.arange(N, device=maze.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embed(pos)

        s_idx = start[:, 0] * self.maze_size + start[:, 1]
        g_idx = goal[:, 0] * self.maze_size + goal[:, 1]
        s_mark = torch.zeros(B, N, device=maze.device, dtype=torch.long)
        s_mark.scatter_(1, s_idx.unsqueeze(1), 1)
        x = x + self.start_embed(s_mark)
        g_mark = torch.zeros(B, N, device=maze.device, dtype=torch.long)
        g_mark.scatter_(1, g_idx.unsqueeze(1), 1)
        x = x + self.goal_embed(g_mark)

        h = self.transformer(x)
        return self.decoder(h)


class StatefulHRMRouter(nn.Module):
    """Make HRMPointerController compatible with PoHBlock router interface."""
    def __init__(self, hrm_controller: HRMPointerController, n_heads: int):
        super().__init__()
        self.hrm = hrm_controller
        self.n_heads = n_heads
        self.state = None

    def forward(self, x_ctrl: torch.Tensor) -> torch.Tensor:
        B, T, _ = x_ctrl.shape
        if self.state is None:
            dev = x_ctrl.device
            self.state = HRMState(
                z_L=torch.zeros(B, self.hrm.d_ctrl, device=dev),
                z_H=torch.zeros(B, self.hrm.d_ctrl, device=dev),
                step=torch.zeros(B, dtype=torch.long, device=dev),
            )
        x_ctrl_mean = x_ctrl.mean(dim=1)
        alphas, state_new, _ = self.hrm(x_ctrl_mean, self.state)
        self.state = HRMState(z_L=state_new.z_L.detach(), z_H=state_new.z_H.detach(), step=state_new.step.detach())
        route_logits = torch.log(alphas.unsqueeze(1).expand(B, T, self.n_heads) + 1e-8)
        return route_logits


class PoHMazeSolver(nn.Module):
    def __init__(self, maze_size: int, d_model: int, n_heads: int, d_ff: int, depth: int, R: int, T: int, dropout: float = 0.1):
        super().__init__()
        self.maze_size = maze_size
        self.d_model = d_model

        self.cell_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Embedding(maze_size * maze_size, d_model)
        self.start_embed = nn.Embedding(2, d_model)
        self.goal_embed = nn.Embedding(2, d_model)

        cfg = PoHConfig(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
        stack = PoHStack(cfg, depth=depth)

        # Replace routers with HRM controller
        for blk in stack.blocks:
            if hasattr(blk, "router"):
                hrm = HRMPointerController(d_model=d_model, n_heads=n_heads, d_ctrl=d_model // 2, T=T)
                blk.router = StatefulHRMRouter(hrm, n_heads)

        self.refiner = IterRefiner(stack, max_inner_iters=R, outer_residual=True, rezero_init=True)
        self.decoder = nn.Linear(d_model, maze_size * maze_size)

    def forward(self, maze, start, goal):
        B = maze.size(0)
        N = self.maze_size * self.maze_size
        x = self.cell_embed(maze.view(B, N, 1))
        pos = torch.arange(N, device=maze.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embed(pos)

        s_idx = start[:, 0] * self.maze_size + start[:, 1]
        g_idx = goal[:, 0] * self.maze_size + goal[:, 1]
        s_mark = torch.zeros(B, N, device=maze.device, dtype=torch.long)
        s_mark.scatter_(1, s_idx.unsqueeze(1), 1)
        x = x + self.start_embed(s_mark)
        g_mark = torch.zeros(B, N, device=maze.device, dtype=torch.long)
        g_mark.scatter_(1, g_idx.unsqueeze(1), 1)
        x = x + self.goal_embed(g_mark)

        h, _ = self.refiner(x)
        return self.decoder(h)


# ------------------ Training/Eval ------------------
def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one(
    model,
    loader,
    device,
    epochs: int = 5,
    lr: float = 1e-4,
    label_smoothing: float = 0.1,
    warmup_steps: int = 1000,
    total_steps: int = None,
) -> None:
    """Train with label smoothing and cosine LR with linear warmup (per-step)."""
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=label_smoothing)

    steps_per_epoch = len(loader)
    if total_steps is None:
        total_steps = max(1, epochs * steps_per_epoch)

    def compute_lr(step_idx: int) -> float:
        # Linear warmup to lr, then cosine decay to 10% of lr
        if step_idx < warmup_steps:
            return lr * float(step_idx) / max(1, warmup_steps)
        rem = max(1, total_steps - warmup_steps)
        prog = float(step_idx - warmup_steps) / rem
        min_lr = 0.1 * lr
        # Cosine from lr -> min_lr
        cos_decay = 0.5 * (1.0 + math.cos(math.pi * prog))
        return min_lr + (lr - min_lr) * cos_decay

    global_step = 0
    for ep in range(epochs):
        tot, num = 0.0, 0
        for maze, start, goal, path, path_len in loader:
            # Update LR per step
            cur_lr = compute_lr(global_step)
            for g in optim.param_groups:
                g["lr"] = cur_lr

            maze = maze.to(device)
            start = start.to(device)
            goal = goal.to(device)
            path = path.to(device)
            optim.zero_grad()
            logits = model(maze, start, goal)  # [B, N, V]
            V = logits.size(-1)
            max_len = path.size(1)
            loss, steps = 0.0, 0
            for i in range(max_len - 1):
                mask = (path[:, i] != -1) & (path[:, i + 1] != -1)
                if mask.any():
                    curr = path[mask, i]
                    nxt = path[mask, i + 1]
                    curr_logits = logits[mask].gather(1, curr.unsqueeze(1).unsqueeze(2).expand(-1, 1, V)).squeeze(1)
                    loss = loss + crit(curr_logits, nxt)
                    steps += 1
            if steps > 0:
                loss = loss / steps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                tot += float(loss.item())
                num += 1
            global_step += 1
        print(f"  Epoch {ep+1}/{epochs}, Loss: {tot / max(1, num):.4f}, LR: {cur_lr:.2e}")


def evaluate(model, loader, device, maze_size: int) -> Tuple[float, float]:
    model.eval()
    correct, optimal, total = 0, 0, 0
    with torch.no_grad():
        for maze, start, goal, path, path_len in loader:
            maze = maze.to(device)
            start = start.to(device)
            goal = goal.to(device)
            B = maze.size(0)
            for b in range(B):
                cur = start[b].cpu().numpy()
                tgt = goal[b].cpu().numpy()
                grid = maze[b].cpu().numpy()
                visited = {tuple(cur)}
                pred_path = [tuple(cur)]
                for _ in range(maze_size * maze_size):
                    if tuple(cur) == tuple(tgt):
                        break
                    # Faster, non-slow path: avoid creating tensor from list of ndarrays
                    cur_tensor = torch.tensor(cur, dtype=torch.long, device=device).unsqueeze(0)
                    logits = model(maze[b:b+1], cur_tensor, goal[b:b+1])
                    idx = cur[0] * maze_size + cur[1]
                    probs = torch.softmax(logits[0, idx], dim=-1).cpu().numpy()
                    candidates = []
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cur[0] + dr, cur[1] + dc
                        if 0 <= nr < maze_size and 0 <= nc < maze_size and grid[nr, nc] < 0.5:
                            if (nr, nc) not in visited:
                                candidates.append((nr, nc, probs[nr * maze_size + nc]))
                    if not candidates:
                        # allow revisit
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = cur[0] + dr, cur[1] + dc
                            if 0 <= nr < maze_size and 0 <= nc < maze_size and grid[nr, nc] < 0.5:
                                candidates.append((nr, nc, probs[nr * maze_size + nc]))
                        if not candidates:
                            break
                    candidates.sort(key=lambda x: x[2], reverse=True)
                    cur = np.array([candidates[0][0], candidates[0][1]])
                    visited.add(tuple(cur))
                    pred_path.append(tuple(cur))
                if tuple(cur) == tuple(tgt):
                    correct += 1
                    true_len = path_len[b].item()
                    pred_len = len(pred_path)
                    if pred_len <= true_len * 1.05:
                        optimal += 1
                total += 1
    acc = 100 * correct / total if total > 0 else 0.0
    opt = 100 * optimal / total if total > 0 else 0.0
    return acc, opt


# ------------------ Main benchmark ------------------
def run_huge_benchmark(
    maze_size: int = 16,
    n_train: int = 2000,
    n_test: int = 200,
    min_path_length: Optional[int] = None,
    epochs: int = 10,
    R: int = 4,
    T: int = 4,
    batch_size: int = 8,
    seed: int = 42,
    output_dir: str = "experiments/results/huge_500m",
    d_model: int = HUGE_CONFIG["d_model"],
    n_heads: int = HUGE_CONFIG["n_heads"],
    d_ff: int = HUGE_CONFIG["d_ff"],
    depth: int = HUGE_CONFIG["depth"],
    lr: float = 1e-4,
    label_smoothing: float = 0.1,
    warmup_steps: int = 2000,
):
    print("\n" + "=" * 80)
    print("HUGE (≈500M) BASELINE vs PoH-HRM BENCHMARK")
    print("=" * 80)
    print(f"Maze size: {maze_size}×{maze_size}")
    print(f"Training samples: {n_train}")
    print(f"Test samples: {n_test}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"Config: d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}, depth={depth}")
    print("=" * 80)

    # Data
    print("\nGenerating training data...")
    train_data = generate_dataset_proper(maze_size, n_train, min_path_length, seed)
    print("\nGenerating test data...")
    test_data = generate_dataset_proper(maze_size, n_test, min_path_length, seed + 10000)

    train_ds = MazeDatasetWrapper(train_data, maze_size)
    test_ds = MazeDatasetWrapper(test_data, maze_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=max(1, batch_size // 2), shuffle=False)

    results = {}

    # Baseline
    print("\n" + "-" * 80)
    print("Training: Baseline Transformer (HUGE)")
    print("-" * 80)
    baseline = BaselineMazeSolver(maze_size, d_model, n_heads, d_ff, depth).to(device)
    base_params = count_parameters(baseline)
    print(f"Parameters: {base_params/1e6:.2f}M")
    total_steps = epochs * len(train_loader)
    train_one(
        baseline,
        train_loader,
        device,
        epochs=epochs,
        lr=lr,
        label_smoothing=label_smoothing,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )
    base_acc, base_opt = evaluate(baseline, test_loader, device, maze_size)
    print(f"Baseline Accuracy: {base_acc:.2f}%, Optimality: {base_opt:.2f}%")

    # PoH-HRM with parity: reduce depth until ≤ 110% baseline params
    print("\n" + "-" * 80)
    print("Training: PoH-HRM (HUGE, parity)")
    print("-" * 80)
    best_poh = None
    best_params = float("inf")
    best_depth = depth
    for trial_depth in range(depth, 0, -1):
        model_trial = PoHMazeSolver(maze_size, d_model, n_heads, d_ff, trial_depth, R, T)
        p = count_parameters(model_trial)
        if p <= base_params * 1.10:  # within 10%
            best_poh = model_trial
            best_params = p
            best_depth = trial_depth
            break
        if abs(p - base_params) < abs(best_params - base_params):
            best_poh = model_trial
            best_params = p
            best_depth = trial_depth

    poh = best_poh.to(device)
    print(f"Parameters: {best_params/1e6:.2f}M (depth={best_depth}, {best_params/base_params*100:.1f}% of baseline)")
    train_one(
        poh,
        train_loader,
        device,
        epochs=epochs,
        lr=lr,
        label_smoothing=label_smoothing,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )
    poh_acc, poh_opt = evaluate(poh, test_loader, device, maze_size)
    print(f"PoH-HRM Accuracy: {poh_acc:.2f}%, Optimality: {poh_opt:.2f}%")

    results = {
        "config": {
            "maze_size": maze_size,
            "n_train": n_train,
            "n_test": n_test,
            "epochs": epochs,
            "batch_size": batch_size,
            "R": R,
            "T": T,
            "device": str(device),
            "d_model": d_model,
            "n_heads": n_heads,
            "d_ff": d_ff,
            "depth": depth,
        },
        "baseline": {
            "params": base_params,
            "accuracy": base_acc,
            "optimality": base_opt,
        },
        "poh": {
            "params": best_params,
            "depth": best_depth,
            "accuracy": poh_acc,
            "optimality": poh_opt,
            "param_ratio": best_params / base_params,
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"huge_benchmark_maze{maze_size}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {out_file}")
    return results


def main():
    ap = argparse.ArgumentParser(description="Huge (~500M) baseline vs PoH-HRM benchmark")
    ap.add_argument("--maze-size", type=int, default=16)
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--min-path", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default="experiments/results/huge_500m")
    ap.add_argument("--d-model", type=int, default=HUGE_CONFIG["d_model"])
    ap.add_argument("--n-heads", type=int, default=HUGE_CONFIG["n_heads"])
    ap.add_argument("--d-ff", type=int, default=HUGE_CONFIG["d_ff"])
    ap.add_argument("--depth", type=int, default=HUGE_CONFIG["depth"])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--warmup-steps", type=int, default=2000)
    args = ap.parse_args()

    run_huge_benchmark(
        maze_size=args.maze_size,
        n_train=args.train,
        n_test=args.test,
        min_path_length=args.min_path,
        epochs=args.epochs,
        R=args.R,
        T=args.T,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        depth=args.depth,
        lr=args.lr,
        label_smoothing=args.label_smoothing,
        warmup_steps=args.warmup_steps,
    )


if __name__ == "__main__":
    main()


