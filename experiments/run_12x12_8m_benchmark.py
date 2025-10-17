#!/usr/bin/env python3
"""
12x12 maze benchmark: ~8M Baseline vs PoH-HRM with parameter parity.

Defaults target ≈8M baseline params by using d_model=384, n_heads=6, d_ff=1536, depth=4.
PoH depth is auto-reduced to match baseline params within ≤10% (favor ≤ baseline if possible).
"""

import os
import sys
import json
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Path setup
try:
    from experiments.setup_colab import setup_pot_paths
    repo_root = setup_pot_paths()
except Exception:
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

print(f"✓ PoT root: {repo_root}")

# Maze dataset
try:
    from maze_dataset import MazeDataset, MazeDatasetConfig
    from maze_dataset.generation import LatticeMazeGenerators
    print("✓ maze-dataset library available")
except ImportError:
    print("✗ maze-dataset not installed. pip install maze-dataset")
    sys.exit(1)

# PoT modules
from src.pot.modules import PoHConfig, PoHStack, IterRefiner
from src.pot.core.hrm_controller import HRMPointerController, HRMState


def device_select(force_cpu: bool = False):
    if force_cpu:
        print("⚙️  Forcing CPU as requested")
        return torch.device('cpu')
    if torch.cuda.is_available():
        print(f"🚀 CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        return torch.device('cuda')
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("🚀 Apple Silicon GPU (MPS) detected")
        return torch.device('mps')
    print("⚠️  No GPU detected, using CPU")
    return torch.device('cpu')


def generate_dataset(maze_size: int, n_samples: int, min_path_length: int, seed: int):
    cfg = MazeDatasetConfig(
        name=f"maze_{maze_size}x{maze_size}_minpath{min_path_length}",
        grid_n=maze_size,
        n_mazes=n_samples * 3,
        maze_ctor=LatticeMazeGenerators.gen_dfs,
        seed=seed,
    )
    dataset = MazeDataset.from_config(cfg, do_generate=True, load_local=False, save_local=False)
    filtered = dataset.filter_by.path_length(min_length=min_path_length)
    if len(filtered) < n_samples:
        print(f"  ⚠️  Only {len(filtered)} mazes met criteria; using all")
        n_samples = len(filtered)
    else:
        filtered = filtered[:n_samples]
    data, lens = [], []
    for sm in filtered:
        mz = sm.maze
        grid = np.ones((maze_size, maze_size), dtype=np.float32)
        for node in mz.get_nodes():
            r, c = (int(node[0]), int(node[1])) if isinstance(node, np.ndarray) else (node.row, node.col)
            grid[r, c] = 0.0
        if hasattr(mz, 'start_pos'):
            start = (mz.start_pos.row, mz.start_pos.col)
            goal = (mz.end_pos.row, mz.end_pos.col)
        else:
            sol = sm.solution
            start, goal = tuple(sol[0]), tuple(sol[-1])
        sol = sm.solution
        path = [tuple(x) for x in sol]
        lens.append(len(path))
        data.append({"maze": grid, "start": start, "goal": goal, "path": path, "length": len(path)})
    print(f"  ✓ Generated {len(data)} mazes | path len: {np.mean(lens):.1f}±{np.std(lens):.1f}")
    return data


class MazeDS(Dataset):
    def __init__(self, data, size):
        self.data = data; self.size = size
        self.max_len = max(len(d['path']) for d in data)
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        it = self.data[i]
        maze = torch.tensor(it['maze'], dtype=torch.float32)
        start = torch.tensor(it['start'], dtype=torch.long)
        goal = torch.tensor(it['goal'], dtype=torch.long)
        idxs = [r * self.size + c for r, c in it['path']]
        pad = idxs + [-1] * (self.max_len - len(idxs))
        return maze, start, goal, torch.tensor(pad, dtype=torch.long), len(idxs)


class Baseline(nn.Module):
    def __init__(self, size, d_model, n_heads, d_ff, depth, dropout=0.1, maze_enc: bool = False):
        super().__init__()
        self.size = size
        self.use_maze_enc = maze_enc
        self.cell = nn.Linear(1, d_model)
        self.pos = nn.Embedding(size*size, d_model)
        self.se = nn.Embedding(2, d_model); self.ge = nn.Embedding(2, d_model)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, num_layers=depth)
        self.out = nn.Linear(d_model, size*size)
        if self.use_maze_enc:
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4,4))
            )
            self.cnn_proj = nn.Linear(32*4*4, d_model)
    def forward(self, maze, start, goal):
        B = maze.size(0); N = self.size*self.size
        x = self.cell(maze.view(B, N, 1))
        pos = torch.arange(N, device=maze.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos(pos)
        if self.use_maze_enc:
            g = maze.unsqueeze(1)
            g = self.cnn(g)
            g = torch.flatten(g, start_dim=1)
            g = self.cnn_proj(g)  # (B, d_model)
            x = x + g.unsqueeze(1)
        s_idx = start[:,0]*self.size + start[:,1]
        g_idx = goal[:,0]*self.size + goal[:,1]
        s_m = torch.zeros(B, N, device=maze.device, dtype=torch.long); s_m.scatter_(1, s_idx.unsqueeze(1), 1)
        g_m = torch.zeros(B, N, device=maze.device, dtype=torch.long); g_m.scatter_(1, g_idx.unsqueeze(1), 1)
        x = x + self.se(s_m) + self.ge(g_m)
        h = self.tr(x)
        return self.out(h)


class StatefulHRMRouter(nn.Module):
    def __init__(self, hrm, n_heads):
        super().__init__(); self.hrm=hrm; self.n_heads=n_heads; self.state=None
    def forward(self, x_ctrl):
        B,T,_ = x_ctrl.shape
        if self.state is None:
            dev = x_ctrl.device
            self.state = HRMState(z_L=torch.zeros(B,self.hrm.d_ctrl,device=dev), z_H=torch.zeros(B,self.hrm.d_ctrl,device=dev), step=torch.zeros(B,dtype=torch.long,device=dev))
        a, st, _ = self.hrm(x_ctrl.mean(dim=1), self.state)
        self.state = HRMState(z_L=st.z_L.detach(), z_H=st.z_H.detach(), step=st.step.detach())
        return torch.log(a.unsqueeze(1).expand(B,T,self.n_heads)+1e-8)


class PoH(nn.Module):
    def __init__(self, size, d_model, n_heads, d_ff, depth, R, T, dropout=0.1, maze_enc: bool = False):
        super().__init__(); self.size=size; self.R=R
        self.use_maze_enc = maze_enc
        self.cell = nn.Linear(1, d_model)
        self.pos = nn.Embedding(size*size, d_model)
        self.se = nn.Embedding(2, d_model); self.ge = nn.Embedding(2, d_model)
        cfg = PoHConfig(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
        stack = PoHStack(cfg, depth=depth)
        for blk in stack.blocks:
            if hasattr(blk,'router'):
                blk.router = StatefulHRMRouter(HRMPointerController(d_model=d_model, n_heads=n_heads, d_ctrl=d_model//2, T=T), n_heads)
        self.ref = IterRefiner(stack, max_inner_iters=R, outer_residual=True, rezero_init=True)
        self.stack = stack  # For O(1) mode
        self.out = nn.Linear(d_model, size*size)
        if self.use_maze_enc:
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4,4))
            )
            self.cnn_proj = nn.Linear(32*4*4, d_model)
    def forward(self, maze, start, goal, last_iter_only=False):
        B = maze.size(0); N = self.size*self.size
        x = self.cell(maze.view(B,N,1))
        pos = torch.arange(N, device=maze.device).unsqueeze(0).expand(B,-1)
        x = x + self.pos(pos)
        if self.use_maze_enc:
            g = maze.unsqueeze(1)
            g = self.cnn(g)
            g = torch.flatten(g, start_dim=1)
            g = self.cnn_proj(g)
            x = x + g.unsqueeze(1)
        s_idx = start[:,0]*self.size + start[:,1]
        g_idx = goal[:,0]*self.size + goal[:,1]
        s_m = torch.zeros(B, N, device=maze.device, dtype=torch.long); s_m.scatter_(1, s_idx.unsqueeze(1), 1)
        g_m = torch.zeros(B, N, device=maze.device, dtype=torch.long); g_m.scatter_(1, g_idx.unsqueeze(1), 1)
        x = x + self.se(s_m) + self.ge(g_m)
        
        if last_iter_only:
            # O(1) memory: only backprop through last iteration
            h = x
            for i in range(self.R):
                h_next = self.stack(h)
                # Handle stack returning tuple (output, stats) or just output
                if isinstance(h_next, tuple):
                    h_next = h_next[0]
                if i < self.R - 1:
                    h = h_next.detach()  # Break gradient flow for intermediate iterations
                else:
                    h = h_next  # Keep gradients for last iteration
        else:
            # O(R) memory: backprop through all iterations (standard)
            h,_ = self.ref(x)
        
        return self.out(h)
    def forward_with_stats(self, maze, start, goal):
        B = maze.size(0); N = self.size*self.size
        x = self.cell(maze.view(B,N,1))
        pos = torch.arange(N, device=maze.device).unsqueeze(0).expand(B,-1)
        x = x + self.pos(pos)
        if self.use_maze_enc:
            g = maze.unsqueeze(1)
            g = self.cnn(g)
            g = torch.flatten(g, start_dim=1)
            g = self.cnn_proj(g)
            x = x + g.unsqueeze(1)
        s_idx = start[:,0]*self.size + start[:,1]
        g_idx = goal[:,0]*self.size + goal[:,1]
        s_m = torch.zeros(B, N, device=maze.device, dtype=torch.long); s_m.scatter_(1, s_idx.unsqueeze(1), 1)
        g_m = torch.zeros(B, N, device=maze.device, dtype=torch.long); g_m.scatter_(1, g_idx.unsqueeze(1), 1)
        x = x + self.se(s_m) + self.ge(g_m)
        h, stats = self.ref(x, return_inner_stats=True)
        return self.out(h), stats


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def train(model, loader, device, epochs=10, lr=1e-3, label_smoothing: float = 0.0, warmup_steps: int = 500, multi_horizon: int = 1, validity_mask: bool = False, route_ent_weight: float = 0.0, ent_anneal: bool = False, size: int = None, supervision_interval: int = 1, last_iter_only: bool = False):
    """
    Train with configurable supervision density and gradient flow.
    
    Args:
        supervision_interval: Supervise every N steps in the path.
            1 = dense (every step)
            2 = sparse (every other step)
            5 = very sparse (every 5th step)
            Forces model to learn longer-term planning.
        last_iter_only: If True, only backprop through last refinement iteration (O(1) memory).
            False = O(R) memory, True = O(1) memory but slower convergence.
    """
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=label_smoothing)
    steps_per_epoch = len(loader)
    total_steps = max(1, epochs * steps_per_epoch)
    def lr_at(step):
        if step < warmup_steps:
            return lr * step / max(1, warmup_steps)
        rem = max(1, total_steps - warmup_steps)
        prog = (step - warmup_steps) / rem
        min_lr = 0.1 * lr
        return min_lr + (lr - min_lr) * 0.5 * (1 + np.cos(np.pi * prog))
    global_step = 0
    for ep in range(epochs):
        tot = 0.0; n=0
        for maze,start,goal,path,_ in loader:
            for g in opt.param_groups:
                g['lr'] = lr_at(global_step)
            maze,start,goal,path = maze.to(device),start.to(device),goal.to(device),path.to(device)
            opt.zero_grad()
            # Forward (optionally with stats for PoH, and with O(1) memory if requested)
            if route_ent_weight>0.0 and hasattr(model, 'forward_with_stats'):
                logits, stats = model.forward_with_stats(maze,start,goal)
            else:
                if hasattr(model, 'forward') and 'last_iter_only' in model.forward.__code__.co_varnames:
                    logits = model(maze,start,goal,last_iter_only=last_iter_only)
                else:
                    logits = model(maze,start,goal)
                stats=None
            V = logits.size(-1); max_len = path.size(1)
            ce_terms = []
            K = max(1, multi_horizon)
            for i in range(0, max_len-1, supervision_interval):  # Sparse supervision
                for k in range(1, K+1):
                    if i+k >= max_len: break
                    m = (path[:,i]!=-1) & (path[:,i+k]!=-1)
                    if not m.any():
                        continue
                    curr, target = path[m,i], path[m,i+k]
                    raw = logits[m].gather(1, curr.unsqueeze(1).unsqueeze(2).expand(-1,1,V)).squeeze(1)
                    if validity_mask and k==1 and size is not None:
                        # Numerically stable masking: set invalid to large negative, then log_softmax + NLL
                        masked = torch.full_like(raw, fill_value=-1e4)
                        for idx_b, cur_idx in enumerate(curr):
                            r = (cur_idx // size).item(); c = (cur_idx % size).item()
                            allowed=[]
                            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = r+dr, c+dc
                                if 0<=nr<size and 0<=nc<size and maze[m][idx_b, nr, nc] < 0.5:
                                    allowed.append(nr*size+nc)
                            if not allowed:
                                pass_cells = (maze[m][idx_b].view(-1) < 0.5).nonzero(as_tuple=False).view(-1)
                                masked[idx_b, pass_cells] = raw[idx_b, pass_cells]
                            else:
                                masked[idx_b, allowed] = raw[idx_b, allowed]
                        logp = torch.log_softmax(masked, dim=-1)
                        nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
                        ce_terms.append(nll.mean())
                    else:
                        ce_terms.append(crit(raw, target))
            # Routing entropy regularization
            if stats is not None and route_ent_weight>0.0:
                # stats is a list per refinement step; each item packs per-block stats averages
                ent_vals=[]
                for s in stats:
                    if 'route_entropy_mean' in s:
                        ent_vals.append(float(s['route_entropy_mean']))
                if ent_vals:
                    ent = torch.tensor(np.mean(ent_vals), dtype=torch.float32, device=device)
                    w = route_ent_weight
                    if ent_anneal:
                        w = w * max(0.0, 1.0 - global_step/float(total_steps))
                    ce_terms.append(w * ent)
            if ce_terms:
                loss = torch.stack(ce_terms).mean()
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
                tot+=loss.item(); n+=1
            global_step += 1
        print(f"  Epoch {ep+1}/{epochs}, Loss(mean CE per token): {tot/max(1,n):.4f}, LR: {opt.param_groups[0]['lr']:.2e}")


@torch.no_grad()
def evaluate(model, loader, device, size):
    model.eval(); correct=0; optimal=0; total=0
    for maze,start,goal,path,plen in loader:
        maze,start,goal = maze.to(device),start.to(device),goal.to(device)
        B = maze.size(0)
        for b in range(B):
            cur = start[b].cpu().numpy(); tgt = goal[b].cpu().numpy(); grid=maze[b].cpu().numpy()
            visited={tuple(cur)}; pred=[tuple(cur)]
            for _ in range(size*size):
                if tuple(cur)==tuple(tgt): break
                logits=model(maze[b:b+1], torch.tensor([cur],dtype=torch.long,device=device), goal[b:b+1])
                idx=cur[0]*size+cur[1]; probs=torch.softmax(logits[0,idx],dim=-1).cpu().numpy()
                cand=[]
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc=cur[0]+dr,cur[1]+dc
                    if 0<=nr<size and 0<=nc<size and grid[nr,nc]<0.5 and (nr,nc) not in visited:
                        cand.append((nr,nc,probs[nr*size+nc]))
                if not cand:
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc=cur[0]+dr,cur[1]+dc
                        if 0<=nr<size and 0<=nc<size and grid[nr,nc]<0.5:
                            cand.append((nr,nc,probs[nr*size+nc]))
                    if not cand: break
                cand.sort(key=lambda x:x[2],reverse=True)
                cur=np.array([cand[0][0],cand[0][1]]); visited.add(tuple(cur)); pred.append(tuple(cur))
            if tuple(cur)==tuple(tgt):
                correct+=1; true_len=plen[b]; pred_len=len(pred)
                if pred_len<=true_len*1.05: optimal+=1
            total+=1
    return 100*correct/max(1,total), 100*optimal/max(1,total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', type=int, default=400)
    ap.add_argument('--test', type=int, default=80)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--R', type=int, default=4)
    ap.add_argument('--T', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', type=str, default='experiments/results/benchmark_12x12_8m')
    ap.add_argument('--cpu', action='store_true', help='Force CPU device')
    # Optimization & training flags
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--label-smoothing', type=float, default=0.0)
    ap.add_argument('--warmup-steps', type=int, default=500)
    ap.add_argument('--multi-horizon', type=int, default=1)
    ap.add_argument('--maze-enc', action='store_true')
    ap.add_argument('--validity-mask', action='store_true')
    ap.add_argument('--route-ent-weight', type=float, default=0.0)
    ap.add_argument('--ent-anneal', action='store_true')
    ap.add_argument('--supervision-interval', type=int, default=1, 
                    help='Supervise every N steps (1=dense, 2=sparse, 5=very sparse)')
    ap.add_argument('--last-iter-only', action='store_true',
                    help='O(1) memory: only backprop through last refinement iteration')
    args = ap.parse_args()

    device = device_select(force_cpu=args.cpu)
    size=12; min_path=int(size*size*0.35)
    print("Generating training data..."); train_data = generate_dataset(size, args.train, min_path, args.seed)
    print("Generating test data..."); test_data = generate_dataset(size, args.test, min_path, args.seed+10000)

    train_loader = DataLoader(MazeDS(train_data, size), batch_size=32, shuffle=True)
    test_loader = DataLoader(MazeDS(test_data, size), batch_size=16, shuffle=False)

    # ~8M baseline config
    d_model=384; n_heads=6; d_ff=1536; depth=4

    print("\nBaseline (≈8M params)")
    base = Baseline(size, d_model, n_heads, d_ff, depth, maze_enc=args.maze_enc).to(device)
    base_p = count_params(base); print(f"Parameters: {base_p/1e6:.2f}M")
    train(base, train_loader, device, epochs=args.epochs, lr=args.lr, label_smoothing=args.label_smoothing, warmup_steps=args.warmup_steps, multi_horizon=args.multi_horizon, validity_mask=args.validity_mask, size=size, supervision_interval=args.supervision_interval, last_iter_only=False)
    base_acc, base_opt = evaluate(base, test_loader, device, size)
    print(f"Baseline: Acc={base_acc:.2f}%, Opt={base_opt:.2f}%")

    print("\nPoH-HRM (parity, depth=4)")
    # Keep depth=4, reduce width (d_model) for parity
    best_poh=None; best_p=float('inf'); best_dm=d_model; best_heads=n_heads
    for trial_heads in [6, 5, 4]:  # Try different head counts
        trial_dm = trial_heads * 64  # Ensure divisibility (d_model = heads * 64)
        if trial_dm < d_model * 0.5: continue  # Don't go too small
        trial_ff = trial_dm * 4
        trial = PoH(size, trial_dm, trial_heads, trial_ff, depth, R=args.R, T=args.T, maze_enc=args.maze_enc)
        p = count_params(trial)
        if p <= base_p * 1.10: best_poh=trial; best_p=p; best_dm=trial_dm; best_heads=trial_heads; break
        if abs(p-base_p) < abs(best_p-base_p): best_poh=trial; best_p=p; best_dm=trial_dm; best_heads=trial_heads
    poh = best_poh.to(device); print(f"Parameters: {best_p/1e6:.2f}M (d_model={best_dm}, n_heads={best_heads}, depth=4)")
    if args.last_iter_only:
        print(f"  Training mode: O(1) memory (last iteration only)")
    else:
        print(f"  Training mode: O(R) memory (all {args.R} iterations)")
    train(poh, train_loader, device, epochs=args.epochs, lr=args.lr, label_smoothing=args.label_smoothing, warmup_steps=args.warmup_steps, multi_horizon=args.multi_horizon, validity_mask=args.validity_mask, route_ent_weight=args.route_ent_weight, ent_anneal=args.ent_anneal, size=size, supervision_interval=args.supervision_interval, last_iter_only=args.last_iter_only)
    poh_acc, poh_opt = evaluate(poh, test_loader, device, size)
    print(f"PoH-HRM: Acc={poh_acc:.2f}%, Opt={poh_opt:.2f}%")

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output,'results.json'),'w') as f:
        json.dump({
            'config': {'size':size, 'train':args.train, 'test':args.test, 'epochs':args.epochs, 'R':args.R, 'T':args.T},
            'baseline': {'params': int(base_p), 'acc': base_acc, 'opt': base_opt},
            'poh': {'params': int(best_p), 'd_model': best_dm, 'n_heads': best_heads, 'depth': 4, 'acc': poh_acc, 'opt': poh_opt},
        }, f, indent=2)
    print(f"\n✓ Results saved to {args.output}/results.json")


if __name__ == '__main__':
    main()


