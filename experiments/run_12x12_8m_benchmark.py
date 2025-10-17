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


def device_select():
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
    def __init__(self, size, d_model, n_heads, d_ff, depth, dropout=0.1):
        super().__init__()
        self.size = size
        self.cell = nn.Linear(1, d_model)
        self.pos = nn.Embedding(size*size, d_model)
        self.se = nn.Embedding(2, d_model); self.ge = nn.Embedding(2, d_model)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, num_layers=depth)
        self.out = nn.Linear(d_model, size*size)
    def forward(self, maze, start, goal):
        B = maze.size(0); N = self.size*self.size
        x = self.cell(maze.view(B, N, 1))
        pos = torch.arange(N, device=maze.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos(pos)
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
    def __init__(self, size, d_model, n_heads, d_ff, depth, R, T, dropout=0.1):
        super().__init__(); self.size=size
        self.cell = nn.Linear(1, d_model)
        self.pos = nn.Embedding(size*size, d_model)
        self.se = nn.Embedding(2, d_model); self.ge = nn.Embedding(2, d_model)
        cfg = PoHConfig(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
        stack = PoHStack(cfg, depth=depth)
        for blk in stack.blocks:
            if hasattr(blk,'router'):
                blk.router = StatefulHRMRouter(HRMPointerController(d_model=d_model, n_heads=n_heads, d_ctrl=d_model//2, T=T), n_heads)
        self.ref = IterRefiner(stack, max_inner_iters=R, outer_residual=True, rezero_init=True)
        self.out = nn.Linear(d_model, size*size)
    def forward(self, maze, start, goal):
        B = maze.size(0); N = self.size*self.size
        x = self.cell(maze.view(B,N,1))
        pos = torch.arange(N, device=maze.device).unsqueeze(0).expand(B,-1)
        x = x + self.pos(pos)
        s_idx = start[:,0]*self.size + start[:,1]
        g_idx = goal[:,0]*self.size + goal[:,1]
        s_m = torch.zeros(B, N, device=maze.device, dtype=torch.long); s_m.scatter_(1, s_idx.unsqueeze(1), 1)
        g_m = torch.zeros(B, N, device=maze.device, dtype=torch.long); g_m.scatter_(1, g_idx.unsqueeze(1), 1)
        x = x + self.se(s_m) + self.ge(g_m)
        h,_ = self.ref(x)
        return self.out(h)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def train(model, loader, device, epochs=10, lr=1e-3):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(ignore_index=-1)
    for ep in range(epochs):
        tot = 0.0; n=0
        for maze,start,goal,path,_ in loader:
            maze,start,goal,path = maze.to(device),start.to(device),goal.to(device),path.to(device)
            opt.zero_grad(); logits = model(maze,start,goal)
            V = logits.size(-1); max_len = path.size(1); loss=0.0; steps=0
            for i in range(max_len-1):
                m = (path[:,i]!=-1) & (path[:,i+1]!=-1)
                if m.any():
                    curr,nextt = path[m,i], path[m,i+1]
                    curr_logits = logits[m].gather(1, curr.unsqueeze(1).unsqueeze(2).expand(-1,1,V)).squeeze(1)
                    loss += crit(curr_logits, nextt); steps+=1
            if steps>0:
                loss/=steps; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
                tot+=loss.item(); n+=1
        print(f"  Epoch {ep+1}/{epochs}, Loss: {tot/max(1,n):.4f}")


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
    args = ap.parse_args()

    device = device_select()
    size=12; min_path=int(size*size*0.35)
    print("Generating training data..."); train_data = generate_dataset(size, args.train, min_path, args.seed)
    print("Generating test data..."); test_data = generate_dataset(size, args.test, min_path, args.seed+10000)

    train_loader = DataLoader(MazeDS(train_data, size), batch_size=32, shuffle=True)
    test_loader = DataLoader(MazeDS(test_data, size), batch_size=16, shuffle=False)

    # ~8M baseline config
    d_model=384; n_heads=6; d_ff=1536; depth=4

    print("\nBaseline (≈8M params)")
    base = Baseline(size, d_model, n_heads, d_ff, depth).to(device)
    base_p = count_params(base); print(f"Parameters: {base_p/1e6:.2f}M")
    train(base, train_loader, device, epochs=args.epochs, lr=1e-3)
    base_acc, base_opt = evaluate(base, test_loader, device, size)
    print(f"Baseline: Acc={base_acc:.2f}%, Opt={base_opt:.2f}%")

    print("\nPoH-HRM (parity)")
    # Reduce PoH depth for parity
    best_poh=None; best_p=float('inf'); best_depth=depth
    for d in range(depth, 0, -1):
        trial = PoH(size, d_model, n_heads, d_ff, d, R=args.R, T=args.T)
        p = count_params(trial)
        if p <= base_p * 1.10: best_poh=trial; best_p=p; best_depth=d; break
        if abs(p-base_p) < abs(best_p-base_p): best_poh=trial; best_p=p; best_depth=d
    poh = best_poh.to(device); print(f"Parameters: {best_p/1e6:.2f}M (depth={best_depth})")
    train(poh, train_loader, device, epochs=args.epochs, lr=1e-3)
    poh_acc, poh_opt = evaluate(poh, test_loader, device, size)
    print(f"PoH-HRM: Acc={poh_acc:.2f}%, Opt={poh_opt:.2f}%")

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output,'results.json'),'w') as f:
        json.dump({
            'config': {'size':size, 'train':args.train, 'test':args.test, 'epochs':args.epochs, 'R':args.R, 'T':args.T},
            'baseline': {'params': int(base_p), 'acc': base_acc, 'opt': base_opt},
            'poh': {'params': int(best_p), 'depth': best_depth, 'acc': poh_acc, 'opt': poh_opt},
        }, f, indent=2)
    print(f"\n✓ Results saved to {args.output}/results.json")


if __name__ == '__main__':
    main()


