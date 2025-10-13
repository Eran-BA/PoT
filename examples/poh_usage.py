#!/usr/bin/env python3
"""
Example: How to use the modular PoH architecture.

Demonstrates:
1. Basic usage (drop-in like TransformerEncoder)
2. Different routing modes (soft vs top-k)
3. Inner refinement with multiple iterations
4. ACT halting for adaptive computation
5. Integration with inner-loop logging

Author: Eran Ben Artzy
Year: 2025
"""

import torch
from src.pot.modules import PoHConfig, PoHStack, IterRefiner
from src.pot.logging import InnerLoopLogger, InnerStepRow, grad_global_norm


def example_basic_usage():
    """Example 1: Basic drop-in usage."""
    print("\n" + "="*60)
    print("Example 1: Basic Usage (Drop-in for TransformerEncoder)")
    print("="*60)
    
    # Configure PoH
    cfg = PoHConfig(
        d_model=512,
        n_heads=8,
        d_ff=2048,
        dropout=0.1,
        route_mode="soft",       # Soft routing
        share_router=True,       # Share router across layers
    )
    
    # Build stack
    stack = PoHStack(cfg, depth=6)
    
    # Forward pass
    x = torch.randn(2, 10, 512)  # [batch=2, seq_len=10, d_model=512]
    out, stats = stack(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Num blocks:   {len(stats)}")
    print(f"Stats keys:   {list(stats[0].keys())}")
    print(f"Route entropy (block 0): {stats[0]['route_entropy_mean']:.4f}")
    print(f"Attn entropy (block 0):  {stats[0]['attn_entropy_mean']:.4f}")


def example_topk_routing():
    """Example 2: Top-k routing (sparse head selection)."""
    print("\n" + "="*60)
    print("Example 2: Top-k Routing (Sparse Head Selection)")
    print("="*60)
    
    cfg = PoHConfig(
        d_model=256,
        n_heads=8,
        route_mode="topk",
        route_topk=2,  # Select top-2 heads per token
    )
    
    stack = PoHStack(cfg, depth=4)
    x = torch.randn(1, 5, 256)
    out, stats = stack(x)
    
    # Inspect routing (should be sparse)
    route = stats[0]['route']  # [B, T, H]
    print(f"Route shape: {route.shape}")
    print(f"Route sparsity: {(route == 0).float().mean():.2%} zeros")
    print(f"Active heads per token: {(route > 0).sum(dim=-1)[0].tolist()}")


def example_inner_refinement():
    """Example 3: Inner refinement (multiple iterations)."""
    print("\n" + "="*60)
    print("Example 3: Inner Refinement (K=3 iterations)")
    print("="*60)
    
    cfg = PoHConfig(d_model=128, n_heads=4, d_ff=256)
    stack = PoHStack(cfg, depth=2)
    refiner = IterRefiner(stack, max_inner_iters=3)
    
    x = torch.randn(1, 8, 128)
    out, inner_stats = refiner(x, return_inner_stats=True)
    
    print(f"Num inner iterations: {len(inner_stats)}")
    for i, s in enumerate(inner_stats):
        print(f"  Iter {i+1}: route_entropy={s.get('route_entropy_mean', 0):.4f}, "
              f"attn_entropy={s.get('attn_entropy_mean', 0):.4f}")


def example_act_halting():
    """Example 4: ACT halting (adaptive computation)."""
    print("\n" + "="*60)
    print("Example 4: ACT Halting (Adaptive Computation)")
    print("="*60)
    
    cfg = PoHConfig(
        d_model=128,
        n_heads=4,
        act_halting=True,
        act_threshold=0.95,
        act_penalty=0.01,
    )
    
    stack = PoHStack(cfg, depth=2)
    refiner = IterRefiner(stack, max_inner_iters=5, act=True)
    
    x = torch.randn(2, 10, 128)
    out, inner_stats = refiner(x, return_inner_stats=True)
    
    print(f"Max iterations: 5")
    print(f"Actual iterations: {len(inner_stats)}")
    for i, s in enumerate(inner_stats):
        halted = s.get('halted_frac', 0)
        print(f"  Iter {i+1}: halted_frac={halted:.2%}")
    
    if inner_stats:
        ponder_cost = inner_stats[-1].get('ponder_cost', 0)
        print(f"Ponder cost: {ponder_cost:.6f}")


def example_with_logging():
    """Example 5: Integration with inner-loop logger."""
    print("\n" + "="*60)
    print("Example 5: Inner-Loop Logging")
    print("="*60)
    
    import tempfile
    import os
    
    cfg = PoHConfig(d_model=64, n_heads=4)
    stack = PoHStack(cfg, depth=2)
    refiner = IterRefiner(stack, max_inner_iters=2)
    
    # Create temporary CSV
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "innerloop_log.csv")
        
        with InnerLoopLogger(csv_path) as logger:
            optimizer = torch.optim.Adam(refiner.parameters(), lr=1e-3)
            
            for step in range(3):  # 3 training steps
                x = torch.randn(2, 5, 64, requires_grad=True)
                target = torch.randn(2, 5, 64)
                
                optimizer.zero_grad()
                out, inner_stats = refiner(x, return_inner_stats=True)
                loss = ((out - target) ** 2).mean()
                loss.backward()
                
                grad_norm = grad_global_norm(refiner)
                optimizer.step()
                
                # Log each inner step
                for s in inner_stats:
                    row = InnerStepRow(
                        run_id="example",
                        epoch=1,
                        global_step=step + 1,
                        inner_step=s["inner_step"],
                        batch_size=2,
                        loss=float(loss.item()),
                        grad_norm=float(grad_norm),
                        attn_entropy_mean=s.get("attn_entropy_mean"),
                    )
                    logger.log(row)
        
        print(f"Logged to: {csv_path}")
        print(f"Total rows: {3 * 2} (3 steps × 2 inner iters)")
        
        # Show first few rows
        with open(csv_path) as f:
            lines = f.readlines()[:4]
            print("\nFirst 3 rows:")
            for line in lines:
                print("  " + line.strip())


def example_param_comparison():
    """Example 6: Parameter count comparison with baseline."""
    print("\n" + "="*60)
    print("Example 6: Parameter Parity Check")
    print("="*60)
    
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    
    d_model, n_heads, d_ff, depth = 512, 8, 2048, 6
    
    # Baseline
    baseline_layer = TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=d_ff,
        batch_first=True
    )
    baseline = TransformerEncoder(baseline_layer, num_layers=depth)
    baseline_params = sum(p.numel() for p in baseline.parameters())
    
    # PoH
    cfg = PoHConfig(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        param_match_baseline=True,
        share_router=True,
    )
    poh_stack = PoHStack(cfg, depth=depth)
    poh_params = sum(p.numel() for p in poh_stack.parameters())
    
    ratio = poh_params / baseline_params
    delta_pct = abs(ratio - 1.0) * 100
    
    print(f"Baseline params: {baseline_params:,}")
    print(f"PoH params:      {poh_params:,}")
    print(f"Ratio:           {ratio:.6f}")
    print(f"Delta:           {delta_pct:.4f}%")
    print(f"✅ Within 1% parity!" if delta_pct <= 1.0 else f"❌ Exceeds 1% parity")


if __name__ == "__main__":
    example_basic_usage()
    example_topk_routing()
    example_inner_refinement()
    example_act_halting()
    example_with_logging()
    example_param_comparison()
    
    print("\n" + "="*60)
    print("All examples completed successfully! 🎉")
    print("="*60 + "\n")

