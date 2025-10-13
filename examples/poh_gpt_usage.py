#!/usr/bin/env python3
"""
Example: PoHGPT - GPT-style autoregressive model with PoH blocks.

Demonstrates:
1. Basic language modeling setup
2. Forward pass with causal masking
3. Autoregressive generation
4. Parameter comparison with baseline GPT
5. Different sampling strategies (temperature, top-k, top-p)

Author: Eran Ben Artzy
Year: 2025
"""

import torch
from src.pot.modules import PoHConfig
from src.pot.models import PoHGPT
from src.pot.models.poh_gpt import BaselineGPT


def example_basic_lm():
    """Example 1: Basic language modeling."""
    print("\n" + "="*60)
    print("Example 1: Basic Language Modeling")
    print("="*60)
    
    # Configure PoHGPT for causal modeling
    cfg = PoHConfig(
        d_model=256,
        n_heads=8,
        d_ff=1024,
        depth=4,
        max_inner_iters=1,  # Single pass (like GPT)
        is_causal=True,     # Enable causal masking
        pos_encoding="absolute",
        max_seq_len=512,
    )
    
    model = PoHGPT(vocab_size=10000, cfg=cfg)
    
    # Forward pass
    input_ids = torch.randint(0, 10000, (2, 20))  # [batch=2, seq_len=20]
    logits, _ = model(input_ids)
    
    print(f"Input shape:  {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")  # [2, 20, 10000]
    print(f"Parameters:   {model.get_num_params():,}")


def example_generation():
    """Example 2: Autoregressive generation."""
    print("\n" + "="*60)
    print("Example 2: Autoregressive Generation")
    print("="*60)
    
    cfg = PoHConfig(
        d_model=128,
        n_heads=4,
        d_ff=512,
        depth=3,
        max_inner_iters=1,
        is_causal=True,
        pos_encoding="absolute",  # Use learned positions
        max_seq_len=512,
    )
    
    model = PoHGPT(vocab_size=5000, cfg=cfg)
    
    # Start with a prompt
    prompt = torch.tensor([[1, 2, 3, 4, 5]])  # [1, 5]
    
    # Generate 15 new tokens
    generated = model.generate(prompt, max_new_tokens=15, temperature=1.0)
    
    print(f"Prompt length:    {prompt.shape[1]}")
    print(f"Generated length: {generated.shape[1]}")  # 5 + 15 = 20
    print(f"Generated IDs:    {generated[0].tolist()[:10]}...")


def example_iterative_refinement():
    """Example 3: Multi-pass iterative refinement."""
    print("\n" + "="*60)
    print("Example 3: Iterative Refinement (K=3)")
    print("="*60)
    
    cfg = PoHConfig(
        d_model=128,
        n_heads=4,
        depth=2,
        max_inner_iters=3,  # 3 refinement passes
        outer_residual=True,
        rezero_init=True,   # Start with identity
        is_causal=True,
        pos_encoding="absolute",
    )
    
    model = PoHGPT(vocab_size=1000, cfg=cfg)
    
    input_ids = torch.randint(0, 1000, (1, 10))
    logits, inner_stats = model(input_ids, return_inner_stats=True)
    
    print(f"Inner iterations: {len(inner_stats)}")
    print(f"Outer residual alpha: {model.refiner.alpha.item():.6f}")
    
    for i, s in enumerate(inner_stats):
        print(f"  Iter {i+1}: route_entropy={s.get('route_entropy_mean', 0):.4f}")


def example_sampling_strategies():
    """Example 4: Different sampling strategies."""
    print("\n" + "="*60)
    print("Example 4: Sampling Strategies")
    print("="*60)
    
    cfg = PoHConfig(d_model=64, n_heads=4, depth=2, is_causal=True)
    model = PoHGPT(vocab_size=100, cfg=cfg)
    
    prompt = torch.tensor([[1, 2, 3]])
    
    # Greedy (temperature=0.1, very deterministic)
    greedy = model.generate(prompt, max_new_tokens=10, temperature=0.1)
    print(f"Greedy (T=0.1):     {greedy[0].tolist()}")
    
    # High temperature (more random)
    random_sample = model.generate(prompt, max_new_tokens=10, temperature=2.0)
    print(f"Random (T=2.0):     {random_sample[0].tolist()}")
    
    # Top-k sampling
    topk = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=10)
    print(f"Top-k (k=10):       {topk[0].tolist()}")
    
    # Top-p (nucleus) sampling
    topp = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_p=0.9)
    print(f"Top-p (p=0.9):      {topp[0].tolist()}")


def example_param_comparison():
    """Example 5: Parameter comparison with baseline GPT."""
    print("\n" + "="*60)
    print("Example 5: Parameter Comparison")
    print("="*60)
    
    d_model, n_heads, d_ff, depth = 256, 8, 1024, 4
    vocab_size = 10000
    
    # Baseline GPT
    baseline = BaselineGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        depth=depth,
    )
    
    # PoHGPT (with routing)
    cfg = PoHConfig(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        depth=depth,
        max_inner_iters=1,
        is_causal=True,
        pos_encoding="absolute",
        share_router=True,
    )
    poh_gpt = PoHGPT(vocab_size=vocab_size, cfg=cfg)
    
    baseline_params = baseline.get_num_params(non_embedding=True)
    poh_params = poh_gpt.get_num_params(non_embedding=True)
    
    ratio = poh_params / baseline_params
    delta_pct = abs(ratio - 1.0) * 100
    
    print(f"Baseline GPT: {baseline_params:,} params")
    print(f"PoHGPT:       {poh_params:,} params")
    print(f"Ratio:        {ratio:.6f}")
    print(f"Delta:        {delta_pct:.4f}%")
    
    if delta_pct <= 1.0:
        print("✅ Within 1% parity!")
    else:
        print(f"⚠️  Exceeds 1% parity by {delta_pct - 1.0:.2f}%")


def example_act_halting():
    """Example 6: ACT halting for adaptive computation."""
    print("\n" + "="*60)
    print("Example 6: ACT Halting (Adaptive Computation)")
    print("="*60)
    
    cfg = PoHConfig(
        d_model=64,
        n_heads=4,
        depth=2,
        max_inner_iters=5,  # Max 5 iterations
        is_causal=True,
        act_halting=True,   # Enable ACT
        act_threshold=0.95,
        act_penalty=0.01,
        pos_encoding="none",
    )
    
    model = PoHGPT(vocab_size=100, cfg=cfg)
    
    input_ids = torch.randint(0, 100, (1, 10))
    logits, inner_stats = model(input_ids, return_inner_stats=True)
    
    print(f"Max iterations: {cfg.max_inner_iters}")
    print(f"Actual iterations: {len(inner_stats)}")
    
    for i, s in enumerate(inner_stats):
        halted = s.get('halted_frac', 0)
        print(f"  Iter {i+1}: halted_frac={halted:.2%}")
    
    if inner_stats:
        ponder_cost = inner_stats[-1].get('ponder_cost', 0)
        print(f"Ponder cost: {ponder_cost:.6f}")


if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    example_basic_lm()
    example_generation()
    example_iterative_refinement()
    example_sampling_strategies()
    example_param_comparison()
    example_act_halting()
    
    print("\n" + "="*60)
    print("All PoHGPT examples completed successfully! 🎉")
    print("="*60 + "\n")

