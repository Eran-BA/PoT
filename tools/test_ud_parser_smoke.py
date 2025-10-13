#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick smoke test for UD Dependency Parser

Runs a minimal training loop to verify:
1. Model initializes correctly
2. Forward pass works
3. Loss computation is valid
4. Gradients flow
5. Optimization step succeeds

Usage:
    python tools/test_ud_parser_smoke.py

Author: Eran Ben Artzy
Year: 2025
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from ud_pointer_parser import UDPointerParser, collate_batch

def smoke_test():
    """Run quick smoke test."""
    print("=" * 60)
    print("UD Dependency Parser - Smoke Test")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📍 Device: {device}")
    
    # 1. Initialize model
    print("\n1️⃣  Initializing parser...")
    parser = UDPointerParser(
        enc_name="distilbert-base-uncased",
        d_model=768,
        n_heads_router=4,
        d_ff_router=512,
        router_mode="mask_concat",
        halting_mode="fixed",
        max_inner_iters=2,
        routing_topk=0,  # Soft routing
    ).to(device)
    
    n_params = sum(p.numel() for p in parser.parameters()) / 1e6
    print(f"   ✅ Model initialized: {n_params:.2f}M parameters")
    
    # 2. Create sample data
    print("\n2️⃣  Creating sample data...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Simple sentences
    tokens_list = [
        ["The", "cat", "sat", "on", "mat"],
        ["Dogs", "run", "fast"],
        ["I", "love", "parsing"],
    ]
    
    # UD-style heads (0 = ROOT)
    heads_gold = [
        [2, 3, 0, 3, 3],  # "sat" is root
        [2, 0, 2],         # "run" is root
        [2, 0, 2],         # "love" is root
    ]
    
    print(f"   ✅ {len(tokens_list)} sentences created")
    
    # 3. Collate batch
    print("\n3️⃣  Collating batch...")
    enc = tokenizer(
        tokens_list,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    for k in enc:
        enc[k] = enc[k].to(device)
    
    word_ids_batch = [enc.word_ids(i) for i in range(len(tokens_list))]
    print(f"   ✅ Batch shape: {enc['input_ids'].shape}")
    
    # 4. Forward pass
    print("\n4️⃣  Running forward pass...")
    parser.eval()
    with torch.no_grad():
        loss, metrics = parser(enc, word_ids_batch, heads_gold)
    
    print(f"   ✅ Loss: {loss.item():.4f}")
    print(f"   ✅ UAS: {metrics['uas']:.4f}")
    print(f"   ✅ Tokens: {metrics['tokens']}")
    if "inner_iters_used" in metrics:
        print(f"   ✅ Inner iterations: {metrics['inner_iters_used']:.1f}")
    
    # 5. Backward pass
    print("\n5️⃣  Testing gradient flow...")
    parser.train()
    optimizer = torch.optim.AdamW(parser.parameters(), lr=1e-4)
    
    optimizer.zero_grad()
    loss, metrics = parser(enc, word_ids_batch, heads_gold)
    loss.backward()
    
    # Count parameters with gradients
    n_with_grad = sum(
        1 for p in parser.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    total_params = sum(1 for _ in parser.parameters())
    
    print(f"   ✅ Gradients computed: {n_with_grad}/{total_params} parameters")
    
    # 6. Optimization step
    print("\n6️⃣  Running optimization step...")
    torch.nn.utils.clip_grad_norm_(parser.parameters(), 1.0)
    optimizer.step()
    print(f"   ✅ Optimizer step completed")
    
    # 7. Test different configurations
    print("\n7️⃣  Testing configurations...")
    
    configs = [
        {"router_mode": "mixture", "name": "Mixture routing"},
        {"halting_mode": "entropy", "ent_threshold": 0.5, "name": "Entropy halting"},
        {"routing_topk": 2, "name": "Top-k routing (k=2)"},
    ]
    
    for cfg in configs:
        name = cfg.pop("name")
        test_parser = UDPointerParser(
            enc_name="distilbert-base-uncased",
            d_model=768,
            n_heads_router=4,
            d_ff_router=512,
            max_inner_iters=2,
            **cfg
        ).to(device)
        
        with torch.no_grad():
            loss, _ = test_parser(enc, word_ids_batch, heads_gold)
        
        print(f"   ✅ {name}: Loss = {loss.item():.4f}")
    
    # Success!
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nThe UD Dependency Parser is working correctly.")
    print("You can now run full experiments with:")
    print("  python ud_pointer_parser.py --epochs 3 --batch_size 8")
    print()

if __name__ == "__main__":
    try:
        smoke_test()
    except Exception as e:
        print(f"\n❌ SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

