"""
Quick A/B test with reduced steps for smoke testing.

Author: Eran Ben Artzy
Year: 2025
"""

import os
import csv
import yaml
import time
from datetime import datetime

from src.models.baseline_gpt import BaselineGPT
from src.pot.models.poh_gpt import PoHGPT
from src.pot.utils.train import Trainer
from src.pot.modules.block import PoHConfig


def quick_ab_test():
    """Run a quick A/B test with minimal steps."""
    
    # Quick config
    base_cfg = {
        "model": {"type": "baseline_gpt", "vocab_size": 1000, "d_model": 128, 
                  "n_heads": 4, "d_ff": 512, "depth": 2, "dropout": 0.1, "max_seq_len": 64},
        "train": {"batch_size": 8, "lr": 3e-4, "max_steps": 100, "eval_interval": 50, "seed": 42},
        "data": {"max_tokens": 1000, "seq_len": 64},
        "save_dir": "experiments/results/lm/quick_test/baseline"
    }
    
    poh_cfg_obj = PoHConfig(
        d_model=128, n_heads=4, d_ff=512, dropout=0.1,
        pos_encoding="absolute", max_seq_len=64,
        is_causal=True, depth=2, max_inner_iters=2,
        outer_residual=True, rezero_init=True
    )
    
    poh_cfg = {
        "model": {"type": "poh_gpt"},
        "train": {"batch_size": 8, "lr": 3e-4, "max_steps": 100, "eval_interval": 50, "seed": 42},
        "data": {"max_tokens": 1000, "seq_len": 64},
        "save_dir": "experiments/results/lm/quick_test/poh"
    }
    
    print("\n🚀 Quick Test: Training Baseline GPT (100 steps)")
    base_model = BaselineGPT(vocab_size=1000, d_model=128, n_heads=4, d_ff=512, 
                             depth=2, dropout=0.1, max_seq_len=64)
    start = time.time()
    base_trainer = Trainer(base_model, base_cfg)
    base_ppl = base_trainer.train_and_eval()
    base_time = (time.time() - start) / 60
    
    print("\n⚙️  Quick Test: Training PoH-GPT (100 steps)")
    poh_model = PoHGPT(vocab_size=1000, cfg=poh_cfg_obj)
    start = time.time()
    poh_trainer = Trainer(poh_model, poh_cfg)
    poh_ppl = poh_trainer.train_and_eval()
    poh_time = (time.time() - start) / 60
    
    delta = (base_ppl - poh_ppl) / base_ppl * 100
    
    print("\n" + "="*50)
    print("Quick A/B Test Summary")
    print("="*50)
    print(f"Baseline GPT: ppl={base_ppl:.2f}, time={base_time:.2f}m")
    print(f"PoH-GPT:      ppl={poh_ppl:.2f}, time={poh_time:.2f}m")
    print(f"Δ improvement: {delta:+.2f}%")
    print("="*50)
    print("\n✅ Quick test completed!")


if __name__ == "__main__":
    quick_ab_test()

