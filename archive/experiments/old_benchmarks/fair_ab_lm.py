"""
Fair A/B comparison: Baseline GPT vs PoH-GPT.

Trains both models with identical hyperparameters and logs results to CSV.

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


# ---------------------------
# Helpers
# ---------------------------

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg):
    m = cfg["model"]
    if m["type"] == "baseline_gpt":
        return BaselineGPT(
            vocab_size=m["vocab_size"],
            d_model=m["d_model"],
            n_heads=m["n_heads"],
            d_ff=m["d_ff"],
            depth=m["depth"],
            dropout=m["dropout"],
            max_seq_len=m["max_seq_len"],
        )
    elif m["type"] == "poh_gpt":
        from src.pot.modules.block import PoHConfig
        
        cfg_poh = PoHConfig(
            d_model=m["d_model"],
            n_heads=m["n_heads"],
            d_ff=m["d_ff"],
            dropout=m["dropout"],
            pos_encoding=m.get("pos_encoding", "absolute"),
            max_seq_len=m["max_seq_len"],
            is_causal=True,
            depth=m["depth"],
            max_inner_iters=m.get("max_inner_iters", 1),
            outer_residual=m.get("outer_residual", False),
            rezero_init=m.get("rezero_init", False),
        )
        
        return PoHGPT(
            vocab_size=m["vocab_size"],
            cfg=cfg_poh,
        )
    else:
        raise ValueError(f"Unknown model type: {m['type']}")


def save_result_csv(row, path="experiments/results/lm/ab_results.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = ["timestamp", "model", "perplexity", "time_min", "delta_vs_baseline"]
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------
# Main A/B Experiment
# ---------------------------

def run_ab():
    cfg_base = load_yaml("experiments/configs/lm/baseline_gpt.yaml")
    cfg_poh = load_yaml("experiments/configs/lm/poh_gpt.yaml")

    base_model = build_model(cfg_base)
    poh_model = build_model(cfg_poh)

    print("\n🚀 Training Baseline GPT")
    start = time.time()
    base_trainer = Trainer(base_model, cfg_base)
    base_ppl = base_trainer.train_and_eval()
    base_time = (time.time() - start) / 60

    print("\n⚙️  Training PoH-GPT")
    start = time.time()
    poh_trainer = Trainer(poh_model, cfg_poh)
    poh_ppl = poh_trainer.train_and_eval()
    poh_time = (time.time() - start) / 60

    delta = (base_ppl - poh_ppl) / base_ppl * 100

    print("\n==============================")
    print("A/B Comparison Summary")
    print("==============================")
    print(f"Baseline GPT: ppl={base_ppl:.2f}, time={base_time:.1f} min")
    print(f"PoH-GPT:      ppl={poh_ppl:.2f}, time={poh_time:.1f} min")
    print(f"Δ improvement: {delta:+.2f}%")
    print("==============================\n")

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_result_csv({
        "timestamp": timestamp,
        "model": "BaselineGPT",
        "perplexity": f"{base_ppl:.2f}",
        "time_min": f"{base_time:.2f}",
        "delta_vs_baseline": "0.00"
    })
    save_result_csv({
        "timestamp": timestamp,
        "model": "PoHGPT",
        "perplexity": f"{poh_ppl:.2f}",
        "time_min": f"{poh_time:.2f}",
        "delta_vs_baseline": f"{delta:.2f}"
    })

    print(f"✅ Results saved to: experiments/results/lm/ab_results.csv")


if __name__ == "__main__":
    run_ab()

