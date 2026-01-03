"""
Unified training utilities for PoH models.

Includes Trainer class with synthetic dataset, optimizer setup,
and fair A/B comparison support.

Author: Eran Ben Artzy
Year: 2025
"""

import os
import time
import json
import math
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


# ---------------------------
# Synthetic dataset for testing (Tiny LM)
# ---------------------------
class SyntheticLMDataset(Dataset):
    """Tiny synthetic dataset for quick LM testing."""
    
    def __init__(self, vocab_size=32000, seq_len=128, size=10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randint(0, self.vocab_size, (self.seq_len,))
        y = torch.roll(x, shifts=-1, dims=0)  # next-token prediction
        return x, y


# ---------------------------
# Trainer Class
# ---------------------------
class Trainer:
    """Unified trainer for both BaselineGPT and PoHGPT."""
    
    def __init__(self, model, cfg, device=None):
        self.model = model
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # default training hyperparams
        train_cfg = cfg.get("train", {})
        self.batch_size = train_cfg.get("batch_size", 16)
        self.lr = train_cfg.get("lr", 3e-4)
        self.weight_decay = train_cfg.get("weight_decay", 0.01)
        self.max_steps = train_cfg.get("max_steps", 1000)
        self.grad_clip = train_cfg.get("grad_clip", 1.0)
        self.eval_interval = train_cfg.get("eval_interval", 200)

        # setup optimizer + scheduler
        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = CosineAnnealingLR(self.opt, T_max=self.max_steps)
        self.criterion = nn.CrossEntropyLoss()

        # synthetic data
        data_cfg = cfg.get("data", {})
        vocab_size = data_cfg.get("vocab_size", getattr(model, "vocab_size", 32000))
        seq_len = data_cfg.get("seq_len", 128)
        dataset = SyntheticLMDataset(vocab_size, seq_len, size=data_cfg.get("max_tokens", 10000))
        self.loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # reproducibility
        seed = train_cfg.get("seed", 42)
        torch.manual_seed(seed)
        random.seed(seed)

        self.save_dir = cfg.get("save_dir", "experiments/results/lm/")
        os.makedirs(self.save_dir, exist_ok=True)

    def _save_config(self):
        config_path = os.path.join(self.save_dir, f"{self.cfg['model']['type']}_config.json")
        with open(config_path, "w") as f:
            json.dump(self.cfg, f, indent=2)

    def _save_checkpoint(self, step):
        path = os.path.join(self.save_dir, f"{self.cfg['model']['type']}_step{step}.pt")
        torch.save(self.model.state_dict(), path)

    def _evaluate(self):
        self.model.eval()
        total_loss, count = 0.0, 0
        with torch.no_grad():
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                # Handle both single logits and (logits, stats) tuple
                logits = output[0] if isinstance(output, tuple) else output
                loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                total_loss += loss.item()
                count += 1
                if count >= 10:  # Quick eval on 10 batches
                    break
        ppl = math.exp(total_loss / count)
        return total_loss / count, ppl

    def train_and_eval(self):
        """Main training loop with periodic evaluation."""
        self._save_config()
        best_val = float("inf")
        self.model.train()
        step = 0

        pbar = tqdm(total=self.max_steps, desc=f"Training {self.cfg['model']['type']}")
        while step < self.max_steps:
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                # Handle both single logits and (logits, stats) tuple
                logits = output[0] if isinstance(output, tuple) else output
                loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.opt.step()
                self.scheduler.step()

                step += 1
                pbar.update(1)
                
                if step % 50 == 0:
                    pbar.set_postfix({"loss": f"{loss.item():.3f}"})
                
                if step % self.eval_interval == 0 or step == self.max_steps:
                    val_loss, val_ppl = self._evaluate()
                    print(f"\n[step {step}] val_loss={val_loss:.3f}, val_ppl={val_ppl:.2f}")
                    if val_loss < best_val:
                        best_val = val_loss
                        self._save_checkpoint(step)
                    self.model.train()
                
                if step >= self.max_steps:
                    break
        
        pbar.close()
        val_loss, val_ppl = self._evaluate()
        print(f"Final validation perplexity: {val_ppl:.2f}")
        return val_ppl

    @staticmethod
    def compare(model_a, model_b, dataset="synthetic_lm"):
        """Convenience A/B test runner."""
        cfg_template = {
            "train": {"max_steps": 200, "eval_interval": 50, "seed": 42},
            "data": {"dataset": dataset, "max_tokens": 2000},
        }
        t1 = Trainer(model_a, {"model": {"type": "baseline_gpt"}, **cfg_template, "save_dir": "experiments/results/lm/baseline"})
        t2 = Trainer(model_b, {"model": {"type": "poh_gpt"}, **cfg_template, "save_dir": "experiments/results/lm/poh"})
        
        print("Running baseline GPT...")
        ppl_a = t1.train_and_eval()
        
        print("\nRunning PoH-GPT...")
        ppl_b = t2.train_and_eval()
        
        print(f"\nBaseline GPT ppl={ppl_a:.2f}, PoH-GPT ppl={ppl_b:.2f}, Δ={(ppl_a - ppl_b)/ppl_a*100:.2f}%")
        return ppl_a, ppl_b

