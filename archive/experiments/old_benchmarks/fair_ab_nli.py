"""
Fair A/B comparison: BERT vs PoH for Natural Language Inference.

Trains both models with identical hyperparameters and logs results to CSV.

Author: Eran Ben Artzy
Year: 2025
"""

import os
import csv
import yaml
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm
from datetime import datetime

from src.models.bert_baseline import BERTForNLI
from src.pot.models.poh_nli import PoHForNLI
from src.pot.tasks.nli import NLIDataLoader, NLIMetrics, create_pair_sequence


# Simple linear warmup scheduler
class LinearWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.step_count = 0
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self):
        self.step_count += 1
        if self.step_count < self.warmup_steps:
            lr = self.base_lr * self.step_count / self.warmup_steps
        else:
            # Linear decay
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.base_lr * (1 - progress)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class NLITrainer:
    """Trainer for NLI models."""
    
    def __init__(self, model, cfg, device=None):
        self.model = model
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Training config
        train_cfg = cfg.get("train", {})
        self.batch_size = int(train_cfg.get("batch_size", 32))
        self.lr = float(train_cfg.get("lr", 2e-5))
        self.weight_decay = float(train_cfg.get("weight_decay", 0.01))
        self.max_steps = int(train_cfg.get("max_steps", 10000))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.eval_interval = int(train_cfg.get("eval_interval", 500))
        self.warmup_steps = int(train_cfg.get("warmup_steps", 1000))
        
        # Data config
        data_cfg = cfg.get("data", {})
        self.use_synthetic = data_cfg.get("synthetic", True)
        self.max_length = data_cfg.get("max_length", 128)
        
        # Setup optimizer + scheduler
        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = LinearWarmupScheduler(self.opt, self.warmup_steps, self.max_steps)
        self.criterion = nn.CrossEntropyLoss()
        
        # Data loader
        self.data_loader_helper = NLIDataLoader(max_length=self.max_length)
        
        # Reproducibility
        seed = train_cfg.get("seed", 42)
        torch.manual_seed(seed)
        
        self.save_dir = cfg.get("save_dir", "experiments/results/nli/")
        os.makedirs(self.save_dir, exist_ok=True)
    
    def _create_batch(self):
        """Create a batch (synthetic for now)."""
        return self.data_loader_helper.create_synthetic_batch(
            batch_size=self.batch_size,
            vocab_size=self.cfg['model']['vocab_size']
        )
    
    def _evaluate(self):
        """Quick evaluation."""
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        count = 0
        
        with torch.no_grad():
            for _ in range(10):  # Evaluate on 10 batches
                batch = self._create_batch().to(self.device)
                
                # Create concatenated sequence
                input_ids, attention_mask = create_pair_sequence(
                    batch.premise_ids,
                    batch.hypothesis_ids,
                )
                
                # Forward
                output = self.model(input_ids, attention_mask)
                logits = output[0] if isinstance(output, tuple) else output
                
                # Loss
                loss = self.criterion(logits, batch.labels)
                total_loss += loss.item()
                
                # Accuracy
                metrics = NLIMetrics.compute(logits, batch.labels)
                total_acc += metrics['accuracy']
                count += 1
        
        return total_loss / count, total_acc / count
    
    def train(self):
        """Main training loop."""
        print(f"Training {self.cfg['model']['type']}...")
        print(f"  Device: {self.device}")
        print(f"  Parameters: {self.model.count_parameters() / 1e6:.2f}M")
        print(f"  Max steps: {self.max_steps}")
        
        self.model.train()
        step = 0
        best_acc = 0.0
        
        pbar = tqdm(total=self.max_steps, desc=f"Training {self.cfg['model']['type']}")
        
        while step < self.max_steps:
            # Create batch
            batch = self._create_batch().to(self.device)
            
            # Create concatenated sequence
            input_ids, attention_mask = create_pair_sequence(
                batch.premise_ids,
                batch.hypothesis_ids,
            )
            
            # Forward
            output = self.model(input_ids, attention_mask)
            logits = output[0] if isinstance(output, tuple) else output
            
            # Loss
            loss = self.criterion(logits, batch.labels)
            
            # Backward
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
                val_loss, val_acc = self._evaluate()
                print(f"\n[step {step}] val_loss={val_loss:.3f}, val_acc={val_acc:.3f}")
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    self._save_checkpoint(step)
                
                self.model.train()
        
        pbar.close()
        
        # Final evaluation
        val_loss, val_acc = self._evaluate()
        print(f"Final validation accuracy: {val_acc:.3f}")
        
        return val_acc
    
    def _save_checkpoint(self, step):
        """Save model checkpoint."""
        path = os.path.join(self.save_dir, f"{self.cfg['model']['type']}_step{step}.pt")
        torch.save(self.model.state_dict(), path)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg):
    m = cfg["model"]
    if m["type"] == "bert_nli":
        return BERTForNLI(
            vocab_size=m["vocab_size"],
            d_model=m["d_model"],
            n_heads=m["n_heads"],
            d_ff=m["d_ff"],
            depth=m["depth"],
            dropout=m["dropout"],
            max_seq_len=m["max_seq_len"],
        )
    elif m["type"] == "poh_nli":
        return PoHForNLI(
            vocab_size=m["vocab_size"],
            d_model=m["d_model"],
            n_heads=m["n_heads"],
            d_ff=m["d_ff"],
            depth=m["depth"],
            dropout=m["dropout"],
            max_seq_len=m["max_seq_len"],
            max_inner_iters=m.get("max_inner_iters", 1),
            route_mode=m.get("route_mode", "soft"),
            route_topk=m.get("route_topk"),
            outer_residual=m.get("outer_residual", False),
            rezero_init=m.get("rezero_init", False),
            share_router=m.get("share_router", True),
        )
    else:
        raise ValueError(f"Unknown model type: {m['type']}")


def save_result_csv(row, path="experiments/results/nli/ab_results.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = ["timestamp", "model", "accuracy", "time_min", "delta_vs_baseline"]
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_ab():
    """Run A/B comparison."""
    cfg_bert = load_yaml("experiments/configs/nli/bert_baseline.yaml")
    cfg_poh = load_yaml("experiments/configs/nli/poh.yaml")
    
    bert_model = build_model(cfg_bert)
    poh_model = build_model(cfg_poh)
    
    print("\n🚀 Training PoH for NLI (Novel Architecture)")
    start = time.time()
    poh_trainer = NLITrainer(poh_model, cfg_poh)
    poh_acc = poh_trainer.train()
    poh_time = (time.time() - start) / 60
    
    print("\n⚙️  Training BERT Baseline for NLI")
    start = time.time()
    bert_trainer = NLITrainer(bert_model, cfg_bert)
    bert_acc = bert_trainer.train()
    bert_time = (time.time() - start) / 60
    
    delta = (poh_acc - bert_acc) / bert_acc * 100
    
    print("\n" + "="*50)
    print("A/B Comparison Summary: NLI")
    print("="*50)
    print(f"PoH:           acc={poh_acc:.3f}, time={poh_time:.1f}m")
    print(f"BERT Baseline: acc={bert_acc:.3f}, time={bert_time:.1f}m")
    print(f"Δ improvement: {delta:+.2f}%")
    print("="*50)
    
    # Save results (PoH first, then baseline)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_result_csv({
        "timestamp": timestamp,
        "model": "PoH",
        "accuracy": f"{poh_acc:.3f}",
        "time_min": f"{poh_time:.2f}",
        "delta_vs_baseline": f"{delta:.2f}"
    })
    save_result_csv({
        "timestamp": timestamp,
        "model": "BERT",
        "accuracy": f"{bert_acc:.3f}",
        "time_min": f"{bert_time:.2f}",
        "delta_vs_baseline": "0.00"
    })
    
    print(f"\n✅ Results saved to: experiments/results/nli/ab_results.csv")


if __name__ == "__main__":
    run_ab()

