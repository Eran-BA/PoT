"""
Real NLI Benchmark: BERT vs PoH on SNLI/MultiNLI.

Full-scale benchmark with real datasets and proper training.

Author: Eran Ben Artzy
Year: 2025
"""

import os
import csv
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from datetime import datetime
from typing import Dict

from src.models.bert_baseline import BERTForNLI
from src.pot.models.poh_nli import PoHForNLI
from src.pot.tasks.nli_dataset import create_nli_dataloader


class LinearWarmupScheduler:
    """Linear warmup + linear decay scheduler."""
    
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.step_count = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        self.step_count += 1
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            
            if self.step_count < self.warmup_steps:
                # Linear warmup
                lr = base_lr * self.step_count / self.warmup_steps
            else:
                # Linear decay
                progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                lr = base_lr * (1 - progress * (1 - self.min_lr_ratio))
            
            param_group['lr'] = max(lr, base_lr * self.min_lr_ratio)
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


class RealNLITrainer:
    """Trainer for real NLI benchmarks."""
    
    def __init__(
        self,
        model,
        model_name: str,
        dataset_name: str = "snli",
        device: str = None,
        # Training config
        batch_size: int = 32,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 20000,
        eval_interval: int = 1000,
        grad_clip: float = 1.0,
        # Data config
        max_train_samples: int = None,
        max_eval_samples: int = 5000,
        max_length: int = 128,
        # Logging
        save_dir: str = "experiments/results/real_nli",
    ):
        self.model = model
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Training config
        self.batch_size = batch_size
        self.lr = lr
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.grad_clip = grad_clip
        
        # Setup optimizer & scheduler
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = LinearWarmupScheduler(self.optimizer, warmup_steps, max_steps)
        self.criterion = nn.CrossEntropyLoss()
        
        # Data loaders
        print(f"\n{'='*60}")
        print(f"Loading {dataset_name.upper()} dataset...")
        print(f"{'='*60}")
        
        self.train_loader = create_nli_dataloader(
            dataset_name=dataset_name,
            split="train",
            batch_size=batch_size,
            max_samples=max_train_samples,
            max_length=max_length,
            shuffle=True,
        )
        
        val_split = "validation" if dataset_name == "snli" else "validation_matched"
        self.val_loader = create_nli_dataloader(
            dataset_name=dataset_name,
            split=val_split,
            batch_size=batch_size,
            max_samples=max_eval_samples,
            max_length=max_length,
            shuffle=False,
        )
        
        # Logging
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Metrics tracking
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_accs = []
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward
        output = self.model(input_ids, attention_mask)
        logits = output[0] if isinstance(output, tuple) else output
        
        # Loss
        loss = self.criterion(logits, labels)
        
        # Backward
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # Per-class accuracy
        class_correct = [0, 0, 0]
        class_total = [0, 0, 0]
        
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward
            output = self.model(input_ids, attention_mask)
            logits = output[0] if isinstance(output, tuple) else output
            
            # Loss
            loss = self.criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            
            # Accuracy
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            # Per-class accuracy
            for i in range(3):
                mask = labels == i
                if mask.sum() > 0:
                    class_correct[i] += (preds[mask] == labels[mask]).sum().item()
                    class_total[i] += mask.sum().item()
        
        metrics = {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples,
            'acc_entailment': class_correct[0] / max(1, class_total[0]),
            'acc_neutral': class_correct[1] / max(1, class_total[1]),
            'acc_contradiction': class_correct[2] / max(1, class_total[2]),
        }
        
        return metrics
    
    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"{'='*60}")
        print(f"  Device: {self.device}")
        print(f"  Parameters: {self.model.count_parameters() / 1e6:.2f}M")
        print(f"  Max steps: {self.max_steps:,}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.lr}")
        print(f"  Train batches per epoch: {len(self.train_loader)}")
        print(f"  Eval batches: {len(self.val_loader)}")
        
        step = 0
        epoch = 0
        pbar = tqdm(total=self.max_steps, desc=f"Training {self.model_name}")
        
        start_time = time.time()
        
        while step < self.max_steps:
            epoch += 1
            
            for batch in self.train_loader:
                loss = self.train_step(batch)
                self.train_losses.append(loss)
                
                step += 1
                pbar.update(1)
                pbar.set_postfix({'loss': f'{loss:.3f}', 'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'})
                
                # Evaluation
                if step % self.eval_interval == 0 or step == self.max_steps:
                    metrics = self.evaluate()
                    self.val_accs.append(metrics['accuracy'])
                    
                    print(f"\n[Step {step:,}/{self.max_steps:,}] Evaluation:")
                    print(f"  Loss: {metrics['loss']:.4f}")
                    print(f"  Accuracy: {metrics['accuracy']:.4f}")
                    print(f"  Entailment: {metrics['acc_entailment']:.4f}")
                    print(f"  Neutral: {metrics['acc_neutral']:.4f}")
                    print(f"  Contradiction: {metrics['acc_contradiction']:.4f}")
                    
                    # Save best model
                    if metrics['accuracy'] > self.best_val_acc:
                        self.best_val_acc = metrics['accuracy']
                        self.save_checkpoint(step, metrics)
                        print(f"  ✅ New best accuracy: {self.best_val_acc:.4f}")
                
                if step >= self.max_steps:
                    break
        
        pbar.close()
        
        total_time = time.time() - start_time
        
        # Final evaluation
        final_metrics = self.evaluate()
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"  Total time: {total_time / 60:.1f} minutes")
        print(f"  Best val accuracy: {self.best_val_acc:.4f}")
        print(f"  Final val accuracy: {final_metrics['accuracy']:.4f}")
        
        return {
            'best_acc': self.best_val_acc,
            'final_acc': final_metrics['accuracy'],
            'time_minutes': total_time / 60,
        }
    
    def save_checkpoint(self, step: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        path = os.path.join(self.save_dir, f"{self.model_name}_best.pt")
        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }, path)


def save_results_csv(results: Dict, path: str = "experiments/results/real_nli/benchmark_results.csv"):
    """Save benchmark results to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    header = ["timestamp", "dataset", "model", "best_acc", "final_acc", "time_min", "delta_vs_baseline"]
    file_exists = os.path.exists(path)
    
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)


def run_benchmark(
    dataset_name: str = "snli",
    max_train_samples: int = None,
    max_steps: int = 20000,
    batch_size: int = 32,
):
    """Run full BERT vs PoH benchmark."""
    
    print("\n" + "="*60)
    print(f"🧠 Real NLI Benchmark: BERT vs PoH on {dataset_name.upper()}")
    print("="*60)
    
    # Build models
    print("\nBuilding models...")
    
    bert = BERTForNLI(
        vocab_size=30522,
        d_model=768,
        n_heads=12,
        d_ff=3072,
        depth=12,
        dropout=0.1,
        max_seq_len=128,
    )
    
    poh = PoHForNLI(
        vocab_size=30522,
        d_model=768,
        n_heads=12,
        d_ff=3072,
        depth=12,
        dropout=0.1,
        max_seq_len=128,
        max_inner_iters=3,
        route_mode="soft",
        outer_residual=True,
        rezero_init=True,
        share_router=True,
    )
    
    print(f"  BERT parameters: {bert.count_parameters() / 1e6:.2f}M")
    print(f"  PoH parameters: {poh.count_parameters() / 1e6:.2f}M")
    
    # Train BERT
    bert_trainer = RealNLITrainer(
        bert,
        model_name="BERT",
        dataset_name=dataset_name,
        batch_size=batch_size,
        max_steps=max_steps,
        max_train_samples=max_train_samples,
    )
    bert_results = bert_trainer.train()
    
    # Train PoH
    poh_trainer = RealNLITrainer(
        poh,
        model_name="PoH",
        dataset_name=dataset_name,
        batch_size=batch_size,
        max_steps=max_steps,
        max_train_samples=max_train_samples,
    )
    poh_results = poh_trainer.train()
    
    # Summary
    print("\n" + "="*60)
    print("📊 BENCHMARK RESULTS")
    print("="*60)
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Training steps: {max_steps:,}")
    print(f"Batch size: {batch_size}")
    print()
    print(f"BERT:")
    print(f"  Best accuracy: {bert_results['best_acc']:.4f}")
    print(f"  Final accuracy: {bert_results['final_acc']:.4f}")
    print(f"  Time: {bert_results['time_minutes']:.1f} minutes")
    print()
    print(f"PoH:")
    print(f"  Best accuracy: {poh_results['best_acc']:.4f}")
    print(f"  Final accuracy: {poh_results['final_acc']:.4f}")
    print(f"  Time: {poh_results['time_minutes']:.1f} minutes")
    print()
    
    delta = (poh_results['best_acc'] - bert_results['best_acc']) / bert_results['best_acc'] * 100
    print(f"Δ PoH improvement: {delta:+.2f}%")
    print("="*60)
    
    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    save_results_csv({
        'timestamp': timestamp,
        'dataset': dataset_name,
        'model': 'BERT',
        'best_acc': f"{bert_results['best_acc']:.4f}",
        'final_acc': f"{bert_results['final_acc']:.4f}",
        'time_min': f"{bert_results['time_minutes']:.2f}",
        'delta_vs_baseline': "0.00"
    })
    
    save_results_csv({
        'timestamp': timestamp,
        'dataset': dataset_name,
        'model': 'PoH',
        'best_acc': f"{poh_results['best_acc']:.4f}",
        'final_acc': f"{poh_results['final_acc']:.4f}",
        'time_min': f"{poh_results['time_minutes']:.2f}",
        'delta_vs_baseline': f"{delta:.2f}"
    })
    
    print(f"\n✅ Results saved to: experiments/results/real_nli/benchmark_results.csv")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Real NLI Benchmark")
    parser.add_argument("--dataset", type=str, default="snli", choices=["snli", "mnli"],
                        help="Dataset to use")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Limit training samples (for quick testing)")
    parser.add_argument("--max_steps", type=int, default=20000,
                        help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    
    args = parser.parse_args()
    
    run_benchmark(
        dataset_name=args.dataset,
        max_train_samples=args.max_train_samples,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
    )

