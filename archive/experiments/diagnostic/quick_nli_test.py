"""
Quick NLI test with reduced steps for smoke testing.

Author: Eran Ben Artzy
Year: 2025
"""

import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from src.models.bert_baseline import BERTForNLI
from src.pot.models.poh_nli import PoHForNLI
from src.pot.tasks.nli import NLIDataLoader, NLIMetrics, create_pair_sequence


class QuickNLITrainer:
    """Quick trainer for smoke testing."""
    
    def __init__(self, model, model_name, device=None):
        self.model = model
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
        self.data_loader = NLIDataLoader(max_length=128)
    
    def train(self, steps=100, batch_size=16):
        """Quick training."""
        print(f"\n{'='*50}")
        print(f"Training {self.model_name}")
        print(f"{'='*50}")
        print(f"  Device: {self.device}")
        print(f"  Parameters: {self.model.count_parameters() / 1e6:.2f}M")
        print(f"  Steps: {steps}")
        
        self.model.train()
        pbar = tqdm(total=steps, desc=self.model_name)
        
        for step in range(steps):
            # Create batch
            batch = self.data_loader.create_synthetic_batch(batch_size, vocab_size=30522)
            batch = batch.to(self.device)
            
            # Create input sequence
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
            self.opt.step()
            
            pbar.update(1)
            if (step + 1) % 25 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.3f}"})
        
        pbar.close()
        
        # Evaluate
        self.model.eval()
        total_acc = 0.0
        
        with torch.no_grad():
            for _ in range(10):
                batch = self.data_loader.create_synthetic_batch(batch_size, vocab_size=30522)
                batch = batch.to(self.device)
                
                input_ids, attention_mask = create_pair_sequence(
                    batch.premise_ids,
                    batch.hypothesis_ids,
                )
                
                output = self.model(input_ids, attention_mask)
                logits = output[0] if isinstance(output, tuple) else output
                
                metrics = NLIMetrics.compute(logits, batch.labels)
                total_acc += metrics['accuracy']
        
        acc = total_acc / 10
        print(f"Validation accuracy: {acc:.3f}")
        
        return acc


def main():
    print("\n🧠 Quick NLI Benchmark: BERT vs PoH")
    print("="*50)
    
    # Small models for quick testing
    bert = BERTForNLI(
        vocab_size=30522,
        d_model=256,
        n_heads=8,
        d_ff=1024,
        depth=4,
        dropout=0.1,
        max_seq_len=128,
    )
    
    poh = PoHForNLI(
        vocab_size=30522,
        d_model=256,
        n_heads=8,
        d_ff=1024,
        depth=4,
        dropout=0.1,
        max_seq_len=128,
        max_inner_iters=12,  # Optimal from diminishing returns analysis
        route_mode="soft",
        outer_residual=True,
        rezero_init=True,
        share_router=True,
    )
    
    # Train PoH first (novel architecture)
    start = time.time()
    poh_trainer = QuickNLITrainer(poh, "PoH-Small")
    poh_acc = poh_trainer.train(steps=100, batch_size=16)
    poh_time = time.time() - start
    
    # Train BERT baseline
    start = time.time()
    bert_trainer = QuickNLITrainer(bert, "BERT-Small")
    bert_acc = bert_trainer.train(steps=100, batch_size=16)
    bert_time = time.time() - start
    
    # Summary
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"PoH-Small:  acc={poh_acc:.3f}, time={poh_time:.1f}s")
    print(f"BERT-Small: acc={bert_acc:.3f}, time={bert_time:.1f}s")
    
    delta = (poh_acc - bert_acc) / bert_acc * 100
    print(f"Δ improvement: {delta:+.2f}%")
    print("="*50)
    
    print("\n✅ Quick test completed!")


if __name__ == "__main__":
    main()

