#!/usr/bin/env python3
"""
Diagnostic script to understand NLI training issues.

This script will:
1. Train with different learning rates
2. Monitor gradient flow
3. Check if model is learning basic patterns
4. Compare against a simple baseline
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time

from src.pot.modules import PoHConfig, PoHStack, IterRefiner

# Import SNLI dataset
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("⚠️  Install datasets: pip install datasets")
    sys.exit(1)


class SNLIDataset(Dataset):
    """Real SNLI dataset."""
    def __init__(self, split='train', max_samples=None):
        print(f"Loading SNLI {split}...")
        self.dataset = load_dataset('snli', split=split)
        self.dataset = self.dataset.filter(lambda x: x['label'] != -1)
        
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        
        print(f"  Loaded {len(self.dataset)} examples")
    
    def _tokenize(self, text, vocab_size=10000, max_len=32):
        words = text.lower().split()[:max_len]
        tokens = [hash(w) % vocab_size for w in words]
        tokens = tokens + [0] * (max_len - len(tokens))
        return torch.tensor(tokens[:max_len], dtype=torch.long)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        premise = self._tokenize(item['premise'])
        hypothesis = self._tokenize(item['hypothesis'])
        label = item['label']
        return premise, hypothesis, label


class SimpleBaseline(nn.Module):
    """Simple LSTM baseline for comparison."""
    def __init__(self, vocab_size=10000, d_model=256, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=2, batch_first=True, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3)
        )
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        self.embed.weight.data[0].zero_()
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, premise, hypothesis):
        p_emb = self.embed(premise)
        h_emb = self.embed(hypothesis)
        
        _, (p_hidden, _) = self.lstm(p_emb)
        _, (h_hidden, _) = self.lstm(h_emb)
        
        p_pooled = p_hidden[-1]  # Last layer
        h_pooled = h_hidden[-1]
        
        combined = torch.cat([p_pooled, h_pooled, p_pooled * h_pooled], dim=-1)
        logits = self.classifier(combined)
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class PoHForNLI(nn.Module):
    """PoH model for NLI."""
    def __init__(self, vocab_size=10000, d_model=256, n_heads=8, d_ff=1024, 
                 depth=4, R=4, T=4, dropout=0.1, max_seq_len=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        cfg = PoHConfig(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            route_mode="soft",
            share_router=True,
            pos_encoding="absolute",
            max_seq_len=max_seq_len,
        )
        
        stack = PoHStack(cfg, depth=depth)
        
        # Set HRM period
        for block in stack.blocks:
            if hasattr(block, 'router'):
                if hasattr(block.router, 'hrm_controller'):
                    block.router.hrm_controller.T = T
                elif hasattr(block.router, 'T'):
                    block.router.T = T
        
        self.refiner = IterRefiner(stack, max_inner_iters=R, outer_residual=True, rezero_init=True)
        
        self.pooler = nn.Linear(d_model, d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        self.embed.weight.data[0].zero_()
        for module in [self.pooler] + list(self.classifier.modules()):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, premise, hypothesis):
        p_emb = self.embed(premise)
        h_emb = self.embed(hypothesis)
        
        p_out, _ = self.refiner(p_emb)
        h_out, _ = self.refiner(h_emb)
        
        p_mask = (premise != 0).float().unsqueeze(-1)
        h_mask = (hypothesis != 0).float().unsqueeze(-1)
        
        p_pooled = (p_out * p_mask).sum(dim=1) / (p_mask.sum(dim=1) + 1e-9)
        h_pooled = (h_out * h_mask).sum(dim=1) / (h_mask.sum(dim=1) + 1e-9)
        
        p_pooled = self.pooler(p_pooled)
        h_pooled = self.pooler(h_pooled)
        
        combined = torch.cat([p_pooled, h_pooled, p_pooled * h_pooled], dim=-1)
        logits = self.classifier(combined)
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


def check_gradient_flow(model, premise, hypothesis, labels, criterion):
    """Check if gradients are flowing properly."""
    model.zero_grad()
    logits = model(premise, hypothesis)
    loss = criterion(logits, labels)
    loss.backward()
    
    # Collect gradient statistics
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    
    return grad_norms


def train_with_lr(model, train_loader, val_loader, lr, max_steps=200, device='cpu'):
    """Train model with specific LR and return validation accuracy."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    step = 0
    pbar = tqdm(total=max_steps, desc=f"LR={lr:.0e}")
    
    while step < max_steps:
        for premise, hypothesis, labels in train_loader:
            if step >= max_steps:
                break
            
            premise = premise.to(device)
            hypothesis = hypothesis.to(device)
            labels = labels.to(device)
            
            logits = model(premise, hypothesis)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            step += 1
            pbar.update(1)
            pbar.set_postfix({'loss': f"{loss.item():.3f}"})
    
    pbar.close()
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for premise, hypothesis, labels in val_loader:
            premise = premise.to(device)
            hypothesis = hypothesis.to(device)
            labels = labels.to(device)
            
            logits = model(premise, hypothesis)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    acc = correct / total
    return acc


def main():
    print("\n" + "="*60)
    print("NLI Training Diagnostic")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load small datasets for quick testing
    train_dataset = SNLIDataset('train', max_samples=5000)
    val_dataset = SNLIDataset('validation', max_samples=1000)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=0)
    
    # Test 1: Gradient flow check
    print("\n" + "="*60)
    print("TEST 1: Gradient Flow Check")
    print("="*60)
    
    poh_model = PoHForNLI(vocab_size=10000, d_model=256, n_heads=8, d_ff=1024, 
                          depth=4, R=4, T=4, dropout=0.1)
    poh_model = poh_model.to(device)
    
    # Get one batch
    premise, hypothesis, labels = next(iter(train_loader))
    premise = premise.to(device)
    hypothesis = hypothesis.to(device)
    labels = labels.to(device)
    
    criterion = nn.CrossEntropyLoss()
    grad_norms = check_gradient_flow(poh_model, premise, hypothesis, labels, criterion)
    
    print("\nGradient norms (top 10 largest):")
    sorted_grads = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)[:10]
    for name, norm in sorted_grads:
        print(f"  {name}: {norm:.6f}")
    
    # Check if any gradients are zero or exploding
    zero_grads = sum(1 for norm in grad_norms.values() if norm < 1e-8)
    large_grads = sum(1 for norm in grad_norms.values() if norm > 100)
    
    print(f"\nZero gradients: {zero_grads}/{len(grad_norms)}")
    print(f"Large gradients (>100): {large_grads}/{len(grad_norms)}")
    
    # Test 2: Learning rate sweep
    print("\n" + "="*60)
    print("TEST 2: Learning Rate Sweep")
    print("="*60)
    
    learning_rates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    results = {}
    
    for lr in learning_rates:
        print(f"\n--- Testing LR = {lr:.0e} ---")
        
        # Fresh model
        model = PoHForNLI(vocab_size=10000, d_model=256, n_heads=8, d_ff=1024, 
                         depth=4, R=4, T=4, dropout=0.1)
        
        acc = train_with_lr(model, train_loader, val_loader, lr=lr, max_steps=200, device=device)
        results[lr] = acc
        print(f"Final validation accuracy: {acc:.3f}")
    
    print("\n" + "="*60)
    print("LR Sweep Results:")
    print("="*60)
    for lr, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  LR={lr:.0e}: {acc:.3f}")
    
    best_lr = max(results.items(), key=lambda x: x[1])[0]
    print(f"\n🏆 Best LR: {best_lr:.0e} (accuracy: {results[best_lr]:.3f})")
    
    # Test 3: PoH vs Simple Baseline
    print("\n" + "="*60)
    print("TEST 3: PoH vs Simple LSTM Baseline")
    print("="*60)
    
    print("\n--- Training Simple LSTM Baseline ---")
    baseline = SimpleBaseline(vocab_size=10000, d_model=256, dropout=0.1)
    print(f"Baseline params: {baseline.count_parameters()/1e6:.2f}M")
    baseline_acc = train_with_lr(baseline, train_loader, val_loader, lr=best_lr, 
                                  max_steps=200, device=device)
    
    print(f"\n--- Training PoH (R=4, T=4) ---")
    poh = PoHForNLI(vocab_size=10000, d_model=256, n_heads=8, d_ff=1024, 
                    depth=4, R=4, T=4, dropout=0.1)
    print(f"PoH params: {poh.count_parameters()/1e6:.2f}M")
    poh_acc = train_with_lr(poh, train_loader, val_loader, lr=best_lr, 
                           max_steps=200, device=device)
    
    print("\n" + "="*60)
    print("Final Comparison:")
    print("="*60)
    print(f"LSTM Baseline: {baseline_acc:.3f}")
    print(f"PoH (R=4, T=4): {poh_acc:.3f}")
    print(f"Improvement: {(poh_acc - baseline_acc):.3f} ({(poh_acc - baseline_acc)/baseline_acc*100:+.1f}%)")
    
    # Test 4: Different R values
    print("\n" + "="*60)
    print("TEST 4: Effect of Refinement Steps (R)")
    print("="*60)
    
    R_values = [1, 2, 4, 8]
    R_results = {}
    
    for R in R_values:
        print(f"\n--- Testing R={R} ---")
        model = PoHForNLI(vocab_size=10000, d_model=256, n_heads=8, d_ff=1024, 
                         depth=4, R=R, T=4, dropout=0.1)
        acc = train_with_lr(model, train_loader, val_loader, lr=best_lr, 
                           max_steps=200, device=device)
        R_results[R] = acc
        print(f"R={R}: {acc:.3f}")
    
    print("\n" + "="*60)
    print("Refinement Steps Analysis:")
    print("="*60)
    for R, acc in sorted(R_results.items()):
        print(f"  R={R}: {acc:.3f}")
    
    best_R = max(R_results.items(), key=lambda x: x[1])[0]
    print(f"\n🏆 Best R: {best_R} (accuracy: {R_results[best_R]:.3f})")
    
    print("\n" + "="*60)
    print("✅ Diagnostic Complete!")
    print("="*60)
    print(f"\nRecommendations:")
    print(f"  - Best learning rate: {best_lr:.0e}")
    print(f"  - Best refinement steps: R={best_R}")
    print(f"  - LSTM baseline: {baseline_acc:.3f}")
    print(f"  - PoH with best R: {R_results[best_R]:.3f}")
    print()


if __name__ == "__main__":
    main()

