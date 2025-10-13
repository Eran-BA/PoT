#!/usr/bin/env python3
"""
A/B Test: PoH vs BERT on NLI

Fair comparison with matched parameters and optimal hyperparameters.

PoH advantages:
- Dynamic routing via HRM controller
- Multi-step refinement (R iterations)
- Adaptive computation

BERT advantages:
- Battle-tested architecture
- Simpler design
- Industry standard
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse

from src.pot.modules import PoHConfig, PoHStack, IterRefiner

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("⚠️  Install datasets: pip install datasets")
    sys.exit(1)


# ========== Dataset ==========

class SNLIDataset(Dataset):
    """SNLI dataset from Hugging Face."""
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


# ========== BERT Baseline ==========

class BERTForNLI(nn.Module):
    """
    Standard BERT architecture for NLI.
    
    Architecture:
    - Token embeddings + positional embeddings
    - N transformer encoder layers (standard attention)
    - Mean pooling
    - Classification head
    
    This is the industry-standard approach.
    """
    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        depth: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Standard transformer encoder (no routing, no refinement)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN like modern transformers
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Classification head
        self.pooler = nn.Linear(d_model, d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        self.token_embed.weight.data[0].zero_()
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)
        
        for module in [self.pooler] + list(self.classifier.modules()):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, premise, hypothesis):
        B, L = premise.size()
        
        # Embed premise
        p_tok = self.token_embed(premise)
        p_pos = self.pos_embed(torch.arange(L, device=premise.device).unsqueeze(0).expand(B, -1))
        p_emb = self.dropout(p_tok + p_pos)
        
        # Embed hypothesis
        h_tok = self.token_embed(hypothesis)
        h_pos = self.pos_embed(torch.arange(L, device=hypothesis.device).unsqueeze(0).expand(B, -1))
        h_emb = self.dropout(h_tok + h_pos)
        
        # Encode with standard transformer
        p_enc = self.encoder(p_emb)
        h_enc = self.encoder(h_emb)
        
        # Pool (mean over non-padding)
        p_mask = (premise != 0).float().unsqueeze(-1)
        h_mask = (hypothesis != 0).float().unsqueeze(-1)
        
        p_pooled = (p_enc * p_mask).sum(dim=1) / (p_mask.sum(dim=1) + 1e-9)
        h_pooled = (h_enc * h_mask).sum(dim=1) / (h_mask.sum(dim=1) + 1e-9)
        
        # Apply pooler
        p_pooled = self.pooler(p_pooled)
        h_pooled = self.pooler(h_pooled)
        
        # Combine: [premise; hypothesis; premise*hypothesis]
        combined = torch.cat([p_pooled, h_pooled, p_pooled * h_pooled], dim=-1)
        
        # Classify
        logits = self.classifier(combined)
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ========== PoH Model ==========

class PoHForNLI(nn.Module):
    """
    PoH (Pointer-over-Heads) model for NLI.
    
    Architecture:
    - Token embeddings
    - PoH stack with HRM-based routing (R refinement iterations)
    - Mean pooling
    - Classification head
    
    Key differences from BERT:
    - Dynamic head routing via HRM controller
    - Multi-step refinement (processes input R times)
    - Adaptive computation patterns
    """
    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        depth: int = 4,
        R: int = 8,  # Optimal from diagnostic
        T: int = 4,  # HRM outer loop period
        dropout: float = 0.1,
        max_seq_len: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.R = R
        self.T = T
        
        # Embeddings (no positional - PoH stack handles it)
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # PoH stack configuration
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
        
        # Create stack
        stack = PoHStack(cfg, depth=depth)
        
        # Set HRM period T in each block
        for block in stack.blocks:
            if hasattr(block, 'router'):
                if hasattr(block.router, 'hrm_controller'):
                    block.router.hrm_controller.T = T
                elif hasattr(block.router, 'T'):
                    block.router.T = T
        
        # Iterative refiner with R refinement steps
        self.refiner = IterRefiner(
            stack,
            max_inner_iters=R,
            outer_residual=True,
            rezero_init=True
        )
        
        # Classification head (same as BERT)
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
        # Embed
        p_emb = self.embed(premise)
        h_emb = self.embed(hypothesis)
        
        # Process with PoH (R refinement steps)
        p_out, _ = self.refiner(p_emb)
        h_out, _ = self.refiner(h_emb)
        
        # Pool
        p_mask = (premise != 0).float().unsqueeze(-1)
        h_mask = (hypothesis != 0).float().unsqueeze(-1)
        
        p_pooled = (p_out * p_mask).sum(dim=1) / (p_mask.sum(dim=1) + 1e-9)
        h_pooled = (h_out * h_mask).sum(dim=1) / (h_mask.sum(dim=1) + 1e-9)
        
        # Apply pooler
        p_pooled = self.pooler(p_pooled)
        h_pooled = self.pooler(h_pooled)
        
        # Combine
        combined = torch.cat([p_pooled, h_pooled, p_pooled * h_pooled], dim=-1)
        
        # Classify
        logits = self.classifier(combined)
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ========== Training ==========

def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_steps: int = 1000,
    lr: float = 1e-3,
    device: str = 'cpu',
):
    """Train model and return results."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    print(f"Parameters: {model.count_parameters()/1e6:.2f}M")
    print(f"Learning rate: {lr:.0e}")
    print(f"Max steps: {max_steps}")
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Warmup + cosine decay scheduler
    warmup_steps = 200
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_val_acc = 0.0
    final_val_acc = 0.0
    start_time = time.time()
    
    step = 0
    eval_interval = 100
    
    pbar = tqdm(total=max_steps, desc=f"{model_name}")
    
    while step < max_steps:
        model.train()
        for premise, hypothesis, labels in train_loader:
            if step >= max_steps:
                break
            
            premise = premise.to(device)
            hypothesis = hypothesis.to(device)
            labels = labels.to(device)
            
            # Forward
            logits = model(premise, hypothesis)
            loss = criterion(logits, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            step += 1
            pbar.update(1)
            pbar.set_postfix({'loss': f"{loss.item():.3f}"})
            
            # Evaluate
            if step % eval_interval == 0 or step == max_steps:
                model.eval()
                correct = 0
                total = 0
                val_loss = 0.0
                num_batches = 0
                
                with torch.no_grad():
                    for p, h, l in val_loader:
                        p, h, l = p.to(device), h.to(device), l.to(device)
                        logits = model(p, h)
                        loss = criterion(logits, l)
                        preds = logits.argmax(dim=-1)
                        
                        correct += (preds == l).sum().item()
                        total += l.size(0)
                        val_loss += loss.item()
                        num_batches += 1
                
                val_acc = correct / total
                val_loss = val_loss / num_batches
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                
                final_val_acc = val_acc
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.3f}",
                    'val_acc': f"{val_acc:.3f}",
                    'best': f"{best_val_acc:.3f}"
                })
                
                model.train()
    
    pbar.close()
    
    elapsed = (time.time() - start_time) / 60
    
    print(f"\n✅ {model_name} Training Complete")
    print(f"  Best val accuracy: {best_val_acc:.4f}")
    print(f"  Final val accuracy: {final_val_acc:.4f}")
    print(f"  Training time: {elapsed:.2f} minutes")
    
    return {
        'model_name': model_name,
        'best_acc': best_val_acc,
        'final_acc': final_val_acc,
        'time_min': elapsed,
        'params_M': model.count_parameters() / 1e6
    }


# ========== Main A/B Test ==========

def main():
    parser = argparse.ArgumentParser(description='PoH vs BERT A/B Test on NLI')
    parser.add_argument('--train-samples', type=int, default=10000, help='Training samples')
    parser.add_argument('--val-samples', type=int, default=2000, help='Validation samples')
    parser.add_argument('--max-steps', type=int, default=1000, help='Training steps')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--R', type=int, default=8, help='PoH refinement steps')
    parser.add_argument('--T', type=int, default=4, help='HRM outer loop period')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("PoH vs BERT A/B Test on NLI")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Training samples: {args.train_samples:,}")
    print(f"  Validation samples: {args.val_samples:,}")
    print(f"  Training steps: {args.max_steps:,}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr:.0e}")
    print(f"  PoH R (refinement steps): {args.R}")
    print(f"  PoH T (HRM period): {args.T}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    # Load data
    print(f"\n{'='*60}")
    print("Loading SNLI Dataset")
    print(f"{'='*60}")
    
    train_dataset = SNLIDataset('train', max_samples=args.train_samples)
    val_dataset = SNLIDataset('validation', max_samples=args.val_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=0)
    
    # Build models with matched architecture
    print(f"\n{'='*60}")
    print("Building Models (Matched Architecture)")
    print(f"{'='*60}")
    
    bert = BERTForNLI(
        vocab_size=10000,
        d_model=256,
        n_heads=8,
        d_ff=1024,
        depth=4,
        dropout=0.1,
        max_seq_len=32,
    )
    
    poh = PoHForNLI(
        vocab_size=10000,
        d_model=256,
        n_heads=8,
        d_ff=1024,
        depth=4,
        R=args.R,
        T=args.T,
        dropout=0.1,
        max_seq_len=32,
    )
    
    print(f"BERT: {bert.count_parameters()/1e6:.2f}M parameters")
    print(f"PoH:  {poh.count_parameters()/1e6:.2f}M parameters")
    
    # Train PoH first (novel architecture)
    poh_results = train_model(
        poh, "PoH", train_loader, val_loader,
        max_steps=args.max_steps, lr=args.lr, device=device
    )
    
    # Train BERT baseline
    bert_results = train_model(
        bert, "BERT", train_loader, val_loader,
        max_steps=args.max_steps, lr=args.lr, device=device
    )
    
    # Summary
    print("\n" + "="*60)
    print("📊 A/B TEST RESULTS")
    print("="*60)
    
    print(f"\n🔬 PoH (Novel Architecture)")
    print(f"  Architecture: {args.R} refinement iterations, HRM routing (T={args.T})")
    print(f"  Parameters: {poh_results['params_M']:.2f}M")
    print(f"  Best accuracy: {poh_results['best_acc']:.4f}")
    print(f"  Final accuracy: {poh_results['final_acc']:.4f}")
    print(f"  Training time: {poh_results['time_min']:.2f} min")
    
    print(f"\n📚 BERT (Industry Standard)")
    print(f"  Architecture: Standard transformer encoder")
    print(f"  Parameters: {bert_results['params_M']:.2f}M")
    print(f"  Best accuracy: {bert_results['best_acc']:.4f}")
    print(f"  Final accuracy: {bert_results['final_acc']:.4f}")
    print(f"  Training time: {bert_results['time_min']:.2f} min")
    
    # Comparison
    delta_acc = poh_results['best_acc'] - bert_results['best_acc']
    delta_pct = (delta_acc / bert_results['best_acc']) * 100 if bert_results['best_acc'] > 0 else 0
    
    print(f"\n{'='*60}")
    print("📈 Comparison")
    print(f"{'='*60}")
    print(f"Accuracy delta: {delta_acc:+.4f} ({delta_pct:+.2f}%)")
    
    if delta_acc > 0:
        print(f"✅ PoH WINS by {delta_acc:.4f} ({delta_pct:.2f}%)")
    elif delta_acc < 0:
        print(f"❌ BERT WINS by {-delta_acc:.4f} ({-delta_pct:.2f}%)")
    else:
        print(f"⚖️  TIE")
    
    # Save results
    import csv
    from datetime import datetime
    
    os.makedirs("experiments/results/poh_vs_bert", exist_ok=True)
    results_file = "experiments/results/poh_vs_bert/ab_results.csv"
    
    file_exists = os.path.exists(results_file)
    
    with open(results_file, 'a', newline='') as f:
        fieldnames = ['timestamp', 'model', 'R', 'T', 'best_acc', 'final_acc', 'time_min', 'params_M', 'delta_vs_bert']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        writer.writerow({
            'timestamp': timestamp,
            'model': 'PoH',
            'R': args.R,
            'T': args.T,
            'best_acc': f"{poh_results['best_acc']:.4f}",
            'final_acc': f"{poh_results['final_acc']:.4f}",
            'time_min': f"{poh_results['time_min']:.2f}",
            'params_M': f"{poh_results['params_M']:.2f}",
            'delta_vs_bert': f"{delta_acc:+.4f}"
        })
        
        writer.writerow({
            'timestamp': timestamp,
            'model': 'BERT',
            'R': '-',
            'T': '-',
            'best_acc': f"{bert_results['best_acc']:.4f}",
            'final_acc': f"{bert_results['final_acc']:.4f}",
            'time_min': f"{bert_results['time_min']:.2f}",
            'params_M': f"{bert_results['params_M']:.2f}",
            'delta_vs_bert': "0.0000"
        })
    
    print(f"\n✅ Results saved to: {results_file}")
    print("="*60)


if __name__ == "__main__":
    main()

