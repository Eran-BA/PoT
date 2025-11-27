"""
Natural Language Inference (NLI) with Pointer-over-Heads Transformer

Task: Given premise and hypothesis, predict: entailment, contradiction, or neutral
Dataset: SNLI (Stanford NLI) - 570k sentence pairs
Difficulty: Requires compositional reasoning, world knowledge, logical inference

Why PoH Should Help:
- Different heads can specialize in different reasoning patterns:
  * Lexical overlap detection
  * Negation/contradiction patterns
  * Semantic similarity
  * Syntactic alignment
- Iterative refinement for hard cases
- Adaptive routing based on sentence complexity

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import argparse
import json
import os
import random
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm


# =========================================================================
# Model Architecture
# =========================================================================

class PoHNLIClassifier(nn.Module):
    """
    NLI classifier with PoH Transformer encoder.
    
    Architecture:
    1. BERT/RoBERTa encoder (frozen or fine-tuned)
    2. PoH Transformer layer (iterative refinement with head routing)
    3. Cross-attention between premise and hypothesis
    4. Classification head (3-way: entailment, contradiction, neutral)
    """
    
    def __init__(
        self,
        encoder_name: str = "bert-base-uncased",
        d_model: int = 768,
        n_heads: int = 12,
        d_ff: int = 3072,
        max_inner_iters: int = 3,
        use_poh: bool = True,
        freeze_encoder: bool = False,
        use_hrm: bool = True,
        temperature_init: float = 2.0,
        temperature_min: float = 0.8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_poh = use_poh
        
        # Pre-trained encoder
        self.encoder = AutoModel.from_pretrained(encoder_name)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # PoH Transformer layer (if enabled)
        if use_poh:
            from experiments.sort_pointer_improved import ImprovedPoHBlock
            self.poh_layer = ImprovedPoHBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                max_iters=max_inner_iters,
                temperature_init=temperature_init,
                temperature_min=temperature_min,
            )
        
        # Cross-attention between premise and hypothesis
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads // 2, dropout=dropout, batch_first=True
        )
        
        # Pooling and classification
        self.pooler = nn.Sequential(
            nn.Linear(d_model * 4, d_model),  # [CLS, mean, max, cross]
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3),  # 3-way classification
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def set_temperature(self, temp: float):
        """Update routing temperature (for annealing)."""
        if self.use_poh:
            self.poh_layer.set_temperature(temp)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass.
        
        Args:
            input_ids: [B, L] token ids ([CLS] premise [SEP] hypothesis [SEP])
            attention_mask: [B, L] attention mask
            token_type_ids: [B, L] segment ids (0=premise, 1=hypothesis)
            return_diagnostics: Return routing diagnostics
        
        Returns:
            logits: [B, 3] class logits
            diagnostics: Optional dict with routing stats
        """
        # Encode with pre-trained model
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = outputs.last_hidden_state  # [B, L, D]
        
        # Apply PoH layer (iterative refinement with head routing)
        diagnostics = None
        if self.use_poh:
            hidden, diagnostics = self.poh_layer(
                hidden, return_diagnostics=return_diagnostics
            )
        
        # Separate premise and hypothesis
        # Assumes format: [CLS] premise [SEP] hypothesis [SEP]
        # token_type_ids: 0 for premise, 1 for hypothesis
        if token_type_ids is not None:
            premise_mask = (token_type_ids == 0) & (attention_mask.bool())
            hypothesis_mask = (token_type_ids == 1) & (attention_mask.bool())
        else:
            # Fallback: split at middle
            L = hidden.size(1)
            mid = L // 2
            premise_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
            hypothesis_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
            premise_mask[:, :mid] = attention_mask[:, :mid].bool()
            hypothesis_mask[:, mid:] = attention_mask[:, mid:].bool()
        
        # Extract premise and hypothesis representations
        premise = hidden * premise_mask.unsqueeze(-1).float()
        hypothesis = hidden * hypothesis_mask.unsqueeze(-1).float()
        
        # Cross-attention: hypothesis attends to premise
        cross_out, _ = self.cross_attn(
            hypothesis, premise, premise,
            key_padding_mask=~premise_mask,
        )
        
        # Pooling strategies
        cls_rep = hidden[:, 0]  # [CLS] token
        
        # Mean pooling (hypothesis)
        hyp_sum = (hypothesis * hypothesis_mask.unsqueeze(-1).float()).sum(dim=1)
        hyp_count = hypothesis_mask.sum(dim=1, keepdim=True).clamp_min(1)
        hyp_mean = hyp_sum / hyp_count
        
        # Max pooling (cross-attention output)
        cross_mask = hypothesis_mask.unsqueeze(-1).float()
        cross_masked = cross_out.masked_fill(~cross_mask.bool(), -1e9)
        cross_max, _ = cross_masked.max(dim=1)
        
        # Mean pooling (cross-attention output)
        cross_sum = (cross_out * cross_mask).sum(dim=1)
        cross_count = hypothesis_mask.sum(dim=1, keepdim=True).clamp_min(1)
        cross_mean = cross_sum / cross_count
        
        # Concatenate all representations
        combined = torch.cat([cls_rep, hyp_mean, cross_max, cross_mean], dim=-1)
        
        # Pooler and classifier
        pooled = self.pooler(combined)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        return logits, diagnostics


# =========================================================================
# Training Utilities
# =========================================================================

def train_epoch(
    model: PoHNLIClassifier,
    dataloader,
    optimizer,
    device,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    progress = tqdm(dataloader, desc="Training")
    for batch in progress:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch.get('token_type_ids', None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        labels = batch['labels'].to(device)
        
        # Forward
        logits, _ = model(input_ids, attention_mask, token_type_ids)
        loss = F.cross_entropy(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += len(labels)
        
        # Update progress
        progress.set_postfix({
            'loss': loss.item(),
            'acc': total_correct / total_samples,
        })
    
    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
    }


def eval_epoch(
    model: PoHNLIClassifier,
    dataloader,
    device,
    return_diagnostics: bool = False,
) -> Dict[str, float]:
    """Evaluate for one epoch."""
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    all_diagnostics = [] if return_diagnostics else None
    
    progress = tqdm(dataloader, desc="Evaluating")
    with torch.no_grad():
        for batch in progress:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch['labels'].to(device)
            
            logits, diagnostics = model(
                input_ids, attention_mask, token_type_ids,
                return_diagnostics=return_diagnostics
            )
            loss = F.cross_entropy(logits, labels)
            
            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)
            
            if return_diagnostics and diagnostics is not None:
                all_diagnostics.append(diagnostics)
            
            progress.set_postfix({
                'loss': loss.item(),
                'acc': total_correct / total_samples,
            })
    
    metrics = {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
    }
    
    # Aggregate diagnostics
    if return_diagnostics and all_diagnostics:
        # Average entropies
        if 'entropies' in all_diagnostics[0]:
            all_entropies = [d['entropies'] for d in all_diagnostics]
            mean_entropy_per_iter = np.mean(all_entropies, axis=0)
            metrics['mean_entropy'] = float(np.mean(mean_entropy_per_iter))
            metrics['entropy_per_iter'] = mean_entropy_per_iter.tolist()
    
    return metrics


# =========================================================================
# Data Loading
# =========================================================================

def load_snli_data(
    tokenizer,
    max_length: int = 128,
    batch_size: int = 32,
    num_train_samples: Optional[int] = None,
    num_val_samples: Optional[int] = None,
):
    """Load SNLI dataset and create dataloaders."""
    
    print("Loading SNLI dataset...")
    dataset = load_dataset("snli")
    
    # Filter out examples with label == -1 (no gold label)
    def filter_fn(example):
        return example['label'] != -1
    
    train_dataset = dataset['train'].filter(filter_fn)
    val_dataset = dataset['validation'].filter(filter_fn)
    test_dataset = dataset['test'].filter(filter_fn)
    
    # Limit samples if specified
    if num_train_samples:
        train_dataset = train_dataset.select(range(min(num_train_samples, len(train_dataset))))
    if num_val_samples:
        val_dataset = val_dataset.select(range(min(num_val_samples, len(val_dataset))))
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Tokenization
    def tokenize_fn(examples):
        return tokenizer(
            examples['premise'],
            examples['hypothesis'],
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors=None,
        )
    
    print("Tokenizing...")
    train_dataset = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=['premise', 'hypothesis'],
    )
    val_dataset = val_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=['premise', 'hypothesis'],
    )
    test_dataset = test_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=['premise', 'hypothesis'],
    )
    
    # Set format
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])
    
    # Rename label column
    train_dataset = train_dataset.rename_column('label', 'labels')
    val_dataset = val_dataset.rename_column('label', 'labels')
    test_dataset = test_dataset.rename_column('label', 'labels')
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, val_loader, test_loader


# =========================================================================
# Main Training Script
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="NLI with PoH Transformer")
    
    # Model
    parser.add_argument('--encoder', type=str, default='bert-base-uncased',
                        help='Pre-trained encoder')
    parser.add_argument('--use_poh', action='store_true',
                        help='Use PoH layer')
    parser.add_argument('--max_inner_iters', type=int, default=3,
                        help='Max inner iterations for PoH')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze pre-trained encoder')
    parser.add_argument('--use_hrm', action='store_true', default=True,
                        help='Use HRM-style gradients in PoH')
    
    # Training
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Max sequence length')
    
    # Data
    parser.add_argument('--num_train_samples', type=int, default=None,
                        help='Limit training samples (for quick testing)')
    parser.add_argument('--num_val_samples', type=int, default=None,
                        help='Limit validation samples')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda)')
    parser.add_argument('--output_dir', type=str, default='experiments/results_nli',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Device: {device}")
    print(f"PoH enabled: {args.use_poh}")
    if args.use_poh:
        print(f"  Max inner iterations: {args.max_inner_iters}")
        print(f"  HRM-style: {args.use_hrm}")
    print("=" * 80)
    
    # Load data
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    train_loader, val_loader, test_loader = load_snli_data(
        tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_train_samples=args.num_train_samples,
        num_val_samples=args.num_val_samples,
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = PoHNLIClassifier(
        encoder_name=args.encoder,
        use_poh=args.use_poh,
        max_inner_iters=args.max_inner_iters,
        freeze_encoder=args.freeze_encoder,
        use_hrm=args.use_hrm,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    best_val_acc = 0.0
    results = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, args.grad_clip)
        print(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        
        # Validate
        val_metrics = eval_epoch(model, val_loader, device, return_diagnostics=args.use_poh)
        print(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        
        if args.use_poh and 'mean_entropy' in val_metrics:
            print(f"Mean routing entropy: {val_metrics['mean_entropy']:.3f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            print(f"✓ New best validation accuracy: {best_val_acc:.4f}")
        
        # Record results
        results.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'mean_entropy': val_metrics.get('mean_entropy', None),
        })
    
    # Final evaluation on test set
    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 80)
    test_metrics = eval_epoch(model, test_loader, device, return_diagnostics=args.use_poh)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    if args.use_poh and 'mean_entropy' in test_metrics:
        print(f"Mean routing entropy: {test_metrics['mean_entropy']:.3f}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f"nli_{'poh' if args.use_poh else 'baseline'}_seed{args.seed}.json"
    )
    
    with open(output_file, 'w') as f:
        json.dump({
            'args': vars(args),
            'best_val_acc': best_val_acc,
            'test_acc': test_metrics['accuracy'],
            'test_loss': test_metrics['loss'],
            'results_per_epoch': results,
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()

