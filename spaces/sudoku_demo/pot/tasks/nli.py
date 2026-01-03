"""
Natural Language Inference (NLI) task adapter for PoH framework.

Supports SNLI and MultiNLI datasets with premise-hypothesis pair encoding.

Author: Eran Ben Artzy
Year: 2025
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class NLIBatch:
    """Batch of NLI examples."""
    premise_ids: torch.Tensor          # [B, T_premise]
    hypothesis_ids: torch.Tensor       # [B, T_hypothesis]
    premise_mask: torch.Tensor         # [B, T_premise]
    hypothesis_mask: torch.Tensor      # [B, T_hypothesis]
    labels: torch.Tensor               # [B] (0=entailment, 1=neutral, 2=contradiction)
    
    def to(self, device):
        return NLIBatch(
            premise_ids=self.premise_ids.to(device),
            hypothesis_ids=self.hypothesis_ids.to(device),
            premise_mask=self.premise_mask.to(device),
            hypothesis_mask=self.hypothesis_mask.to(device),
            labels=self.labels.to(device),
        )


class NLIClassificationHead(nn.Module):
    """Classification head for NLI (3-way: entailment, neutral, contradiction)."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 3)  # 3 classes
    
    def forward(self, sequence_output: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            sequence_output: [B, T, D] - encoder output
            mask: [B, T] - attention mask
        Returns:
            logits: [B, 3] - class logits
        """
        # Use [CLS] token (first token) for classification
        cls_output = sequence_output[:, 0, :]  # [B, D]
        
        pooled = self.pooler(cls_output)       # [B, D]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)       # [B, 3]
        
        return logits


class NLIMetrics:
    """Compute accuracy for NLI."""
    
    @staticmethod
    def compute(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Args:
            logits: [B, 3] - predicted logits
            labels: [B] - true labels (0, 1, or 2)
        Returns:
            dict with 'accuracy'
        """
        preds = logits.argmax(dim=-1)  # [B]
        accuracy = (preds == labels).float().mean().item()
        
        return {"accuracy": accuracy}
    
    @staticmethod
    def compute_per_class(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Compute per-class accuracy."""
        preds = logits.argmax(dim=-1)
        
        metrics = {}
        for cls_idx, cls_name in enumerate(["entailment", "neutral", "contradiction"]):
            mask = labels == cls_idx
            if mask.sum() > 0:
                cls_acc = (preds[mask] == labels[mask]).float().mean().item()
                metrics[f"acc_{cls_name}"] = cls_acc
        
        metrics["accuracy"] = (preds == labels).float().mean().item()
        return metrics


def create_pair_sequence(
    premise_ids: torch.Tensor,
    hypothesis_ids: torch.Tensor,
    cls_token_id: int = 101,  # [CLS]
    sep_token_id: int = 102,  # [SEP]
    pad_token_id: int = 0,
    max_seq_len: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create BERT-style [CLS] premise [SEP] hypothesis [SEP] sequence.
    
    Args:
        premise_ids: [B, T_p]
        hypothesis_ids: [B, T_h]
        cls_token_id: ID for [CLS] token
        sep_token_id: ID for [SEP] token
        pad_token_id: ID for [PAD] token
        max_seq_len: maximum sequence length
    
    Returns:
        input_ids: [B, T] where T = 1 + T_p + 1 + T_h + 1
        attention_mask: [B, T]
    """
    B = premise_ids.size(0)
    device = premise_ids.device
    
    # Get lengths (excluding padding)
    premise_lens = (premise_ids != pad_token_id).sum(dim=1)  # [B]
    hypothesis_lens = (hypothesis_ids != pad_token_id).sum(dim=1)  # [B]
    
    # Max total length (capped at max_seq_len)
    max_len = min((1 + premise_lens.max() + 1 + hypothesis_lens.max() + 1).item(), max_seq_len)
    
    # Initialize with padding
    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
    
    for i in range(B):
        p_len = min(premise_lens[i].item(), max_len - 3)  # Reserve space for [CLS] and 2x [SEP]
        h_len = min(hypothesis_lens[i].item(), max_len - 3 - p_len)
        
        # [CLS] premise [SEP] hypothesis [SEP]
        pos = 0
        input_ids[i, pos] = cls_token_id
        attention_mask[i, pos] = 1
        pos += 1
        
        if p_len > 0:
            input_ids[i, pos:pos+p_len] = premise_ids[i, :p_len]
            attention_mask[i, pos:pos+p_len] = 1
            pos += p_len
        
        if pos < max_len:
            input_ids[i, pos] = sep_token_id
            attention_mask[i, pos] = 1
            pos += 1
        
        if h_len > 0 and pos < max_len:
            input_ids[i, pos:pos+h_len] = hypothesis_ids[i, :h_len]
            attention_mask[i, pos:pos+h_len] = 1
            pos += h_len
        
        if pos < max_len:
            input_ids[i, pos] = sep_token_id
            attention_mask[i, pos] = 1
    
    return input_ids, attention_mask


class NLIDataLoader:
    """Simple NLI data loader wrapper."""
    
    def __init__(self, dataset_name: str = "snli", split: str = "train", max_length: int = 128):
        """
        Args:
            dataset_name: 'snli' or 'mnli'
            split: 'train', 'validation', or 'test'
            max_length: max sequence length
        """
        self.dataset_name = dataset_name
        self.split = split
        self.max_length = max_length
        
        # Label mapping
        self.label_map = {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2,
        }
    
    def load_from_huggingface(self):
        """Load dataset from Hugging Face datasets library."""
        try:
            from datasets import load_dataset
            
            if self.dataset_name == "snli":
                dataset = load_dataset("snli", split=self.split)
            elif self.dataset_name == "mnli":
                dataset = load_dataset("multi_nli", split=self.split)
            else:
                raise ValueError(f"Unknown dataset: {self.dataset_name}")
            
            # Filter out examples with label == -1 (no label)
            dataset = dataset.filter(lambda x: x['label'] != -1)
            
            return dataset
        
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
    
    def create_synthetic_batch(self, batch_size: int = 16, vocab_size: int = 30000) -> NLIBatch:
        """Create a synthetic batch for testing (no external deps)."""
        premise_len = torch.randint(10, self.max_length // 2, (batch_size,))
        hypothesis_len = torch.randint(10, self.max_length // 2, (batch_size,))
        
        max_p = premise_len.max().item()
        max_h = hypothesis_len.max().item()
        
        premise_ids = torch.randint(3, vocab_size, (batch_size, max_p))
        hypothesis_ids = torch.randint(3, vocab_size, (batch_size, max_h))
        
        # Create masks
        premise_mask = torch.zeros(batch_size, max_p, dtype=torch.long)
        hypothesis_mask = torch.zeros(batch_size, max_h, dtype=torch.long)
        
        for i in range(batch_size):
            premise_mask[i, :premise_len[i]] = 1
            hypothesis_mask[i, :hypothesis_len[i]] = 1
        
        # Pad sequences
        for i in range(batch_size):
            premise_ids[i, premise_len[i]:] = 0
            hypothesis_ids[i, hypothesis_len[i]:] = 0
        
        labels = torch.randint(0, 3, (batch_size,))
        
        return NLIBatch(
            premise_ids=premise_ids,
            hypothesis_ids=hypothesis_ids,
            premise_mask=premise_mask,
            hypothesis_mask=hypothesis_mask,
            labels=labels,
        )

