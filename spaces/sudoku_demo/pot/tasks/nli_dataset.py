"""
Real NLI dataset loaders for SNLI and MultiNLI.

Uses Hugging Face datasets library for loading standard benchmarks.

Author: Eran Ben Artzy
Year: 2025
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class NLIExample:
    """Single NLI example."""
    premise: str
    hypothesis: str
    label: int  # 0=entailment, 1=neutral, 2=contradiction
    

class SNLIDataset(Dataset):
    """Stanford Natural Language Inference dataset."""
    
    def __init__(self, split: str = "train", max_samples: Optional[int] = None):
        """
        Args:
            split: 'train', 'validation', or 'test'
            max_samples: limit dataset size (for quick experiments)
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Please install datasets library:\n"
                "  pip install datasets"
            )
        
        print(f"Loading SNLI {split} split...")
        dataset = load_dataset("snli", split=split)
        
        # Filter out examples with no gold label (-1)
        dataset = dataset.filter(lambda x: x['label'] != -1)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        self.examples = []
        for item in dataset:
            self.examples.append(NLIExample(
                premise=item['premise'],
                hypothesis=item['hypothesis'],
                label=item['label']
            ))
        
        print(f"  Loaded {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


class MultiNLIDataset(Dataset):
    """Multi-Genre Natural Language Inference dataset."""
    
    def __init__(self, split: str = "train", max_samples: Optional[int] = None):
        """
        Args:
            split: 'train', 'validation_matched', 'validation_mismatched'
            max_samples: limit dataset size
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Please install datasets library:\n"
                "  pip install datasets"
            )
        
        print(f"Loading MultiNLI {split} split...")
        dataset = load_dataset("multi_nli", split=split)
        
        # Filter out examples with no gold label
        dataset = dataset.filter(lambda x: x['label'] != -1)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        self.examples = []
        for item in dataset:
            self.examples.append(NLIExample(
                premise=item['premise'],
                hypothesis=item['hypothesis'],
                label=item['label']
            ))
        
        print(f"  Loaded {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def collate_nli_batch(
    examples,
    tokenizer,
    max_length: int = 128,
    cls_token_id: int = 101,
    sep_token_id: int = 102,
    pad_token_id: int = 0,
):
    """
    Collate function for NLI batches.
    
    Args:
        examples: list of NLIExample
        tokenizer: tokenizer with encode() method
        max_length: max sequence length
        cls_token_id, sep_token_id, pad_token_id: special tokens
    
    Returns:
        dict with input_ids, attention_mask, labels
    """
    batch_size = len(examples)
    
    # Tokenize premises and hypotheses
    premise_ids = []
    hypothesis_ids = []
    labels = []
    
    for ex in examples:
        # Simple whitespace tokenization (replace with real tokenizer in production)
        p_tokens = ex.premise.lower().split()[:max_length // 2 - 2]
        h_tokens = ex.hypothesis.lower().split()[:max_length // 2 - 2]
        
        # Convert to token IDs (simple hash for demo, use real tokenizer in production)
        p_ids = [hash(t) % 30000 + 100 for t in p_tokens]
        h_ids = [hash(t) % 30000 + 100 for t in h_tokens]
        
        premise_ids.append(p_ids)
        hypothesis_ids.append(h_ids)
        labels.append(ex.label)
    
    # Pad to same length
    max_p_len = max(len(p) for p in premise_ids) if premise_ids else 1
    max_h_len = max(len(h) for h in hypothesis_ids) if hypothesis_ids else 1
    
    # Create [CLS] premise [SEP] hypothesis [SEP] sequences
    max_seq_len = min(max_length, 1 + max_p_len + 1 + max_h_len + 1)
    
    input_ids = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    
    for i in range(batch_size):
        pos = 0
        
        # [CLS]
        input_ids[i, pos] = cls_token_id
        attention_mask[i, pos] = 1
        pos += 1
        
        # Premise
        p_len = min(len(premise_ids[i]), max_seq_len - pos - 2)
        if p_len > 0:
            input_ids[i, pos:pos+p_len] = torch.tensor(premise_ids[i][:p_len])
            attention_mask[i, pos:pos+p_len] = 1
            pos += p_len
        
        # [SEP]
        if pos < max_seq_len:
            input_ids[i, pos] = sep_token_id
            attention_mask[i, pos] = 1
            pos += 1
        
        # Hypothesis
        h_len = min(len(hypothesis_ids[i]), max_seq_len - pos - 1)
        if h_len > 0:
            input_ids[i, pos:pos+h_len] = torch.tensor(hypothesis_ids[i][:h_len])
            attention_mask[i, pos:pos+h_len] = 1
            pos += h_len
        
        # [SEP]
        if pos < max_seq_len:
            input_ids[i, pos] = sep_token_id
            attention_mask[i, pos] = 1
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': torch.tensor(labels, dtype=torch.long)
    }


class SimpleTokenizer:
    """Simple tokenizer for demo purposes."""
    
    def __init__(self, vocab_size: int = 30522):
        self.vocab_size = vocab_size
    
    def encode(self, text: str, max_length: int = 128):
        """Encode text to token IDs."""
        tokens = text.lower().split()[:max_length]
        # Simple hash-based encoding (replace with real tokenizer)
        token_ids = [hash(t) % (self.vocab_size - 100) + 100 for t in tokens]
        return token_ids
    
    def batch_encode(self, texts, max_length: int = 128, padding: bool = True):
        """Batch encode texts."""
        all_ids = [self.encode(text, max_length) for text in texts]
        
        if padding:
            max_len = max(len(ids) for ids in all_ids)
            all_ids = [ids + [0] * (max_len - len(ids)) for ids in all_ids]
        
        return all_ids


def create_nli_dataloader(
    dataset_name: str = "snli",
    split: str = "train",
    batch_size: int = 32,
    max_samples: Optional[int] = None,
    max_length: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
):
    """
    Create DataLoader for NLI dataset.
    
    Args:
        dataset_name: 'snli' or 'mnli'
        split: dataset split
        batch_size: batch size
        max_samples: limit number of samples
        max_length: max sequence length
        shuffle: shuffle data
        num_workers: number of workers
    
    Returns:
        DataLoader
    """
    # Load dataset
    if dataset_name == "snli":
        dataset = SNLIDataset(split=split, max_samples=max_samples)
    elif dataset_name == "mnli":
        dataset = MultiNLIDataset(split=split, max_samples=max_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create tokenizer
    tokenizer = SimpleTokenizer()
    
    # Create collate function
    def collate_fn(examples):
        return collate_nli_batch(examples, tokenizer, max_length=max_length)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    return dataloader

