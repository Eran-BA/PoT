"""
Batching and collation functions for dependency parsing.

Handles tokenization, word-ID alignment, and label conversion for batching
dependency parsing examples.

Functions:
    collate_batch: Collate examples into batched tensors

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

from typing import List, Dict, Union, Optional, Tuple, Any

import torch
from transformers import AutoTokenizer


def collate_batch(
    examples: Union[List[Dict], Dict],
    tokenizer: AutoTokenizer,
    device: torch.device,
    label_vocab: Optional[Dict[str, int]] = None,
    return_deprels: bool = False
) -> Union[
    Tuple[Dict[str, torch.Tensor], List[List[Optional[int]]], List[List[int]], Optional[List[List[int]]]],
    Tuple[Dict[str, torch.Tensor], List[List[Optional[int]]], List[List[int]], Optional[List[List[int]]], Optional[List[List[str]]]]
]:
    """Collate examples into batched tensors for training/evaluation.
    
    Handles both HuggingFace dict format (dict of lists) and standard format
    (list of dicts). Tokenizes text, tracks word-to-subword alignment, and
    optionally converts dependency labels to indices.
    
    Args:
        examples: Batch of examples in either format:
                 - List[Dict]: [{"tokens": [...], "head": [...], "deprel": [...]}, ...]
                 - Dict: {"tokens": [[...], [...]], "head": [[...], [...]]}
        tokenizer: HuggingFace tokenizer for subword tokenization
        device: Device to place tensors on
        label_vocab: Optional vocabulary mapping label strings to indices
        return_deprels: If True, also return original string labels for punct masking
        
    Returns:
        Tuple containing:
        - enc: Tokenized inputs (dict with 'input_ids', 'attention_mask', etc.)
        - word_ids: Word indices for each subword [B, S]
        - heads: Gold dependency heads [B, T]
        - labels: Gold dependency labels [B, T] (indices or original)
        - (if return_deprels) deprels_str: Original string labels for masking
        
    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        >>> examples = [
        ...     {"tokens": ["The", "cat"], "head": [0, 1], "deprel": ["det", "root"]},
        ...     {"tokens": ["Dogs", "bark"], "head": [0, 1], "deprel": ["nsubj", "root"]}
        ... ]
        >>> label_vocab = {"det": 0, "root": 1, "nsubj": 2, "<UNK>": 3}
        >>> enc, word_ids, heads, labels = collate_batch(
        ...     examples, tokenizer, device='cpu', label_vocab=label_vocab
        ... )
        >>> print(enc['input_ids'].shape)  # [2, max_subword_len]
        >>> print(len(word_ids))  # 2 (batch size)
        >>> print(heads)  # [[0, 1], [0, 1]]
        >>> print(labels)  # [[0, 1], [2, 1]] (converted to indices)
        
    Note:
        - Handles both string and pre-indexed labels
        - Supports punctuation masking via return_deprels=True
        - Pads to maximum sequence length in batch
        - Truncates to 512 tokens by default
    """
    # Handle both formats: dict of lists or list of dicts
    if isinstance(examples, dict):
        # HuggingFace format: dict of lists
        toks = examples["tokens"]
        heads = examples["head"]
        labels = examples.get("deprel", None)
    else:
        # Standard format: list of dicts
        toks = [ex["tokens"] for ex in examples]
        heads = [ex["head"] for ex in examples]
        
        # Check if any example has labels
        if any("deprel" in ex for ex in examples):
            labels = [
                ex.get("deprel", [0] * len(ex["tokens"]))
                for ex in examples
            ]
        else:
            labels = None
    
    # Tokenize with word-ID tracking
    enc = tokenizer(
        toks,
        is_split_into_words=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )
    
    # Get word IDs for subword-to-word alignment
    word_ids = [enc.word_ids(i) for i in range(len(toks))]
    
    # Move tensors to device
    for k in enc:
        enc[k] = enc[k].to(device)
    
    # Keep original string labels for punctuation masking if needed
    deprels_str = None
    if labels is not None:
        # Check if labels contain strings (not already indexed)
        has_strings = any(
            isinstance(lbl, str)
            for sent in labels
            for lbl in (sent if isinstance(sent, list) else [sent])
        )
        if has_strings:
            deprels_str = labels
    
    # Convert string labels to indices if vocabulary provided
    if labels is not None and label_vocab is not None:
        label_indices = []
        for sent_labels in labels:
            indices = [
                label_vocab.get(lbl, 0) if isinstance(lbl, str) else lbl
                for lbl in sent_labels
            ]
            label_indices.append(indices)
        labels = label_indices
    
    if return_deprels:
        return enc, word_ids, heads, labels, deprels_str
    return enc, word_ids, heads, labels

