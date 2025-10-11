"""
Helper utilities for dependency parsing.

Common functions for processing tokenized sequences, padding batches,
and preparing target tensors.

Functions:
    mean_pool_subwords: Pool subword tokens to word-level representations
    pad_words: Pad variable-length word sequences to batch
    make_targets: Create padded target tensors from lists

Example:
    >>> from src.utils.helpers import mean_pool_subwords, pad_words, make_targets
    >>> # After tokenizing: last_hidden [B, S, D], word_ids [B, S]
    >>> words = mean_pool_subwords(last_hidden, word_ids)
    >>> X, mask = pad_words(words)
    >>> Y, pad_mask = make_targets(heads_gold, X.size(1), device)

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

from typing import List, Tuple

import torch


def mean_pool_subwords(
    last_hidden: torch.Tensor,
    word_ids: List[List[int]]
) -> List[torch.Tensor]:
    """Pool subword token representations to word-level representations.
    
    Averages all subword pieces belonging to the same word. This is essential
    for dependency parsing, which operates at the word level, not subword level.
    
    Args:
        last_hidden: Subword representations from encoder [B, S, D]
                    where S is the subword sequence length
        word_ids: Word indices for each subword [B, S].
                 Each element is either an int (word index) or None (special token).
                 
    Returns:
        List of word-level tensors, one per batch element.
        Each tensor has shape [num_words, D].
        
    Example:
        >>> # Sentence: "The dog" -> subwords: ["The", "do", "##g"]
        >>> # word_ids: [None, 0, 1, 1, None]  # CLS, "The", "do", "##g", SEP
        >>> last_hidden = torch.randn(1, 5, 768)
        >>> word_ids = [[None, 0, 1, 1, None]]
        >>> words = mean_pool_subwords(last_hidden, word_ids)
        >>> print(words[0].shape)  # [2, 768] - two words
        
    Note:
        - Special tokens with word_id=None are ignored
        - Empty sequences (no valid words) return empty tensor [0, D]
        - Handles variable number of words per batch element
    """
    B, S, D = last_hidden.shape
    outs = []
    
    for b in range(B):
        ids = word_ids[b]
        accum, cnt = {}, {}
        
        # Accumulate subword vectors for each word
        for s, wid in enumerate(ids):
            if wid is None:
                continue  # Skip special tokens
            if wid not in accum:
                accum[wid] = last_hidden[b, s]
                cnt[wid] = 1
            else:
                accum[wid] += last_hidden[b, s]
                cnt[wid] += 1
        
        # Handle empty sequence
        if not accum:
            outs.append(torch.zeros(0, D, device=last_hidden.device))
            continue
        
        # Average accumulated vectors
        n_words = max(accum.keys()) + 1
        M = torch.zeros(n_words, D, device=last_hidden.device)
        for wid, vec in accum.items():
            M[wid] = vec / cnt[wid]
        outs.append(M)
        
    return outs


def pad_words(
    word_batches: List[torch.Tensor],
    pad_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length word sequences to uniform batch tensor.
    
    Collates a list of word-level representations (each potentially different length)
    into a single padded tensor with an attention mask.
    
    Args:
        word_batches: List of word tensors [num_words_i, D]
        pad_value: Value to use for padding (default: 0.0)
        
    Returns:
        Tuple containing:
        - X: Padded tensor [B, max_len, D]
        - mask: Boolean mask [B, max_len] (True = valid word, False = padding)
        
    Example:
        >>> words = [torch.randn(5, 768), torch.randn(3, 768), torch.randn(7, 768)]
        >>> X, mask = pad_words(words)
        >>> print(X.shape)  # [3, 7, 768] - padded to max length 7
        >>> print(mask.shape)  # [3, 7]
        >>> print(mask[0].sum())  # 5 - first sequence has 5 valid words
        
    Note:
        - Handles empty sequences (returns [B, 0, 1] and empty mask)
        - All sequences padded to length of longest sequence in batch
        - Mask is essential for masking padding in attention and loss computation
    """
    B = len(word_batches)
    max_len = max(w.size(0) for w in word_batches) if B > 0 else 0
    D = word_batches[0].size(1) if max_len > 0 else 1
    
    # Create padded tensor
    X = word_batches[0].new_full((B, max_len, D), pad_value)
    mask = torch.zeros(B, max_len, dtype=torch.bool, device=X.device)
    
    # Fill in actual word representations
    for b, w in enumerate(word_batches):
        L = w.size(0)
        if L > 0:
            X[b, :L] = w
            mask[b, :L] = True
            
    return X, mask


def make_targets(
    heads: List[List[int]],
    max_len: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create padded target tensors from variable-length head lists.
    
    Converts list of gold dependency heads to padded batch tensors.
    Heads are 0-indexed (0 = ROOT, 1-n = word positions).
    
    Args:
        heads: List of head sequences [B]. Each sequence is List[int] of length n_words.
        max_len: Maximum sequence length to pad to
        device: Device to place tensors on
        
    Returns:
        Tuple containing:
        - Y: Target tensor [B, max_len] with head indices
        - pad: Boolean mask [B, max_len] (True = valid position, False = padding)
        
    Example:
        >>> # Two sentences with different lengths
        >>> heads = [[0, 2, 0], [0, 1, 0, 3]]  # First has 3 words, second has 4
        >>> Y, pad = make_targets(heads, max_len=5, device='cpu')
        >>> print(Y.shape)  # [2, 5]
        >>> print(pad.shape)  # [2, 5]
        >>> print(Y[0, :3])  # [0, 2, 0] - first sentence heads
        >>> print(pad[0])  # [T, T, T, F, F] - 3 valid, 2 padding
        
    Note:
        - Sequences longer than max_len are truncated
        - Padded positions have value 0 but pad mask is False
        - Always use pad mask to exclude padding from loss computation
    """
    B = len(heads)
    Y = torch.zeros(B, max_len, dtype=torch.long, device=device)
    pad = torch.zeros(B, max_len, dtype=torch.bool, device=device)
    
    for b, h in enumerate(heads):
        n = min(len(h), max_len)
        if n > 0:
            Y[b, :n] = torch.tensor(h[:n], device=device)
            pad[b, :n] = True
            
    return Y, pad

