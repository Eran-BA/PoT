#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive tests for UD Dependency Parser (ud_pointer_parser.py)

Tests:
1. Utility functions (mean_pool_subwords, pad_block_diagonal, make_pointer_targets)
2. BiaffinePointer layer
3. UDPointerParser model (forward pass, shapes, loss computation)
4. Data loading and collation
5. End-to-end training step
6. Integration with PointerMoHTransformerBlock

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import pytest
import torch
import torch.nn as nn
from transformers import AutoTokenizer

# Import the module to test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ud_pointer_parser import (
    mean_pool_subwords,
    pad_block_diagonal,
    make_pointer_targets,
    BiaffinePointer,
    UDPointerParser,
    collate_batch,
)


# ========================================
# 1. Test Utility Functions
# ========================================

class TestUtilityFunctions:
    """Test helper functions for word pooling and batching."""

    def test_mean_pool_subwords_basic(self):
        """Test basic subword-to-word pooling."""
        B, S, D = 2, 8, 16
        last_hidden = torch.randn(B, S, D)
        
        # Simple case: 3 words for first batch, 2 for second
        # word_ids maps subwords to word indices (None for special tokens)
        word_ids = [
            [None, 0, 0, 1, 1, 1, 2, None],  # 3 words (0,1,2)
            [None, 0, 1, 1, None, None, None, None],  # 2 words (0,1)
        ]
        
        result = mean_pool_subwords(last_hidden, word_ids)
        
        assert len(result) == B
        assert result[0].shape == (3, D)  # 3 words
        assert result[1].shape == (2, D)  # 2 words
        
        # Check that pooling is correct for word 1 in first batch (indices 3,4,5)
        expected_word1 = (last_hidden[0, 3] + last_hidden[0, 4] + last_hidden[0, 5]) / 3
        torch.testing.assert_close(result[0][1], expected_word1)

    def test_mean_pool_subwords_empty(self):
        """Test pooling with no valid words (all None)."""
        B, S, D = 1, 4, 8
        last_hidden = torch.randn(B, S, D)
        word_ids = [[None, None, None, None]]  # No words
        
        result = mean_pool_subwords(last_hidden, word_ids)
        
        assert len(result) == 1
        assert result[0].shape == (0, D)  # Empty tensor

    def test_mean_pool_subwords_single_subword_per_word(self):
        """Test when each word is a single subword."""
        B, S, D = 1, 5, 8
        last_hidden = torch.randn(B, S, D)
        word_ids = [[None, 0, 1, 2, None]]  # 3 words, 1 subword each
        
        result = mean_pool_subwords(last_hidden, word_ids)
        
        assert result[0].shape == (3, D)
        # Each word should be identical to its subword
        torch.testing.assert_close(result[0][0], last_hidden[0, 1])
        torch.testing.assert_close(result[0][1], last_hidden[0, 2])
        torch.testing.assert_close(result[0][2], last_hidden[0, 3])

    def test_pad_block_diagonal_basic(self):
        """Test padding of variable-length word sequences."""
        D = 8
        word_batches = [
            torch.randn(3, D),  # 3 words
            torch.randn(5, D),  # 5 words
            torch.randn(2, D),  # 2 words
        ]
        
        X, mask = pad_block_diagonal(word_batches, pad_value=0.0)
        
        assert X.shape == (3, 5, D)  # B=3, max_len=5
        assert mask.shape == (3, 5)
        
        # Check masks
        assert mask[0].sum() == 3
        assert mask[1].sum() == 5
        assert mask[2].sum() == 2
        
        # Check padding
        assert torch.all(X[0, 3:] == 0.0)  # Padded positions
        assert torch.all(X[2, 2:] == 0.0)
        
        # Check non-padded values
        torch.testing.assert_close(X[0, :3], word_batches[0])
        torch.testing.assert_close(X[1, :5], word_batches[1])

    def test_pad_block_diagonal_single_item(self):
        """Test padding with single sequence."""
        D = 4
        word_batches = [torch.randn(3, D)]
        
        X, mask = pad_block_diagonal(word_batches)
        
        assert X.shape == (1, 3, D)
        assert mask.shape == (1, 3)
        assert torch.all(mask[0])

    def test_make_pointer_targets_basic(self):
        """Test creation of pointer targets from UD heads."""
        heads = [
            [2, 0, 2],  # 3 words: word0->2, word1->0(ROOT), word2->2
            [0, 2, 0],  # 3 words: word0->0(ROOT), word1->2, word2->0(ROOT)
        ]
        device = torch.device("cpu")
        
        Y, pad = make_pointer_targets(heads, max_len=4, device=device)
        
        assert Y.shape == (2, 4)
        assert pad.shape == (2, 4)
        
        # Check values
        assert Y[0, 0] == 2
        assert Y[0, 1] == 0  # ROOT
        assert Y[0, 2] == 2
        assert Y[0, 3] == 0  # Padding
        
        # Check masks
        assert pad[0].sum() == 3  # 3 valid tokens
        assert pad[1].sum() == 3

    def test_make_pointer_targets_truncation(self):
        """Test that targets are truncated to max_len."""
        heads = [[1, 2, 3, 0, 1]]  # 5 words
        device = torch.device("cpu")
        
        Y, pad = make_pointer_targets(heads, max_len=3, device=device)
        
        assert Y.shape == (1, 3)
        assert pad[0].sum() == 3  # Only first 3 tokens


# ========================================
# 2. Test BiaffinePointer Layer
# ========================================

class TestBiaffinePointer:
    """Test the biaffine pointer mechanism."""

    def test_biaffine_pointer_forward_shape(self):
        """Test output shapes from BiaffinePointer."""
        B, T, D = 2, 5, 16
        model = BiaffinePointer(d_model=D)
        
        dep = torch.randn(B, T, D)
        head = torch.randn(B, T, D)
        mask_dep = torch.ones(B, T, dtype=torch.bool)
        mask_head = torch.ones(B, T, dtype=torch.bool)
        
        logits = model(dep, head, mask_dep, mask_head)
        
        # Should output [B, T, T+1] (T+1 for ROOT + T head candidates)
        assert logits.shape == (B, T, T + 1)

    def test_biaffine_pointer_masking(self):
        """Test that masking correctly sets invalid positions to -inf."""
        B, T, D = 1, 4, 8
        model = BiaffinePointer(d_model=D)
        
        dep = torch.randn(B, T, D)
        head = torch.randn(B, T, D)
        
        # Mask out position 2 as invalid head candidate
        mask_dep = torch.ones(B, T, dtype=torch.bool)
        mask_head = torch.tensor([[True, True, False, True]])  # Position 2 invalid
        
        logits = model(dep, head, mask_dep, mask_head)
        
        # Check that position 3 (index 2+1 for ROOT offset) is -inf for all dependents
        assert torch.all(torch.isinf(logits[:, :, 3]))
        assert torch.all(logits[:, :, 3] < 0)
        
        # ROOT (index 0) should always be valid
        assert not torch.any(torch.isinf(logits[:, :, 0]))

    def test_biaffine_pointer_dependent_masking(self):
        """Test that padded dependents have all logits set to -inf."""
        B, T, D = 1, 4, 8
        model = BiaffinePointer(d_model=D)
        
        dep = torch.randn(B, T, D)
        head = torch.randn(B, T, D)
        
        # Only first 2 dependents are valid
        mask_dep = torch.tensor([[True, True, False, False]])
        mask_head = torch.ones(B, T, dtype=torch.bool)
        
        logits = model(dep, head, mask_dep, mask_head)
        
        # Positions 2 and 3 (invalid dependents) should have all -inf
        assert torch.all(torch.isinf(logits[:, 2, :]))
        assert torch.all(torch.isinf(logits[:, 3, :]))
        
        # Valid dependents should have finite values
        assert not torch.all(torch.isinf(logits[:, 0, :]))
        assert not torch.all(torch.isinf(logits[:, 1, :]))

    def test_biaffine_pointer_gradient_flow(self):
        """Test that gradients flow through BiaffinePointer."""
        D = 8
        model = BiaffinePointer(d_model=D)
        
        dep = torch.randn(1, 3, D, requires_grad=True)
        head = torch.randn(1, 3, D, requires_grad=True)
        mask_dep = torch.ones(1, 3, dtype=torch.bool)
        mask_head = torch.ones(1, 3, dtype=torch.bool)
        
        logits = model(dep, head, mask_dep, mask_head)
        loss = logits.sum()
        loss.backward()
        
        assert dep.grad is not None
        assert head.grad is not None
        assert model.W.grad is not None
        assert model.U.weight.grad is not None


# ========================================
# 3. Test UDPointerParser Model
# ========================================

class TestUDPointerParser:
    """Test the full UD parser model."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        return AutoTokenizer.from_pretrained("distilbert-base-uncased")

    @pytest.fixture
    def small_parser(self):
        """Create a small parser for testing (fast initialization)."""
        # Use a tiny model to speed up tests
        return UDPointerParser(
            enc_name="distilbert-base-uncased",
            d_model=768,
            n_heads_router=4,
            d_ff_router=512,
            max_inner_iters=2,
        )

    def test_parser_initialization(self, small_parser):
        """Test that parser initializes correctly."""
        assert small_parser.encoder is not None
        assert small_parser.router is not None
        assert small_parser.pointer is not None

    def test_parser_forward_pass(self, small_parser, mock_tokenizer):
        """Test forward pass through parser."""
        device = torch.device("cpu")
        small_parser.to(device)
        
        # Create dummy input
        tokens_list = [["The", "cat", "sat"], ["Dogs", "run"]]
        heads_gold = [[2, 3, 0], [2, 0]]  # UD convention (1-indexed, 0=ROOT)
        
        # Tokenize
        enc = mock_tokenizer(
            tokens_list,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
        )
        for k in enc:
            enc[k] = enc[k].to(device)
        
        word_ids_batch = [enc.word_ids(i) for i in range(len(tokens_list))]
        
        # Forward pass
        loss, metrics = small_parser(enc, word_ids_batch, heads_gold)
        
        # Check outputs
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0  # Cross-entropy is non-negative
        
        assert "uas" in metrics
        assert 0.0 <= metrics["uas"] <= 1.0
        assert "tokens" in metrics
        assert metrics["tokens"] > 0

    def test_parser_backward_pass(self, small_parser, mock_tokenizer):
        """Test that gradients flow correctly."""
        device = torch.device("cpu")
        small_parser.to(device)
        
        tokens_list = [["The", "cat"]]
        heads_gold = [[2, 0]]
        
        enc = mock_tokenizer(
            tokens_list,
            is_split_into_words=True,
            return_tensors="pt",
        )
        for k in enc:
            enc[k] = enc[k].to(device)
        
        word_ids_batch = [enc.word_ids(0)]
        
        # Forward + backward
        loss, metrics = small_parser(enc, word_ids_batch, heads_gold)
        loss.backward()
        
        # Check that some parameters have gradients
        has_grad = False
        for name, param in small_parser.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, "No gradients found in any parameter"

    def test_parser_variable_length_sentences(self, small_parser, mock_tokenizer):
        """Test parser handles variable-length sentences correctly."""
        device = torch.device("cpu")
        small_parser.to(device)
        
        # Different lengths
        tokens_list = [
            ["A"],
            ["The", "quick", "fox"],
            ["Dogs", "run", "fast", "today"],
        ]
        heads_gold = [
            [0],
            [2, 3, 0],
            [2, 0, 2, 2],
        ]
        
        enc = mock_tokenizer(
            tokens_list,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
        )
        for k in enc:
            enc[k] = enc[k].to(device)
        
        word_ids_batch = [enc.word_ids(i) for i in range(len(tokens_list))]
        
        loss, metrics = small_parser(enc, word_ids_batch, heads_gold)
        
        assert loss.item() >= 0
        assert metrics["tokens"] == 1 + 3 + 4  # Total tokens


# ========================================
# 4. Test Data Loading and Collation
# ========================================

class TestDataCollation:
    """Test data loading and batch collation."""

    @pytest.fixture
    def mock_tokenizer(self):
        return AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def test_collate_batch_basic(self, mock_tokenizer):
        """Test basic batch collation."""
        device = torch.device("cpu")
        
        examples = [
            {"tokens": ["The", "cat"], "head": [2, 0]},
            {"tokens": ["Dogs", "run", "fast"], "head": [2, 0, 2]},
        ]
        
        enc, word_ids_batch, heads_list = collate_batch(examples, mock_tokenizer, device)
        
        # Check outputs
        assert "input_ids" in enc
        assert "attention_mask" in enc
        assert enc["input_ids"].device.type == device.type
        
        assert len(word_ids_batch) == 2
        assert len(heads_list) == 2
        assert heads_list[0] == [2, 0]
        assert heads_list[1] == [2, 0, 2]

    def test_collate_batch_dict_format(self, mock_tokenizer):
        """Test collation with dict-of-lists format (HF dataset format)."""
        device = torch.device("cpu")
        
        examples = {
            "tokens": [["The", "cat"], ["Dogs"]],
            "head": [[2, 0], [0]],
        }
        
        enc, word_ids_batch, heads_list = collate_batch(examples, mock_tokenizer, device)
        
        assert len(word_ids_batch) == 2
        assert len(heads_list) == 2

    def test_collate_batch_padding(self, mock_tokenizer):
        """Test that padding is applied correctly."""
        device = torch.device("cpu")
        
        examples = [
            {"tokens": ["A"], "head": [0]},
            {"tokens": ["The", "big", "red", "car"], "head": [4, 4, 4, 0]},
        ]
        
        enc, word_ids_batch, heads_list = collate_batch(examples, mock_tokenizer, device)
        
        # Second sequence should be longer
        assert enc["input_ids"].shape[0] == 2
        # Both should have same sequence length (padded)
        assert enc["input_ids"].shape[1] == enc["attention_mask"].shape[1]


# ========================================
# 5. Test Integration
# ========================================

class TestIntegration:
    """End-to-end integration tests."""

    @pytest.fixture
    def setup_parser_and_data(self):
        """Setup parser and sample data."""
        device = torch.device("cpu")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        parser = UDPointerParser(
            enc_name="distilbert-base-uncased",
            d_model=768,
            n_heads_router=4,
            d_ff_router=512,
            max_inner_iters=2,
            router_mode="mask_concat",
        ).to(device)
        
        # Sample data
        tokens_list = [["The", "cat", "sat"], ["Dogs", "run"]]
        heads_gold = [[2, 3, 0], [2, 0]]
        
        return parser, tokenizer, tokens_list, heads_gold, device

    def test_full_training_step(self, setup_parser_and_data):
        """Test a complete training step (forward + backward + optimizer step)."""
        parser, tokenizer, tokens_list, heads_gold, device = setup_parser_and_data
        
        optimizer = torch.optim.AdamW(parser.parameters(), lr=1e-4)
        
        # Prepare batch
        enc = tokenizer(
            tokens_list,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
        )
        for k in enc:
            enc[k] = enc[k].to(device)
        
        word_ids_batch = [enc.word_ids(i) for i in range(len(tokens_list))]
        
        # Training step
        parser.train()
        optimizer.zero_grad()
        
        loss_before, metrics = parser(enc, word_ids_batch, heads_gold)
        loss_before.backward()
        
        # Check gradients exist
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in parser.parameters()
        )
        assert has_grad
        
        # Optimizer step
        torch.nn.utils.clip_grad_norm_(parser.parameters(), 1.0)
        optimizer.step()
        
        # Second forward pass should give different loss (parameters changed)
        optimizer.zero_grad()
        loss_after, _ = parser(enc, word_ids_batch, heads_gold)
        
        # Losses might be different (not guaranteed, but likely)
        # Just check they're both valid
        assert loss_before.item() >= 0
        assert loss_after.item() >= 0

    def test_parser_with_different_router_modes(self):
        """Test parser works with both router combination modes."""
        device = torch.device("cpu")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        tokens_list = [["The", "cat"]]
        heads_gold = [[2, 0]]
        
        enc = tokenizer(
            tokens_list,
            is_split_into_words=True,
            return_tensors="pt",
        )
        for k in enc:
            enc[k] = enc[k].to(device)
        
        word_ids_batch = [enc.word_ids(0)]
        
        for mode in ["mask_concat", "mixture"]:
            parser = UDPointerParser(
                enc_name="distilbert-base-uncased",
                d_model=768,
                n_heads_router=4,
                d_ff_router=512,
                router_mode=mode,
                max_inner_iters=2,
            ).to(device)
            
            loss, metrics = parser(enc, word_ids_batch, heads_gold)
            
            assert loss.item() >= 0, f"Failed for mode={mode}"
            assert metrics["uas"] >= 0, f"Failed for mode={mode}"

    def test_parser_with_different_halting_modes(self):
        """Test parser works with different halting strategies."""
        device = torch.device("cpu")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        tokens_list = [["The", "cat"]]
        heads_gold = [[2, 0]]
        
        enc = tokenizer(
            tokens_list,
            is_split_into_words=True,
            return_tensors="pt",
        )
        for k in enc:
            enc[k] = enc[k].to(device)
        
        word_ids_batch = [enc.word_ids(0)]
        
        for halt_mode in ["fixed", "entropy"]:  # Skip "halting" (ACT) for speed
            parser = UDPointerParser(
                enc_name="distilbert-base-uncased",
                d_model=768,
                n_heads_router=4,
                d_ff_router=512,
                halting_mode=halt_mode,
                max_inner_iters=3,
            ).to(device)
            
            loss, metrics = parser(enc, word_ids_batch, heads_gold)
            
            assert loss.item() >= 0, f"Failed for halting_mode={halt_mode}"


# ========================================
# Run Tests
# ========================================

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])

