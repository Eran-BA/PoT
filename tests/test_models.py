#!/usr/bin/env python3
"""
Basic unit tests for PoT models.

Tests model initialization, forward passes, and basic functionality.

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import unittest
import torch
from transformers import AutoTokenizer

from src.models import PoHParser, BaselineParser
from src.models.pointer_block import PointerMoHTransformerBlock
from src.models.layers import BiaffinePointer, BiaffineLabeler
from src.data.loaders import create_dummy_dataset
from src.data.collate import collate_batch


class TestPointerBlock(unittest.TestCase):
    """Tests for PointerMoHTransformerBlock."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 768
        self.n_heads = 8
        self.batch_size = 2
        self.seq_len = 10
        self.device = torch.device("cpu")
    
    def test_initialization(self):
        """Test block initialization."""
        block = PointerMoHTransformerBlock(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=2048,
            halting_mode="fixed",
            max_inner_iters=1
        )
        self.assertIsNotNone(block)
        self.assertEqual(block.d_model, self.d_model)
        self.assertEqual(block.n_heads, self.n_heads)
    
    def test_forward_fixed(self):
        """Test forward pass with fixed halting."""
        block = PointerMoHTransformerBlock(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=2048,
            halting_mode="fixed",
            max_inner_iters=1
        ).to(self.device)
        
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, device=self.device)
        y, aux = block(x)
        
        # Check output shape
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.d_model))
        
        # Check auxiliary outputs
        self.assertIn('alphas', aux)
        self.assertIn('inner_iters_used', aux)
        self.assertEqual(aux['inner_iters_used'], 1)
    
    def test_forward_multiple_iters(self):
        """Test forward pass with multiple iterations."""
        block = PointerMoHTransformerBlock(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=2048,
            halting_mode="fixed",
            max_inner_iters=3
        ).to(self.device)
        
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, device=self.device)
        y, aux = block(x)
        
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertEqual(aux['inner_iters_used'], 3)
    
    def test_collect_all(self):
        """Test collect_all mode for deep supervision."""
        block = PointerMoHTransformerBlock(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=2048,
            max_inner_iters=2
        ).to(self.device)
        
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, device=self.device)
        y, aux = block(x, collect_all=True)
        
        # Check routed sequence is collected
        self.assertIn('routed', aux)
        routed = aux['routed']
        self.assertEqual(routed.shape, (self.batch_size, 2, self.seq_len, self.d_model))


class TestLayers(unittest.TestCase):
    """Tests for layer components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 768
        self.batch_size = 2
        self.seq_len = 10
        self.n_labels = 50
        self.device = torch.device("cpu")
    
    def test_biaffine_pointer(self):
        """Test BiaffinePointer layer."""
        pointer = BiaffinePointer(d=self.d_model).to(self.device)
        
        dep = torch.randn(self.batch_size, self.seq_len, self.d_model, device=self.device)
        head = torch.randn(self.batch_size, self.seq_len, self.d_model, device=self.device)
        mask_dep = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool, device=self.device)
        mask_head = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool, device=self.device)
        
        logits = pointer(dep, head, mask_dep, mask_head)
        
        # Should output [B, T, T+1] (T+1 for ROOT)
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.seq_len + 1))
    
    def test_biaffine_labeler(self):
        """Test BiaffineLabeler layer."""
        labeler = BiaffineLabeler(d=self.d_model, n_labels=self.n_labels).to(self.device)
        
        dep = torch.randn(self.batch_size, self.seq_len, self.d_model, device=self.device)
        head = torch.randn(self.batch_size, self.seq_len + 1, self.d_model, device=self.device)  # +1 for ROOT
        head_indices = torch.randint(0, self.seq_len + 1, (self.batch_size, self.seq_len), device=self.device)
        mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool, device=self.device)
        
        logits = labeler(dep, head, head_indices, mask)
        
        # Should output [B, T, n_labels]
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.n_labels))


class TestParsers(unittest.TestCase):
    """Tests for full parser models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.n_labels = 10
        
        # Create minimal dummy data
        self.data = create_dummy_dataset(n_samples=4)
    
    def test_poh_parser_initialization(self):
        """Test PoH parser initialization."""
        parser = PoHParser(
            enc_name="distilbert-base-uncased",
            d_model=768,
            n_heads=8,
            d_ff=2048,
            n_labels=self.n_labels,
            use_labels=True
        ).to(self.device)
        
        self.assertIsNotNone(parser)
        self.assertEqual(parser.d_model, 768)
    
    def test_baseline_parser_initialization(self):
        """Test baseline parser initialization."""
        parser = BaselineParser(
            enc_name="distilbert-base-uncased",
            d_model=768,
            n_heads=8,
            d_ff=2048,
            n_labels=self.n_labels,
            use_labels=True
        ).to(self.device)
        
        self.assertIsNotNone(parser)
        self.assertEqual(parser.d_model, 768)
    
    def test_poh_parser_forward(self):
        """Test PoH parser forward pass."""
        parser = PoHParser(
            enc_name="distilbert-base-uncased",
            n_labels=self.n_labels,
            use_labels=True,
            max_inner_iters=1
        ).to(self.device)
        
        # Collate batch
        enc, word_ids, heads, labels = collate_batch(
            self.data[:2],
            self.tokenizer,
            self.device
        )
        
        # Forward pass
        loss, metrics = parser(enc, word_ids, heads, labels)
        
        # Check outputs
        self.assertIsInstance(loss.item(), float)
        self.assertIn('uas', metrics)
        self.assertIn('las', metrics)
        self.assertIn('tokens', metrics)
        self.assertGreaterEqual(metrics['uas'], 0.0)
        self.assertLessEqual(metrics['uas'], 1.0)
    
    def test_baseline_parser_forward(self):
        """Test baseline parser forward pass."""
        parser = BaselineParser(
            enc_name="distilbert-base-uncased",
            n_labels=self.n_labels,
            use_labels=True
        ).to(self.device)
        
        # Collate batch
        enc, word_ids, heads, labels = collate_batch(
            self.data[:2],
            self.tokenizer,
            self.device
        )
        
        # Forward pass
        loss, metrics = parser(enc, word_ids, heads, labels)
        
        # Check outputs
        self.assertIsInstance(loss.item(), float)
        self.assertIn('uas', metrics)
        self.assertIn('las', metrics)
        self.assertGreaterEqual(metrics['uas'], 0.0)
        self.assertLessEqual(metrics['uas'], 1.0)
    
    def test_poh_eval_mode(self):
        """Test PoH parser in eval mode returns predictions."""
        parser = PoHParser(
            enc_name="distilbert-base-uncased",
            n_labels=self.n_labels,
            use_labels=True
        ).to(self.device)
        
        parser.eval()
        
        # Collate batch
        enc, word_ids, heads, labels = collate_batch(
            self.data[:2],
            self.tokenizer,
            self.device
        )
        
        # Forward pass
        with torch.no_grad():
            loss, metrics = parser(enc, word_ids, heads, labels)
        
        # Check predictions are returned
        self.assertIn('pred_heads', metrics)
        self.assertIn('pred_labels', metrics)
        self.assertIsInstance(metrics['pred_heads'], list)
        self.assertIsInstance(metrics['pred_labels'], list)


class TestDataPipeline(unittest.TestCase):
    """Tests for data loading and collation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    def test_dummy_dataset_creation(self):
        """Test dummy dataset creation."""
        data = create_dummy_dataset(n_samples=10)
        
        self.assertEqual(len(data), 10)
        self.assertIn('tokens', data[0])
        self.assertIn('head', data[0])
        self.assertIn('deprel', data[0])
    
    def test_collate_batch(self):
        """Test batch collation."""
        data = create_dummy_dataset(n_samples=4)
        
        enc, word_ids, heads, labels = collate_batch(
            data,
            self.tokenizer,
            self.device
        )
        
        # Check shapes
        self.assertIn('input_ids', enc)
        self.assertIsInstance(word_ids, list)
        self.assertIsInstance(heads, list)
        self.assertIsInstance(labels, list)
        self.assertEqual(len(word_ids), 4)
        self.assertEqual(len(heads), 4)


def run_tests():
    """Run all tests."""
    print("=" * 80)
    print("Running PoT Model Tests")
    print("=" * 80)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestPointerBlock))
    suite.addTests(loader.loadTestsFromTestCase(TestLayers))
    suite.addTests(loader.loadTestsFromTestCase(TestParsers))
    suite.addTests(loader.loadTestsFromTestCase(TestDataPipeline))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print()
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)

