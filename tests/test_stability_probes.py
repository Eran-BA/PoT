"""
Tests for Stability Probes.

Tests the Phase 0 diagnostic probes that measure solver convergence behavior:
- Fixed-point residual (E_fp)
- Per-step delta (convergence curve)
- Noise sensitivity (E_noise)

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def small_model():
    """Create a small HybridPoHHRMSolver for testing."""
    from src.pot.models.sudoku_solver import HybridPoHHRMSolver
    
    model = HybridPoHHRMSolver(
        vocab_size=10,
        d_model=64,
        n_heads=4,
        H_layers=1,
        L_layers=1,
        d_ff=128,
        H_cycles=1,
        L_cycles=2,
        T=2,
        halt_max_steps=3,  # Need >1 for probes to work
        injection_mode="broadcast_memory",
        injection_kwargs={"memory_size": 4, "n_heads": 2},
    )
    model.eval()
    return model


@pytest.fixture
def small_model_no_act():
    """Create a small HybridPoHHRMSolver without ACT."""
    from src.pot.models.sudoku_solver import HybridPoHHRMSolver
    
    model = HybridPoHHRMSolver(
        vocab_size=10,
        d_model=64,
        n_heads=4,
        H_layers=1,
        L_layers=1,
        d_ff=128,
        H_cycles=1,
        L_cycles=2,
        T=2,
        halt_max_steps=1,  # No ACT
    )
    model.eval()
    return model


@pytest.fixture
def dummy_dataloader():
    """Create a small dataloader for testing."""
    B = 4
    inputs = torch.randint(0, 10, (B, 81))
    labels = torch.randint(1, 10, (B, 81))
    puzzle_ids = torch.zeros(B, dtype=torch.long)
    
    # Match the dict format expected by the probe
    class DictDataset:
        def __init__(self, inp, lbl, pid):
            self.inp = inp
            self.lbl = lbl
            self.pid = pid
        def __len__(self):
            return len(self.inp)
        def __getitem__(self, idx):
            return {'input': self.inp[idx], 'label': self.lbl[idx], 'puzzle_id': self.pid[idx]}
    
    dataset = DictDataset(inputs, labels, puzzle_ids)
    return DataLoader(dataset, batch_size=2)


class TestForwardWithProbes:
    """Test the forward_with_probes method."""
    
    def test_returns_required_keys(self, small_model):
        """forward_with_probes should return all required keys."""
        inputs = torch.randint(0, 10, (2, 81))
        puzzle_ids = torch.zeros(2, dtype=torch.long)
        
        with torch.no_grad():
            result = small_model.forward_with_probes(inputs, puzzle_ids)
        
        assert 'logits' in result
        assert 'intermediate_hiddens' in result
        assert 'final_carry' in result
        assert 'input_emb' in result
        assert 'steps' in result
    
    def test_final_carry_is_act_carry(self, small_model):
        """final_carry should be an ACTCarry with z_H and z_L."""
        from src.pot.models.hybrid_hrm import ACTCarry
        
        inputs = torch.randint(0, 10, (2, 81))
        puzzle_ids = torch.zeros(2, dtype=torch.long)
        
        with torch.no_grad():
            result = small_model.forward_with_probes(inputs, puzzle_ids)
        
        carry = result['final_carry']
        assert isinstance(carry, ACTCarry)
        assert carry.z_H.shape == (2, 81, 64)
        assert carry.z_L.shape == (2, 81, 64)
    
    def test_intermediate_hiddens_length(self, small_model):
        """Should have one hidden per ACT step."""
        inputs = torch.randint(0, 10, (2, 81))
        puzzle_ids = torch.zeros(2, dtype=torch.long)
        
        with torch.no_grad():
            result = small_model.forward_with_probes(inputs, puzzle_ids)
        
        assert len(result['intermediate_hiddens']) == 3  # halt_max_steps=3
    
    def test_no_act_returns_none_carry(self, small_model_no_act):
        """Without ACT, final_carry should be None."""
        inputs = torch.randint(0, 10, (2, 81))
        puzzle_ids = torch.zeros(2, dtype=torch.long)
        
        with torch.no_grad():
            result = small_model_no_act.forward_with_probes(inputs, puzzle_ids)
        
        assert result['final_carry'] is None


class TestComputeStabilityProbes:
    """Test the compute_stability_probes function."""
    
    def test_returns_probe_metrics(self, small_model, dummy_dataloader):
        """Should return dict with probe/ prefixed keys."""
        from src.training.stability_probes import compute_stability_probes
        
        result = compute_stability_probes(
            small_model, dummy_dataloader, torch.device('cpu'), max_batches=2
        )
        
        assert 'probe/E_fp_H' in result
        assert 'probe/E_fp_L' in result
        assert 'probe/E_noise' in result
        assert 'probe/delta_ratio' in result
    
    def test_delta_steps_present(self, small_model, dummy_dataloader):
        """Should have delta_step entries for each consecutive pair."""
        from src.training.stability_probes import compute_stability_probes
        
        result = compute_stability_probes(
            small_model, dummy_dataloader, torch.device('cpu'), max_batches=2
        )
        
        # halt_max_steps=3 means 3 intermediate hiddens, 2 deltas
        assert 'probe/delta_step_1' in result
        assert 'probe/delta_step_2' in result
    
    def test_values_are_nonnegative(self, small_model, dummy_dataloader):
        """All probe values should be non-negative (they're norms)."""
        from src.training.stability_probes import compute_stability_probes
        
        result = compute_stability_probes(
            small_model, dummy_dataloader, torch.device('cpu'), max_batches=2
        )
        
        for key, value in result.items():
            assert value >= 0, f"{key} is negative: {value}"
    
    def test_no_act_returns_empty(self, small_model_no_act, dummy_dataloader):
        """Without ACT, probes should return empty dict."""
        from src.training.stability_probes import compute_stability_probes
        
        result = compute_stability_probes(
            small_model_no_act, dummy_dataloader, torch.device('cpu'), max_batches=2
        )
        
        assert result == {}
    
    def test_max_batches_limits_computation(self, small_model, dummy_dataloader):
        """max_batches=1 should still produce valid results."""
        from src.training.stability_probes import compute_stability_probes
        
        result = compute_stability_probes(
            small_model, dummy_dataloader, torch.device('cpu'), max_batches=1
        )
        
        assert len(result) > 0
        assert 'probe/E_fp_H' in result
    
    def test_noise_std_affects_e_noise(self, small_model, dummy_dataloader):
        """Higher noise_std should generally produce larger E_noise."""
        from src.training.stability_probes import compute_stability_probes
        
        torch.manual_seed(42)
        result_low = compute_stability_probes(
            small_model, dummy_dataloader, torch.device('cpu'),
            max_batches=2, noise_std=0.001
        )
        
        torch.manual_seed(42)
        result_high = compute_stability_probes(
            small_model, dummy_dataloader, torch.device('cpu'),
            max_batches=2, noise_std=1.0
        )
        
        # Much higher noise should produce larger sensitivity
        assert result_high['probe/E_noise'] > result_low['probe/E_noise']


class TestActForwardReturnFinalCarry:
    """Test that act_forward correctly returns final carry."""
    
    def test_carry_not_returned_by_default(self, small_model):
        """By default, final_carry should not be in result."""
        inputs = torch.randint(0, 10, (2, 81))
        puzzle_ids = torch.zeros(2, dtype=torch.long)
        input_emb = small_model._compute_input_embedding(inputs, puzzle_ids)
        
        with torch.no_grad():
            result = small_model.act_forward(input_emb)
        
        assert 'final_carry' not in result
    
    def test_carry_returned_when_requested(self, small_model):
        """When return_final_carry=True, carry should be in result."""
        inputs = torch.randint(0, 10, (2, 81))
        puzzle_ids = torch.zeros(2, dtype=torch.long)
        input_emb = small_model._compute_input_embedding(inputs, puzzle_ids)
        
        with torch.no_grad():
            result = small_model.act_forward(input_emb, return_final_carry=True)
        
        assert 'final_carry' in result
        assert result['final_carry'].z_H.shape == (2, 81, 64)
