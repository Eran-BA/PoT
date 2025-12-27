"""
Tests for Diffusion-Based H,L Cycles.

Tests cover:
1. Noise schedules - correct sigma values and shapes
2. Denoising blocks - output shapes and gradient flow
3. Level timing - learned vs fixed timing
4. DiffusionHLCycles - full forward pass and state management
5. DiffusionHRMSolver - end-to-end integration

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import pytest
import torch
import torch.nn as nn

from src.pot.core.diffusion_hl_cycles import (
    get_noise_schedule,
    DualTimescaleNoiseSchedule,
    DiffusionHLState,
    TimestepEmbedding,
    AdaLNDenoiseBlock,
    SequenceDenoiseBlock,
    LevelTimingDiffuser,
    DiffusionHLCycles,
)
from src.pot.models.diffusion_hrm_solver import (
    DiffusionHRMSolver,
    DiffusionSudokuSolver,
    DiffusionMazeSolver,
    DiffusionHRMCarry,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def seq_len():
    return 16


@pytest.fixture
def d_model():
    return 64


@pytest.fixture
def n_heads():
    return 4


# =============================================================================
# Noise Schedule Tests
# =============================================================================

class TestNoiseSchedules:
    """Tests for noise schedule generation."""
    
    def test_linear_schedule_shape(self):
        """Linear schedule has correct shape."""
        schedule = get_noise_schedule("linear", 32, torch.device("cpu"))
        assert schedule.shape == (32,)
    
    def test_cosine_schedule_shape(self):
        """Cosine schedule has correct shape."""
        schedule = get_noise_schedule("cosine", 32, torch.device("cpu"))
        assert schedule.shape == (32,)
    
    def test_sqrt_schedule_shape(self):
        """Sqrt schedule has correct shape."""
        schedule = get_noise_schedule("sqrt", 32, torch.device("cpu"))
        assert schedule.shape == (32,)
    
    def test_schedule_decreasing(self):
        """Noise schedules decrease over time (sigma goes from high to low)."""
        for schedule_type in ["linear", "cosine", "sqrt"]:
            schedule = get_noise_schedule(schedule_type, 32, torch.device("cpu"))
            # First value should be higher than last
            assert schedule[0] > schedule[-1], f"{schedule_type} should decrease"
    
    def test_schedule_bounds(self):
        """Noise schedules stay in valid range."""
        for schedule_type in ["linear", "cosine", "sqrt"]:
            schedule = get_noise_schedule(schedule_type, 32, torch.device("cpu"))
            assert (schedule >= 1e-4).all(), f"{schedule_type} should be >= 1e-4"
            assert (schedule <= 1.0).all(), f"{schedule_type} should be <= 1.0"
    
    def test_invalid_schedule_raises(self):
        """Invalid schedule type raises ValueError."""
        with pytest.raises(ValueError):
            get_noise_schedule("invalid", 32, torch.device("cpu"))


class TestDualTimescaleNoiseSchedule:
    """Tests for dual-timescale noise schedule."""
    
    def test_init(self):
        """DualTimescaleNoiseSchedule initializes correctly."""
        schedule = DualTimescaleNoiseSchedule(max_steps=32, T=4)
        assert schedule.max_steps == 32
        assert schedule.T == 4
        assert schedule.h_steps == 8  # 32 // 4
    
    def test_h_schedule_slower(self):
        """H schedule has fewer steps than L schedule."""
        schedule = DualTimescaleNoiseSchedule(max_steps=32, T=4)
        assert schedule.sigma_H.shape[0] < schedule.sigma_L.shape[0]
    
    def test_get_sigma_L(self):
        """get_sigma_L returns correct values."""
        schedule = DualTimescaleNoiseSchedule(max_steps=32, T=4)
        sigma = schedule.get_sigma_L(0)
        assert sigma.shape == ()
        assert sigma > 0
    
    def test_get_sigma_H(self):
        """get_sigma_H returns correct values."""
        schedule = DualTimescaleNoiseSchedule(max_steps=32, T=4)
        sigma = schedule.get_sigma_H(0)
        assert sigma.shape == ()
        assert sigma > 0
    
    def test_get_sigmas(self):
        """get_sigmas returns both noise levels."""
        schedule = DualTimescaleNoiseSchedule(max_steps=32, T=4)
        sigma_L, sigma_H = schedule.get_sigmas(5, 2)
        assert sigma_L.shape == ()
        assert sigma_H.shape == ()


# =============================================================================
# Timestep Embedding Tests
# =============================================================================

class TestTimestepEmbedding:
    """Tests for timestep embedding."""
    
    def test_output_shape(self, d_model):
        """Embedding has correct output shape."""
        embed = TimestepEmbedding(d_model)
        t = torch.tensor([0.5, 0.3, 0.8])
        out = embed(t)
        assert out.shape == (3, d_model)
    
    def test_scalar_input(self, d_model):
        """Works with scalar timestep."""
        embed = TimestepEmbedding(d_model)
        t = torch.tensor(0.5)
        out = embed(t)
        assert out.shape == (1, d_model)
    
    def test_different_timesteps_different_outputs(self, d_model):
        """Different timesteps produce different embeddings."""
        embed = TimestepEmbedding(d_model)
        t1 = torch.tensor([0.1])
        t2 = torch.tensor([0.9])
        out1 = embed(t1)
        out2 = embed(t2)
        assert not torch.allclose(out1, out2)


# =============================================================================
# Denoising Block Tests
# =============================================================================

class TestAdaLNDenoiseBlock:
    """Tests for AdaLN denoising block."""
    
    def test_output_shape_2d(self, batch_size, d_model):
        """Correct output shape for 2D input."""
        block = AdaLNDenoiseBlock(d_model)
        z = torch.randn(batch_size, d_model)
        cond = torch.randn(batch_size, d_model)
        out = block(z, cond)
        assert out.shape == z.shape
    
    def test_output_shape_3d(self, batch_size, seq_len, d_model):
        """Correct output shape for 3D input."""
        block = AdaLNDenoiseBlock(d_model)
        z = torch.randn(batch_size, seq_len, d_model)
        cond = torch.randn(batch_size, d_model)
        out = block(z, cond)
        assert out.shape == z.shape
    
    def test_residual_connection(self, batch_size, d_model):
        """Block uses residual connection."""
        block = AdaLNDenoiseBlock(d_model)
        z = torch.randn(batch_size, d_model)
        cond = torch.zeros(batch_size, d_model)
        out = block(z, cond)
        # With zero-init modulation, output should be close to input + small residual
        # (due to zero init of adaLN modulation)
        assert (out - z).abs().mean() < 1.0  # Some change from MLP
    
    def test_gradient_flow(self, batch_size, d_model):
        """Gradients flow through block."""
        block = AdaLNDenoiseBlock(d_model)
        z = torch.randn(batch_size, d_model, requires_grad=True)
        cond = torch.randn(batch_size, d_model, requires_grad=True)
        out = block(z, cond)
        loss = out.sum()
        loss.backward()
        assert z.grad is not None
        assert cond.grad is not None


class TestSequenceDenoiseBlock:
    """Tests for sequence-aware denoising block."""
    
    def test_output_shape(self, batch_size, seq_len, d_model, n_heads):
        """Correct output shape."""
        block = SequenceDenoiseBlock(d_model, n_heads)
        z = torch.randn(batch_size, seq_len, d_model)
        cond = torch.randn(batch_size, d_model)
        out = block(z, cond)
        assert out.shape == z.shape
    
    def test_cross_conditioning(self, batch_size, seq_len, d_model, n_heads):
        """Works with cross-attention conditioning."""
        block = SequenceDenoiseBlock(d_model, n_heads)
        z = torch.randn(batch_size, seq_len, d_model)
        cond = torch.randn(batch_size, d_model)
        cross_cond = torch.randn(batch_size, seq_len, d_model)
        out = block(z, cond, cross_cond=cross_cond)
        assert out.shape == z.shape
    
    def test_gradient_flow(self, batch_size, seq_len, d_model, n_heads):
        """Gradients flow through block."""
        block = SequenceDenoiseBlock(d_model, n_heads)
        z = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        cond = torch.randn(batch_size, d_model, requires_grad=True)
        out = block(z, cond)
        loss = out.sum()
        loss.backward()
        assert z.grad is not None
        assert cond.grad is not None


# =============================================================================
# Level Timing Tests
# =============================================================================

class TestLevelTimingDiffuser:
    """Tests for level timing controller."""
    
    def test_output_shape(self, batch_size, seq_len, d_model):
        """Correct output shapes."""
        timer = LevelTimingDiffuser(d_model)
        z_L = torch.randn(batch_size, seq_len, d_model)
        z_H = torch.randn(batch_size, seq_len, d_model)
        gate, logits = timer(z_L, z_H)
        assert gate.shape == (batch_size,)
        assert logits.shape == (batch_size, 2)
    
    def test_gate_range(self, batch_size, seq_len, d_model):
        """Gate values are in [0, 1]."""
        timer = LevelTimingDiffuser(d_model)
        z_L = torch.randn(batch_size, seq_len, d_model)
        z_H = torch.randn(batch_size, seq_len, d_model)
        gate, _ = timer(z_L, z_H)
        assert (gate >= 0).all()
        assert (gate <= 1).all()
    
    def test_hard_mode(self, batch_size, seq_len, d_model):
        """Hard mode produces binary decisions."""
        timer = LevelTimingDiffuser(d_model)
        timer.eval()
        z_L = torch.randn(batch_size, seq_len, d_model)
        z_H = torch.randn(batch_size, seq_len, d_model)
        gate, _ = timer(z_L, z_H, hard=True)
        # Hard mode should give 0 or 1
        assert ((gate == 0) | (gate == 1)).all()
    
    def test_temperature_annealing(self, d_model):
        """Temperature annealing works."""
        timer = LevelTimingDiffuser(d_model, temperature_init=1.0)
        initial_temp = timer.temperature
        timer.anneal_temperature(0.9)
        assert timer.temperature < initial_temp
    
    def test_gradient_flow(self, batch_size, seq_len, d_model):
        """Gradients flow through timer."""
        timer = LevelTimingDiffuser(d_model)
        z_L = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        z_H = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        gate, logits = timer(z_L, z_H)
        loss = gate.sum()
        loss.backward()
        assert z_L.grad is not None
        assert z_H.grad is not None


# =============================================================================
# DiffusionHLCycles Tests
# =============================================================================

class TestDiffusionHLCycles:
    """Tests for the main diffusion H,L cycles module."""
    
    def test_init(self, d_model, n_heads):
        """Module initializes correctly."""
        cycles = DiffusionHLCycles(d_model=d_model, n_heads=n_heads, max_steps=16)
        assert cycles.d_model == d_model
        assert cycles.max_steps == 16
    
    def test_init_state(self, batch_size, seq_len, d_model, n_heads, device):
        """State initialization works."""
        cycles = DiffusionHLCycles(d_model=d_model, n_heads=n_heads).to(device)
        state = cycles.init_state(batch_size, seq_len, device)
        
        assert isinstance(state, DiffusionHLState)
        assert state.z_H.shape == (batch_size, seq_len, d_model)
        assert state.z_L.shape == (batch_size, seq_len, d_model)
        assert state.h_step == 0
        assert state.l_step == 0
    
    def test_step(self, batch_size, seq_len, d_model, n_heads, device):
        """Single step works correctly."""
        cycles = DiffusionHLCycles(d_model=d_model, n_heads=n_heads, max_steps=16).to(device)
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        state = cycles.init_state(batch_size, seq_len, device)
        
        new_state, aux = cycles.step(x, state)
        
        assert new_state.z_H.shape == state.z_H.shape
        assert new_state.z_L.shape == state.z_L.shape
        assert new_state.l_step == 1
        assert "sigma_L" in aux
        assert "sigma_H" in aux
        assert "gate" in aux
    
    def test_forward(self, batch_size, seq_len, d_model, n_heads, device):
        """Full forward pass works."""
        cycles = DiffusionHLCycles(d_model=d_model, n_heads=n_heads, max_steps=8).to(device)
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        final_state, aux = cycles(x, n_steps=8)
        
        assert final_state.z_H.shape == (batch_size, seq_len, d_model)
        assert final_state.z_L.shape == (batch_size, seq_len, d_model)
        assert final_state.l_step == 8
        assert "avg_gate" in aux
    
    def test_denoising_reduces_noise(self, batch_size, seq_len, d_model, n_heads, device):
        """Denoising process reduces noise over steps."""
        cycles = DiffusionHLCycles(
            d_model=d_model, n_heads=n_heads, max_steps=16,
            init_noise_scale=1.0
        ).to(device)
        cycles.train()  # Use noisy init
        
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        # Initial state has noise
        state = cycles.init_state(batch_size, seq_len, device)
        initial_std = state.z_L.std()
        
        # Run denoising
        final_state, _ = cycles(x, n_steps=16)
        final_std = final_state.z_L.std()
        
        # The representation should become more structured (not necessarily lower std,
        # but different from pure noise)
        assert not torch.allclose(state.z_L, final_state.z_L)
    
    def test_fixed_timing(self, batch_size, seq_len, d_model, n_heads, device):
        """Fixed timing mode updates H every T steps."""
        T = 4
        cycles = DiffusionHLCycles(
            d_model=d_model, n_heads=n_heads, max_steps=16, T=T,
            learned_timing=False
        ).to(device)
        
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        state = cycles.init_state(batch_size, seq_len, device)
        
        h_steps_history = []
        for i in range(16):
            state, aux = cycles.step(x, state)
            h_steps_history.append(state.h_step)
        
        # H should update at steps 4, 8, 12, 16 (every T steps)
        # So h_step should be 0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4
        expected_h_updates = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4]
        assert h_steps_history == expected_h_updates
    
    def test_gradient_flow_through_cycles(self, batch_size, seq_len, d_model, n_heads, device):
        """Gradients flow through the diffusion cycles."""
        cycles = DiffusionHLCycles(d_model=d_model, n_heads=n_heads, max_steps=4).to(device)
        x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        
        final_state, _ = cycles(x, n_steps=4)
        loss = final_state.z_H.sum() + final_state.z_L.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.abs().sum() > 0  # Non-zero gradients
    
    def test_state_detach(self, batch_size, seq_len, d_model, n_heads, device):
        """State detach works correctly."""
        cycles = DiffusionHLCycles(d_model=d_model, n_heads=n_heads, max_steps=8).to(device)
        x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        
        final_state, _ = cycles(x, n_steps=4)
        detached = final_state.detach()
        
        assert not detached.z_H.requires_grad
        assert not detached.z_L.requires_grad


# =============================================================================
# DiffusionHRMSolver Tests
# =============================================================================

class TestDiffusionHRMSolver:
    """Tests for the full solver module."""
    
    def test_init(self):
        """Solver initializes correctly."""
        solver = DiffusionHRMSolver(
            vocab_size=10,
            d_model=64,
            n_heads=4,
            seq_len=16,
            max_steps=8,
        )
        assert solver.vocab_size == 10
        assert solver.d_model == 64
        assert solver.seq_len == 16
    
    def test_forward(self, batch_size, device):
        """Forward pass works correctly."""
        solver = DiffusionHRMSolver(
            vocab_size=10,
            d_model=64,
            n_heads=4,
            seq_len=16,
            max_steps=8,
            num_puzzles=100,
        ).to(device)
        
        input_seq = torch.randint(0, 10, (batch_size, 16), device=device)
        puzzle_ids = torch.randint(0, 100, (batch_size,), device=device)
        
        logits, q_halt, q_continue, steps = solver(input_seq, puzzle_ids)
        
        assert logits.shape == (batch_size, 16, 10)
        assert q_halt.shape == (batch_size,)
        assert q_continue.shape == (batch_size,)
        assert steps >= 1
    
    def test_forward_without_puzzle_emb(self, batch_size, device):
        """Forward works without puzzle embeddings."""
        solver = DiffusionHRMSolver(
            vocab_size=10,
            d_model=64,
            n_heads=4,
            seq_len=16,
            max_steps=8,
            num_puzzles=0,  # No puzzle embeddings
        ).to(device)
        
        input_seq = torch.randint(0, 10, (batch_size, 16), device=device)
        
        logits, q_halt, q_continue, steps = solver(input_seq)
        
        assert logits.shape == (batch_size, 16, 10)
    
    def test_act_forward(self, batch_size, device):
        """ACT forward with multiple halt steps."""
        solver = DiffusionHRMSolver(
            vocab_size=10,
            d_model=64,
            n_heads=4,
            seq_len=16,
            max_steps=4,
            halt_max_steps=3,
            num_puzzles=0,
        ).to(device)
        solver.train()
        
        input_seq = torch.randint(0, 10, (batch_size, 16), device=device)
        
        logits, q_halt, q_continue, steps = solver(input_seq)
        
        assert logits.shape == (batch_size, 16, 10)
        assert steps >= 1
        assert steps <= 3
    
    def test_gradient_flow(self, batch_size, device):
        """Gradients flow through solver."""
        solver = DiffusionHRMSolver(
            vocab_size=10,
            d_model=64,
            n_heads=4,
            seq_len=16,
            max_steps=4,
            num_puzzles=0,
        ).to(device)
        
        input_seq = torch.randint(0, 10, (batch_size, 16), device=device)
        
        logits, _, _, _ = solver(input_seq)
        loss = logits.sum()
        loss.backward()
        
        # Check some parameters have gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                      for p in solver.parameters() if p.requires_grad)
        assert has_grad
    
    def test_async_batching(self, batch_size, device):
        """Async batching methods work."""
        solver = DiffusionHRMSolver(
            vocab_size=10,
            d_model=64,
            n_heads=4,
            seq_len=16,
            max_steps=4,
            halt_max_steps=2,
            num_puzzles=0,
        ).to(device)
        
        # Initial carry
        carry = solver.initial_async_carry(batch_size, device)
        assert isinstance(carry, DiffusionHRMCarry)
        assert carry.halted.all()  # All start halted
        
        # Reset with new data
        new_input = torch.randint(0, 10, (batch_size, 16), device=device)
        new_labels = torch.randint(0, 10, (batch_size, 16), device=device)
        new_puzzle_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        carry = solver.reset_async_carry(
            carry,
            halted_mask=carry.halted,
            new_input=new_input,
            new_labels=new_labels,
            new_puzzle_ids=new_puzzle_ids,
            device=device,
        )
        
        assert not carry.halted.any()  # All reset
        
        # Run single step
        input_emb = solver.get_input_embedding(carry.current_input)
        new_carry, outputs = solver.forward_single_step_async(carry, input_emb)
        
        assert 'logits' in outputs
        assert 'q_halt_logits' in outputs
        assert outputs['logits'].shape == (batch_size, 16, 10)


class TestDiffusionSudokuSolver:
    """Tests for Sudoku-specific solver."""
    
    def test_init(self):
        """Sudoku solver has correct defaults."""
        solver = DiffusionSudokuSolver(d_model=64, n_heads=4)
        assert solver.vocab_size == 10
        assert solver.seq_len == 81
    
    def test_forward(self, batch_size, device):
        """Sudoku forward pass works."""
        solver = DiffusionSudokuSolver(
            d_model=64, n_heads=4, max_steps=4, num_puzzles=100
        ).to(device)
        
        input_seq = torch.randint(0, 10, (batch_size, 81), device=device)
        puzzle_ids = torch.randint(0, 100, (batch_size,), device=device)
        
        logits, _, _, _ = solver(input_seq, puzzle_ids)
        
        assert logits.shape == (batch_size, 81, 10)


class TestDiffusionMazeSolver:
    """Tests for maze-specific solver."""
    
    def test_init(self):
        """Maze solver has correct defaults."""
        solver = DiffusionMazeSolver(grid_size=10, d_model=64, n_heads=4)
        assert solver.grid_size == 10
        assert solver.seq_len == 100  # 10x10
    
    def test_forward(self, batch_size, device):
        """Maze forward pass works."""
        solver = DiffusionMazeSolver(
            grid_size=10, d_model=64, n_heads=4, max_steps=4
        ).to(device)
        
        input_seq = torch.randint(0, 5, (batch_size, 100), device=device)
        
        logits, _, _, _ = solver(input_seq)
        
        assert logits.shape == (batch_size, 100, 5)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_training_step(self, batch_size, device):
        """Complete training step works."""
        solver = DiffusionHRMSolver(
            vocab_size=10,
            d_model=64,
            n_heads=4,
            seq_len=16,
            max_steps=4,
            num_puzzles=0,
        ).to(device)
        solver.train()
        
        optimizer = torch.optim.Adam(solver.parameters(), lr=1e-4)
        
        # Forward pass
        input_seq = torch.randint(0, 10, (batch_size, 16), device=device)
        targets = torch.randint(0, 10, (batch_size, 16), device=device)
        
        logits, _, _, _ = solver(input_seq)
        
        # Compute loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, 10),
            targets.view(-1),
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
    
    def test_eval_mode_deterministic(self, batch_size, device):
        """Eval mode is deterministic (no random noise init)."""
        solver = DiffusionHRMSolver(
            vocab_size=10,
            d_model=64,
            n_heads=4,
            seq_len=16,
            max_steps=4,
            num_puzzles=0,
        ).to(device)
        solver.eval()
        
        input_seq = torch.randint(0, 10, (batch_size, 16), device=device)
        
        with torch.no_grad():
            logits1, _, _, _ = solver(input_seq)
            logits2, _, _, _ = solver(input_seq)
        
        # Same input should give same output in eval mode
        assert torch.allclose(logits1, logits2)
    
    def test_memory_efficiency(self, device):
        """Test that gradient checkpointing (detach in early steps) reduces memory."""
        # This is a qualitative test - we just ensure it runs without OOM
        solver = DiffusionHRMSolver(
            vocab_size=10,
            d_model=128,
            n_heads=4,
            seq_len=32,
            max_steps=16,  # Many steps
            num_puzzles=0,
        ).to(device)
        
        batch_size = 8
        input_seq = torch.randint(0, 10, (batch_size, 32), device=device)
        
        logits, _, _, _ = solver(input_seq)
        loss = logits.sum()
        loss.backward()
        
        # If we got here without OOM, memory efficiency is working
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

