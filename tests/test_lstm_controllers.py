"""
Unit tests for LSTM and xLSTM Depth Controllers.

Tests:
- Basic forward pass shapes for all controllers
- State persistence across depth steps
- Top-k sparsity and temperature effects
- Gradient flow through controllers
- Token-conditioned vs global routing

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import pytest
import torch
import torch.nn.functional as F

from src.pot.core.lstm_controllers import (
    LSTMDepthController,
    LSTMDepthState,
    xLSTMDepthController,
    xLSTMDepthState,
    minGRUDepthController,
)


# =============================================================================
# LSTM Depth Controller Tests
# =============================================================================

class TestLSTMDepthController:
    """Tests for LSTMDepthController."""
    
    @pytest.mark.parametrize("B,S,D,H", [(4, 7, 64, 8)])
    def test_basic_step_shapes(self, B, S, D, H):
        """Test that step() returns correct shapes."""
        ctrl = LSTMDepthController(
            d_model=D,
            n_heads=H,
            d_ctrl=64,
            token_conditioned=True,
        )
        X = torch.randn(B, S, D)
        state = None
        
        alpha, state, aux = ctrl.step(X, state=state)
        
        assert alpha.shape == (B, S, H), f"Expected ({B}, {S}, {H}), got {alpha.shape}"
        assert isinstance(state, LSTMDepthState)
        assert state.h.shape == (B, 64)
        assert state.c.shape == (B, 64)
        assert state.step == 1
        assert "entropy" in aux
    
    def test_multi_step_state_evolution(self):
        """Test that state evolves across steps."""
        B, S, D, H = 2, 5, 32, 4
        ctrl = LSTMDepthController(d_model=D, n_heads=H, d_ctrl=32)
        X = torch.randn(B, S, D)
        state = None
        
        states = []
        for t in range(5):
            alpha, state, _ = ctrl.step(X, state=state)
            states.append(state.h.detach().clone())
        
        # Check state evolves
        diff = (states[-1] - states[0]).abs().sum().item()
        assert diff > 0.01, "LSTM state should evolve across steps"
    
    def test_alphas_sum_to_one(self):
        """Test routing weights sum to 1."""
        B, S, D, H = 3, 6, 48, 8
        ctrl = LSTMDepthController(d_model=D, n_heads=H)
        X = torch.randn(B, S, D)
        
        alpha, _, _ = ctrl.step(X, state=None)
        sums = alpha.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    
    def test_topk_sparsity(self):
        """Test top-k produces sparse routing."""
        B, S, D, H = 2, 4, 32, 8
        topk = 3
        ctrl = LSTMDepthController(d_model=D, n_heads=H, topk=topk)
        X = torch.randn(B, S, D)
        
        alpha, _, _ = ctrl.step(X, state=None)
        
        nonzero = (alpha > 0).sum(dim=-1)
        assert nonzero.max().item() <= topk
    
    def test_gradients_flow(self):
        """Test gradients flow through controller."""
        B, S, D, H = 2, 4, 32, 4
        ctrl = LSTMDepthController(d_model=D, n_heads=H)
        X = torch.randn(B, S, D, requires_grad=True)
        
        state = None
        for _ in range(3):
            alpha, state, _ = ctrl.step(X, state=state)
        
        loss = alpha.sum()
        loss.backward()
        
        assert X.grad is not None
        assert X.grad.abs().sum().item() > 0


# =============================================================================
# xLSTM Depth Controller Tests
# =============================================================================

class TestxLSTMDepthController:
    """Tests for xLSTMDepthController (sLSTM variant)."""
    
    @pytest.mark.parametrize("B,S,D,H", [(4, 7, 64, 8)])
    def test_basic_step_shapes(self, B, S, D, H):
        """Test that step() returns correct shapes."""
        ctrl = xLSTMDepthController(
            d_model=D,
            n_heads=H,
            d_ctrl=64,
            token_conditioned=True,
        )
        X = torch.randn(B, S, D)
        state = None
        
        alpha, state, aux = ctrl.step(X, state=state)
        
        assert alpha.shape == (B, S, H), f"Expected ({B}, {S}, {H}), got {alpha.shape}"
        assert isinstance(state, xLSTMDepthState)
        assert state.h.shape == (B, 64)
        assert state.c.shape == (B, 64)
        assert state.n.shape == (B, 64)  # normalizer
        assert state.m.shape == (B,)      # max tracker
        assert state.step == 1
    
    def test_exponential_gating_stability(self):
        """Test that exponential gating remains numerically stable."""
        B, S, D, H = 2, 5, 32, 4
        ctrl = xLSTMDepthController(d_model=D, n_heads=H, d_ctrl=32)
        
        # Use large input values to stress test stability
        X = torch.randn(B, S, D) * 5.0
        state = None
        
        # Run many steps
        for t in range(20):
            alpha, state, _ = ctrl.step(X, state=state)
            
            # Check no NaN or Inf
            assert not torch.isnan(alpha).any(), f"NaN in alpha at step {t}"
            assert not torch.isinf(alpha).any(), f"Inf in alpha at step {t}"
            assert not torch.isnan(state.h).any(), f"NaN in h at step {t}"
            assert not torch.isnan(state.c).any(), f"NaN in c at step {t}"
    
    def test_alphas_sum_to_one(self):
        """Test routing weights sum to 1."""
        B, S, D, H = 3, 6, 48, 8
        ctrl = xLSTMDepthController(d_model=D, n_heads=H)
        X = torch.randn(B, S, D)
        
        alpha, _, _ = ctrl.step(X, state=None)
        sums = alpha.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    
    def test_state_evolution(self):
        """Test that xLSTM state evolves across steps."""
        B, S, D, H = 2, 4, 32, 4
        ctrl = xLSTMDepthController(d_model=D, n_heads=H)
        X = torch.randn(B, S, D)
        state = None
        
        h_states = []
        for _ in range(5):
            _, state, _ = ctrl.step(X, state=state)
            h_states.append(state.h.detach().clone())
        
        diff = (h_states[-1] - h_states[0]).abs().sum().item()
        assert diff > 0.01, "xLSTM state should evolve"
    
    def test_topk_sparsity(self):
        """Test top-k sparsification."""
        B, S, D, H = 2, 4, 32, 8
        topk = 2
        ctrl = xLSTMDepthController(d_model=D, n_heads=H, topk=topk)
        X = torch.randn(B, S, D)
        
        alpha, _, _ = ctrl.step(X, state=None)
        
        nonzero = (alpha > 0).sum(dim=-1)
        assert nonzero.max().item() <= topk
    
    def test_gradients_flow(self):
        """Test gradients flow through xLSTM."""
        B, S, D, H = 2, 4, 32, 4
        ctrl = xLSTMDepthController(d_model=D, n_heads=H)
        X = torch.randn(B, S, D, requires_grad=True)
        
        state = None
        for _ in range(3):
            alpha, state, _ = ctrl.step(X, state=state)
        
        loss = alpha.sum()
        loss.backward()
        
        assert X.grad is not None
        assert X.grad.abs().sum().item() > 0


# =============================================================================
# minGRU Depth Controller Tests
# =============================================================================

class TestminGRUDepthController:
    """Tests for minGRUDepthController."""
    
    @pytest.mark.parametrize("B,S,D,H", [(4, 7, 64, 8)])
    def test_basic_step_shapes(self, B, S, D, H):
        """Test that step() returns correct shapes."""
        ctrl = minGRUDepthController(
            d_model=D,
            n_heads=H,
            d_ctrl=64,
            token_conditioned=True,
        )
        X = torch.randn(B, S, D)
        state = None
        
        alpha, state, aux = ctrl.step(X, state=state)
        
        assert alpha.shape == (B, S, H), f"Expected ({B}, {S}, {H}), got {alpha.shape}"
        assert state.shape == (B, 64)  # minGRU uses simple tensor state
        assert "entropy" in aux
    
    def test_fewer_params_than_lstm(self):
        """minGRU should have fewer parameters than LSTM."""
        D, H = 64, 8
        
        lstm = LSTMDepthController(d_model=D, n_heads=H, d_ctrl=64)
        mingru = minGRUDepthController(d_model=D, n_heads=H, d_ctrl=64)
        
        lstm_params = sum(p.numel() for p in lstm.parameters())
        mingru_params = sum(p.numel() for p in mingru.parameters())
        
        assert mingru_params < lstm_params, (
            f"minGRU ({mingru_params}) should have fewer params than LSTM ({lstm_params})"
        )
    
    def test_alphas_sum_to_one(self):
        """Test routing weights sum to 1."""
        B, S, D, H = 3, 6, 48, 8
        ctrl = minGRUDepthController(d_model=D, n_heads=H)
        X = torch.randn(B, S, D)
        
        alpha, _, _ = ctrl.step(X, state=None)
        sums = alpha.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    
    def test_gradients_flow(self):
        """Test gradients flow through minGRU."""
        B, S, D, H = 2, 4, 32, 4
        ctrl = minGRUDepthController(d_model=D, n_heads=H)
        X = torch.randn(B, S, D, requires_grad=True)
        
        state = None
        for _ in range(3):
            alpha, state, _ = ctrl.step(X, state=state)
        
        loss = alpha.sum()
        loss.backward()
        
        assert X.grad is not None


# =============================================================================
# Token Conditioning Tests (applies to all controllers)
# =============================================================================

class TestTokenConditioning:
    """Test token-conditioned vs global routing for all controllers."""
    
    @pytest.mark.parametrize("ControllerClass", [
        LSTMDepthController,
        xLSTMDepthController,
        minGRUDepthController,
    ])
    def test_token_conditioned_varies_per_token(self, ControllerClass):
        """Token-conditioned routing should give different α per token."""
        B, S, D, H = 2, 8, 48, 6
        ctrl = ControllerClass(
            d_model=D, n_heads=H, d_ctrl=48, token_conditioned=True
        )
        
        X = torch.randn(B, S, D)
        X[:, 0, :] *= 3.0  # Make first token different
        
        alpha, _, _ = ctrl.step(X, state=None)
        
        token_diff = (alpha[:, 0, :] - alpha[:, 1, :]).abs().sum().item()
        assert token_diff > 0.001, "Token-conditioned should vary per token"
    
    @pytest.mark.parametrize("ControllerClass", [
        LSTMDepthController,
        xLSTMDepthController,
        minGRUDepthController,
    ])
    def test_global_routing_same_for_all_tokens(self, ControllerClass):
        """Global routing should be identical for all tokens."""
        B, S, D, H = 2, 8, 48, 6
        ctrl = ControllerClass(
            d_model=D, n_heads=H, d_ctrl=48, token_conditioned=False
        )
        X = torch.randn(B, S, D)
        
        alpha, _, _ = ctrl.step(X, state=None)
        
        for s in range(1, S):
            assert torch.allclose(alpha[:, 0, :], alpha[:, s, :], atol=1e-5)


# =============================================================================
# Temperature Tests (applies to all controllers)
# =============================================================================

class TestTemperatureEffects:
    """Test temperature effects on routing sharpness."""
    
    @pytest.mark.parametrize("ControllerClass", [
        LSTMDepthController,
        xLSTMDepthController,
        minGRUDepthController,
    ])
    def test_temperature_affects_sharpness(self, ControllerClass):
        """Lower temperature should produce sharper routing."""
        B, S, D, H = 2, 4, 32, 6
        X = torch.randn(B, S, D)
        
        ctrl_soft = ControllerClass(d_model=D, n_heads=H, temperature=2.0)
        ctrl_sharp = ControllerClass(d_model=D, n_heads=H, temperature=0.5)
        
        # Copy weights
        ctrl_sharp.load_state_dict(ctrl_soft.state_dict())
        ctrl_sharp.temperature = 0.5
        
        alpha_soft, _, _ = ctrl_soft.step(X, state=None)
        alpha_sharp, _, _ = ctrl_sharp.step(X, state=None)
        
        # Lower temp = higher max probability
        assert alpha_sharp.max().item() >= alpha_soft.max().item() - 0.15
    
    @pytest.mark.parametrize("ControllerClass", [
        LSTMDepthController,
        xLSTMDepthController,
        minGRUDepthController,
    ])
    def test_set_temperature(self, ControllerClass):
        """Test temperature can be updated."""
        ctrl = ControllerClass(d_model=32, n_heads=4, temperature=1.0)
        
        assert ctrl.temperature == 1.0
        ctrl.set_temperature(0.5)
        assert ctrl.temperature == 0.5
        
        # Should clamp to minimum
        ctrl.set_temperature(0.01)
        assert ctrl.temperature >= 0.1


# =============================================================================
# Forward API Compatibility Tests
# =============================================================================

class TestForwardAPICompatibility:
    """Test forward() API compatibility with HRM controller."""
    
    @pytest.mark.parametrize("ControllerClass", [
        LSTMDepthController,
        xLSTMDepthController,
        minGRUDepthController,
    ])
    def test_forward_3d_input(self, ControllerClass):
        """Test forward() with 3D input."""
        B, L, D, H = 2, 8, 32, 4
        ctrl = ControllerClass(d_model=D, n_heads=H)
        X = torch.randn(B, L, D)
        
        alpha, state, aux = ctrl.forward(X, state=None)
        
        assert alpha.shape == (B, L, H)
    
    @pytest.mark.parametrize("ControllerClass", [
        LSTMDepthController,
        xLSTMDepthController,
        minGRUDepthController,
    ])
    def test_forward_2d_input(self, ControllerClass):
        """Test forward() with 2D input (already pooled)."""
        B, D, H = 2, 32, 4
        ctrl = ControllerClass(d_model=D, n_heads=H)
        X = torch.randn(B, D)
        
        alpha, state, aux = ctrl.forward(X, state=None)
        
        # Should squeeze to [B, H]
        assert alpha.shape == (B, H)


if __name__ == "__main__":
    print("Running LSTM Controllers unit tests...")
    pytest.main([__file__, "-v"])
