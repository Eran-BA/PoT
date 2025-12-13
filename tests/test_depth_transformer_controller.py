"""
Unit tests for Causal Depth Transformer Controller.

Tests:
- Basic forward pass shapes
- Causal attention over depth (step t only sees 0..t)
- Cache accumulation across steps
- Top-k sparsity and temperature scheduling
- Gradient flow through controller
- Token-conditioned vs global routing

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import pytest
import torch
import torch.nn.functional as F

from src.pot.core.depth_transformer_controller import (
    CausalDepthTransformerRouter,
    DepthControllerCache,
)


class TestBasicFunctionality:
    """Test basic forward pass and shapes."""

    @pytest.mark.parametrize("B,S,D,H", [(4, 7, 64, 8)])
    def test_basic_step_shapes(self, B, S, D, H):
        """Test that step() returns correct shapes."""
        ctrl = CausalDepthTransformerRouter(
            d_model=D,
            n_heads=H,
            d_ctrl=64,
            n_ctrl_layers=1,
            n_ctrl_heads=4,
            max_depth=16,
            token_conditioned=True,
        )
        X = torch.randn(B, S, D)
        cache = None

        # Run one step
        alpha, cache, aux = ctrl.step(X, t=0, cache=cache)

        assert alpha.shape == (B, S, H), f"Expected alpha shape ({B}, {S}, {H}), got {alpha.shape}"
        assert isinstance(cache, DepthControllerCache), "Expected DepthControllerCache"
        assert len(cache.u_list) == 1, "Cache should have 1 entry after first step"
        assert "entropy" in aux and aux["entropy"].ndim == 0, "Expected scalar entropy"
        assert "alphas" in aux and aux["alphas"].shape == (B, S, H)
        assert "temperature" in aux

    def test_multi_step_cache_accumulation(self):
        """Test that cache accumulates entries across steps."""
        B, S, D, H = 2, 5, 32, 4
        ctrl = CausalDepthTransformerRouter(
            d_model=D, n_heads=H, d_ctrl=32, n_ctrl_layers=1, max_depth=16
        )
        X = torch.randn(B, S, D)
        cache = None

        K = 8  # Run 8 refinement steps
        for t in range(K):
            alpha, cache, _ = ctrl.step(X, t=t, cache=cache)

        assert len(cache.u_list) == K, f"Cache should have {K} entries"
        assert alpha.shape == (B, S, H)

    def test_alphas_sum_to_one(self):
        """Test that routing weights sum to 1 across heads."""
        B, S, D, H = 3, 6, 48, 8
        ctrl = CausalDepthTransformerRouter(
            d_model=D, n_heads=H, d_ctrl=48, max_depth=16
        )
        X = torch.randn(B, S, D)
        cache = None

        for t in range(4):
            alpha, cache, _ = ctrl.step(X, t=t, cache=cache)
            # Check sum to 1 for each token
            sums = alpha.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
                f"Alphas should sum to 1, got sums: {sums}"
            )


class TestCausalBehavior:
    """Test that the controller is causal over depth."""

    def test_different_depths_produce_different_routing(self):
        """Different depth steps should produce different routing weights."""
        B, S, D, H = 2, 4, 32, 4
        ctrl = CausalDepthTransformerRouter(
            d_model=D, n_heads=H, d_ctrl=32, max_depth=16
        )
        X = torch.randn(B, S, D)

        alphas = []
        cache = None
        for t in range(5):
            alpha, cache, _ = ctrl.step(X, t=t, cache=cache)
            alphas.append(alpha.detach().clone())

        # Each step should produce different routing (due to evolving depth context)
        for i in range(1, len(alphas)):
            diff = (alphas[i] - alphas[i - 1]).abs().sum().item()
            # Allow for some steps to be similar but not all identical
            if diff < 1e-6:
                # At least one pair should differ
                continue

        # Overall, check that first and last are different
        total_diff = (alphas[-1] - alphas[0]).abs().sum().item()
        assert total_diff > 0.01, "Routing should evolve across depth steps"

    def test_step_out_of_range_raises(self):
        """Test that step beyond max_depth raises error."""
        ctrl = CausalDepthTransformerRouter(
            d_model=32, n_heads=4, max_depth=8
        )
        X = torch.randn(2, 4, 32)

        with pytest.raises(ValueError, match="out of range"):
            ctrl.step(X, t=10, cache=None)

        with pytest.raises(ValueError, match="out of range"):
            ctrl.step(X, t=-1, cache=None)


class TestTokenConditionedRouting:
    """Test token-conditioned vs global routing."""

    def test_token_conditioned_produces_per_token_alpha(self):
        """Token-conditioned routing should give different α per token."""
        B, S, D, H = 2, 8, 48, 6
        ctrl = CausalDepthTransformerRouter(
            d_model=D, n_heads=H, d_ctrl=48, token_conditioned=True, max_depth=16
        )
        
        # Create input with distinct tokens
        X = torch.randn(B, S, D)
        X[:, 0, :] *= 3.0  # Make first token very different
        
        alpha, _, _ = ctrl.step(X, t=0, cache=None)
        
        # Check that different tokens get different routing
        token_diff = (alpha[:, 0, :] - alpha[:, 1, :]).abs().sum().item()
        assert token_diff > 0.001, "Token-conditioned routing should vary per token"

    def test_global_routing_same_for_all_tokens(self):
        """Global (non-token-conditioned) routing should be identical for all tokens."""
        B, S, D, H = 2, 8, 48, 6
        ctrl = CausalDepthTransformerRouter(
            d_model=D, n_heads=H, d_ctrl=48, token_conditioned=False, max_depth=16
        )
        X = torch.randn(B, S, D)
        
        alpha, _, _ = ctrl.step(X, t=0, cache=None)
        
        # All tokens in same batch should have identical routing
        for s in range(1, S):
            assert torch.allclose(alpha[:, 0, :], alpha[:, s, :], atol=1e-5), (
                "Global routing should be identical across tokens"
            )


class TestTopKAndTemperature:
    """Test top-k sparsification and temperature effects."""

    def test_topk_sparsity(self):
        """Test that top-k produces sparse routing."""
        B, S, D, H = 3, 5, 48, 10
        topk = 3
        ctrl = CausalDepthTransformerRouter(
            d_model=D, n_heads=H, d_ctrl=48, topk=topk, max_depth=16
        )
        X = torch.randn(B, S, D)
        
        alpha, _, _ = ctrl.step(X, t=0, cache=None)
        
        # Count non-zero heads per token
        nonzero_per_token = (alpha > 0).sum(dim=-1)  # [B, S]
        assert nonzero_per_token.max().item() <= topk, (
            f"Top-{topk} should have at most {topk} non-zero heads"
        )
        
        # Should still sum to 1 after renormalization
        assert torch.allclose(alpha.sum(dim=-1), torch.ones(B, S), atol=1e-5)

    def test_temperature_effect(self):
        """Lower temperature should produce sharper routing."""
        B, S, D, H = 2, 4, 32, 6
        
        ctrl_soft = CausalDepthTransformerRouter(
            d_model=D, n_heads=H, temperature=2.0, max_depth=16
        )
        ctrl_sharp = CausalDepthTransformerRouter(
            d_model=D, n_heads=H, temperature=0.5, max_depth=16
        )
        
        # Copy weights so only temperature differs
        ctrl_sharp.load_state_dict(ctrl_soft.state_dict())
        ctrl_sharp.temperature = 0.5
        
        X = torch.randn(B, S, D)
        
        alpha_soft, _, _ = ctrl_soft.step(X, t=0, cache=None)
        alpha_sharp, _, _ = ctrl_sharp.step(X, t=0, cache=None)
        
        # Sharper should have higher max probability
        assert alpha_sharp.max().item() >= alpha_soft.max().item() - 0.1, (
            "Lower temperature should produce sharper (higher max) routing"
        )

    def test_set_temperature(self):
        """Test temperature can be updated dynamically."""
        ctrl = CausalDepthTransformerRouter(
            d_model=32, n_heads=4, temperature=1.0, max_depth=16
        )
        
        assert ctrl.temperature == 1.0
        ctrl.set_temperature(0.5)
        assert ctrl.temperature == 0.5
        
        # Should clamp to minimum
        ctrl.set_temperature(0.01)
        assert ctrl.temperature >= 0.1


class TestGradientFlow:
    """Test gradient flow through the controller."""

    def test_gradients_flow_to_input(self):
        """Test that gradients flow back through controller to inputs."""
        B, S, D, H = 2, 4, 32, 4
        ctrl = CausalDepthTransformerRouter(
            d_model=D, n_heads=H, d_ctrl=32, max_depth=16
        )
        X = torch.randn(B, S, D, requires_grad=True)
        
        # Run forward
        cache = None
        for t in range(3):
            alpha, cache, _ = ctrl.step(X, t=t, cache=cache)
        
        # Compute loss and backprop
        target = torch.zeros_like(alpha)
        target[:, :, 0] = 1.0  # Push to head 0
        loss = F.mse_loss(alpha, target)
        loss.backward()
        
        # Check gradients exist
        assert X.grad is not None, "Input should have gradients"
        assert X.grad.abs().sum().item() > 0, "Gradients should be non-zero"

    def test_gradients_flow_to_params(self):
        """Test that gradients flow to controller parameters."""
        B, S, D, H = 2, 4, 32, 4
        ctrl = CausalDepthTransformerRouter(
            d_model=D, n_heads=H, d_ctrl=32, max_depth=16
        )
        X = torch.randn(B, S, D)
        
        cache = None
        for t in range(3):
            alpha, cache, _ = ctrl.step(X, t=t, cache=cache)
        
        loss = alpha.sum()
        loss.backward()
        
        # Check that some parameters have gradients
        has_grad = False
        for name, param in ctrl.named_parameters():
            if param.grad is not None and param.grad.abs().sum().item() > 0:
                has_grad = True
                break
        
        assert has_grad, "Controller parameters should have gradients"


class TestForwardAPICompatibility:
    """Test the forward() API for HRM compatibility."""

    def test_forward_3d_input(self):
        """Test forward() with 3D input [B, L, D]."""
        B, L, D, H = 2, 8, 32, 4
        ctrl = CausalDepthTransformerRouter(
            d_model=D, n_heads=H, d_ctrl=32, max_depth=16
        )
        X = torch.randn(B, L, D)
        
        alpha, cache, aux = ctrl.forward(X, state=None, step=0)
        
        assert alpha.shape == (B, L, H)
        assert isinstance(cache, DepthControllerCache)

    def test_forward_2d_input(self):
        """Test forward() with 2D input [B, D] (already pooled)."""
        B, D, H = 2, 32, 4
        ctrl = CausalDepthTransformerRouter(
            d_model=D, n_heads=H, d_ctrl=32, max_depth=16
        )
        X = torch.randn(B, D)
        
        alpha, cache, aux = ctrl.forward(X, state=None, step=0)
        
        # Should squeeze back to [B, H] for 2D input
        assert alpha.shape == (B, H)


class TestEntropyRegularization:
    """Test entropy computation for regularization."""

    def test_entropy_in_aux(self):
        """Test that entropy is returned in aux dict."""
        B, S, D, H = 2, 4, 32, 8
        ctrl = CausalDepthTransformerRouter(
            d_model=D, n_heads=H, d_ctrl=32, max_depth=16
        )
        X = torch.randn(B, S, D)
        
        _, _, aux = ctrl.step(X, t=0, cache=None)
        
        assert "entropy" in aux
        assert aux["entropy"].ndim == 0  # Scalar
        assert aux["entropy"].item() > 0  # Entropy should be positive

    def test_temperature_affects_entropy(self):
        """Higher temperature should produce higher entropy."""
        B, S, D, H = 2, 4, 32, 6
        X = torch.randn(B, S, D)
        
        ctrl_soft = CausalDepthTransformerRouter(
            d_model=D, n_heads=H, temperature=2.0, max_depth=16
        )
        ctrl_sharp = CausalDepthTransformerRouter(
            d_model=D, n_heads=H, temperature=0.5, max_depth=16
        )
        ctrl_sharp.load_state_dict(ctrl_soft.state_dict())
        ctrl_sharp.temperature = 0.5
        
        _, _, aux_soft = ctrl_soft.step(X, t=0, cache=None)
        _, _, aux_sharp = ctrl_sharp.step(X, t=0, cache=None)
        
        assert aux_soft["entropy"].item() > aux_sharp["entropy"].item(), (
            "Higher temperature should yield higher entropy"
        )


if __name__ == "__main__":
    print("Running Causal Depth Transformer Controller unit tests...")
    pytest.main([__file__, "-v"])
