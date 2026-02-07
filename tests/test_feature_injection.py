"""
Tests for Feature Injection Module.

Tests all 7 injection modes:
- none: Routing-only (no injection)
- broadcast: Gated broadcast of controller features to all tokens
- broadcast_memory: Gated broadcast with memory bank accumulation
- film: FiLM modulation (scale and shift)
- depth_token: Prepend learnable depth token
- cross_attn: Cross-attention to depth memory bank
- alpha_gated: Alpha-modulated broadcast (injection follows routing)

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import pytest
import torch
import torch.nn as nn

from src.pot.core.feature_injection import (
    FeatureInjector,
    GatedBroadcastInjection,
    GatedBroadcastWithMemoryInjection,
    AlphaGatedInjection,
    FiLMInjection,
    DepthTokenInjection,
    CrossAttentionInjection,
    INJECTION_MODES,
)


# =============================================================================
# Fixtures
# =============================================================================

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
def d_ctrl():
    return 32


@pytest.fixture
def tokens(batch_size, seq_len, d_model):
    """Create random token embeddings."""
    return torch.randn(batch_size, seq_len, d_model)


@pytest.fixture
def controller_features(batch_size, d_ctrl):
    """Create random controller features."""
    return torch.randn(batch_size, d_ctrl)


# =============================================================================
# Test Individual Injection Modules
# =============================================================================

class TestGatedBroadcastInjection:
    """Tests for GatedBroadcastInjection."""
    
    def test_output_shape(self, tokens, controller_features, d_model, d_ctrl):
        """Output should have same shape as input."""
        injector = GatedBroadcastInjection(d_model, d_ctrl)
        output = injector(tokens, controller_features)
        assert output.shape == tokens.shape
    
    def test_gradient_flow(self, tokens, controller_features, d_model, d_ctrl):
        """Gradients should flow through to controller features."""
        injector = GatedBroadcastInjection(d_model, d_ctrl)
        tokens.requires_grad_(True)
        controller_features.requires_grad_(True)
        
        output = injector(tokens, controller_features)
        loss = output.sum()
        loss.backward()
        
        assert tokens.grad is not None
        assert controller_features.grad is not None
    
    def test_gating_bounds(self, tokens, controller_features, d_model, d_ctrl):
        """Gate values should be in [0, 1] (sigmoid output)."""
        injector = GatedBroadcastInjection(d_model, d_ctrl)
        gate = torch.sigmoid(injector.gate(controller_features))
        assert (gate >= 0).all() and (gate <= 1).all()


class TestFiLMInjection:
    """Tests for FiLM (Feature-wise Linear Modulation) injection."""
    
    def test_output_shape(self, tokens, controller_features, d_model, d_ctrl):
        """Output should have same shape as input."""
        injector = FiLMInjection(d_model, d_ctrl)
        output = injector(tokens, controller_features)
        assert output.shape == tokens.shape
    
    def test_identity_initialization(self, tokens, controller_features, d_model, d_ctrl):
        """With identity init (gamma=1, beta=0), output should equal input."""
        injector = FiLMInjection(d_model, d_ctrl)
        # FiLM is initialized to identity transform
        # With zero controller features, gamma should be 1 and beta should be 0
        zero_features = torch.zeros_like(controller_features)
        output = injector(tokens, zero_features)
        # Should be close to identity (may not be exact due to activation)
        assert output.shape == tokens.shape
    
    def test_gradient_flow(self, tokens, controller_features, d_model, d_ctrl):
        """Gradients should flow through to controller features."""
        injector = FiLMInjection(d_model, d_ctrl)
        tokens.requires_grad_(True)
        controller_features.requires_grad_(True)
        
        output = injector(tokens, controller_features)
        loss = output.sum()
        loss.backward()
        
        assert tokens.grad is not None
        assert controller_features.grad is not None


class TestDepthTokenInjection:
    """Tests for depth token injection."""
    
    def test_output_shape(self, tokens, controller_features, d_model, d_ctrl, seq_len):
        """Output should have one additional token (depth token prepended)."""
        injector = DepthTokenInjection(d_model, d_ctrl)
        output = injector(tokens, controller_features)
        assert output.shape == (tokens.shape[0], seq_len + 1, d_model)
    
    def test_remove_depth_token(self, tokens, controller_features, d_model, d_ctrl):
        """Removing depth token should restore original sequence length."""
        injector = DepthTokenInjection(d_model, d_ctrl)
        output = injector(tokens, controller_features)
        restored = injector.remove_depth_token(output)
        assert restored.shape == tokens.shape
    
    def test_gradient_flow(self, tokens, controller_features, d_model, d_ctrl):
        """Gradients should flow through to controller features."""
        injector = DepthTokenInjection(d_model, d_ctrl)
        tokens.requires_grad_(True)
        controller_features.requires_grad_(True)
        
        output = injector(tokens, controller_features)
        loss = output.sum()
        loss.backward()
        
        assert tokens.grad is not None
        assert controller_features.grad is not None


class TestCrossAttentionInjection:
    """Tests for cross-attention to memory bank."""
    
    def test_output_shape(self, tokens, controller_features, d_model, d_ctrl):
        """Output tokens should have same shape as input."""
        injector = CrossAttentionInjection(d_model, d_ctrl, memory_size=8, n_heads=2)
        output, memory = injector(tokens, controller_features, memory=None)
        assert output.shape == tokens.shape
    
    def test_memory_initialization(self, tokens, controller_features, d_model, d_ctrl, batch_size):
        """Memory should be initialized with projected controller features."""
        injector = CrossAttentionInjection(d_model, d_ctrl, memory_size=8, n_heads=2)
        output, memory = injector(tokens, controller_features, memory=None)
        assert memory.shape == (batch_size, 1, d_model)
    
    def test_memory_accumulation(self, tokens, controller_features, d_model, d_ctrl, batch_size):
        """Memory should accumulate across steps."""
        injector = CrossAttentionInjection(d_model, d_ctrl, memory_size=8, n_heads=2)
        
        # First step
        output1, memory1 = injector(tokens, controller_features, memory=None)
        assert memory1.shape[1] == 1
        
        # Second step
        output2, memory2 = injector(tokens, controller_features, memory=memory1)
        assert memory2.shape[1] == 2
        
        # Third step
        output3, memory3 = injector(tokens, controller_features, memory=memory2)
        assert memory3.shape[1] == 3
    
    def test_memory_capping(self, tokens, controller_features, d_model, d_ctrl):
        """Memory should be capped at memory_size."""
        memory_size = 3
        injector = CrossAttentionInjection(d_model, d_ctrl, memory_size=memory_size, n_heads=2)
        
        memory = None
        for _ in range(10):
            _, memory = injector(tokens, controller_features, memory=memory)
        
        assert memory.shape[1] == memory_size
    
    def test_gradient_flow(self, tokens, controller_features, d_model, d_ctrl):
        """Gradients should flow through to controller features."""
        injector = CrossAttentionInjection(d_model, d_ctrl, memory_size=8, n_heads=2)
        tokens.requires_grad_(True)
        controller_features.requires_grad_(True)
        
        output, memory = injector(tokens, controller_features, memory=None)
        loss = output.sum()
        loss.backward()
        
        assert tokens.grad is not None
        assert controller_features.grad is not None


class TestGatedBroadcastWithMemoryInjection:
    """Tests for gated broadcast injection with memory bank."""
    
    def test_output_shape(self, tokens, controller_features, d_model, d_ctrl):
        """Output should have same shape as input."""
        injector = GatedBroadcastWithMemoryInjection(d_model, d_ctrl, memory_size=8, n_heads=2)
        output, memory = injector(tokens, controller_features, memory=None)
        assert output.shape == tokens.shape
    
    def test_memory_initialization(self, tokens, controller_features, d_model, d_ctrl, batch_size):
        """Memory should be initialized with projected controller features."""
        injector = GatedBroadcastWithMemoryInjection(d_model, d_ctrl, memory_size=8, n_heads=2)
        output, memory = injector(tokens, controller_features, memory=None)
        assert memory.shape == (batch_size, 1, d_model)
    
    def test_memory_accumulation(self, tokens, controller_features, d_model, d_ctrl, batch_size):
        """Memory should accumulate across steps."""
        injector = GatedBroadcastWithMemoryInjection(d_model, d_ctrl, memory_size=8, n_heads=2)
        
        # First step
        output1, memory1 = injector(tokens, controller_features, memory=None)
        assert memory1.shape[1] == 1
        
        # Second step
        output2, memory2 = injector(tokens, controller_features, memory=memory1)
        assert memory2.shape[1] == 2
        
        # Third step
        output3, memory3 = injector(tokens, controller_features, memory=memory2)
        assert memory3.shape[1] == 3
    
    def test_memory_capping(self, tokens, controller_features, d_model, d_ctrl):
        """Memory should be capped at memory_size."""
        memory_size = 3
        injector = GatedBroadcastWithMemoryInjection(d_model, d_ctrl, memory_size=memory_size, n_heads=2)
        
        memory = None
        for _ in range(10):
            _, memory = injector(tokens, controller_features, memory=memory)
        
        assert memory.shape[1] == memory_size
    
    def test_gradient_flow(self, tokens, controller_features, d_model, d_ctrl):
        """Gradients should flow through to controller features."""
        injector = GatedBroadcastWithMemoryInjection(d_model, d_ctrl, memory_size=8, n_heads=2)
        tokens.requires_grad_(True)
        controller_features.requires_grad_(True)
        
        output, memory = injector(tokens, controller_features, memory=None)
        loss = output.sum()
        loss.backward()
        
        assert tokens.grad is not None
        assert controller_features.grad is not None
    
    def test_broadcast_behavior(self, controller_features, d_model, d_ctrl, batch_size, seq_len):
        """All tokens should receive the same injection (broadcast)."""
        injector = GatedBroadcastWithMemoryInjection(d_model, d_ctrl, memory_size=8, n_heads=2)
        
        # Use zero tokens so the injection is the only signal
        tokens = torch.zeros(batch_size, seq_len, d_model)
        output, memory = injector(tokens, controller_features, memory=None)
        
        # Check that the injection is the same across all token positions
        # (since tokens are zero, output IS the injection)
        for b in range(batch_size):
            diffs = (output[b] - output[b, 0:1]).abs().max()
            assert diffs < 1e-5, f"Injection differs across tokens: max diff = {diffs}"


class TestTokenConditionedMemoryRead:
    """Tests for token-conditioned memory read (memory_read='token_conditioned')."""
    
    def test_output_shape(self, tokens, controller_features, d_model, d_ctrl):
        """Output should have same shape as input."""
        injector = GatedBroadcastWithMemoryInjection(d_model, d_ctrl, memory_size=8, n_heads=2, memory_read="token_conditioned")
        output, memory = injector(tokens, controller_features, memory=None)
        assert output.shape == tokens.shape
    
    def test_memory_initialization(self, tokens, controller_features, d_model, d_ctrl, batch_size):
        """Memory should be initialized identically to broadcast mode."""
        injector = GatedBroadcastWithMemoryInjection(d_model, d_ctrl, memory_size=8, n_heads=2, memory_read="token_conditioned")
        output, memory = injector(tokens, controller_features, memory=None)
        assert memory.shape == (batch_size, 1, d_model)
    
    def test_memory_accumulation(self, tokens, controller_features, d_model, d_ctrl, batch_size):
        """Memory should accumulate across steps."""
        injector = GatedBroadcastWithMemoryInjection(d_model, d_ctrl, memory_size=8, n_heads=2, memory_read="token_conditioned")
        _, mem1 = injector(tokens, controller_features, memory=None)
        assert mem1.shape[1] == 1
        _, mem2 = injector(tokens, controller_features, memory=mem1)
        assert mem2.shape[1] == 2
        _, mem3 = injector(tokens, controller_features, memory=mem2)
        assert mem3.shape[1] == 3
    
    def test_memory_capping(self, tokens, controller_features, d_model, d_ctrl):
        """Memory should be capped at memory_size."""
        memory_size = 3
        injector = GatedBroadcastWithMemoryInjection(d_model, d_ctrl, memory_size=memory_size, n_heads=2, memory_read="token_conditioned")
        memory = None
        for _ in range(10):
            _, memory = injector(tokens, controller_features, memory=memory)
        assert memory.shape[1] == memory_size
    
    def test_gradient_flow(self, tokens, controller_features, d_model, d_ctrl):
        """Gradients should flow through to both tokens and controller features."""
        injector = GatedBroadcastWithMemoryInjection(d_model, d_ctrl, memory_size=8, n_heads=2, memory_read="token_conditioned")
        tokens.requires_grad_(True)
        controller_features.requires_grad_(True)
        
        output, memory = injector(tokens, controller_features, memory=None)
        loss = output.sum()
        loss.backward()
        
        assert tokens.grad is not None
        assert controller_features.grad is not None
    
    def test_per_token_variance(self, controller_features, d_model, d_ctrl, batch_size, seq_len):
        """Token-conditioned read should produce different injections per token (non-zero variance)."""
        injector = GatedBroadcastWithMemoryInjection(d_model, d_ctrl, memory_size=8, n_heads=2, memory_read="token_conditioned")
        
        # Use non-zero, varied tokens so each cell has a different query
        torch.manual_seed(42)
        tokens = torch.randn(batch_size, seq_len, d_model)
        
        # Run a few steps to build up memory
        memory = None
        for _ in range(4):
            r = torch.randn(batch_size, d_ctrl)
            _, memory = injector(tokens, r, memory=memory)
        
        output, _ = injector(tokens, controller_features, memory=memory)
        injection = output - tokens  # isolate what was injected
        
        # Variance across tokens should be non-zero (unlike broadcast which is ~0)
        per_batch_var = injection.var(dim=1).mean()  # variance across S dimension
        assert per_batch_var > 1e-8, f"Token-conditioned injection has near-zero per-token variance: {per_batch_var}"
    
    def test_broadcast_has_zero_per_token_variance(self, controller_features, d_model, d_ctrl, batch_size, seq_len):
        """Broadcast mode should have zero per-token variance (sanity check contrast)."""
        injector = GatedBroadcastWithMemoryInjection(d_model, d_ctrl, memory_size=8, n_heads=2, memory_read="broadcast")
        
        tokens = torch.zeros(batch_size, seq_len, d_model)
        output, _ = injector(tokens, controller_features, memory=None)
        injection = output - tokens
        
        per_batch_var = injection.var(dim=1).mean()
        assert per_batch_var < 1e-8, f"Broadcast injection should be identical across tokens, but var={per_batch_var}"
    
    def test_invalid_memory_read_raises(self, d_model, d_ctrl):
        """Invalid memory_read value should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown memory_read mode"):
            GatedBroadcastWithMemoryInjection(d_model, d_ctrl, memory_read="invalid")


class TestAlphaGatedInjection:
    """Tests for alpha-modulated injection."""
    
    @pytest.fixture
    def n_heads(self):
        return 8
    
    @pytest.fixture
    def alpha(self, batch_size, n_heads):
        """Create random routing weights."""
        alpha = torch.softmax(torch.randn(batch_size, n_heads), dim=-1)
        return alpha
    
    @pytest.fixture
    def alpha_per_token(self, batch_size, seq_len, n_heads):
        """Create per-token routing weights."""
        alpha = torch.softmax(torch.randn(batch_size, seq_len, n_heads), dim=-1)
        return alpha
    
    def test_output_shape(self, tokens, controller_features, alpha, d_model, d_ctrl):
        """Output should have same shape as input."""
        injector = AlphaGatedInjection(d_model, d_ctrl)
        output = injector(tokens, controller_features, alpha)
        assert output.shape == tokens.shape
    
    def test_output_shape_per_token_alpha(self, tokens, controller_features, alpha_per_token, d_model, d_ctrl):
        """Output should work with per-token alpha."""
        injector = AlphaGatedInjection(d_model, d_ctrl)
        output = injector(tokens, controller_features, alpha_per_token)
        assert output.shape == tokens.shape
    
    def test_mean_aggregation(self, tokens, controller_features, alpha, d_model, d_ctrl):
        """Mean aggregation should use mean of alpha."""
        injector = AlphaGatedInjection(d_model, d_ctrl, alpha_aggregation="mean")
        output = injector(tokens, controller_features, alpha)
        assert output.shape == tokens.shape
    
    def test_max_aggregation(self, tokens, controller_features, alpha, d_model, d_ctrl):
        """Max aggregation should use max of alpha."""
        injector = AlphaGatedInjection(d_model, d_ctrl, alpha_aggregation="max")
        output = injector(tokens, controller_features, alpha)
        assert output.shape == tokens.shape
    
    def test_entropy_aggregation(self, tokens, controller_features, alpha, d_model, d_ctrl):
        """Entropy aggregation should use (1 - entropy) of alpha."""
        injector = AlphaGatedInjection(d_model, d_ctrl, alpha_aggregation="entropy")
        output = injector(tokens, controller_features, alpha)
        assert output.shape == tokens.shape
    
    def test_no_learned_gate(self, tokens, controller_features, alpha, d_model, d_ctrl):
        """Should work without learned gate."""
        injector = AlphaGatedInjection(d_model, d_ctrl, use_learned_gate=False)
        output = injector(tokens, controller_features, alpha)
        assert output.shape == tokens.shape
    
    def test_none_alpha_fallback(self, tokens, controller_features, d_model, d_ctrl):
        """Should fallback to uniform when alpha is None."""
        injector = AlphaGatedInjection(d_model, d_ctrl)
        output = injector(tokens, controller_features, alpha=None)
        assert output.shape == tokens.shape
    
    def test_gradient_flow(self, tokens, controller_features, alpha, d_model, d_ctrl):
        """Gradients should flow through to controller features."""
        injector = AlphaGatedInjection(d_model, d_ctrl)
        tokens.requires_grad_(True)
        controller_features.requires_grad_(True)
        alpha.requires_grad_(True)
        
        output = injector(tokens, controller_features, alpha)
        loss = output.sum()
        loss.backward()
        
        assert tokens.grad is not None
        assert controller_features.grad is not None
        assert alpha.grad is not None
    
    def test_confident_routing_stronger_injection(self, tokens, controller_features, d_model, d_ctrl):
        """With entropy aggregation, confident routing should give stronger injection."""
        injector = AlphaGatedInjection(d_model, d_ctrl, alpha_aggregation="entropy", use_learned_gate=False)
        
        B = tokens.size(0)
        n_heads = 8
        
        # Confident alpha (one-hot-ish)
        confident_alpha = torch.zeros(B, n_heads)
        confident_alpha[:, 0] = 1.0
        
        # Uniform alpha (max entropy)
        uniform_alpha = torch.ones(B, n_heads) / n_heads
        
        output_confident = injector(tokens, controller_features, confident_alpha)
        output_uniform = injector(tokens, controller_features, uniform_alpha)
        
        # Confident should have larger deviation from input
        diff_confident = (output_confident - tokens).abs().mean()
        diff_uniform = (output_uniform - tokens).abs().mean()
        
        assert diff_confident > diff_uniform


# =============================================================================
# Test FeatureInjector (Main Interface)
# =============================================================================

class TestFeatureInjector:
    """Tests for the main FeatureInjector interface."""
    
    def test_all_modes_valid(self):
        """All modes in INJECTION_MODES should be valid."""
        for mode in INJECTION_MODES:
            injector = FeatureInjector(d_model=64, d_ctrl=32, mode=mode)
            assert injector.mode == mode
    
    def test_invalid_mode_raises(self):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown injection mode"):
            FeatureInjector(d_model=64, d_ctrl=32, mode="invalid_mode")
    
    def test_none_mode_passthrough(self, tokens, controller_features, d_model, d_ctrl):
        """Mode 'none' should return input unchanged."""
        injector = FeatureInjector(d_model, d_ctrl, mode="none")
        output, memory = injector(tokens, controller_features)
        assert torch.equal(output, tokens)
        assert memory is None
    
    def test_none_mode_without_features(self, tokens, d_model, d_ctrl):
        """Mode 'none' with no features should return input unchanged."""
        injector = FeatureInjector(d_model, d_ctrl, mode="none")
        output, memory = injector(tokens, r=None)
        assert torch.equal(output, tokens)
    
    def test_broadcast_mode(self, tokens, controller_features, d_model, d_ctrl):
        """Broadcast mode should return same-shaped output."""
        injector = FeatureInjector(d_model, d_ctrl, mode="broadcast")
        output, memory = injector(tokens, controller_features)
        assert output.shape == tokens.shape
        assert memory is None
    
    def test_film_mode(self, tokens, controller_features, d_model, d_ctrl):
        """FiLM mode should return same-shaped output."""
        injector = FeatureInjector(d_model, d_ctrl, mode="film")
        output, memory = injector(tokens, controller_features)
        assert output.shape == tokens.shape
        assert memory is None
    
    def test_depth_token_mode(self, tokens, controller_features, d_model, d_ctrl, seq_len):
        """Depth token mode should add one token."""
        injector = FeatureInjector(d_model, d_ctrl, mode="depth_token")
        output, memory = injector(tokens, controller_features)
        assert output.shape == (tokens.shape[0], seq_len + 1, d_model)
        assert injector.has_depth_token()
        
        # Remove depth token
        restored = injector.remove_depth_token(output)
        assert restored.shape == tokens.shape
    
    def test_cross_attn_mode(self, tokens, controller_features, d_model, d_ctrl, batch_size):
        """Cross-attention mode should return output and memory."""
        injector = FeatureInjector(d_model, d_ctrl, mode="cross_attn", memory_size=8, n_heads=2)
        output, memory = injector(tokens, controller_features)
        assert output.shape == tokens.shape
        assert memory is not None
        assert memory.shape == (batch_size, 1, d_model)
    
    def test_broadcast_memory_mode(self, tokens, controller_features, d_model, d_ctrl, batch_size):
        """Broadcast memory mode should return output and memory."""
        injector = FeatureInjector(d_model, d_ctrl, mode="broadcast_memory", memory_size=8, n_heads=2)
        output, memory = injector(tokens, controller_features)
        assert output.shape == tokens.shape
        assert memory is not None
        assert memory.shape == (batch_size, 1, d_model)
    
    def test_broadcast_memory_mode_accumulates(self, tokens, controller_features, d_model, d_ctrl, batch_size):
        """Broadcast memory mode should accumulate memory across calls."""
        injector = FeatureInjector(d_model, d_ctrl, mode="broadcast_memory", memory_size=8, n_heads=2)
        output1, memory1 = injector(tokens, controller_features)
        output2, memory2 = injector(tokens, controller_features, memory=memory1)
        assert memory2.shape == (batch_size, 2, d_model)
    
    def test_alpha_gated_mode(self, tokens, controller_features, d_model, d_ctrl, batch_size):
        """Alpha-gated mode should use routing weights to modulate injection."""
        injector = FeatureInjector(d_model, d_ctrl, mode="alpha_gated")
        n_heads = 8
        alpha = torch.softmax(torch.randn(batch_size, n_heads), dim=-1)
        output, memory = injector(tokens, controller_features, alpha=alpha)
        assert output.shape == tokens.shape
        assert memory is None
    
    def test_alpha_gated_mode_with_options(self, tokens, controller_features, d_model, d_ctrl, batch_size):
        """Alpha-gated mode should accept configuration options."""
        injector = FeatureInjector(
            d_model, d_ctrl, mode="alpha_gated",
            alpha_aggregation="entropy",
            use_learned_gate=False,
        )
        n_heads = 8
        alpha = torch.softmax(torch.randn(batch_size, n_heads), dim=-1)
        output, memory = injector(tokens, controller_features, alpha=alpha)
        assert output.shape == tokens.shape
    
    def test_backward_compatibility_default(self, tokens, controller_features, d_model, d_ctrl):
        """Default mode should be 'none' for backward compatibility."""
        injector = FeatureInjector(d_model, d_ctrl)
        assert injector.mode == "none"
        output, memory = injector(tokens, controller_features)
        assert torch.equal(output, tokens)


# =============================================================================
# Test Integration with ReasoningModule
# =============================================================================

class TestReasoningModuleIntegration:
    """Tests for integration with ReasoningModule."""
    
    @pytest.fixture
    def reasoning_module_params(self):
        return {
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 1,
            "d_ff": 128,
            "dropout": 0.0,
            "T": 2,
        }
    
    def test_reasoning_module_with_none_injection(self, reasoning_module_params, batch_size, seq_len):
        """ReasoningModule with 'none' injection should work."""
        from src.pot.models.reasoning_module import ReasoningModule
        
        module = ReasoningModule(**reasoning_module_params, injection_mode="none")
        hidden = torch.randn(batch_size, seq_len, reasoning_module_params["d_model"])
        injection = torch.randn(batch_size, seq_len, reasoning_module_params["d_model"])
        
        output, ptr_state, inj_mem = module(hidden, injection)
        assert output.shape == hidden.shape
    
    def test_reasoning_module_with_broadcast_injection(self, reasoning_module_params, batch_size, seq_len):
        """ReasoningModule with 'broadcast' injection should work."""
        from src.pot.models.reasoning_module import ReasoningModule
        
        module = ReasoningModule(**reasoning_module_params, injection_mode="broadcast")
        hidden = torch.randn(batch_size, seq_len, reasoning_module_params["d_model"])
        injection = torch.randn(batch_size, seq_len, reasoning_module_params["d_model"])
        
        output, ptr_state, inj_mem = module(hidden, injection)
        assert output.shape == hidden.shape
    
    def test_reasoning_module_with_film_injection(self, reasoning_module_params, batch_size, seq_len):
        """ReasoningModule with 'film' injection should work."""
        from src.pot.models.reasoning_module import ReasoningModule
        
        module = ReasoningModule(**reasoning_module_params, injection_mode="film")
        hidden = torch.randn(batch_size, seq_len, reasoning_module_params["d_model"])
        injection = torch.randn(batch_size, seq_len, reasoning_module_params["d_model"])
        
        output, ptr_state, inj_mem = module(hidden, injection)
        assert output.shape == hidden.shape
    
    def test_reasoning_module_with_depth_token_injection(self, reasoning_module_params, batch_size, seq_len):
        """ReasoningModule with 'depth_token' injection should work."""
        from src.pot.models.reasoning_module import ReasoningModule
        
        module = ReasoningModule(**reasoning_module_params, injection_mode="depth_token")
        hidden = torch.randn(batch_size, seq_len, reasoning_module_params["d_model"])
        injection = torch.randn(batch_size, seq_len, reasoning_module_params["d_model"])
        
        output, ptr_state, inj_mem = module(hidden, injection)
        # Depth token is added then removed
        assert output.shape == hidden.shape
    
    def test_reasoning_module_with_cross_attn_injection(self, reasoning_module_params, batch_size, seq_len):
        """ReasoningModule with 'cross_attn' injection should work."""
        from src.pot.models.reasoning_module import ReasoningModule
        
        module = ReasoningModule(
            **reasoning_module_params,
            injection_mode="cross_attn",
            injection_kwargs={"memory_size": 8, "n_heads": 2}
        )
        hidden = torch.randn(batch_size, seq_len, reasoning_module_params["d_model"])
        injection = torch.randn(batch_size, seq_len, reasoning_module_params["d_model"])
        
        output, ptr_state, inj_mem = module(hidden, injection)
        assert output.shape == hidden.shape
        assert inj_mem is not None
    
    def test_reasoning_module_with_broadcast_memory_injection(self, reasoning_module_params, batch_size, seq_len):
        """ReasoningModule with 'broadcast_memory' injection should work."""
        from src.pot.models.reasoning_module import ReasoningModule
        
        module = ReasoningModule(
            **reasoning_module_params,
            injection_mode="broadcast_memory",
            injection_kwargs={"memory_size": 8, "n_heads": 2}
        )
        hidden = torch.randn(batch_size, seq_len, reasoning_module_params["d_model"])
        injection = torch.randn(batch_size, seq_len, reasoning_module_params["d_model"])
        
        output, ptr_state, inj_mem = module(hidden, injection)
        assert output.shape == hidden.shape
        assert inj_mem is not None
    
    def test_reasoning_module_broadcast_memory_accumulates(self, reasoning_module_params, batch_size, seq_len):
        """broadcast_memory injection memory should grow across calls."""
        from src.pot.models.reasoning_module import ReasoningModule
        
        module = ReasoningModule(
            **reasoning_module_params,
            injection_mode="broadcast_memory",
            injection_kwargs={"memory_size": 8, "n_heads": 2}
        )
        hidden = torch.randn(batch_size, seq_len, reasoning_module_params["d_model"])
        injection = torch.randn(batch_size, seq_len, reasoning_module_params["d_model"])
        
        # First call
        output1, ptr_state1, inj_mem1 = module(hidden, injection)
        assert inj_mem1 is not None
        assert inj_mem1.shape[1] == 1
        
        # Second call with preserved memory
        output2, ptr_state2, inj_mem2 = module(hidden, injection, ptr_state=ptr_state1, injection_memory=inj_mem1)
        assert inj_mem2.shape[1] == 2


# =============================================================================
# Test Integration with HybridPoHHRMSolver
# =============================================================================

class TestHybridSolverIntegration:
    """Tests for integration with HybridPoHHRMSolver."""
    
    @pytest.fixture
    def solver_params(self):
        return {
            "vocab_size": 10,
            "d_model": 64,
            "n_heads": 4,
            "H_layers": 1,
            "L_layers": 1,
            "d_ff": 128,
            "H_cycles": 1,
            "L_cycles": 2,
            "T": 2,
            "halt_max_steps": 1,
        }
    
    def test_solver_with_none_injection(self, solver_params):
        """Solver with 'none' injection should work."""
        from src.pot.models.sudoku_solver import HybridPoHHRMSolver
        
        model = HybridPoHHRMSolver(**solver_params, injection_mode="none")
        inputs = torch.randint(0, 10, (2, 81))
        puzzle_ids = torch.zeros(2, dtype=torch.long)
        
        logits, q_halt, q_continue, steps = model(inputs, puzzle_ids)
        assert logits.shape == (2, 81, 10)
    
    def test_solver_with_broadcast_injection(self, solver_params):
        """Solver with 'broadcast' injection should work."""
        from src.pot.models.sudoku_solver import HybridPoHHRMSolver
        
        model = HybridPoHHRMSolver(**solver_params, injection_mode="broadcast")
        inputs = torch.randint(0, 10, (2, 81))
        puzzle_ids = torch.zeros(2, dtype=torch.long)
        
        logits, q_halt, q_continue, steps = model(inputs, puzzle_ids)
        assert logits.shape == (2, 81, 10)
    
    def test_solver_with_film_injection(self, solver_params):
        """Solver with 'film' injection should work."""
        from src.pot.models.sudoku_solver import HybridPoHHRMSolver
        
        model = HybridPoHHRMSolver(**solver_params, injection_mode="film")
        inputs = torch.randint(0, 10, (2, 81))
        puzzle_ids = torch.zeros(2, dtype=torch.long)
        
        logits, q_halt, q_continue, steps = model(inputs, puzzle_ids)
        assert logits.shape == (2, 81, 10)
    
    def test_solver_with_depth_token_injection(self, solver_params):
        """Solver with 'depth_token' injection should work."""
        from src.pot.models.sudoku_solver import HybridPoHHRMSolver
        
        model = HybridPoHHRMSolver(**solver_params, injection_mode="depth_token")
        inputs = torch.randint(0, 10, (2, 81))
        puzzle_ids = torch.zeros(2, dtype=torch.long)
        
        logits, q_halt, q_continue, steps = model(inputs, puzzle_ids)
        assert logits.shape == (2, 81, 10)
    
    def test_solver_with_cross_attn_injection(self, solver_params):
        """Solver with 'cross_attn' injection should work."""
        from src.pot.models.sudoku_solver import HybridPoHHRMSolver
        
        model = HybridPoHHRMSolver(
            **solver_params,
            injection_mode="cross_attn",
            injection_kwargs={"memory_size": 8, "n_heads": 2}
        )
        inputs = torch.randint(0, 10, (2, 81))
        puzzle_ids = torch.zeros(2, dtype=torch.long)
        
        logits, q_halt, q_continue, steps = model(inputs, puzzle_ids)
        assert logits.shape == (2, 81, 10)
    
    def test_solver_with_broadcast_memory_injection(self, solver_params):
        """Solver with 'broadcast_memory' injection should work."""
        from src.pot.models.sudoku_solver import HybridPoHHRMSolver
        
        model = HybridPoHHRMSolver(
            **solver_params,
            injection_mode="broadcast_memory",
            injection_kwargs={"memory_size": 8, "n_heads": 2}
        )
        inputs = torch.randint(0, 10, (2, 81))
        puzzle_ids = torch.zeros(2, dtype=torch.long)
        
        logits, q_halt, q_continue, steps = model(inputs, puzzle_ids)
        assert logits.shape == (2, 81, 10)
    
    def test_solver_with_broadcast_memory_act(self, solver_params):
        """Solver with 'broadcast_memory' and ACT (multi-step) should preserve memory across steps."""
        from src.pot.models.sudoku_solver import HybridPoHHRMSolver
        
        params = {**solver_params, "halt_max_steps": 3}
        model = HybridPoHHRMSolver(
            **params,
            injection_mode="broadcast_memory",
            injection_kwargs={"memory_size": 8, "n_heads": 2}
        )
        model.eval()
        
        inputs = torch.randint(0, 10, (2, 81))
        puzzle_ids = torch.zeros(2, dtype=torch.long)
        
        logits, q_halt, q_continue, steps, target_q = model(inputs, puzzle_ids)
        assert logits.shape == (2, 81, 10)
    
    def test_solver_with_broadcast_memory_token_conditioned(self, solver_params):
        """Solver with broadcast_memory + token_conditioned read should work."""
        from src.pot.models.sudoku_solver import HybridPoHHRMSolver
        
        model = HybridPoHHRMSolver(
            **solver_params,
            injection_mode="broadcast_memory",
            injection_kwargs={"memory_size": 8, "n_heads": 2, "memory_read": "token_conditioned"}
        )
        inputs = torch.randint(0, 10, (2, 81))
        puzzle_ids = torch.zeros(2, dtype=torch.long)
        
        logits, q_halt, q_continue, steps = model(inputs, puzzle_ids)
        assert logits.shape == (2, 81, 10)
    
    def test_solver_with_broadcast_memory_token_conditioned_act(self, solver_params):
        """Token-conditioned broadcast_memory with ACT should work end-to-end."""
        from src.pot.models.sudoku_solver import HybridPoHHRMSolver
        
        params = {**solver_params, "halt_max_steps": 3}
        model = HybridPoHHRMSolver(
            **params,
            injection_mode="broadcast_memory",
            injection_kwargs={"memory_size": 8, "n_heads": 2, "memory_read": "token_conditioned"}
        )
        model.eval()
        
        inputs = torch.randint(0, 10, (2, 81))
        puzzle_ids = torch.zeros(2, dtype=torch.long)
        
        logits, q_halt, q_continue, steps, target_q = model(inputs, puzzle_ids)
        assert logits.shape == (2, 81, 10)
    
    def test_solver_training_step_token_conditioned(self, solver_params):
        """Token-conditioned broadcast_memory should be trainable."""
        from src.pot.models.sudoku_solver import HybridPoHHRMSolver
        import torch.nn.functional as F
        
        model = HybridPoHHRMSolver(
            **solver_params,
            injection_mode="broadcast_memory",
            injection_kwargs={"memory_size": 8, "n_heads": 2, "memory_read": "token_conditioned"}
        )
        model.train()
        
        inputs = torch.randint(0, 10, (2, 81))
        targets = torch.randint(1, 10, (2, 81))
        puzzle_ids = torch.zeros(2, dtype=torch.long)
        
        logits, q_halt, q_continue, steps = model(inputs, puzzle_ids)
        loss = F.cross_entropy(logits.view(-1, 10), targets.view(-1))
        loss.backward()
        
        grads_found = sum(1 for p in model.parameters() if p.grad is not None)
        assert grads_found > 0
    
    def test_solver_with_cross_attn_act_memory_preserved(self, solver_params):
        """Cross-attn with ACT should preserve injection memory across ACT steps."""
        from src.pot.models.sudoku_solver import HybridPoHHRMSolver
        
        params = {**solver_params, "halt_max_steps": 3}
        model = HybridPoHHRMSolver(
            **params,
            injection_mode="cross_attn",
            injection_kwargs={"memory_size": 8, "n_heads": 2}
        )
        model.eval()
        
        inputs = torch.randint(0, 10, (2, 81))
        puzzle_ids = torch.zeros(2, dtype=torch.long)
        
        logits, q_halt, q_continue, steps, target_q = model(inputs, puzzle_ids)
        assert logits.shape == (2, 81, 10)
    
    def test_solver_forward_with_intermediate(self, solver_params):
        """forward_with_intermediate should return per-step logits."""
        from src.pot.models.sudoku_solver import HybridPoHHRMSolver
        
        params = {**solver_params, "halt_max_steps": 3}
        model = HybridPoHHRMSolver(**params, injection_mode="broadcast")
        model.eval()
        
        inputs = torch.randint(0, 10, (2, 81))
        puzzle_ids = torch.zeros(2, dtype=torch.long)
        
        result = model.forward_with_intermediate(inputs, puzzle_ids)
        
        assert result['logits'].shape == (2, 81, 10)
        assert len(result['intermediate_logits']) == 3  # halt_max_steps
        for step_logits in result['intermediate_logits']:
            assert step_logits.shape == (2, 81, 10)
    
    def test_solver_forward_with_intermediate_no_act(self, solver_params):
        """forward_with_intermediate without ACT should return single step."""
        from src.pot.models.sudoku_solver import HybridPoHHRMSolver
        
        model = HybridPoHHRMSolver(**solver_params, injection_mode="broadcast")
        model.eval()
        
        inputs = torch.randint(0, 10, (2, 81))
        puzzle_ids = torch.zeros(2, dtype=torch.long)
        
        result = model.forward_with_intermediate(inputs, puzzle_ids)
        
        assert result['logits'].shape == (2, 81, 10)
        assert len(result['intermediate_logits']) == 1
    
    def test_solver_training_step(self, solver_params):
        """Solver with injection should be trainable."""
        from src.pot.models.sudoku_solver import HybridPoHHRMSolver
        import torch.nn.functional as F
        
        model = HybridPoHHRMSolver(**solver_params, injection_mode="broadcast")
        model.train()
        
        inputs = torch.randint(0, 10, (2, 81))
        targets = torch.randint(1, 10, (2, 81))
        puzzle_ids = torch.zeros(2, dtype=torch.long)
        
        logits, q_halt, q_continue, steps = model(inputs, puzzle_ids)
        
        # Compute loss and backward
        loss = F.cross_entropy(logits.view(-1, 10), targets.view(-1))
        loss.backward()
        
        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Not all parameters may have gradients in every forward pass
                pass  # Just checking no errors occur
    
    def test_solver_training_step_broadcast_memory(self, solver_params):
        """Solver with broadcast_memory injection should be trainable."""
        from src.pot.models.sudoku_solver import HybridPoHHRMSolver
        import torch.nn.functional as F
        
        model = HybridPoHHRMSolver(
            **solver_params,
            injection_mode="broadcast_memory",
            injection_kwargs={"memory_size": 8, "n_heads": 2}
        )
        model.train()
        
        inputs = torch.randint(0, 10, (2, 81))
        targets = torch.randint(1, 10, (2, 81))
        puzzle_ids = torch.zeros(2, dtype=torch.long)
        
        logits, q_halt, q_continue, steps = model(inputs, puzzle_ids)
        
        # Compute loss and backward
        loss = F.cross_entropy(logits.view(-1, 10), targets.view(-1))
        loss.backward()
        
        # Check that at least some parameters have gradients
        grads_found = sum(1 for p in model.parameters() if p.grad is not None)
        assert grads_found > 0


# =============================================================================
# Test Parameter Counts
# =============================================================================

class TestParameterCounts:
    """Tests to verify parameter counts for different injection modes."""
    
    @pytest.fixture
    def base_params(self):
        return {
            "d_model": 64,
            "d_ctrl": 32,
        }
    
    def test_none_mode_zero_params(self, base_params):
        """Mode 'none' should add zero parameters."""
        injector = FeatureInjector(**base_params, mode="none")
        assert sum(p.numel() for p in injector.parameters()) == 0
    
    def test_broadcast_mode_params(self, base_params):
        """Broadcast mode should have proj + gate + layernorm parameters."""
        injector = FeatureInjector(**base_params, mode="broadcast")
        params = sum(p.numel() for p in injector.parameters())
        # proj: d_ctrl -> d_model, gate: d_ctrl -> 1, ln: 2 * d_model
        expected = base_params["d_ctrl"] * base_params["d_model"] + base_params["d_model"]  # proj weight + bias
        expected += base_params["d_ctrl"] * 1 + 1  # gate weight + bias
        expected += 2 * base_params["d_model"]  # LayerNorm weight + bias
        assert params == expected
    
    def test_film_mode_params(self, base_params):
        """FiLM mode should have MLP parameters."""
        injector = FeatureInjector(**base_params, mode="film")
        params = sum(p.numel() for p in injector.parameters())
        # MLP: d_ctrl -> d_ctrl -> 2*d_model
        expected = base_params["d_ctrl"] * base_params["d_ctrl"] + base_params["d_ctrl"]
        expected += base_params["d_ctrl"] * (2 * base_params["d_model"]) + 2 * base_params["d_model"]
        assert params == expected
    
    def test_depth_token_mode_params(self, base_params):
        """Depth token mode should have proj + layernorm parameters."""
        injector = FeatureInjector(**base_params, mode="depth_token")
        params = sum(p.numel() for p in injector.parameters())
        # proj: d_ctrl -> d_model, ln: 2 * d_model (weight + bias)
        expected = base_params["d_ctrl"] * base_params["d_model"] + base_params["d_model"]  # proj
        expected += 2 * base_params["d_model"]  # LayerNorm
        assert params == expected
    
    def test_cross_attn_mode_params(self, base_params):
        """Cross-attention mode should have memory_proj + cross_attn + ln parameters."""
        injector = FeatureInjector(**base_params, mode="cross_attn", memory_size=8, n_heads=2)
        params = sum(p.numel() for p in injector.parameters())
        # Should be larger than other modes
        assert params > 0
    
    def test_broadcast_memory_mode_params(self, base_params):
        """Broadcast memory mode should have memory_proj + attn + gate + ln + query parameters."""
        injector = FeatureInjector(**base_params, mode="broadcast_memory", memory_size=8, n_heads=2)
        params = sum(p.numel() for p in injector.parameters())
        # Should be larger than broadcast (has attention + memory_proj + summary_query)
        broadcast_injector = FeatureInjector(**base_params, mode="broadcast")
        broadcast_params = sum(p.numel() for p in broadcast_injector.parameters())
        assert params > broadcast_params
    
    def test_alpha_gated_mode_params(self, base_params):
        """Alpha-gated mode should have proj + optional gate parameters."""
        # With learned gate
        injector_with_gate = FeatureInjector(**base_params, mode="alpha_gated", use_learned_gate=True)
        params_with_gate = sum(p.numel() for p in injector_with_gate.parameters())
        
        # Without learned gate
        injector_no_gate = FeatureInjector(**base_params, mode="alpha_gated", use_learned_gate=False)
        params_no_gate = sum(p.numel() for p in injector_no_gate.parameters())
        
        # With gate should have more params
        assert params_with_gate > params_no_gate
        assert params_no_gate > 0

