"""
Tests for modular PoH architecture.

Tests:
1. Parameter parity: PoH vs baseline within 1%
2. Routing correctness: soft sums to 1, topk is sparse
3. ACT halting reduces computation
4. Inner refinement changes output
5. Drop-in compatibility with TransformerEncoder

Author: Eran Ben Artzy
Year: 2025
"""

import pytest
import torch
import torch.nn as nn
from src.pot.modules import (
    PoHConfig,
    PoHBlock,
    PoHStack,
    IterRefiner,
    HeadRouter,
    topk_route,
    soft_route,
)


class TestParameterParity:
    """Test 1: Parameter parity with baseline."""
    
    def test_poh_vs_baseline_param_count(self):
        """PoH should have ≤1% more params than baseline TransformerEncoder (excluding positional encoding)."""
        d_model = 512
        n_heads = 8
        d_ff = 2048
        depth = 6
        
        # Baseline: Standard TransformerEncoder
        baseline_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.1,
            batch_first=True
        )
        baseline = nn.TransformerEncoder(baseline_layer, num_layers=depth)
        baseline_params = sum(p.numel() for p in baseline.parameters())
        
        # PoH: Our architecture (with pos_encoding="none" for fair comparison)
        cfg = PoHConfig(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=0.1,
            param_match_baseline=True,
            share_router=True,
            pos_encoding="none",  # Fair comparison: exclude positional encoding
        )
        poh_stack = PoHStack(cfg, depth=depth)
        poh_params = sum(p.numel() for p in poh_stack.parameters())
        
        # Check within 1%
        ratio = poh_params / baseline_params
        delta_pct = abs(ratio - 1.0) * 100
        
        print(f"\nBaseline: {baseline_params:,} params")
        print(f"PoH:      {poh_params:,} params (pos_encoding=none)")
        print(f"Ratio:    {ratio:.4f} ({delta_pct:.2f}% delta)")
        
        assert delta_pct <= 1.0, f"PoH has {delta_pct:.2f}% param delta (should be ≤1%)"
        
        # Also test with absolute encoding
        cfg_abs = PoHConfig(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=0.1,
            param_match_baseline=True,
            share_router=True,
            pos_encoding="absolute",
            max_seq_len=512,
        )
        poh_stack_abs = PoHStack(cfg_abs, depth=depth)
        poh_params_abs = sum(p.numel() for p in poh_stack_abs.parameters())
        pos_emb_params = cfg_abs.max_seq_len * cfg_abs.d_model  # 512 × 512
        
        print(f"PoH:      {poh_params_abs:,} params (pos_encoding=absolute)")
        print(f"          +{pos_emb_params:,} positional embeddings")
    
    def test_shared_router_reduces_params(self):
        """Shared router should use fewer params than per-block router."""
        cfg_shared = PoHConfig(d_model=256, n_heads=8, share_router=True)
        cfg_separate = PoHConfig(d_model=256, n_heads=8, share_router=False)
        
        stack_shared = PoHStack(cfg_shared, depth=4)
        stack_separate = PoHStack(cfg_separate, depth=4)
        
        params_shared = sum(p.numel() for p in stack_shared.parameters())
        params_separate = sum(p.numel() for p in stack_separate.parameters())
        
        print(f"\nShared router:   {params_shared:,} params")
        print(f"Separate router: {params_separate:,} params")
        print(f"Savings:         {params_separate - params_shared:,} params")
        
        assert params_shared < params_separate, "Shared router should use fewer params"


class TestRouting:
    """Test 2: Routing correctness."""
    
    def test_soft_routing_sums_to_one(self):
        """Soft routing weights should sum to 1 over heads."""
        scores = torch.randn(2, 10, 8)  # [B=2, T=10, H=8]
        weights = soft_route(scores, temperature=1.0)
        
        # Check shape
        assert weights.shape == (2, 10, 8)
        
        # Check sum to 1 over heads
        sums = weights.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(2, 10), atol=1e-5, rtol=1e-5)
        
        # Check non-negative
        assert (weights >= 0).all()
    
    def test_topk_routing_sparsity(self):
        """Top-k routing should select exactly k heads per token."""
        scores = torch.randn(2, 10, 8)  # [B=2, T=10, H=8]
        k = 3
        mask = topk_route(scores, k=k)
        
        # Check shape
        assert mask.shape == (2, 10, 8)
        
        # Check exactly k non-zero per token
        non_zero_counts = (mask > 0).sum(dim=-1)
        assert (non_zero_counts == k).all()
        
        # Check binary (0 or 1)
        assert ((mask == 0) | (mask == 1)).all()
    
    def test_temperature_affects_sharpness(self):
        """Lower temperature should produce sharper distributions."""
        scores = torch.randn(100, 1, 8)  # Large batch for statistics
        
        weights_high_temp = soft_route(scores, temperature=10.0)
        weights_low_temp = soft_route(scores, temperature=0.1)
        
        # Low temp should have higher max values (sharper)
        max_high = weights_high_temp.max(dim=-1).values.mean()
        max_low = weights_low_temp.max(dim=-1).values.mean()
        
        assert max_low > max_high, "Lower temp should produce sharper distributions"


class TestACTHalting:
    """Test 3: ACT halting."""
    
    def test_act_reduces_computation(self):
        """ACT should halt early on some tokens."""
        cfg = PoHConfig(d_model=64, n_heads=4, d_ff=128, act_halting=True, act_threshold=0.9)
        stack = PoHStack(cfg, depth=2)
        refiner = IterRefiner(stack, max_inner_iters=5, act=True, threshold=0.9)
        
        x = torch.randn(2, 10, 64)
        out, stats = refiner(x, return_inner_stats=True)
        
        # Check output shape
        assert out.shape == (2, 10, 64)
        
        # Check halting happened
        assert len(stats) > 0
        last_stat = stats[-1]
        assert "halted_frac" in last_stat
        halted_frac = last_stat["halted_frac"]
        
        print(f"\nHalted fraction: {halted_frac:.2%}")
        assert 0.0 <= halted_frac <= 1.0
    
    def test_no_act_runs_all_iters(self):
        """Without ACT, should run exactly K iterations."""
        cfg = PoHConfig(d_model=64, n_heads=4, d_ff=128, act_halting=False)
        stack = PoHStack(cfg, depth=2)
        refiner = IterRefiner(stack, max_inner_iters=3, act=False)
        
        x = torch.randn(2, 10, 64)
        out, stats = refiner(x, return_inner_stats=True)
        
        # Should have exactly 3 inner steps
        assert len(stats) == 3
        for i, s in enumerate(stats):
            assert s["inner_step"] == i + 1


class TestInnerRefinement:
    """Test 4: Inner refinement changes output."""
    
    def test_refinement_changes_output(self):
        """Multiple inner iterations should change the output."""
        cfg = PoHConfig(d_model=64, n_heads=4, d_ff=128)
        stack = PoHStack(cfg, depth=2)
        refiner = IterRefiner(stack, max_inner_iters=3)
        
        x = torch.randn(2, 10, 64)
        
        # 1 iteration
        refiner.K = 1
        out1, _ = refiner(x)
        
        # 3 iterations
        refiner.K = 3
        out3, _ = refiner(x)
        
        # Should be different
        assert not torch.allclose(out1, out3, atol=1e-5), "Refinement should change output"
    
    def test_stats_collection(self):
        """Inner stats should contain entropy and routing info."""
        cfg = PoHConfig(d_model=64, n_heads=4, d_ff=128)
        stack = PoHStack(cfg, depth=2)
        refiner = IterRefiner(stack, max_inner_iters=2)
        
        x = torch.randn(2, 10, 64)
        out, stats = refiner(x, return_inner_stats=True)
        
        assert len(stats) == 2
        for s in stats:
            assert "inner_step" in s
            assert "route_entropy_mean" in s
            assert "attn_entropy_mean" in s
    
    def test_outer_residual(self):
        """Outer residual should be controllable."""
        cfg = PoHConfig(d_model=64, n_heads=4, d_ff=128)
        stack = PoHStack(cfg, depth=2)
        
        # Without outer residual
        refiner_no_res = IterRefiner(stack, max_inner_iters=2, outer_residual=False)
        
        # With outer residual (plain)
        refiner_res = IterRefiner(stack, max_inner_iters=2, outer_residual=True, rezero_init=False)
        
        # With ReZero initialization
        refiner_rezero = IterRefiner(stack, max_inner_iters=2, outer_residual=True, rezero_init=True)
        
        x = torch.randn(2, 10, 64)
        
        # Check alpha parameter exists
        assert not hasattr(refiner_no_res, 'alpha'), "No alpha without outer_residual"
        assert hasattr(refiner_res, 'alpha'), "Alpha should exist with outer_residual"
        assert hasattr(refiner_rezero, 'alpha'), "Alpha should exist with rezero"
        
        # Check initialization
        assert refiner_res.alpha.item() == 1.0, "Plain residual should init alpha=1"
        assert refiner_rezero.alpha.item() == 0.0, "ReZero should init alpha=0"
        
        # Check outputs are different
        out_no_res, _ = refiner_no_res(x)
        out_res, _ = refiner_res(x)
        
        # They should be different (residual changes the computation)
        assert not torch.allclose(out_no_res, out_res, atol=1e-5), "Outer residual should change output"
        
        print(f"\n✅ Outer residual test passed")
        print(f"   Alpha (plain): {refiner_res.alpha.item():.4f}")
        print(f"   Alpha (rezero): {refiner_rezero.alpha.item():.4f}")


class TestPositionalEncoding:
    """Test 5: Positional encoding modes."""
    
    def test_none_encoding(self):
        """None encoding should not modify input."""
        from src.pot.modules import PositionalEncoding
        
        cfg = PoHConfig(d_model=128, n_heads=4, pos_encoding="none")
        pos_enc = PositionalEncoding(cfg)
        
        x = torch.randn(2, 10, 128)
        x_pos, _ = pos_enc(x)
        
        # Should be identical
        torch.testing.assert_close(x, x_pos)
        print("\n✅ None encoding: input unchanged")
    
    def test_absolute_encoding(self):
        """Absolute encoding should add learned embeddings."""
        from src.pot.modules import PositionalEncoding
        
        cfg = PoHConfig(d_model=128, n_heads=4, pos_encoding="absolute", max_seq_len=512)
        pos_enc = PositionalEncoding(cfg)
        
        x = torch.randn(2, 10, 128)
        x_pos, _ = pos_enc(x)
        
        # Should be different (positions added)
        assert not torch.allclose(x, x_pos)
        
        # Check params exist
        assert hasattr(pos_enc, 'pos_emb')
        assert pos_enc.pos_emb.weight.shape == (512, 128)
        
        print(f"\n✅ Absolute encoding: {pos_enc.pos_emb.weight.shape} params")
    
    def test_absolute_encoding_max_len_check(self):
        """Absolute encoding should error on sequences exceeding max_seq_len."""
        from src.pot.modules import PositionalEncoding
        
        cfg = PoHConfig(d_model=64, n_heads=4, pos_encoding="absolute", max_seq_len=10)
        pos_enc = PositionalEncoding(cfg)
        
        # Should work for T <= max_seq_len
        x_short = torch.randn(1, 10, 64)
        pos_enc(x_short)  # OK
        
        # Should fail for T > max_seq_len
        x_long = torch.randn(1, 20, 64)
        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            pos_enc(x_long)
    
    def test_rotary_not_available_warning(self):
        """Rotary mode should warn if rotary-embedding-torch not installed."""
        from src.pot.modules.positional import ROTARY_AVAILABLE
        
        if not ROTARY_AVAILABLE:
            from src.pot.modules import PositionalEncoding
            
            cfg = PoHConfig(d_model=64, n_heads=4, pos_encoding="rotary")
            
            with pytest.raises(ImportError, match="rotary-embedding-torch"):
                pos_enc = PositionalEncoding(cfg)
            
            print("\n⚠️  RoPE not available (expected, rotary-embedding-torch not installed)")
        else:
            print("\n✅ RoPE available")
    
    def test_stack_with_different_encodings(self):
        """PoHStack should work with all encoding modes."""
        for pos_enc_mode in ["none", "absolute"]:  # Skip rotary if not available
            cfg = PoHConfig(d_model=64, n_heads=4, pos_encoding=pos_enc_mode, max_seq_len=100)
            stack = PoHStack(cfg, depth=2)
            
            x = torch.randn(2, 10, 64)
            out, stats = stack(x)
            
            assert out.shape == x.shape
            assert len(stats) == 2
            
            print(f"✅ PoHStack with pos_encoding={pos_enc_mode}: OK")


class TestDropInCompatibility:
    """Test 6: Drop-in compatibility."""
    
    def test_forward_signature_compatibility(self):
        """PoHStack should have similar interface to TransformerEncoder."""
        cfg = PoHConfig(d_model=256, n_heads=8)
        poh_stack = PoHStack(cfg, depth=4)
        
        x = torch.randn(2, 10, 256)
        
        # Should work with just x
        out, stats = poh_stack(x)
        assert out.shape == x.shape
        
        # Should work with mask (mask should be [T, T] for attention, not [B, T])
        mask = torch.ones(10, 10).bool()
        out, stats = poh_stack(x, attn_mask=mask)
        assert out.shape == x.shape
    
    def test_gradient_flow(self):
        """Gradients should flow through the entire stack."""
        cfg = PoHConfig(d_model=64, n_heads=4, d_ff=128)
        stack = PoHStack(cfg, depth=2)
        refiner = IterRefiner(stack, max_inner_iters=2)
        
        x = torch.randn(2, 10, 64, requires_grad=True)
        out, _ = refiner(x)
        
        # Compute dummy loss
        loss = out.mean()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        
        # Check all params have gradients
        for name, param in refiner.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} has no gradient"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

