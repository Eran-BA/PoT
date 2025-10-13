"""
Unit tests for HRM-style Pointer Controller.

Tests:
- Basic forward pass shapes
- Multi-timescale H-module updates (every T steps)
- Top-k sparsity and temperature scheduling
- Gradient flow through controller

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import pytest
import torch
import torch.nn.functional as F

from src.models.layers import HRMPointerController, HRMState


@pytest.mark.parametrize("B,L,D,H", [(4, 7, 32, 6)])
def test_basic_forward_shapes(B, L, D, H):
    """Test basic forward pass returns correct shapes."""
    ctrl = HRMPointerController(d_model=D, n_heads=H, d_ctrl=32, T=3, topk=None)
    x = torch.randn(B, L, D)
    head_feats = torch.randn(B, H, L, D // H)  # [B,H,L,d_head]
    state = ctrl.init_state(B, x.device)

    alphas, new_state, aux = ctrl(x, head_outputs=head_feats, state=state)
    
    assert alphas.shape == (B, H), f"Expected alphas shape ({B}, {H}), got {alphas.shape}"
    assert isinstance(new_state, HRMState), "Expected HRMState returned"
    assert "entropy" in aux and aux["entropy"].ndim == 0, "Expected scalar entropy in aux"
    assert "alphas" in aux and aux["alphas"].shape == (B, H)
    assert "router_logits" in aux and aux["router_logits"].shape == (B, H)
    assert "temperature" in aux and isinstance(aux["temperature"], float)


@pytest.mark.parametrize("B,D,H,T", [(2, 32, 8, 4)])
def test_high_level_updates_every_T_steps(B, D, H, T):
    """Test that H-module (slow) updates only every T steps."""
    ctrl = HRMPointerController(d_model=D, n_heads=H, d_ctrl=32, T=T)
    x = torch.randn(B, D)  # pooled input
    head_feats = torch.randn(B, H, 1, D // H)
    state = ctrl.init_state(B, x.device)

    def step_once(state_):
        alphas, state2, _ = ctrl(x, head_outputs=head_feats, state=state_)
        return state2

    zH_vals = []
    for t in range(T + 2):
        state = step_once(state)
        zH_vals.append(state.z_H.detach().clone())

    # z_H should change at steps where (step % T) == 0
    # We started at step=0; after first call step=1
    # Changes expected at steps that are multiples of T
    changed_at_T = (zH_vals[T] - zH_vals[T - 1]).abs().sum().item()
    assert changed_at_T > 0.0, "z_H should update at the T-th step"
    
    # Check that between steps 1..(T-1), there is little or no change
    accum = 0.0
    for i in range(1, T):
        accum += (zH_vals[i] - zH_vals[i - 1]).abs().sum().item()
    assert accum < changed_at_T + 1e-6, (
        f"z_H should be mostly constant until step T. "
        f"Accumulated change before T: {accum:.6f}, change at T: {changed_at_T:.6f}"
    )


def test_topk_sparsity_and_temperature():
    """Test top-k sparsification and temperature effect on routing sharpness."""
    B, D, H = 3, 48, 10
    ctrl = HRMPointerController(
        d_model=D,
        n_heads=H,
        d_ctrl=32,
        T=2,
        topk=3,
        temperature_init=2.0,
        temperature_min=0.7,
    )
    x = torch.randn(B, D)
    head_feats = torch.randn(B, H, 1, D // H)
    state = ctrl.init_state(B, x.device)

    # High temperature → softer distribution
    ctrl.set_temperature(2.0)
    a_soft, state, _ = ctrl(x, head_outputs=head_feats, state=state)
    assert torch.allclose(a_soft.sum(-1), torch.ones(B), atol=1e-5), "Alphas should sum to 1"

    # Low temperature → sharper distribution with same top-k
    ctrl.set_temperature(0.7)
    a_sharp, state, _ = ctrl(x, head_outputs=head_feats, state=state)
    
    # Both are top-k sparse (at most 3 non-zero heads per batch)
    assert (a_soft > 0).sum(dim=-1).max().item() <= 3, "Soft routing should be top-3 sparse"
    assert (a_sharp > 0).sum(dim=-1).max().item() <= 3, "Sharp routing should be top-3 sparse"
    
    # Sharper means higher max prob
    assert a_sharp.max().item() >= a_soft.max().item(), (
        f"Lower temperature should produce sharper routing. "
        f"Soft max: {a_soft.max().item():.4f}, Sharp max: {a_sharp.max().item():.4f}"
    )


def test_temperature_schedule():
    """Test temperature scheduling and clamping to minimum."""
    B, D, H = 2, 32, 6
    ctrl = HRMPointerController(
        d_model=D, n_heads=H, temperature_init=2.0, temperature_min=0.5
    )
    
    # Set above min
    ctrl.set_temperature(1.0)
    assert abs(torch.exp(ctrl.log_temperature).item() - 1.0) < 1e-5
    
    # Try to set below min (should clamp)
    ctrl.set_temperature(0.3)
    assert torch.exp(ctrl.log_temperature).item() >= 0.5, "Should clamp to temperature_min"


def test_gradients_flow():
    """Test that gradients flow back through controller to inputs."""
    B, D, H = 2, 32, 6
    ctrl = HRMPointerController(d_model=D, n_heads=H, d_ctrl=32, T=3)
    x = torch.randn(B, D, requires_grad=True)
    head_feats = torch.randn(B, H, 1, D // H)
    state = ctrl.init_state(B, x.device)

    # Simple target: push probability mass to head 0
    alphas, state, _ = ctrl(x, head_outputs=head_feats, state=state)
    target = torch.zeros_like(alphas)
    target[:, 0] = 1.0
    loss = F.kl_div((alphas + 1e-9).log(), target, reduction="batchmean")
    loss.backward()

    # Controller params should have non-zero grad
    has_grad = False
    for n, p in ctrl.named_parameters():
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            has_grad = True
            break
    assert has_grad, "Expected non-zero gradients through controller"
    
    # Input should also have grad
    assert x.grad is not None and x.grad.abs().sum().item() > 0, (
        "Gradients should flow back to input x"
    )


def test_state_persistence():
    """Test that state properly persists across iterations."""
    B, D, H = 2, 32, 6
    ctrl = HRMPointerController(d_model=D, n_heads=H, d_ctrl=32, T=3)
    x = torch.randn(B, D)
    head_feats = torch.randn(B, H, 1, D // H)
    
    state = ctrl.init_state(B, x.device)
    assert state.step.sum().item() == 0, "Initial step should be 0"
    
    # Run 5 iterations
    for t in range(5):
        _, state, _ = ctrl(x, head_outputs=head_feats, state=state)
    
    assert state.step.min().item() == 5, f"After 5 iterations, step should be 5, got {state.step}"
    assert state.z_L.shape == (B, 32)
    assert state.z_H.shape == (B, 32)


def test_entropy_regularization():
    """Test that entropy is computed and changes with routing sharpness."""
    B, D, H = 4, 32, 8
    ctrl = HRMPointerController(
        d_model=D, n_heads=H, temperature_init=3.0, temperature_min=0.5
    )
    x = torch.randn(B, D)
    head_feats = torch.randn(B, H, 1, D // H)
    state = ctrl.init_state(B, x.device)
    
    # High temperature → higher entropy
    ctrl.set_temperature(3.0)
    _, state, aux_soft = ctrl(x, head_outputs=head_feats, state=state)
    entropy_soft = aux_soft["entropy"].item()
    
    # Low temperature → lower entropy (sharper distribution)
    ctrl.set_temperature(0.5)
    _, state, aux_sharp = ctrl(x, head_outputs=head_feats, state=state)
    entropy_sharp = aux_sharp["entropy"].item()
    
    assert entropy_soft > entropy_sharp, (
        f"Higher temperature should yield higher entropy. "
        f"Soft: {entropy_soft:.4f}, Sharp: {entropy_sharp:.4f}"
    )


def test_different_input_shapes():
    """Test controller handles both pooled and sequence inputs."""
    B, L, D, H = 2, 10, 32, 4
    ctrl = HRMPointerController(d_model=D, n_heads=H)
    head_feats = torch.randn(B, H, 1, D // H)
    state = ctrl.init_state(B, "cpu")
    
    # Pooled input [B, D]
    x_pooled = torch.randn(B, D)
    alphas1, state, _ = ctrl(x_pooled, head_outputs=head_feats, state=state)
    assert alphas1.shape == (B, H)
    
    # Sequence input [B, L, D] with mean pooling
    x_seq = torch.randn(B, L, D)
    alphas2, state, _ = ctrl(
        x_seq, head_outputs=head_feats, state=state, per_token_pool="mean"
    )
    assert alphas2.shape == (B, H)
    
    # Sequence input with CLS pooling
    alphas3, state, _ = ctrl(
        x_seq, head_outputs=head_feats, state=state, per_token_pool="cls"
    )
    assert alphas3.shape == (B, H)


def test_masked_pooling():
    """Test that masking correctly handles variable-length sequences."""
    B, L, D, H = 2, 8, 32, 4
    ctrl = HRMPointerController(d_model=D, n_heads=H)
    x = torch.randn(B, L, D)
    head_feats = torch.randn(B, H, 1, D // H)
    state = ctrl.init_state(B, "cpu")
    
    # Create mask: first sequence full, second sequence half
    mask = torch.ones(B, L)
    mask[1, 4:] = 0  # Mask out last half of second sequence
    
    alphas, state, _ = ctrl(x, head_outputs=head_feats, state=state, mask=mask)
    assert alphas.shape == (B, H)
    
    # Results should differ between masked and unmasked
    alphas_no_mask, _, _ = ctrl(
        x, head_outputs=head_feats, state=ctrl.init_state(B, "cpu")
    )
    assert not torch.allclose(alphas, alphas_no_mask, atol=1e-5), (
        "Masking should affect pooling and thus routing"
    )


@pytest.mark.parametrize("topk", [None, 2, 4])
def test_topk_variants(topk):
    """Test different top-k settings."""
    B, D, H = 2, 32, 8
    ctrl = HRMPointerController(d_model=D, n_heads=H, topk=topk)
    x = torch.randn(B, D)
    head_feats = torch.randn(B, H, 1, D // H)
    state = ctrl.init_state(B, "cpu")
    
    alphas, state, aux = ctrl(x, head_outputs=head_feats, state=state)
    
    if topk is None:
        # Dense routing: all heads may be used
        assert (alphas > 0).sum().item() > 0
    else:
        # Sparse routing: at most topk heads per batch element
        assert (alphas > 0).sum(dim=-1).max().item() <= topk
        if "topk_idx" in aux:
            assert aux["topk_idx"].shape == (B, topk)


if __name__ == "__main__":
    # Run basic smoke test
    print("Running HRM Controller unit tests...")
    pytest.main([__file__, "-v"])

