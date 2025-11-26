"""
Core neural network layers for Pointer-over-Heads Transformer.

This module provides the fundamental building blocks for the PoH architecture:
- Utility functions for routing and entropy computation
- Multi-head self-attention with per-head outputs
- Pointer controller for head routing

Classes:
    MultiHeadSelfAttention: Standard MHA returning per-head outputs
    PointerOverHeadsController: Computes routing logits over attention heads

Functions:
    entropy_from_logits: Compute categorical entropy from logits
    gumbel_softmax_topk: Top-k straight-through Gumbel-Softmax routing

Example:
    >>> from src.models.layers import MultiHeadSelfAttention, PointerOverHeadsController
    >>> mha = MultiHeadSelfAttention(d_model=768, n_heads=8)
    >>> controller = PointerOverHeadsController(d_model=768, n_heads=8, recurrent=True)
    >>> x = torch.randn(2, 10, 768)
    >>> heads_out, attn = mha(x)
    >>> routing_logits = controller(x, provisional_heads=heads_out)

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HRMState:
    """Persistent controller state (per sequence or per layer, depending on where you keep it)."""
    z_L: torch.Tensor  # [B, d_ctrl]
    z_H: torch.Tensor  # [B, d_ctrl]
    step: torch.Tensor  # [B] or scalar long; counts inner-iteration steps


class HRMPointerController(nn.Module):
    """
    HRM-style two-timescale controller that produces routing logits over attention heads.

    - Low-level module f_L: updates EVERY inner step.
    - High-level module f_H: updates every T steps (multi-timescale), conditions f_L via context.
    - Produces head logits -> softmax (temperature) -> optional top-k -> alphas.

    Note on routing granularity:
        This controller produces per-SEQUENCE routing weights [B, n_heads] that are
        broadcast to all tokens. For true per-TOKEN routing [B, T, H], use
        PointerOverHeadsController instead.

    Intended as a drop-in replacement for your existing Pointer Controller in PoT.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads to route over
        d_ctrl: Controller hidden dimension (default: d_model)
        T: Period for H-module updates (slow timescale, default: 4)
        topk: Optional sparsification (route to top-k heads only)
        temperature_init: Initial softmax temperature (default: 2.0)
        temperature_min: Minimum temperature (default: 0.7)
        entropy_reg: Entropy regularization coefficient (default: 1e-3)
        use_layernorm: Apply LayerNorm to states (default: True)
        dropout: Dropout probability (default: 0.0)
        detach_H_update: If True, block gradients through slow-timescale H updates
                        for memory efficiency. If False (default), H-module is fully
                        differentiable as in the original HRM paper.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ctrl: Optional[int] = None,
        *,
        T: int = 4,                      # H-module update period (slow timescale)
        topk: Optional[int] = None,      # if set, sparsify routing to top-k heads
        temperature_init: float = 2.0,   # start soft, cool down during training
        temperature_min: float = 0.7,
        entropy_reg: float = 1e-3,       # small entropy reg; decay in trainer if desired
        use_layernorm: bool = True,
        dropout: float = 0.0,
        detach_H_update: bool = False,   # If True, block gradients through H updates
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ctrl = d_ctrl or d_model
        self.T = int(T)
        self.topk = topk
        self.temperature_init = temperature_init
        self.temperature_min = temperature_min
        self.entropy_reg = entropy_reg
        self.detach_H_update = detach_H_update

        # Input adapters: summarize token/context into controller space
        self.inp_proj = nn.Linear(d_model, self.d_ctrl)

        # ---------- HRM core ----------
        # Low-level (fast) recurrent cell
        self.f_L = nn.GRUCell(input_size=self.d_ctrl * 2, hidden_size=self.d_ctrl)
        # High-level (slow) recurrent cell
        self.f_H = nn.GRUCell(input_size=self.d_ctrl, hidden_size=self.d_ctrl)

        # Optional norms
        self.ln_L = nn.LayerNorm(self.d_ctrl) if use_layernorm else nn.Identity()
        self.ln_H = nn.LayerNorm(self.d_ctrl) if use_layernorm else nn.Identity()

        # Head router (maps low-level state -> logits over heads)
        self.router = nn.Linear(self.d_ctrl, n_heads)

        # Optionally condition router on H as well
        self.mix_gate = nn.Linear(self.d_ctrl, self.d_ctrl, bias=False)  # modulate z_L by z_H

        self.drop = nn.Dropout(dropout)

        # Temperature as a learnable/log-scheduled scalar (kept simple here; you can schedule externally)
        self.log_temperature = nn.Parameter(torch.tensor(self._to_logtemp(temperature_init)), requires_grad=False)

    # -------------- Utilities --------------
    def _to_logtemp(self, T: float) -> float:
        return float(torch.log(torch.tensor(T)))

    def set_temperature(self, T: float):
        """Optionally called by the trainer to schedule temperature per epoch."""
        T = max(self.temperature_min, float(T))
        with torch.no_grad():
            self.log_temperature.copy_(torch.log(torch.tensor(T)))

    def init_state(self, batch_size: int, device: torch.device) -> HRMState:
        z0 = torch.zeros(batch_size, self.d_ctrl, device=device)
        step = torch.zeros(batch_size, dtype=torch.long, device=device)
        return HRMState(z_L=z0, z_H=z0, step=step)

    # -------------- Forward --------------
    def _maybe_update_H(self, x_ctrl: torch.Tensor, state: HRMState) -> HRMState:
        """Update H-module only when (step % T) == 0 (per-example).

        Simplest variant: if any in batch needs H-update, we compute for all (keeps it batched).

        If detach_H_update=True, gradients are blocked through this update for memory efficiency.
        If detach_H_update=False (default), the H-module is fully differentiable.
        """
        needs = (state.step % self.T) == 0
        if needs.any():
            ctx = torch.no_grad() if self.detach_H_update else nullcontext()
            with ctx:
                # standard GRUCell expects [B, in_dim]
                z_H_new = self.f_H(x_ctrl, state.z_H)
                state = HRMState(z_L=state.z_L, z_H=self.ln_H(z_H_new), step=state.step)
        return state

    def forward(
        self,
        x: torch.Tensor,                      # [B, L, d_model] or [B, d_model] if pooled
        head_outputs: torch.Tensor,           # [B, n_heads, ...] precomputed head features to be mixed
        *,
        state: Optional[HRMState] = None,
        per_token_pool: str = "mean",         # how to summarize sequence to controller, if x is [B, L, d]
        mask: Optional[torch.Tensor] = None,  # [B, L] attention mask if you want masked pool
        return_aux: bool = True
    ) -> Tuple[torch.Tensor, HRMState, Dict[str, Any]]:
        """
        Returns:
          alphas: [B, n_heads] routing weights
          new_state: updated HRMState
          aux: dict with 'entropy', 'logits', 'temperature', (optionally 'topk_idx')
        """
        B = x.size(0)
        device = x.device
        if state is None:
            state = self.init_state(B, device)

        # ----- summarize x into controller space -----
        if x.dim() == 3:
            if mask is not None:
                # masked mean
                m = mask.float().unsqueeze(-1)  # [B, L, 1]
                x_sum = (x * m).sum(dim=1)
                x_den = m.sum(dim=1).clamp_min(1e-6)
                x_pooled = x_sum / x_den
            else:
                if per_token_pool == "mean":
                    x_pooled = x.mean(dim=1)
                elif per_token_pool == "cls":
                    x_pooled = x[:, 0]
                else:
                    x_pooled = x.mean(dim=1)
        else:
            x_pooled = x  # already [B, d_model]

        x_ctrl = self.inp_proj(self.drop(x_pooled))  # [B, d_ctrl]

        # ----- HRM updates: H slow, L fast -----
        state = self._maybe_update_H(x_ctrl, state)

        # L takes current x and context from H
        l_inp = torch.cat([x_ctrl, state.z_H], dim=-1)        # [B, 2*d_ctrl]
        z_L_new = self.f_L(l_inp, state.z_L)                  # [B, d_ctrl]
        z_L_new = self.ln_L(z_L_new)

        # small cross-conditioning: modulate z_L by z_H (FiLM-lite)
        z_L_cond = z_L_new + self.mix_gate(state.z_H)

        # ----- Routing over heads -----
        logits = self.router(self.drop(z_L_cond))             # [B, n_heads]
        T = torch.exp(self.log_temperature).clamp(min=self.temperature_min)
        probs = F.softmax(logits / T, dim=-1)                 # [B, n_heads]

        # Optional top-k sparsification (straightforward masking)
        topk_idx = None
        if self.topk is not None and self.topk < self.n_heads:
            topk_vals, topk_idx = probs.topk(self.topk, dim=-1)
            mask_topk = torch.zeros_like(probs)
            mask_topk.scatter_(dim=-1, index=topk_idx, value=1.0)
            probs = probs * mask_topk
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # ----- Entropy regularization (return term; your trainer can add to loss and decay weight) -----
        entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1).mean()

        # ----- Advance step counter -----
        new_step = state.step + 1
        new_state = HRMState(z_L=z_L_new, z_H=state.z_H, step=new_step)

        # ----- Mix heads (outside controller: you may already do it) -----
        # In many PoT blocks you already multiply alphas with head outputs.
        # We return alphas and let the caller do the actual mixing for shape consistency.
        alphas = probs  # [B, n_heads]

        aux = {}
        if return_aux:
            aux = {
                "router_logits": logits.detach(),
                "alphas": alphas.detach(),
                "entropy": entropy.detach(),
                "temperature": float(T.detach().cpu()),
            }
            if topk_idx is not None:
                aux["topk_idx"] = topk_idx.detach()

        return alphas, new_state, aux





# -------------------------
# Utility Functions
# -------------------------


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute mean categorical entropy from routing logits.

    Measures uncertainty in routing decisions. Higher entropy indicates
    more uniform distribution across heads, lower entropy indicates
    strong preference for specific heads.

    Args:
        logits: Routing logits [..., H] where H is number of heads

    Returns:
        Scalar tensor with mean entropy across all positions

    Example:
        >>> logits = torch.randn(2, 10, 8)  # [B, T, H]
        >>> ent = entropy_from_logits(logits)
        >>> print(f"Mean routing entropy: {ent:.3f}")
    """
    p = F.softmax(logits, dim=-1)
    lp = F.log_softmax(logits, dim=-1)
    ent = -(p * lp).sum(dim=-1)
    return ent.mean()


def gumbel_softmax_topk(
    logits: torch.Tensor, tau: float = 1.0, topk: int = 1, hard: bool = True
) -> torch.Tensor:
    """Top-k straight-through Gumbel-Softmax for hard routing.

    Implements differentiable top-k selection using Gumbel-Softmax trick
    with straight-through estimator. Returns k-hot vector normalized to
    sum to 1 (convex combination of top-k heads).

    Args:
        logits: Routing logits [*, H]
        tau: Temperature parameter (lower = more peaked)
        topk: Number of heads to select
        hard: If True, use straight-through estimator for discrete output

    Returns:
        k-hot routing weights [*, H] summing to 1.0

    Note:
        During forward pass, returns hard k-hot selection.
        During backward pass, gradients flow through soft Gumbel-Softmax.

    Example:
        >>> logits = torch.randn(2, 10, 8)  # [B, T, H]
        >>> weights = gumbel_softmax_topk(logits, tau=1.0, topk=2, hard=True)
        >>> assert (weights > 0).sum(dim=-1).allclose(torch.tensor(2.0))  # Exactly 2 heads selected
        >>> assert weights.sum(dim=-1).allclose(torch.tensor(1.0))  # Normalized
    """
    # Sample Gumbel noise
    g = -torch.empty_like(logits).exponential_().log()  # ~ Gumbel(0,1)
    y = (logits + g) / max(tau, 1e-6)
    y_soft = F.softmax(y, dim=-1)

    if not hard:
        return y_soft

    # Hard top-k selection with straight-through
    k = min(topk, y_soft.size(-1))
    vals, idx = torch.topk(y_soft, k=k, dim=-1)
    mask = torch.zeros_like(y_soft)
    mask.scatter_(-1, idx, 1.0)

    # Straight-through estimator: forward uses hard, backward uses soft
    y_hard = mask + (y_soft - y_soft.detach())

    # Normalize k-hot to sum to 1 (convex combination)
    y_hard = y_hard / (y_hard.sum(dim=-1, keepdim=True) + 1e-8)
    return y_hard


# -------------------------
# Multi-Head Self-Attention
# -------------------------


class MultiHeadSelfAttention(nn.Module):
    """Standard Multi-Head Self-Attention with per-head outputs.

    Unlike typical MHA implementations that concatenate and project heads,
    this version returns individual head outputs [B, T, H, Dh] to enable
    head-level routing in the PoH architecture.

    Args:
        d_model: Model hidden dimension (must be divisible by n_heads)
        n_heads: Number of attention heads
        bias: Whether to use bias in linear projections
        attn_dropout: Dropout probability for attention weights

    Attributes:
        d_head: Dimension per head (d_model // n_heads)
        q, k, v: Query, key, value projections
        out: Output projection (applied after head combination)

    Example:
        >>> mha = MultiHeadSelfAttention(d_model=768, n_heads=8)
        >>> x = torch.randn(2, 10, 768)  # [batch, seq_len, d_model]
        >>> heads_out, attn_probs = mha(x)
        >>> print(heads_out.shape)  # [2, 10, 8, 96]
        >>> print(attn_probs.shape)  # [2, 8, 10, 10]
    """

    def __init__(self, d_model: int, n_heads: int, *, bias: bool = True, attn_dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q = nn.Linear(d_model, d_model, bias=bias)
        self.k = nn.Linear(d_model, d_model, bias=bias)
        self.v = nn.Linear(d_model, d_model, bias=bias)
        self.out = nn.Linear(d_model, d_model, bias=bias)
        self.drop = nn.Dropout(attn_dropout)

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-head self-attention with per-head outputs.

        Args:
            x: Input representations [B, T, D]
            attn_mask: Optional attention mask [B, H, T, T].
                      Additive mask (use -inf where masked).

        Returns:
            Tuple containing:
            - heads_out: Per-head outputs [B, T, H, Dh] (before concatenation)
            - attn_probs: Attention probabilities [B, H, T, T]

        Note:
            The per-head outputs enable dynamic routing in PoH.
            Use heads_out for routing, not the concatenated version.
        """
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head

        # Project and reshape to [B, H, T, Dh]
        Q = self.q(x).view(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]
        K = self.k(x).view(B, T, H, Dh).transpose(1, 2)
        V = self.v(x).view(B, T, H, Dh).transpose(1, 2)

        # Scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(Dh)  # [B, H, T, T]

        if attn_mask is not None:
            scores = scores + attn_mask

        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.drop(attn_probs)

        # Apply attention to values
        heads_out = attn_probs @ V  # [B, H, T, Dh]

        # Return in [B, T, H, Dh] format for easier routing
        heads_out = heads_out.transpose(1, 2)  # [B, T, H, Dh]

        return heads_out, attn_probs


# -------------------------
# Pointer Controller
# -------------------------


class PointerOverHeadsController(nn.Module):
    """Computes routing logits over attention heads.

    The controller decides which attention heads to use at each position
    by computing routing logits. Supports both feed-forward (static) and
    recurrent (feedback) modes.

    In recurrent mode, the controller receives feedback from provisional
    head outputs, enabling adaptive routing based on what each head computed.

    Args:
        d_model: Model hidden dimension
        n_heads: Number of attention heads to route over
        recurrent: If True, use feedback from provisional heads
        summary: How to summarize head outputs ('mean' or 'max')
        hidden_factor: Size of hidden layer relative to d_model

    Attributes:
        ln: LayerNorm for input normalization
        mlp_static: Static routing MLP (always used)
        mlp_recur: Recurrent routing MLP (if recurrent=True)

    Example:
        >>> # Static (feed-forward) controller
        >>> controller = PointerOverHeadsController(d_model=768, n_heads=8, recurrent=False)
        >>> x = torch.randn(2, 10, 768)
        >>> logits = controller(x)
        >>> print(logits.shape)  # [2, 10, 8]

        >>> # Recurrent controller with feedback
        >>> controller = PointerOverHeadsController(d_model=768, n_heads=8, recurrent=True)
        >>> heads_out = torch.randn(2, 10, 8, 96)  # Provisional head outputs
        >>> logits = controller(x, provisional_heads=heads_out)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        recurrent: bool = False,
        summary: str = "mean",
        hidden_factor: float = 0.5,
    ):
        super().__init__()
        self.recurrent = recurrent
        self.summary = summary
        self.n_heads = n_heads

        # Hidden layer size (at least 32 dimensions)
        hid = max(32, int(d_model * hidden_factor))

        # Layer normalization for stable training
        self.ln = nn.LayerNorm(d_model)

        # Static routing path (always used)
        self.mlp_static = nn.Sequential(nn.Linear(d_model, hid), nn.GELU(), nn.Linear(hid, n_heads))

        # Recurrent routing path (conditional on provisional heads)
        if recurrent:
            self.mlp_recur = nn.Sequential(
                nn.Linear(d_model + n_heads, hid), nn.GELU(), nn.Linear(hid, n_heads)
            )

    def summarize_heads(self, heads_out: torch.Tensor) -> torch.Tensor:
        """Summarize per-head outputs to scalars for routing feedback.

        Args:
            heads_out: Per-head outputs [B, T, H, Dh]

        Returns:
            Head summaries [B, T, H] (one scalar per head per position)

        Raises:
            ValueError: If summary method is not 'mean' or 'max'
        """
        if self.summary == "mean":
            return heads_out.mean(dim=-1)
        if self.summary == "max":
            return heads_out.amax(dim=-1)
        raise ValueError(f"summary must be 'mean' or 'max', got {self.summary}")

    def forward(
        self, token_ctx: torch.Tensor, provisional_heads: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute routing logits over attention heads.

        Args:
            token_ctx: Token representations [B, T, D]
            provisional_heads: Optional provisional head outputs [B, T, H, Dh].
                             Required if recurrent=True.

        Returns:
            Routing logits [B, T, H] (one logit per head per position)

        Raises:
            AssertionError: If recurrent=True but provisional_heads not provided

        Note:
            Logits are not normalized here. Apply softmax or gumbel_softmax_topk
            to get routing weights.
        """
        h = self.ln(token_ctx)
        logits = self.mlp_static(h)  # [B, T, H]

        if self.recurrent:
            assert provisional_heads is not None, "recurrent=True requires provisional_heads"
            # Summarize what each head computed
            s = self.summarize_heads(provisional_heads)  # [B, T, H]
            # Combine context with head feedback
            logits = logits + self.mlp_recur(torch.cat([h, s], dim=-1))

        return logits


# -------------------------
# Biaffine Layers
# -------------------------


class BiaffinePointer(nn.Module):
    """Biaffine attention for dependency head prediction.

    Computes biaffine scores between dependent and candidate head positions.
    Includes a special ROOT token at position 0. Used for unlabeled attachment.

    Args:
        d: Model hidden dimension

    Attributes:
        W: Biaffine weight matrix [D, D]
        U: Bias term for head candidates
        root: Learnable ROOT representation

    Example:
        >>> pointer = BiaffinePointer(d=768)
        >>> dep = torch.randn(2, 10, 768)  # Dependent representations
        >>> head = torch.randn(2, 10, 768)  # Head candidate representations
        >>> mask_dep = torch.ones(2, 10, dtype=torch.bool)
        >>> mask_head = torch.ones(2, 10, dtype=torch.bool)
        >>> logits = pointer(dep, head, mask_dep, mask_head)
        >>> print(logits.shape)  # [2, 10, 11] - each word chooses from ROOT + 10 words
    """

    def __init__(self, d: int):
        super().__init__()
        # Biaffine weight matrix
        self.W = nn.Parameter(torch.empty(d, d))
        nn.init.xavier_uniform_(self.W)

        # Head bias term
        self.U = nn.Linear(d, 1, bias=True)

        # Learnable ROOT representation
        self.root = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.root, std=0.02)

    def forward(
        self, dep: torch.Tensor, head: torch.Tensor, mask_dep: torch.Tensor, mask_head: torch.Tensor
    ) -> torch.Tensor:
        """Compute biaffine head selection logits.

        Args:
            dep: Dependent representations [B, T, D]
            head: Head candidate representations [B, T, D]
            mask_dep: Valid dependent mask [B, T]
            mask_head: Valid head mask [B, T]

        Returns:
            Logits for head selection [B, T, T+1] where T+1 includes ROOT at position 0

        Note:
            - Position 0 in output corresponds to ROOT
            - Positions 1..T correspond to word indices
            - Invalid positions are masked with -inf
        """
        B, T, D = dep.shape

        # Prepend ROOT to head candidates
        root = self.root.view(1, 1, D).expand(B, 1, D)
        heads_all = torch.cat([root, head], dim=1)  # [B, T+1, D]

        # Biaffine attention: dep^T W head
        bil = (dep @ self.W) @ heads_all.transpose(1, 2)  # [B, T, T+1]

        # Add head bias
        u = self.U(heads_all).squeeze(-1).unsqueeze(1).expand(B, T, T + 1)
        logits = bil + u

        # Mask invalid positions
        # Create candidate mask (ROOT always valid + head mask)
        C = torch.ones(B, T + 1, dtype=torch.bool, device=dep.device)
        C[:, 1:] = mask_head
        logits = logits.masked_fill(~C.unsqueeze(1), float("-inf"))
        logits = logits.masked_fill((~mask_dep).unsqueeze(-1), float("-inf"))

        return logits


class BiaffineLabeler(nn.Module):
    """Biaffine classifier for dependency relation labels.

    Given predicted head attachments, classifies the dependency relation type.
    Uses smaller projected dimensions for efficiency and a biaffine scorer
    for each label class.

    Args:
        d: Model hidden dimension
        n_labels: Number of dependency relation labels

    Attributes:
        dep_proj: Projection for dependent representations
        head_proj: Projection for head representations
        W: Biaffine weight tensor [n_labels, d_label, d_label]
        bias: Bias term for each label

    Example:
        >>> labeler = BiaffineLabeler(d=768, n_labels=50)
        >>> dep = torch.randn(2, 10, 768)
        >>> head = torch.randn(2, 11, 768)  # Includes ROOT at position 0
        >>> head_indices = torch.randint(0, 11, (2, 10))  # Predicted heads
        >>> mask = torch.ones(2, 10, dtype=torch.bool)
        >>> logits = labeler(dep, head, head_indices, mask)
        >>> print(logits.shape)  # [2, 10, 50] - label scores for each word
    """

    def __init__(self, d: int, n_labels: int):
        super().__init__()
        # Project to smaller dimensions for efficiency
        d_label = d // 2
        self.dep_proj = nn.Linear(d, d_label)
        self.head_proj = nn.Linear(d, d_label)

        # Biaffine scoring for each label
        self.W = nn.Parameter(torch.empty(n_labels, d_label, d_label))
        self.bias = nn.Parameter(torch.zeros(n_labels))
        nn.init.xavier_uniform_(self.W)

    def forward(
        self, dep: torch.Tensor, head: torch.Tensor, head_indices: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute label classification logits for predicted heads.

        Args:
            dep: Dependent representations [B, T, D]
            head: Head representations [B, T+1, D] (includes ROOT at position 0)
            head_indices: Predicted head indices [B, T] (values in 0..T)
            mask: Valid token mask [B, T]

        Returns:
            Label logits [B, T, n_labels] for each token's predicted head

        Note:
            - Projects to smaller dimensions (d//2) for efficiency
            - Uses biaffine scoring: dep^T W_label head for each label
            - Only computes scores for the predicted head (not all candidates)
        """
        B, T, D = dep.shape

        # Project to label space
        dep_label = self.dep_proj(dep)  # [B, T, d_label]
        head_label = self.head_proj(head)  # [B, T+1, d_label]

        # Gather head representations based on predicted indices
        head_indices_expanded = head_indices.unsqueeze(-1).expand(-1, -1, head_label.size(-1))
        selected_heads = torch.gather(head_label, 1, head_indices_expanded)  # [B, T, d_label]

        # Biaffine scoring: dep^T W head for each label
        dep_expanded = dep_label.unsqueeze(2)  # [B, T, 1, d_label]
        head_expanded = selected_heads.unsqueeze(-1)  # [B, T, d_label, 1]

        # Compute biaffine scores for all labels
        logits = torch.einsum("btid,nde,btef->btn", dep_expanded, self.W, head_expanded).squeeze(-1)
        logits = logits + self.bias.view(1, 1, -1)  # [B, T, n_labels]

        return logits
