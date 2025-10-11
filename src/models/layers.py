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
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    logits: torch.Tensor,
    tau: float = 1.0,
    topk: int = 1,
    hard: bool = True
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
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        bias: bool = True,
        attn_dropout: float = 0.0
    ):
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
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
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
        hidden_factor: float = 0.5
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
        self.mlp_static = nn.Sequential(
            nn.Linear(d_model, hid),
            nn.GELU(),
            nn.Linear(hid, n_heads)
        )
        
        # Recurrent routing path (conditional on provisional heads)
        if recurrent:
            self.mlp_recur = nn.Sequential(
                nn.Linear(d_model + n_heads, hid),
                nn.GELU(),
                nn.Linear(hid, n_heads)
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
        self,
        token_ctx: torch.Tensor,
        provisional_heads: Optional[torch.Tensor] = None
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
            assert provisional_heads is not None, \
                "recurrent=True requires provisional_heads"
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
        self,
        dep: torch.Tensor,
        head: torch.Tensor,
        mask_dep: torch.Tensor,
        mask_head: torch.Tensor
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
        u = self.U(heads_all).squeeze(-1).unsqueeze(1).expand(B, T, T+1)
        logits = bil + u
        
        # Mask invalid positions
        # Create candidate mask (ROOT always valid + head mask)
        C = torch.ones(B, T+1, dtype=torch.bool, device=dep.device)
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
        self,
        dep: torch.Tensor,
        head: torch.Tensor,
        head_indices: torch.Tensor,
        mask: torch.Tensor
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
        logits = torch.einsum('btid,nde,btef->btn', dep_expanded, self.W, head_expanded).squeeze(-1)
        logits = logits + self.bias.view(1, 1, -1)  # [B, T, n_labels]
        
        return logits

