"""
Pointer-over-Heads dependency parser with adaptive routing.

Implements the PoH parser with support for multiple training modes:
- Standard fixed iterations
- Entropy-based early stopping
- ACT-style learned halting
- Deep supervision for iterative refinement
- TRM-style outer supervision steps

Classes:
    PoHParser: Main PoH parser with dynamic head routing

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import ParserBase
from src.models.pointer_block import PointerMoHTransformerBlock
from src.models.layers import BiaffinePointer, BiaffineLabeler
from src.utils.helpers import mean_pool_subwords, pad_words, make_targets


class PoHParser(ParserBase):
    """Pointer-over-Heads dependency parser with dynamic head routing.

    Extends the baseline parser with adaptive multi-head attention routing.
    The controller learns to dynamically select which attention heads to use
    based on the input, enabling more flexible and efficient processing.

    Architecture:
        1. Pretrained encoder (e.g., DistilBERT)
        2. Subword-to-word pooling
        3. PoH transformer block with adaptive routing
        4. Biaffine head prediction
        5. Optional biaffine label classification

    Training Modes:
        - Standard: Fixed iterations with final-state loss
        - Deep Supervision: Loss at each iteration with linear ramp
        - ACT Halting: Adaptive Computation Time with ponder cost
        - Combined ACT + Deep Supervision: Recommended for best results
        - TRM Mode: Outer supervision steps with inner refinement loops

    Args:
        enc_name: HuggingFace model name
        d_model: Model hidden dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward network dimension
        halting_mode: Stopping criterion ('fixed', 'entropy', 'halting')
        max_inner_iters: Maximum inner iterations
        routing_topk: Number of heads to select (0 = soft routing)
        combination: Head combination mode ('mask_concat' or 'mixture')
        ent_threshold: Entropy threshold for early stopping
        n_labels: Number of dependency labels
        use_labels: Whether to predict labels (LAS) or just heads (UAS)
        deep_supervision: Enable deep supervision mode
        act_halting: Enable ACT-style differentiable halting
        ponder_coef: Coefficient for ACT ponder cost
        ramp_strength: Strength of deep supervision ramp (0=flat, 1=full)
        grad_mode: Gradient mode ('full' BPTT or 'last' iterate)

    Attributes:
        block: PoH transformer block with adaptive routing
        pointer: Biaffine pointer for head prediction
        labeler: Optional biaffine labeler for relation classification
        use_labels: Whether label prediction is enabled
        n_labels: Number of dependency relation labels
        deep_supervision: Deep supervision enabled flag
        act_halting: ACT halting enabled flag

    Example:
        >>> # Basic PoH parser with entropy-based halting
        >>> parser = PoHParser(
        ...     enc_name="distilbert-base-uncased",
        ...     d_model=768,
        ...     n_heads=8,
        ...     d_ff=2048,
        ...     halting_mode="entropy",
        ...     max_inner_iters=3,
        ...     routing_topk=0,  # Soft routing
        ...     n_labels=50,
        ...     use_labels=True
        ... )

        >>> # With deep supervision + ACT halting (recommended)
        >>> parser = PoHParser(
        ...     enc_name="distilbert-base-uncased",
        ...     d_model=768,
        ...     halting_mode="halting",
        ...     deep_supervision=True,
        ...     act_halting=True,
        ...     ponder_coef=1e-3,
        ...     ramp_strength=1.0,
        ...     grad_mode="full"
        ... )

        >>> # Training
        >>> loss, metrics = parser(subw_tokens, word_ids, heads_gold, labels_gold)
        >>> print(f"UAS: {metrics['uas']:.3f}, Iterations: {metrics['inner_iters_used']}")
    """

    def __init__(
        self,
        enc_name: str = "distilbert-base-uncased",
        d_model: int = 768,
        n_heads: int = 8,
        d_ff: int = 2048,
        halting_mode: str = "entropy",
        max_inner_iters: int = 3,
        routing_topk: int = 2,
        combination: str = "mask_concat",
        ent_threshold: float = 0.8,
        n_labels: int = 50,
        use_labels: bool = True,
        deep_supervision: bool = False,
        act_halting: bool = False,
        ponder_coef: float = 1e-3,
        ramp_strength: float = 1.0,
        grad_mode: str = "full",
    ):
        super().__init__(enc_name, d_model)

        # PoH transformer block with adaptive routing
        self.block = PointerMoHTransformerBlock(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            halting_mode=halting_mode,
            max_inner_iters=max_inner_iters,
            min_inner_iters=1,
            ent_threshold=ent_threshold,
            routing_topk=routing_topk,
            combination=combination,
            controller_recurrent=True,
            controller_summary="mean",
            grad_mode=grad_mode,
        )

        # Head prediction
        self.pointer = BiaffinePointer(d_model)

        # Optional label prediction
        self.use_labels = use_labels
        self.n_labels = n_labels
        if use_labels:
            self.labeler = BiaffineLabeler(d_model, n_labels)

        # Iterative refinement modes
        self.deep_supervision = deep_supervision
        self.act_halting = act_halting
        self.ponder_coef = ponder_coef
        self.ramp_strength = ramp_strength

    def forward(
        self,
        subw: Dict[str, torch.Tensor],
        word_ids: List[List[Optional[int]]],
        heads_gold: List[List[int]],
        labels_gold: Optional[List[List[int]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Forward pass for dependency parsing with adaptive routing.

        Supports multiple training modes for iterative refinement:
        - Standard: Uses final iteration output only
        - Deep Supervision: Applies loss at each iteration with linear ramp
        - ACT Halting: Learns when to stop with ponder cost
        - Combined: ACT + deep supervision (recommended)

        Args:
            subw: Subword tokens (dict with 'input_ids', 'attention_mask', etc.)
            word_ids: Word indices for each subword [B, S]
            heads_gold: Gold dependency heads [B, T] (0 = ROOT)
            labels_gold: Optional gold dependency labels [B, T]

        Returns:
            Tuple containing:
            - loss: Combined head + label loss (scalar tensor)
            - metrics: Dictionary with:
                - uas: Unlabeled Attachment Score
                - las: Labeled Attachment Score (if use_labels=True)
                - tokens: Number of tokens
                - inner_iters_used: Number of iterations executed
                - (if act_halting) ponder_cost: ACT ponder cost
                - (eval mode) pred_heads: Predicted heads
                - (eval mode) pred_labels: Predicted labels

        Note:
            The collect_all mode is automatically enabled during training
            when deep_supervision or act_halting is True. This ensures
            full gradient flow through all iterations.
        """
        device = next(self.parameters()).device

        # Encode and pool to word level
        last_hidden = self.encoder(**subw).last_hidden_state
        words = mean_pool_subwords(last_hidden, word_ids)
        X, mask = pad_words(words)

        # Use collect_all mode if deep_supervision or act_halting enabled
        # This collects all intermediate states for loss computation
        collect_all = self.training and (self.deep_supervision or self.act_halting)
        X, aux = self.block(X, attn_mask=None, return_aux=True, collect_all=collect_all)

        Y_heads, pad = make_targets(heads_gold, X.size(1), device)

        # === Iterative refinement loss computation ===
        if collect_all and "routed" in aux:
            # Deep supervision or ACT halting mode
            routed_seq = aux["routed"]  # [B, iters, T, D]

            # Pointer function for loss computation
            def pointer_fn(x, _, m, __):
                return self.pointer(x, x, m, m)

            if self.act_halting and self.deep_supervision and "halt_logits" in aux:
                # COMBINED: ACT + deep supervision (recommended)
                from src.utils.iterative_losses import act_deep_supervision_loss

                halt_logits_seq = aux["halt_logits"]

                head_loss, diagnostics = act_deep_supervision_loss(
                    routed_seq,
                    halt_logits_seq,
                    pointer_fn,
                    Y_heads,
                    pad,
                    ponder_coef=self.ponder_coef,
                    ramp_strength=self.ramp_strength,
                )
                # Store diagnostics for logging
                for key, value in diagnostics.items():
                    aux[key] = value

            elif self.act_halting and "halt_logits" in aux:
                # ACT-style expected loss only (no deep supervision ramp)
                from src.utils.iterative_losses import act_expected_loss

                halt_logits_seq = aux["halt_logits"]

                head_loss, ponder_cost = act_expected_loss(
                    routed_seq,
                    halt_logits_seq,
                    pointer_fn,
                    Y_heads,
                    pad,
                    ponder_coef=self.ponder_coef,
                    per_token=False,  # Per-sequence halting (simpler)
                )
                # Store ponder cost for logging
                aux["ponder_cost"] = ponder_cost.item()

            elif self.deep_supervision:
                # Deep supervision only (no ACT)
                from src.utils.iterative_losses import deep_supervision_loss

                head_loss = deep_supervision_loss(
                    routed_seq,
                    pointer_fn,
                    Y_heads,
                    pad,
                    weight_schedule="linear",  # Ramp from 0.3 to 1.0
                )
            else:
                # Fallback: just use final state (shouldn't happen)
                head_logits = self.pointer(X, X, mask, mask)
                head_loss = F.cross_entropy(
                    head_logits.view(-1, head_logits.size(-1))[pad.view(-1)],
                    Y_heads.view(-1)[pad.view(-1)],
                )

            # For metrics, always use final iteration
            head_logits = self.pointer(X, X, mask, mask)
        else:
            # Standard single-pass mode
            head_logits = self.pointer(X, X, mask, mask)
            head_loss = F.cross_entropy(
                head_logits.view(-1, head_logits.size(-1))[pad.view(-1)],
                Y_heads.view(-1)[pad.view(-1)],
            )

        # Compute UAS (always on final output)
        with torch.no_grad():
            pred_heads = head_logits.argmax(-1)
            uas = (pred_heads[pad] == Y_heads[pad]).float().mean().item()

        # Label prediction (only on final output for simplicity)
        total_loss = head_loss
        las = uas  # Default to UAS if no labels
        pred_labels = None

        if self.use_labels and labels_gold is not None:
            # Use gold heads for training, predicted heads for evaluation
            head_indices = Y_heads if self.training else pred_heads

            # Prepend ROOT representation
            root_repr = self.pointer.root.view(1, 1, -1).expand(X.size(0), 1, X.size(-1))
            X_with_root = torch.cat([root_repr, X], dim=1)

            # Compute label logits
            label_logits = self.labeler(X, X_with_root, head_indices, mask)
            Y_labels, label_pad = make_targets(labels_gold, X.size(1), device)
            label_loss = F.cross_entropy(
                label_logits.view(-1, label_logits.size(-1))[pad.view(-1)],
                Y_labels.view(-1)[pad.view(-1)],
            )
            total_loss = head_loss + label_loss

            # Compute LAS
            with torch.no_grad():
                pred_labels = label_logits.argmax(-1)
                las = (
                    ((pred_heads[pad] == Y_heads[pad]) & (pred_labels[pad] == Y_labels[pad]))
                    .float()
                    .mean()
                    .item()
                )

        # Prepare metrics
        metrics = {"uas": uas, "las": las, "tokens": pad.sum().item()}

        # Add auxiliary metrics
        if "inner_iters_used" in aux:
            metrics["inner_iters_used"] = aux["inner_iters_used"].item()
        if "ponder_cost" in aux:
            metrics["ponder_cost"] = aux["ponder_cost"]

        # Add predictions for CoNLL-U export (evaluation only)
        if not self.training:
            metrics["pred_heads"] = [
                pred_heads[b, : pad[b].sum()].cpu().tolist() for b in range(pred_heads.size(0))
            ]
            if pred_labels is not None:
                metrics["pred_labels"] = [
                    pred_labels[b, : pad[b].sum()].cpu().tolist()
                    for b in range(pred_labels.size(0))
                ]

        return total_loss, metrics

    def forward_trm(
        self,
        subw: Dict[str, torch.Tensor],
        word_ids: List[List[Optional[int]]],
        heads_gold: List[List[int]],
        labels_gold: Optional[List[List[int]]] = None,
        args=None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """TRM-style forward pass with outer supervision steps.

        Implements the Tiny Recursive Model approach: performs multiple
        outer supervision steps, each consisting of n inner refinement
        updates followed by a pointer refresh. Deep supervision is applied
        across these outer steps with an optional ramp.

        Based on "Less is More: Recursive Reasoning with Tiny Networks"
        (arXiv:2510.04871)

        Args:
            subw: Subword tokens
            word_ids: Word indices for each subword
            heads_gold: Gold dependency heads
            labels_gold: Optional gold dependency labels
            args: Arguments object with TRM configuration:
                - trm_supervision_steps: Number of outer supervision steps
                - trm_inner_updates: Inner updates per step
                - trm_ramp_strength: Deep supervision ramp strength

        Returns:
            Tuple containing:
            - loss: Combined head + label loss
            - metrics: Dictionary with standard metrics plus:
                - trm_supervision_steps: Number of outer steps performed

        Note:
            If args.trm_inner_updates is None, uses block.max_inner_iters.
            Can be combined with HRM-style last-iterate gradients by setting
            block.grad_mode = "last".
        """
        device = next(self.parameters()).device

        # Encode and pool to word level
        last_hidden = self.encoder(**subw).last_hidden_state
        words = mean_pool_subwords(last_hidden, word_ids)
        X, mask = pad_words(words)
        Y_heads, pad = make_targets(heads_gold, X.size(1), device)

        # Decide inner updates per TRM step
        inner_updates = (
            args.trm_inner_updates
            if args.trm_inner_updates is not None
            else self.block.max_inner_iters
        )

        # Collect latent states z_t from each supervision step
        Zs = []
        z = X

        for s in range(args.trm_supervision_steps):
            # Run one TRM step: n inner updates
            old_iters = self.block.max_inner_iters
            self.block.max_inner_iters = inner_updates
            y, aux = self.block(
                z, attn_mask=None, return_aux=True, collect_all=False, return_final_z=True
            )
            self.block.max_inner_iters = old_iters

            z_next = aux["z_final"]  # [B, T, D]
            Zs.append(z_next)

            # HRM-style across steps if grad_mode==last
            if (
                hasattr(self.block, "grad_mode")
                and self.block.grad_mode == "last"
                and s < args.trm_supervision_steps - 1
            ):
                z_next = z_next.detach()
            z = z_next

        # Deep supervision over refreshes
        from src.utils.trm_losses import trm_supervised_loss

        head_loss, head_logits = trm_supervised_loss(
            self.pointer, Zs, Y_heads, pad, ramp_strength=args.trm_ramp_strength
        )

        # Compute UAS (from final refresh)
        with torch.no_grad():
            pred_heads = head_logits.argmax(-1)
            uas = (pred_heads[pad] == Y_heads[pad]).float().mean().item()

        # Label prediction (only on final z for simplicity)
        total_loss = head_loss
        las = uas
        pred_labels = None

        if self.use_labels and labels_gold is not None:
            z_final = Zs[-1]
            head_indices = Y_heads if self.training else pred_heads

            # Prepend ROOT representation
            root_repr = self.pointer.root.view(1, 1, -1).expand(
                z_final.size(0), 1, z_final.size(-1)
            )
            X_with_root = torch.cat([root_repr, z_final], dim=1)

            # Compute label logits
            label_logits = self.labeler(z_final, X_with_root, head_indices, pad)
            Y_labels, label_pad = make_targets(labels_gold, z_final.size(1), device)
            label_loss = F.cross_entropy(
                label_logits.view(-1, label_logits.size(-1))[pad.view(-1)],
                Y_labels.view(-1)[pad.view(-1)],
            )
            total_loss = head_loss + label_loss

            # Compute LAS
            with torch.no_grad():
                pred_labels = label_logits.argmax(-1)
                las = (
                    ((pred_heads[pad] == Y_heads[pad]) & (pred_labels[pad] == Y_labels[pad]))
                    .float()
                    .mean()
                    .item()
                )

        # Prepare metrics
        metrics = {
            "uas": uas,
            "las": las,
            "tokens": pad.sum().item(),
            "inner_iters_used": float(inner_updates * args.trm_supervision_steps),
            "trm_supervision_steps": args.trm_supervision_steps,
        }

        # Add predictions for CoNLL-U export (evaluation only)
        if not self.training:
            metrics["pred_heads"] = [
                pred_heads[b, : pad[b].sum()].cpu().tolist() for b in range(pred_heads.size(0))
            ]
            if pred_labels is not None:
                metrics["pred_labels"] = [
                    pred_labels[b, : pad[b].sum()].cpu().tolist()
                    for b in range(pred_labels.size(0))
                ]

        return total_loss, metrics
