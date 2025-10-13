"""
Improved Pointer Network Decoder for Sorting with All Enhancements

Fixes implemented:
1. Mask-aware ranking loss (RankNet pairwise)
2. Deep supervision (average loss over inner iterations)
3. Temperature scheduling for routing
4. Entropy regularization with decay
5. Two-optimizer setup (encoder vs controller)
6. Differential gradient clipping
7. Controller warm-up
8. Per-iteration diagnostics

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List


class ImprovedPoHBlock(nn.Module):
    """Enhanced PoH block with temperature control, entropy tracking, and deep supervision support."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_iters: int,
        temperature_init: float = 2.0,
        temperature_min: float = 0.8,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_iters = max_iters
        
        # Temperature control
        self.register_buffer('temperature', torch.tensor(temperature_init))
        self.temperature_min = temperature_min

        # Controller for routing
        self.controller = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_heads)
        )

        # Per-head attention projections
        self.q_proj = nn.ModuleList(
            [nn.Linear(d_model, d_model // n_heads) for _ in range(n_heads)]
        )
        self.k_proj = nn.ModuleList(
            [nn.Linear(d_model, d_model // n_heads) for _ in range(n_heads)]
        )
        self.v_proj = nn.ModuleList(
            [nn.Linear(d_model, d_model // n_heads) for _ in range(n_heads)]
        )
        self.out_proj = nn.Linear(d_model, d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def set_temperature(self, temp: float):
        """Update routing temperature (for annealing)."""
        self.temperature.fill_(max(temp, self.temperature_min))
    
    def forward(
        self,
        z: torch.Tensor,
        return_diagnostics: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Iteratively refine latent z with deep supervision support.
        
        Returns:
            z_refined: Final refined latent [B, T, D]
            diagnostics: Optional dict with per-iteration stats
        """
        B, T, D = z.shape
        
        # Storage for diagnostics
        if return_diagnostics:
            all_alphas = []
            all_entropies = []
            all_z = []
        
        for iter_idx in range(self.max_iters):
            # Controller routing
            logits = self.controller(z)  # [B, T, n_heads]
            alphas = F.softmax(logits / self.temperature, dim=-1)  # [B, T, n_heads]
            
            if return_diagnostics:
                all_alphas.append(alphas.detach())
                # Entropy per position
                entropy = -(alphas * (alphas + 1e-10).log()).sum(dim=-1)  # [B, T]
                all_entropies.append(entropy.mean().item())

            # Per-head attention
            head_outs = []
            for h_idx in range(self.n_heads):
                q = self.q_proj[h_idx](z)  # [B, T, d_model//n_heads]
                k = self.k_proj[h_idx](z)
                v = self.v_proj[h_idx](z)

                scores = torch.einsum("btd,bsd->bts", q, k) / (
                    (self.d_model // self.n_heads) ** 0.5
                )
                attn = F.softmax(scores, dim=-1)
                out = torch.einsum("bts,bsd->btd", attn, v)
                head_outs.append(out)

            # Stack and concat all heads
            head_outs = torch.cat(head_outs, dim=-1)  # [B, T, d_model]
            
            # Weight by routing alphas
            head_outs = head_outs.view(B, T, self.n_heads, -1)  # [B, T, H, d//H]
            weighted = (alphas.unsqueeze(-1) * head_outs).sum(dim=2)  # [B, T, d//H]
            
            # But we need full d_model, so we actually need to re-concat
            # Better: apply alphas as scalars on each head output before concat
            head_outs_list = [head_outs[:, :, h, :] for h in range(self.n_heads)]
            weighted_heads = [alphas[:, :, h:h+1] * head_outs_list[h] for h in range(self.n_heads)]
            weighted_concat = torch.cat(weighted_heads, dim=-1)  # [B, T, D]
            
            attn_out = self.out_proj(weighted_concat)

            # Residual + norm
            z = self.ln1(z + attn_out)
            z_pre_ffn = z  # Store for diagnostics
            
            if return_diagnostics:
                all_z.append(z_pre_ffn)
            
            # FFN
            z = self.ln2(z + self.ffn(z))

        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                'alphas': torch.stack(all_alphas, dim=1),  # [B, iters, T, H]
                'entropies': all_entropies,  # List of scalars
                'z_per_iter': torch.stack(all_z, dim=1),  # [B, iters, T, D]
            }
        
        return z, diagnostics


class ImprovedPointerDecoderSort(nn.Module):
    """
    Enhanced pointer decoder with:
    - Mask-aware ranking loss
    - Deep supervision
    - Temperature scheduling
    - Diagnostic outputs
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        max_inner_iters: int = 3,
        use_poh: bool = False,
        temperature_init: float = 2.0,
        temperature_min: float = 0.8,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_poh = use_poh

        # Encoder: embed input values + positions
        self.value_embed = nn.Sequential(
            nn.Linear(1, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, d_model)
        )
        self.pos_embed = nn.Embedding(100, d_model)

        # Encoder transformer
        self.encoder_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.encoder_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.encoder_ln1 = nn.LayerNorm(d_model)
        self.encoder_ln2 = nn.LayerNorm(d_model)

        # Decoder: rank embeddings
        self.rank_embed = nn.Embedding(100, d_model)

        # PoH block for iterative refinement (if enabled)
        if use_poh:
            self.poh_block = ImprovedPoHBlock(
                d_model, n_heads, d_ff, max_inner_iters, temperature_init, temperature_min
            )

        # Pointer mechanism
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
    
    def set_temperature(self, temp: float):
        """Set routing temperature (for annealing schedule)."""
        if self.use_poh:
            self.poh_block.set_temperature(temp)

    def encode(self, x):
        """Encode input array values."""
        B, N, _ = x.shape

        # Embed values + positions
        positions = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.value_embed(x.squeeze(-1).unsqueeze(-1)) + self.pos_embed(positions)

        # Encoder transformer
        attn_out, _ = self.encoder_attn(h, h, h)
        h = self.encoder_ln1(h + attn_out)
        h = self.encoder_ln2(h + self.encoder_ffn(h))

        return h

    def decode_step(self, rank_t, z, mask):
        """
        Single decoder step: rank t chooses an input position.
        
        Args:
            rank_t: Scalar rank index
            z: [B, T, d_model] current latent (possibly refined by PoH)
            mask: [B, T] coverage mask
        """
        B, T, _ = z.shape

        # Query for this rank
        rank_idx = torch.full((B,), rank_t, device=z.device, dtype=torch.long)
        q_t = self.rank_embed(rank_idx)  # [B, d_model]

        # Pointer attention over refined latent
        q = self.query_proj(q_t)  # [B, d_model]
        k = self.key_proj(z)  # [B, T, d_model]
        logits = torch.einsum("bd,btd->bt", q, k)  # [B, T]

        # Apply coverage mask
        logits = logits.masked_fill(mask == 0, float("-inf"))

        return logits

    def forward(
        self,
        x: torch.Tensor,  # [B, N, 1]
        targets: Optional[torch.Tensor] = None,  # [B, N]
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        """
        Full pointer decoder with optional diagnostics.
        
        Returns:
            all_logits: [B, N, N] pointer logits for each decode step
            loss: Scalar loss (None if targets=None)
            diagnostics: Optional dict with routing stats, per-iter outputs
        """
        B, N, _ = x.shape

        # Encode
        z = self.encode(x)  # [B, N, d_model]

        # Optionally refine with PoH
        diagnostics = None
        if self.use_poh:
            z, diagnostics = self.poh_block(z, return_diagnostics=return_diagnostics)

        # Coverage mask: all positions available initially
        mask = torch.ones(B, N, device=x.device, dtype=torch.float32)

        # Decode: loop over output ranks
        all_logits = []
        for t in range(N):
            logits_t = self.decode_step(t, z, mask)  # [B, N]
            all_logits.append(logits_t)

            if targets is not None:
                # Teacher forcing
                chosen_idx = targets[:, t]
                mask[torch.arange(B, device=x.device), chosen_idx] = 0.0
            else:
                # Greedy inference
                chosen_idx = logits_t.argmax(dim=-1)
                mask[torch.arange(B, device=x.device), chosen_idx] = 0.0

        all_logits = torch.stack(all_logits, dim=1)  # [B, N, N]

        # Compute loss (will be done externally with ranking loss)
        loss = None

        return all_logits, loss, diagnostics
    
    def get_encoder_parameters(self):
        """Return encoder parameters (for separate optimizer)."""
        params = []
        params.extend(self.value_embed.parameters())
        params.extend(self.pos_embed.parameters())
        params.extend(self.encoder_attn.parameters())
        params.extend(self.encoder_ffn.parameters())
        params.extend(self.encoder_ln1.parameters())
        params.extend(self.encoder_ln2.parameters())
        params.extend(self.rank_embed.parameters())
        params.extend(self.query_proj.parameters())
        params.extend(self.key_proj.parameters())
        return params
    
    def get_controller_parameters(self):
        """Return controller parameters (for separate optimizer)."""
        if self.use_poh:
            return list(self.poh_block.controller.parameters())
        return []

