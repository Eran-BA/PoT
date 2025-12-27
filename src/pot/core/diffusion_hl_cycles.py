"""
Diffusion-Based H,L Cycles

A novel architecture that replaces deterministic HRM H,L cycles with
diffusion-based denoising processes operating at dual timescales.

Key Innovations:
1. Dual-Timescale Diffusion: z_H and z_L are progressively denoised at different rates
   - L-diffusion: Fast denoising (updates every step)
   - H-diffusion: Slow denoising (updates every T steps on average)

2. Learned Timing: Instead of fixed T-step H updates, a diffusion-based gate
   learns when to trigger H-level updates based on input complexity

3. Coupled Denoising: L denoises conditioned on H; H denoises conditioned on L
   (after L update), creating hierarchical information flow

4. adaLN Conditioning: Both denoisers use adaptive LayerNorm conditioned on
   timestep + input features (DiT-style)

The architecture can be viewed as:
- A hierarchical diffusion model where coarse structure (H) and fine details (L)
  are refined at different rates
- An extension of HRM where the two-timescale reasoning becomes iterative denoising
- A unification of diffusion models and recurrent reasoning

Reference:
- HRM: "Hierarchical Reasoning Model" (two-timescale processing)
- DiT: "Scalable Diffusion Models with Transformers" (adaLN conditioning)
- DDPM: "Denoising Diffusion Probabilistic Models" (noise schedules)

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# State Classes
# =============================================================================

@dataclass
class DiffusionHLState:
    """
    State for Diffusion H,L cycles.
    
    Maintains the progressively denoised H and L latent representations,
    along with step counters and timing state.
    
    Attributes:
        z_H: [B, S, d_model] H-level latent (slow timescale, strategic)
        z_L: [B, S, d_model] L-level latent (fast timescale, tactical)
        h_step: H-diffusion timestep counter
        l_step: L-diffusion timestep counter
        level_logits: [B, 2] logits for H/L timing diffusion
        cumulative_gate: [B] accumulated H-update probability
    """
    z_H: torch.Tensor
    z_L: torch.Tensor
    h_step: int
    l_step: int
    level_logits: Optional[torch.Tensor] = None
    cumulative_gate: Optional[torch.Tensor] = None
    
    def detach(self) -> 'DiffusionHLState':
        """Detach all tensors from computation graph."""
        return DiffusionHLState(
            z_H=self.z_H.detach(),
            z_L=self.z_L.detach(),
            h_step=self.h_step,
            l_step=self.l_step,
            level_logits=self.level_logits.detach() if self.level_logits is not None else None,
            cumulative_gate=self.cumulative_gate.detach() if self.cumulative_gate is not None else None,
        )


# =============================================================================
# Noise Schedules
# =============================================================================

def get_noise_schedule(
    schedule_type: str,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate noise schedule for diffusion process.
    
    Args:
        schedule_type: "linear", "cosine", or "sqrt"
        num_steps: Maximum number of diffusion steps
        device: Target device
        
    Returns:
        [num_steps] tensor of noise levels (sigma values, decreasing from 1 to ~0)
    """
    t = torch.linspace(0, 1, num_steps, device=device)
    
    if schedule_type == "linear":
        # Linear decrease from 1 to 0
        sigma = 1.0 - t
    elif schedule_type == "cosine":
        # Cosine schedule (smoother, better for generation)
        sigma = torch.cos(t * math.pi / 2)
    elif schedule_type == "sqrt":
        # Square root schedule (faster denoising at start)
        sigma = torch.sqrt(1.0 - t)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    return sigma.clamp(min=1e-4)


class DualTimescaleNoiseSchedule(nn.Module):
    """
    Dual noise schedules for H (slow) and L (fast) diffusion processes.
    
    The L-level uses a fine-grained schedule with max_steps timesteps,
    while the H-level uses a coarser schedule with max_steps // T timesteps.
    
    This creates the two-timescale effect: L denoises quickly with small
    incremental changes, while H denoises slowly with larger jumps.
    
    Args:
        max_steps: Maximum L-diffusion steps
        T: Timescale ratio (H updates ~T times slower than L)
        schedule_type: Type of noise schedule ("linear", "cosine", "sqrt")
    """
    
    def __init__(
        self,
        max_steps: int = 32,
        T: int = 4,
        schedule_type: Literal["linear", "cosine", "sqrt"] = "cosine",
    ):
        super().__init__()
        self.max_steps = max_steps
        self.T = T
        self.schedule_type = schedule_type
        
        # H-level has fewer steps (slower denoising)
        self.h_steps = max(max_steps // T, 1)
        
        # Register schedules as buffers (not parameters)
        # These will be created on first use to get the right device
        self.register_buffer(
            "sigma_L",
            get_noise_schedule(schedule_type, max_steps, torch.device("cpu")),
            persistent=False,
        )
        self.register_buffer(
            "sigma_H", 
            get_noise_schedule(schedule_type, self.h_steps, torch.device("cpu")),
            persistent=False,
        )
    
    def get_sigma_L(self, step: int) -> torch.Tensor:
        """Get L-level noise for given step."""
        step = min(step, self.max_steps - 1)
        return self.sigma_L[step]
    
    def get_sigma_H(self, step: int) -> torch.Tensor:
        """Get H-level noise for given step."""
        step = min(step, self.h_steps - 1)
        return self.sigma_H[step]
    
    def get_sigmas(self, l_step: int, h_step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get both noise levels for current steps."""
        return self.get_sigma_L(l_step), self.get_sigma_H(h_step)


# =============================================================================
# Timestep Embedding
# =============================================================================

class TimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding with MLP projection.
    
    Encodes the diffusion timestep as a learnable embedding that can
    condition the denoising network.
    
    Args:
        d_embed: Embedding dimension
        max_period: Maximum period for sinusoidal encoding
    """
    
    def __init__(self, d_embed: int, max_period: int = 10000):
        super().__init__()
        self.d_embed = d_embed
        self.max_period = max_period
        
        # MLP to process raw sinusoidal embedding
        self.mlp = nn.Sequential(
            nn.Linear(d_embed, d_embed * 4),
            nn.GELU(),
            nn.Linear(d_embed * 4, d_embed),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Embed timestep.
        
        Args:
            t: [B] or scalar timestep (can be float in [0, 1] or sigma value)
            
        Returns:
            [B, d_embed] timestep embedding
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        half_dim = self.d_embed // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=t.device) / half_dim
        )
        
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.d_embed % 2:
            embedding = F.pad(embedding, (0, 1))
        
        return self.mlp(embedding)


# =============================================================================
# Denoising Blocks
# =============================================================================

class AdaLNDenoiseBlock(nn.Module):
    """
    Denoising block with Adaptive Layer Normalization (adaLN).
    
    Based on DiT (Diffusion Transformers) architecture. The block conditions
    on both the current state and external conditioning (timestep + context).
    
    Architecture:
        z → LayerNorm → scale/shift by cond → MLP → residual
        
    The adaptive modulation allows the denoising behavior to change based
    on the current timestep (noise level) and input features.
    
    Args:
        d_model: Model dimension
        d_cond: Conditioning dimension (timestep + context)
        dropout: Dropout probability
        n_layers: Number of MLP layers
    """
    
    def __init__(
        self,
        d_model: int,
        d_cond: Optional[int] = None,
        dropout: float = 0.0,
        n_layers: int = 2,
    ):
        super().__init__()
        d_cond = d_cond or d_model
        
        # Layer norm (without learnable affine - we use adaLN instead)
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        
        # Main processing MLP
        layers = []
        for i in range(n_layers):
            in_dim = d_model if i == 0 else d_model * 4
            out_dim = d_model if i == n_layers - 1 else d_model * 4
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU() if i < n_layers - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < n_layers - 1 else nn.Identity(),
            ])
        self.mlp = nn.Sequential(*layers)
        
        # Adaptive LayerNorm modulation: conditioning → (scale, shift) pairs
        # We produce 4 values: scale1, shift1 (for input), scale2, shift2 (for output)
        self.adaLN_modulation = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_cond, d_model * 4),
        )
        
        # Zero-initialize the modulation output for stable training
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(
        self,
        z: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Denoise one step.
        
        Args:
            z: [B, S, d_model] or [B, d_model] noisy latent
            cond: [B, d_cond] conditioning vector
            
        Returns:
            [B, S, d_model] or [B, d_model] denoised latent (residual added)
        """
        # Handle both [B, S, D] and [B, D] inputs
        squeeze_output = z.dim() == 2
        if squeeze_output:
            z = z.unsqueeze(1)  # [B, 1, D]
        
        # Get adaptive LN parameters from conditioning
        B = z.shape[0]
        modulation = self.adaLN_modulation(cond)  # [B, d_model * 4]
        scale1, shift1, scale2, shift2 = modulation.chunk(4, dim=-1)  # Each [B, d_model]
        
        # Expand for sequence dimension
        scale1 = scale1.unsqueeze(1)  # [B, 1, d_model]
        shift1 = shift1.unsqueeze(1)
        scale2 = scale2.unsqueeze(1)
        shift2 = shift2.unsqueeze(1)
        
        # Normalize and apply first modulation
        h = self.norm(z)
        h = h * (1 + scale1) + shift1
        
        # MLP processing
        h = self.mlp(h)
        
        # Apply second modulation
        h = h * (1 + scale2) + shift2
        
        # Residual connection
        out = z + h
        
        if squeeze_output:
            out = out.squeeze(1)
        
        return out


class SequenceDenoiseBlock(nn.Module):
    """
    Sequence-aware denoising block with self-attention.
    
    For denoising sequence latents [B, S, D], we use self-attention
    to allow information flow across positions during denoising.
    
    Architecture:
        z → SelfAttn(z, conditioned by t) → FFN → residual
        
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_cond: Conditioning dimension
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_cond: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        d_cond = d_cond or d_model
        
        # Self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # FFN with SwiGLU-like gating
        self.ffn_gate = nn.Linear(d_model, d_model * 4, bias=False)
        self.ffn_up = nn.Linear(d_model, d_model * 4, bias=False)
        self.ffn_down = nn.Linear(d_model * 4, d_model, bias=False)
        self.ffn_drop = nn.Dropout(dropout)
        
        # Norms (without affine for adaLN)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        
        # AdaLN modulation for both attention and FFN outputs
        self.adaLN_modulation = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_cond, d_model * 6),  # scale/shift for norm1, norm2, and output
        )
        
        # Zero-init for stability
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(
        self,
        z: torch.Tensor,
        cond: torch.Tensor,
        cross_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Denoise with self-attention.
        
        Args:
            z: [B, S, d_model] noisy sequence latent
            cond: [B, d_cond] timestep conditioning
            cross_cond: [B, S, d_model] optional cross-attention conditioning
            
        Returns:
            [B, S, d_model] denoised sequence
        """
        # Get modulation parameters
        modulation = self.adaLN_modulation(cond)  # [B, d_model * 6]
        (scale1, shift1, scale2, shift2, 
         gate_scale, gate_shift) = modulation.chunk(6, dim=-1)
        
        # Expand for sequence dim
        scale1 = scale1.unsqueeze(1)
        shift1 = shift1.unsqueeze(1)
        scale2 = scale2.unsqueeze(1)
        shift2 = shift2.unsqueeze(1)
        gate_scale = gate_scale.unsqueeze(1)
        gate_shift = gate_shift.unsqueeze(1)
        
        # Self-attention branch
        h = self.norm1(z)
        h = h * (1 + scale1) + shift1
        
        # Use cross_cond as key/value if provided (for H conditioning L)
        if cross_cond is not None:
            attn_out, _ = self.attn(h, cross_cond, cross_cond, need_weights=False)
        else:
            attn_out, _ = self.attn(h, h, h, need_weights=False)
        
        z = z + attn_out
        
        # FFN branch with gating (SwiGLU-style)
        h = self.norm2(z)
        h = h * (1 + scale2) + shift2
        
        gate = F.silu(self.ffn_gate(h))
        value = self.ffn_up(h)
        h = gate * value
        h = self.ffn_drop(self.ffn_down(h))
        
        # Output with gating
        h = h * (1 + gate_scale) + gate_shift
        
        return z + h


# =============================================================================
# Level Timing Controller
# =============================================================================

class LevelTimingDiffuser(nn.Module):
    """
    Diffusion-based controller for H/L transition timing.
    
    Instead of fixed "update H every T steps", we learn when to update H
    via a discrete diffusion process over {skip_H, update_H}.
    
    The controller outputs a soft gate that determines how much to update
    the H-level at each step. During training, we use Gumbel-softmax for
    differentiability. During inference, we can use hard decisions.
    
    Architecture:
        [z_L, z_H] → MLP → logits → Gumbel-softmax → gate
        
    The gate can be viewed as a "denoised" version of the H-update decision,
    where the noise comes from uncertainty about the optimal timing.
    
    Args:
        d_model: Model dimension
        d_hidden: Hidden dimension for timing MLP
        temperature_init: Initial Gumbel-softmax temperature
        temperature_min: Minimum temperature (annealed during training)
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        d_model: int,
        d_hidden: Optional[int] = None,
        temperature_init: float = 1.0,
        temperature_min: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()
        d_hidden = d_hidden or d_model
        
        self.temperature_init = temperature_init
        self.temperature_min = temperature_min
        
        # Learnable log-temperature for annealing
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(temperature_init))
        )
        
        # Pool sequence to single vector
        self.pool_norm = nn.LayerNorm(d_model)
        
        # Combine z_L and z_H information
        self.combine = nn.Sequential(
            nn.Linear(d_model * 2, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Output logits for {skip_H, update_H}
        self.logit_head = nn.Linear(d_hidden, 2)
        
        # Initialize to slight preference for skipping (like fixed T=4 behavior)
        nn.init.zeros_(self.logit_head.weight)
        nn.init.constant_(self.logit_head.bias[0], 0.5)  # skip
        nn.init.constant_(self.logit_head.bias[1], -0.5)  # update
    
    @property
    def temperature(self) -> float:
        """Current Gumbel-softmax temperature."""
        return max(self.temperature_min, math.exp(self.log_temperature.item()))
    
    def set_temperature(self, T: float):
        """Set temperature directly."""
        T = max(self.temperature_min, T)
        self.log_temperature.data.fill_(math.log(T))
    
    def anneal_temperature(self, factor: float = 0.99):
        """Anneal temperature by multiplicative factor."""
        new_temp = max(self.temperature_min, self.temperature * factor)
        self.set_temperature(new_temp)
    
    def forward(
        self,
        z_L: torch.Tensor,
        z_H: torch.Tensor,
        hard: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute H-update gate.
        
        Args:
            z_L: [B, S, d_model] L-level latent
            z_H: [B, S, d_model] H-level latent
            hard: If True, use hard (argmax) decisions instead of soft
            
        Returns:
            gate: [B] soft gate in [0, 1] for H-update (0 = skip, 1 = update)
            logits: [B, 2] raw logits for {skip, update}
        """
        # Pool to single vector per sample
        z_L_pooled = self.pool_norm(z_L).mean(dim=1)  # [B, d_model]
        z_H_pooled = self.pool_norm(z_H).mean(dim=1)  # [B, d_model]
        
        # Combine and compute logits
        combined = torch.cat([z_L_pooled, z_H_pooled], dim=-1)  # [B, 2*d_model]
        hidden = self.combine(combined)  # [B, d_hidden]
        logits = self.logit_head(hidden)  # [B, 2]
        
        # Gumbel-softmax for differentiable sampling
        if self.training and not hard:
            probs = F.gumbel_softmax(logits, tau=self.temperature, hard=False)
            gate = probs[:, 1]  # P(update_H)
        else:
            # Hard decision during inference
            probs = F.softmax(logits, dim=-1)
            if hard:
                gate = (logits[:, 1] > logits[:, 0]).float()
            else:
                gate = probs[:, 1]
        
        return gate, logits


# =============================================================================
# Main Diffusion H,L Cycles Module
# =============================================================================

class DiffusionHLCycles(nn.Module):
    """
    Diffusion-based H,L cycles for two-timescale reasoning.
    
    This module replaces the deterministic HRM H,L cycles with diffusion-based
    denoising at dual timescales. Both z_H and z_L are treated as latent
    variables that are progressively denoised, with learned timing for when
    to update the H-level.
    
    Architecture:
        1. Initialize z_H, z_L with noise (or zeros during inference)
        2. For each step t:
           a. Denoise z_L conditioned on (z_H, x, sigma_L(t))
           b. Compute gate = P(update_H | z_L, z_H, t)
           c. If gate > threshold: Denoise z_H conditioned on (z_L, sigma_H(h_step))
           d. Update step counters
        3. Return final denoised z_H, z_L
    
    The coupling between H and L creates hierarchical information flow:
    - L learns fast, tactical responses conditioned on slow H guidance
    - H learns slow, strategic representations from aggregated L information
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads for sequence denoisers
        max_steps: Maximum L-diffusion steps
        T: Base timescale ratio (H is ~T times slower than L)
        noise_schedule: Type of noise schedule
        n_denoise_layers: Number of layers in denoise blocks
        dropout: Dropout probability
        use_sequence_denoiser: If True, use attention-based sequence denoiser
        learned_timing: If True, learn when to update H (vs fixed T)
        init_noise_scale: Scale for initial noise during training
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        max_steps: int = 32,
        T: int = 4,
        noise_schedule: Literal["linear", "cosine", "sqrt"] = "cosine",
        n_denoise_layers: int = 2,
        dropout: float = 0.0,
        use_sequence_denoiser: bool = True,
        learned_timing: bool = True,
        init_noise_scale: float = 1.0,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_steps = max_steps
        self.T = T
        self.init_noise_scale = init_noise_scale
        self.use_sequence_denoiser = use_sequence_denoiser
        self.learned_timing = learned_timing
        
        # Dual-timescale noise schedules
        self.noise_schedule = DualTimescaleNoiseSchedule(
            max_steps=max_steps,
            T=T,
            schedule_type=noise_schedule,
        )
        
        # Timestep embeddings (separate for H and L)
        self.time_embed_L = TimestepEmbedding(d_model)
        self.time_embed_H = TimestepEmbedding(d_model)
        
        # Input conditioning projection
        self.input_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        
        # L-level denoiser (fast, every step)
        if use_sequence_denoiser:
            self.L_denoiser = SequenceDenoiseBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_cond=d_model,
                dropout=dropout,
            )
        else:
            self.L_denoiser = AdaLNDenoiseBlock(
                d_model=d_model,
                d_cond=d_model,
                dropout=dropout,
                n_layers=n_denoise_layers,
            )
        
        # H-level denoiser (slow, conditional)
        if use_sequence_denoiser:
            self.H_denoiser = SequenceDenoiseBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_cond=d_model,
                dropout=dropout,
            )
        else:
            self.H_denoiser = AdaLNDenoiseBlock(
                d_model=d_model,
                d_cond=d_model,
                dropout=dropout,
                n_layers=n_denoise_layers,
            )
        
        # Conditioning fusion: combines input + timestep
        self.L_cond_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),  # input + t_embed + z_H_pooled
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.H_cond_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # t_embed + z_L_pooled
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # Level timing controller
        if learned_timing:
            self.level_timer = LevelTimingDiffuser(
                d_model=d_model,
                dropout=dropout,
            )
        else:
            self.level_timer = None
    
    def init_state(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> DiffusionHLState:
        """
        Initialize diffusion state.
        
        During training, we initialize with noise.
        During inference, we can start with zeros for determinism.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            device: Target device
            
        Returns:
            Initial DiffusionHLState
        """
        if self.training:
            # Training: start with noise
            z_H = torch.randn(batch_size, seq_len, self.d_model, device=device)
            z_H = z_H * self.init_noise_scale
            z_L = torch.randn(batch_size, seq_len, self.d_model, device=device)
            z_L = z_L * self.init_noise_scale
        else:
            # Inference: start with zeros for determinism
            z_H = torch.zeros(batch_size, seq_len, self.d_model, device=device)
            z_L = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        return DiffusionHLState(
            z_H=z_H,
            z_L=z_L,
            h_step=0,
            l_step=0,
            level_logits=None,
            cumulative_gate=torch.zeros(batch_size, device=device),
        )
    
    def _pool(self, z: torch.Tensor) -> torch.Tensor:
        """Pool sequence to single vector."""
        return z.mean(dim=1)  # [B, d_model]
    
    def step(
        self,
        x: torch.Tensor,
        state: DiffusionHLState,
        force_h_update: Optional[bool] = None,
    ) -> Tuple[DiffusionHLState, Dict[str, Any]]:
        """
        One diffusion step: denoise L, conditionally denoise H.
        
        Args:
            x: [B, S, d_model] input features (used as conditioning)
            state: Current DiffusionHLState
            force_h_update: If not None, override learned timing
            
        Returns:
            new_state: Updated state with denoised z_L, z_H
            aux: Dict with auxiliary information
        """
        B, S, D = x.shape
        device = x.device
        
        z_L = state.z_L
        z_H = state.z_H
        l_step = state.l_step
        h_step = state.h_step
        
        # Get noise levels
        sigma_L = self.noise_schedule.get_sigma_L(l_step)
        sigma_H = self.noise_schedule.get_sigma_H(h_step)
        
        # Timestep embeddings
        t_embed_L = self.time_embed_L(sigma_L.expand(B))  # [B, d_model]
        t_embed_H = self.time_embed_H(sigma_H.expand(B))  # [B, d_model]
        
        # Project input
        x_proj = self.input_proj(x)  # [B, S, d_model]
        x_pooled = self._pool(x_proj)  # [B, d_model]
        
        # ===== L-level denoising (always happens) =====
        # Conditioning: input + timestep + H-level context
        z_H_pooled = self._pool(z_H)  # [B, d_model]
        L_cond = self.L_cond_fusion(torch.cat([x_pooled, t_embed_L, z_H_pooled], dim=-1))
        
        # HRM-style input injection: add input to z_L before denoising
        # This keeps the original input signal flowing through at every step
        z_L_injected = z_L + x_proj
        
        # Denoise L (with input-injected state)
        if self.use_sequence_denoiser:
            z_L_new = self.L_denoiser(z_L_injected, L_cond, cross_cond=z_H)
        else:
            z_L_new = self.L_denoiser(z_L_injected, L_cond)
        
        # ===== H-level update decision =====
        if force_h_update is not None:
            update_H = force_h_update
            gate = torch.ones(B, device=device) if update_H else torch.zeros(B, device=device)
            level_logits = None
        elif self.learned_timing:
            gate, level_logits = self.level_timer(z_L_new, z_H)
            # Accumulate gate for soft scheduling
            cumulative = state.cumulative_gate + gate if state.cumulative_gate is not None else gate
            # Update H when cumulative gate exceeds 1 (roughly every T steps on average)
            update_H_mask = cumulative >= 1.0
            # Reset cumulative for samples that updated
            cumulative = cumulative - update_H_mask.float()
        else:
            # Fixed timing: update every T steps
            update_H = (l_step + 1) % self.T == 0
            gate = torch.ones(B, device=device) if update_H else torch.zeros(B, device=device)
            level_logits = None
            cumulative = state.cumulative_gate
            update_H_mask = gate > 0.5
        
        # ===== H-level denoising (conditional) =====
        if self.learned_timing:
            # Soft update: interpolate between old and new based on gate
            z_L_pooled = self._pool(z_L_new)
            H_cond = self.H_cond_fusion(torch.cat([t_embed_H, z_L_pooled], dim=-1))
            
            if self.use_sequence_denoiser:
                z_H_denoised = self.H_denoiser(z_H, H_cond, cross_cond=z_L_new)
            else:
                z_H_denoised = self.H_denoiser(z_H, H_cond)
            
            # Soft gate: interpolate
            gate_expanded = gate.view(B, 1, 1)
            z_H_new = gate_expanded * z_H_denoised + (1 - gate_expanded) * z_H
            
            # Increment h_step for samples that updated (using hard threshold)
            if update_H_mask.any():
                h_step_new = h_step + 1
            else:
                h_step_new = h_step
        else:
            if update_H:
                z_L_pooled = self._pool(z_L_new)
                H_cond = self.H_cond_fusion(torch.cat([t_embed_H, z_L_pooled], dim=-1))
                
                if self.use_sequence_denoiser:
                    z_H_new = self.H_denoiser(z_H, H_cond, cross_cond=z_L_new)
                else:
                    z_H_new = self.H_denoiser(z_H, H_cond)
                h_step_new = h_step + 1
            else:
                z_H_new = z_H
                h_step_new = h_step
            cumulative = state.cumulative_gate
        
        # Build new state
        new_state = DiffusionHLState(
            z_H=z_H_new,
            z_L=z_L_new,
            h_step=h_step_new,
            l_step=l_step + 1,
            level_logits=level_logits,
            cumulative_gate=cumulative if self.learned_timing else None,
        )
        
        # Auxiliary info
        aux = {
            "sigma_L": sigma_L.item(),
            "sigma_H": sigma_H.item(),
            "gate": gate.mean().item() if isinstance(gate, torch.Tensor) else float(gate),
            "l_step": l_step + 1,
            "h_step": h_step_new,
        }
        if level_logits is not None:
            aux["level_logits"] = level_logits.detach()
        
        return new_state, aux
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[DiffusionHLState] = None,
        n_steps: Optional[int] = None,
        return_all_states: bool = False,
    ) -> Tuple[DiffusionHLState, Dict[str, Any]]:
        """
        Run full diffusion denoising loop.
        
        Args:
            x: [B, S, d_model] input features
            state: Optional initial state (created if None)
            n_steps: Number of steps (default: max_steps)
            return_all_states: If True, return states from all steps
            
        Returns:
            final_state: Final denoised state
            aux: Dict with metrics and optional intermediate states
        """
        B, S, D = x.shape
        device = x.device
        
        if state is None:
            state = self.init_state(B, S, device)
        
        n_steps = n_steps or self.max_steps
        
        all_states = [] if return_all_states else None
        all_aux = []
        
        # Main denoising loop
        for step_idx in range(n_steps):
            is_last = step_idx == n_steps - 1
            
            # Gradient handling: only last few steps get gradients (like HRM)
            if not is_last and step_idx < n_steps - 2:
                with torch.no_grad():
                    state, aux = self.step(x, state)
                state = state.detach()
            else:
                state, aux = self.step(x, state)
            
            if return_all_states:
                all_states.append(state)
            all_aux.append(aux)
        
        # Aggregate auxiliary info
        final_aux = {
            "final_sigma_L": all_aux[-1]["sigma_L"],
            "final_sigma_H": all_aux[-1]["sigma_H"],
            "avg_gate": sum(a["gate"] for a in all_aux) / len(all_aux),
            "total_l_steps": state.l_step,
            "total_h_steps": state.h_step,
        }
        
        if return_all_states:
            final_aux["all_states"] = all_states
        
        return state, final_aux

