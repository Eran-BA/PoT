"""
Diffusion Depth Controller

A diffusion-based controller that iteratively denoises routing weights, inspired
by Large Diffusion Models (LDM) and diffusion transformers. The controller learns
to progressively refine noisy routing weights through a learned denoising process.

Key Features:
- Iterative denoising of routing weights across depth steps
- Learned noise schedule conditioned on depth
- Denoising network that conditions on input features
- Smooth, temporally coherent routing evolution

Reference: Inspired by diffusion models and diffusion transformers
- Rombach et al. (2022) "High-Resolution Image Synthesis with Latent Diffusion Models"
- Peebles & Xie (2023) "Scalable Diffusion Models with Transformers (DiT)"

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
# State Class
# =============================================================================

@dataclass
class DiffusionDepthState:
    """State for Diffusion depth controller.
    
    Attributes:
        z: [B, d_ctrl] current latent representation (progressively denoised)
        step: Current depth step counter
    """
    z: torch.Tensor
    step: int


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
        [num_steps] tensor of noise levels (sigma values)
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


# =============================================================================
# Denoising Network Components
# =============================================================================

class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding for conditioning."""
    
    def __init__(self, d_embed: int, max_period: int = 10000):
        super().__init__()
        self.d_embed = d_embed
        self.max_period = max_period
        
        # MLP to process raw embedding
        self.mlp = nn.Sequential(
            nn.Linear(d_embed, d_embed * 4),
            nn.GELU(),
            nn.Linear(d_embed * 4, d_embed),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] or scalar timestep (can be float in [0, 1])
            
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


class DenoiseBlock(nn.Module):
    """
    Denoising block with adaptive layer norm (adaLN).
    
    Conditions on both input features and timestep embedding.
    Similar to DiT blocks but simplified for routing.
    """
    
    def __init__(self, d_ctrl: int, dropout: float = 0.0):
        super().__init__()
        
        # Self-processing
        self.norm1 = nn.LayerNorm(d_ctrl, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_ctrl, d_ctrl * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ctrl * 4, d_ctrl),
            nn.Dropout(dropout),
        )
        
        # Adaptive LayerNorm conditioning (scale and shift from timestep)
        self.adaLN_modulation = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_ctrl, d_ctrl * 4),  # scale1, shift1, scale2, shift2
        )
        
    def forward(
        self,
        z: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z: [B, d_ctrl] noisy latent
            cond: [B, d_ctrl] conditioning (timestep + input features)
            
        Returns:
            [B, d_ctrl] denoised latent
        """
        # Get adaptive LN parameters
        modulation = self.adaLN_modulation(cond)
        scale1, shift1, scale2, shift2 = modulation.chunk(4, dim=-1)
        
        # First norm + modulation
        h = self.norm1(z) * (1 + scale1) + shift1
        
        # MLP
        h = self.mlp(h)
        
        # Second modulation on output
        h = h * (1 + scale2) + shift2
        
        return z + h


# =============================================================================
# Diffusion Depth Controller
# =============================================================================

class DiffusionDepthController(nn.Module):
    """
    Diffusion-based depth controller with iterative denoising.
    
    Operates across the DEPTH axis (refinement iterations). At each depth step,
    the controller denoises the routing representation, progressively refining
    from a noisy initial state to clean routing weights.
    
    Architecture:
        X^(t) [B, S, d_model] → Pool → x_ctrl [B, d_ctrl]
        z^(t) = Denoise(z^(t-1), sigma(t), x_ctrl)
        Router(z^(t), X^(t)) → α^(t) [B, S, H]
    
    The denoising process follows:
        1. Initialize z^(0) ~ N(0, I) (or zeros)
        2. At each step t, denoise: z^(t) = f_θ(z^(t-1), σ(t), x_ctrl)
        3. Route based on final z: α = softmax(Router(z))
    
    Args:
        d_model: Model dimension (token representation size)
        n_heads: Number of attention heads to route over (H)
        d_ctrl: Controller hidden dimension (default: d_model)
        n_denoise_layers: Number of denoising blocks (default: 2)
        noise_schedule: Type of noise schedule ("linear", "cosine", "sqrt")
        max_depth: Maximum depth steps (default: 32)
        dropout: Dropout probability (default: 0.0)
        token_conditioned: If True, α depends on both token x_i and denoised state
        temperature: Softmax temperature for routing (default: 1.0)
        topk: Optional top-k sparsification (default: None)
        use_layernorm: Apply LayerNorm to outputs (default: True)
        entropy_reg: Entropy regularization coefficient (default: 1e-3)
        init_noise_scale: Scale for initial noise (default: 1.0)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ctrl: Optional[int] = None,
        n_denoise_layers: int = 2,
        noise_schedule: Literal["linear", "cosine", "sqrt"] = "cosine",
        max_depth: int = 32,
        dropout: float = 0.0,
        token_conditioned: bool = True,
        temperature: float = 1.0,
        topk: Optional[int] = None,
        use_layernorm: bool = True,
        entropy_reg: float = 1e-3,
        init_noise_scale: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ctrl = d_ctrl or d_model
        self.n_denoise_layers = n_denoise_layers
        self.noise_schedule_type = noise_schedule
        self.max_depth = max_depth
        self.temperature = float(temperature)
        self.token_conditioned = token_conditioned
        self.topk = topk
        self.entropy_reg = entropy_reg
        self.init_noise_scale = init_noise_scale
        
        if topk is not None:
            assert 1 <= topk <= n_heads, "topk must be in [1, n_heads]"
        
        # Input projection and pooling
        self.pool_ln = nn.LayerNorm(d_model)
        self.inp_proj = nn.Linear(d_model, self.d_ctrl)
        
        # Timestep embedding
        self.time_embed = TimestepEmbedding(self.d_ctrl)
        
        # Conditioning fusion (combines input features + timestep)
        self.cond_fusion = nn.Sequential(
            nn.Linear(self.d_ctrl * 2, self.d_ctrl),
            nn.GELU(),
            nn.Linear(self.d_ctrl, self.d_ctrl),
        )
        
        # Denoising network (stack of DenoiseBlocks)
        self.denoise_blocks = nn.ModuleList([
            DenoiseBlock(self.d_ctrl, dropout=dropout)
            for _ in range(n_denoise_layers)
        ])
        
        # Final projection after denoising
        self.final_proj = nn.Linear(self.d_ctrl, self.d_ctrl)
        
        # Optional layer norm
        self.ln_z = nn.LayerNorm(self.d_ctrl) if use_layernorm else nn.Identity()
        
        # Router
        self.drop = nn.Dropout(dropout)
        if token_conditioned:
            self.router = nn.Sequential(
                nn.LayerNorm(d_model + self.d_ctrl),
                nn.Linear(d_model + self.d_ctrl, self.d_ctrl),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.d_ctrl, n_heads),
            )
        else:
            self.router = nn.Sequential(
                nn.LayerNorm(self.d_ctrl),
                nn.Linear(self.d_ctrl, self.d_ctrl),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.d_ctrl, n_heads),
            )
        
        # Register noise schedule as buffer
        self.register_buffer(
            "sigma_schedule",
            get_noise_schedule(noise_schedule, max_depth, torch.device("cpu"))
        )
    
    def init_state(self, batch_size: int, device: torch.device) -> DiffusionDepthState:
        """Initialize state with noise or zeros."""
        if self.training:
            # During training, start with noise
            z0 = torch.randn(batch_size, self.d_ctrl, device=device) * self.init_noise_scale
        else:
            # During inference, can start with zeros (deterministic)
            z0 = torch.zeros(batch_size, self.d_ctrl, device=device)
        return DiffusionDepthState(z=z0, step=0)
    
    def _pool(self, X: torch.Tensor) -> torch.Tensor:
        """Pool tokens to single vector per batch."""
        Xn = self.pool_ln(X)
        return Xn.mean(dim=1)  # [B, d_model]
    
    def _topk_mask_renorm(self, alpha: torch.Tensor) -> torch.Tensor:
        """Apply top-k masking and renormalize."""
        if self.topk is None or self.topk >= alpha.size(-1):
            return alpha
        
        topv, topi = torch.topk(alpha, k=self.topk, dim=-1)
        masked = torch.zeros_like(alpha)
        masked.scatter_(-1, topi, topv)
        denom = masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return masked / denom
    
    def _compute_entropy(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute routing entropy."""
        return -(alpha * alpha.clamp_min(1e-12).log()).sum(dim=-1).mean()
    
    def set_temperature(self, T: float):
        """Update softmax temperature."""
        self.temperature = max(0.1, float(T))
    
    def _get_sigma(self, step: int, device: torch.device) -> torch.Tensor:
        """Get noise level for given step."""
        step = min(step, self.max_depth - 1)
        return self.sigma_schedule[step].to(device)
    
    def _denoise_step(
        self,
        z: torch.Tensor,
        x_ctrl: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        One denoising step.
        
        Args:
            z: [B, d_ctrl] current noisy latent
            x_ctrl: [B, d_ctrl] input conditioning
            sigma: [] noise level
            
        Returns:
            [B, d_ctrl] denoised latent
        """
        B = z.shape[0]
        
        # Get timestep embedding (sigma encodes the "time" in diffusion)
        t_embed = self.time_embed(sigma.expand(B))  # [B, d_ctrl]
        
        # Fuse input conditioning with timestep
        cond = self.cond_fusion(torch.cat([x_ctrl, t_embed], dim=-1))  # [B, d_ctrl]
        
        # Apply denoising blocks
        for block in self.denoise_blocks:
            z = block(z, cond)
        
        # Final projection
        z = self.final_proj(z)
        
        return z
    
    def step(
        self,
        X: torch.Tensor,
        state: Optional[DiffusionDepthState] = None,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, DiffusionDepthState, Dict[str, Any]]:
        """
        One refinement step (one denoising iteration).
        
        Args:
            X: [B, S, d_model] token representations
            state: Previous DiffusionDepthState (or None to initialize)
            return_aux: If True, return auxiliary metrics
            
        Returns:
            alpha: [B, S, H] routing weights
            state: Updated DiffusionDepthState
            aux: Dict with metrics
        """
        B, S, D = X.shape
        device = X.device
        
        if state is None:
            state = self.init_state(B, device)
        
        # Pool and project input
        g = self._pool(X)  # [B, d_model]
        x_ctrl = self.inp_proj(self.drop(g))  # [B, d_ctrl]
        
        # Get noise level for current step
        sigma = self._get_sigma(state.step, device)
        
        # Denoise step
        z_new = self._denoise_step(state.z, x_ctrl, sigma)
        z_new = self.ln_z(z_new)
        
        # Route to heads
        if self.token_conditioned:
            z_tok = z_new[:, None, :].expand(B, S, self.d_ctrl)
            logits = self.router(torch.cat([X, z_tok], dim=-1))  # [B, S, H]
        else:
            logits_global = self.router(z_new)  # [B, H]
            logits = logits_global[:, None, :].expand(B, S, self.n_heads)
        
        alpha = F.softmax(logits / self.temperature, dim=-1)
        alpha = self._topk_mask_renorm(alpha)
        
        entropy = self._compute_entropy(alpha)
        
        new_state = DiffusionDepthState(z=z_new, step=state.step + 1)
        
        aux: Dict[str, Any] = {}
        if return_aux:
            aux = {
                "router_logits": logits.detach(),
                "alphas": alpha.detach(),
                "entropy": entropy.detach(),
                "temperature": self.temperature,
                "depth_step": state.step,
                "sigma": sigma.item(),
            }
        
        return alpha, new_state, aux
    
    def forward(
        self,
        x: torch.Tensor,
        head_outputs: Optional[torch.Tensor] = None,
        *,
        state: Optional[DiffusionDepthState] = None,
        per_token_pool: str = "mean",
        mask: Optional[torch.Tensor] = None,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, DiffusionDepthState, Dict[str, Any]]:
        """Forward pass compatible with HRMPointerController API."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        alpha, state, aux = self.step(x, state, return_aux)
        
        if alpha.size(1) == 1:
            alpha = alpha.squeeze(1)
        
        return alpha, state, aux

