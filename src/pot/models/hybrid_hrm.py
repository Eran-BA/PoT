"""
HybridHRM - Two-timescale reasoning with PoT head routing.

This module implements the Hybrid PoT-HRM architecture that combines:
- HRM's two-timescale reasoning (H_level slow, L_level fast)
- PoT's pointer-based head routing inside each reasoning module
- ACT (Adaptive Computation Time) wrapper for dynamic outer steps

The architecture uses:
- z_H, z_L: Persistent hidden states
- L_level: Fast reasoning, updates every inner step
- H_level: Slow reasoning, updates every outer step
- Both use PoT head routing for dynamic attention
- ACT wrapper: Adaptive outer steps via Q-learning (like HRM)

Key HRM implementation details matched:
1. Embedding scaling by sqrt(d_model) - amplifies signal ~22x
2. H_init/L_init as buffers (NOT learned parameters)
3. Zero-initialized task embeddings
4. Q head initialized to near-zero (bias=-5)
5. No dropout in transformer blocks
6. Post-norm architecture with RMSNorm and SwiGLU
7. Only final iteration gets gradients (all others in no_grad)
8. ACT: Fixed inner cycles (H_cycles, L_cycles) + adaptive outer steps

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from src.pot.models.reasoning_module import ReasoningModule
from src.pot.models.hrm_layers import RMSNorm
from src.pot.core.hrm_controller import HRMState
from src.pot.core.depth_transformer_controller import DepthControllerCache
from src.pot.core.lstm_controllers import LSTMDepthState, xLSTMDepthState
from src.pot.modules.rope import RotaryEmbedding, CosSin


@dataclass
class ACTCarry:
    """Carry state for ACT wrapper."""
    z_H: torch.Tensor  # [B, seq_len, d_model]
    z_L: torch.Tensor  # [B, seq_len, d_model]
    L_ptr_state: Any
    H_ptr_state: Any
    L_inj_mem: Any = None  # Injection memory for L-level (cross_attn / broadcast_memory)
    H_inj_mem: Any = None  # Injection memory for H-level (cross_attn / broadcast_memory)


@dataclass
class PoTAsyncCarry:
    """
    Carry state for HRM-style async batching.
    
    Enables samples that halt early to be immediately replaced with new puzzles,
    improving GPU utilization during ACT training.
    
    Fields:
        z_H: H-level hidden states [B, seq_len, d_model]
        z_L: L-level hidden states [B, seq_len, d_model]
        L_ptr_state: Pointer controller state for L-level
        H_ptr_state: Pointer controller state for H-level
        steps: Current ACT step count per sample [B]
        halted: Which samples have halted [B]
        current_input: Current input embedding per sample [B, seq_len]
        current_labels: Current target labels per sample [B, seq_len]
        current_puzzle_ids: Current puzzle IDs per sample [B]
        L_inj_mem: Injection memory for L-level (cross_attn / broadcast_memory)
        H_inj_mem: Injection memory for H-level (cross_attn / broadcast_memory)
    """
    z_H: torch.Tensor           # [B, seq_len, d_model]
    z_L: torch.Tensor           # [B, seq_len, d_model]
    L_ptr_state: Any
    H_ptr_state: Any
    steps: torch.Tensor         # [B] int32
    halted: torch.Tensor        # [B] bool
    current_input: torch.Tensor # [B, seq_len]
    current_labels: torch.Tensor # [B, seq_len]
    current_puzzle_ids: torch.Tensor  # [B]
    L_inj_mem: Any = None       # Injection memory for L-level
    H_inj_mem: Any = None       # Injection memory for H-level

    def detach(self) -> 'PoTAsyncCarry':
        """Detach all tensors from computation graph to prevent gradient accumulation across steps."""
        def _detach_state(state):
            """Recursively detach state tensors."""
            if state is None:
                return None
            if isinstance(state, torch.Tensor):
                return state.detach()
            if hasattr(state, 'u_list'):  # DepthControllerCache
                return type(state)(u_list=[u.detach() for u in state.u_list])
            if hasattr(state, 'h') and hasattr(state, 'c'):  # LSTM-like states
                if hasattr(state, 'n'):  # xLSTMDepthState
                    return type(state)(
                        h=state.h.detach(), c=state.c.detach(),
                        n=state.n.detach(), m=state.m.detach(),
                        step=state.step
                    )
                else:  # LSTMDepthState
                    return type(state)(h=state.h.detach(), c=state.c.detach(), step=state.step)
            if hasattr(state, 'z_L') and hasattr(state, 'z_H'):  # HRMState
                return type(state)(
                    z_L=state.z_L.detach(), z_H=state.z_H.detach(), step=state.step
                )
            return state  # Unknown type, return as-is
        
        return PoTAsyncCarry(
            z_H=self.z_H.detach(),
            z_L=self.z_L.detach(),
            L_ptr_state=_detach_state(self.L_ptr_state),
            H_ptr_state=_detach_state(self.H_ptr_state),
            steps=self.steps.detach(),
            halted=self.halted.detach(),
            current_input=self.current_input.detach(),
            current_labels=self.current_labels.detach(),
            current_puzzle_ids=self.current_puzzle_ids.detach(),
            L_inj_mem=_detach_state(self.L_inj_mem),
            H_inj_mem=_detach_state(self.H_inj_mem),
        )


class HybridHRMBase(nn.Module):
    """
    Base class for Hybrid PoT-HRM models with ACT wrapper.
    
    Provides the core two-timescale reasoning loop with PoT head routing
    and adaptive computation time (ACT) for dynamic outer steps.
    
    Architecture:
    - z_H, z_L: Persistent hidden states over seq_len positions
    - L_level: Fast reasoning, updates every inner step (fixed L_cycles)
    - H_level: Slow reasoning, updates every outer step (fixed H_cycles)
    - Both use PoT head routing for dynamic attention
    - ACT: Adaptive number of outer steps via Q-learning
    
    Like HRM:
    - Fixed inner cycles (H_cycles x L_cycles) per ACT step
    - Adaptive outer steps (halt_max_steps) via Q-learning during training
    - During evaluation: runs halt_max_steps unless allow_early_halt_eval=True
    
    Args:
        d_model: Hidden dimension size
        n_heads: Number of attention heads
        H_layers: Number of layers in H_level module
        L_layers: Number of layers in L_level module
        d_ff: Feedforward hidden dimension
        seq_len: Sequence length (e.g., 81 for Sudoku)
        H_cycles: Number of H_level cycles per ACT step (fixed)
        L_cycles: Number of L_level cycles per H_cycle (fixed)
        dropout: Dropout rate (default 0.0 to match HRM)
        T: HRM period for pointer controller
        hrm_grad_style: If True, only last L+H calls get gradients (HRM-style).
        halt_max_steps: Maximum ACT outer steps (default 1 = no ACT)
        halt_exploration_prob: Exploration probability for Q-learning
        allow_early_halt_eval: If True, enable Q-learning based early halting during eval
        controller_type: Type of depth controller ("gru", "lstm", "xlstm", "mingru", "transformer", "pot_transformer", "swin", "mamba", "diffusion")
        controller_kwargs: Additional kwargs for controller creation
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        H_layers: int = 2,
        L_layers: int = 2,
        d_ff: int = 2048,
        seq_len: int = 81,
        H_cycles: int = 2,
        L_cycles: int = 8,
        dropout: float = 0.0,
        T: int = 4,
        hrm_grad_style: bool = False,
        halt_max_steps: int = 1,
        halt_exploration_prob: float = 0.1,
        allow_early_halt_eval: bool = False,
        controller_type: str = "gru",
        controller_kwargs: dict = None,
        injection_mode: str = "none",
        injection_kwargs: dict = None,
        use_rope: bool = False,
        rope_base: float = 10000.0,
        use_flash_attn: bool = True,
        use_edit_mask: bool = False,
        randomize_steps: bool = False,
        min_L_cycles: int = None,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.randomize_steps = randomize_steps
        self.min_L_cycles = min_L_cycles if min_L_cycles is not None else max(2, L_cycles // 2)
        self.hrm_grad_style = hrm_grad_style
        self.halt_max_steps = halt_max_steps
        self.halt_exploration_prob = halt_exploration_prob
        self.allow_early_halt_eval = allow_early_halt_eval
        self.controller_type = controller_type
        self.injection_mode = injection_mode
        self.use_rope = use_rope
        self.use_flash_attn = use_flash_attn
        self.use_edit_mask = use_edit_mask
        self.verbose = False  # Set to True to see inner-loop progress
        
        # Embedding scaling factor (CRITICAL: HRM uses this!)
        self.embed_scale = d_model ** 0.5  # sqrt(512) ≈ 22.6
        
        # RoPE (Rotary Position Embeddings) - like HRM
        if use_rope:
            head_dim = d_model // n_heads
            self.rotary_emb = RotaryEmbedding(
                dim=head_dim,
                max_position_embeddings=seq_len,
                base=rope_base,
            )
        else:
            self.rotary_emb = None
        
        # Initial hidden states as BUFFERS (NOT learned, like HRM)
        # HRM uses trunc_normal with std=1
        self.register_buffer('H_init', torch.randn(d_model) * 1.0)
        self.register_buffer('L_init', torch.randn(d_model) * 1.0)
        
        # Two-timescale reasoning modules with PoT head routing and feature injection
        ctrl_kwargs = controller_kwargs or {}
        inj_kwargs = injection_kwargs or {}
        self.L_level = ReasoningModule(
            d_model, n_heads, L_layers, d_ff, dropout, T,
            controller_type=controller_type,
            controller_kwargs=ctrl_kwargs,
            injection_mode=injection_mode,
            injection_kwargs=inj_kwargs,
            use_rope=use_rope,
            use_flash_attn=use_flash_attn,
            use_edit_mask=use_edit_mask,
        )
        self.H_level = ReasoningModule(
            d_model, n_heads, H_layers, d_ff, dropout, T,
            controller_type=controller_type,
            controller_kwargs=ctrl_kwargs,
            injection_mode=injection_mode,
            injection_kwargs=inj_kwargs,
            use_rope=use_rope,
            use_flash_attn=use_flash_attn,
            use_edit_mask=use_edit_mask,
        )
        
        # Output normalization
        self.final_norm = RMSNorm(d_model)
        
        # Q-halting with HRM-style initialization
        self.q_head = nn.Linear(d_model, 2)
        # Zero weights and bias=-5 for faster Q-learning bootstrap
        nn.init.zeros_(self.q_head.weight)
        nn.init.constant_(self.q_head.bias, -5.0)
    
    def get_input_embedding(self, *args, **kwargs) -> torch.Tensor:
        """
        Override in subclass to compute input embedding.
        
        Should return tensor of shape [B, seq_len, d_model] with HRM-style scaling applied.
        """
        raise NotImplementedError("Subclass must implement get_input_embedding")
    
    def get_output_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Override in subclass to compute output logits from hidden state.
        
        Args:
            hidden: Final hidden state [B, seq_len, d_model]
            
        Returns:
            logits: Output logits [B, seq_len, vocab_size]
        """
        raise NotImplementedError("Subclass must implement get_output_logits")
    
    def _get_rope_cos_sin(self, seq_len: int = None) -> Optional[CosSin]:
        """Get RoPE cos/sin embeddings if using RoPE, else None."""
        if self.rotary_emb is None:
            return None
        return self.rotary_emb(seq_len)
    
    def reasoning_loop(
        self, 
        input_emb: torch.Tensor,
        hrm_grad_style: Optional[bool] = None,
        causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Core two-timescale reasoning loop.
        
        Args:
            input_emb: Scaled input embedding [B, seq_len, d_model]
            hrm_grad_style: If True, only last L_level + H_level calls get gradients (HRM-style).
                           If False, all calls in last H_cycle get gradients.
                           If None (default), uses self.hrm_grad_style.
            causal: If True, apply causal masking in attention (for trajectory mode).
            
        Returns:
            hidden: Final hidden state [B, seq_len, d_model]
            q_halt: Q-values for halting [B]
            q_continue: Q-values for continuing [B]
            steps: Total number of reasoning steps
        """
        B = input_emb.size(0)
        device = input_emb.device
        
        # Use instance config if not specified
        if hrm_grad_style is None:
            hrm_grad_style = self.hrm_grad_style
        
        # Use actual input sequence length (supports variable-length trajectories)
        S = input_emb.size(1)
        cos_sin = self._get_rope_cos_sin(S)
        
        # Initialize hidden states from BUFFERS (not learned, like HRM)
        z_H = self.H_init.view(1, 1, -1).expand(B, S, -1).clone()
        z_L = self.L_init.view(1, 1, -1).expand(B, S, -1).clone()
        
        # Initialize pointer controller states
        L_ptr_state = self._init_controller_state(self.L_level.pointer_controller, B, device)
        H_ptr_state = self._init_controller_state(self.H_level.pointer_controller, B, device)
        
        # Initialize injection memory (for cross_attn mode)
        L_inj_mem = None
        H_inj_mem = None
        
        if hrm_grad_style:
            # HRM-style: ONLY the very last L_level call and very last H_level call get gradients
            with torch.no_grad():
                for H_step in range(self.H_cycles):
                    for L_step in range(self.L_cycles):
                        # Skip the very last L_step of the very last H_step
                        is_last_L = (H_step == self.H_cycles - 1) and (L_step == self.L_cycles - 1)
                        if not is_last_L:
                            z_L, L_ptr_state, L_inj_mem = self.L_level(z_L, z_H + input_emb, L_ptr_state, injection_memory=L_inj_mem, cos_sin=cos_sin, causal=causal)
                    
                    # Skip the very last H_step
                    is_last_H = (H_step == self.H_cycles - 1)
                    if not is_last_H:
                        z_H, H_ptr_state, H_inj_mem = self.H_level(z_H, z_L, H_ptr_state, injection_memory=H_inj_mem, cos_sin=cos_sin, causal=causal)
            
            # Detach to cut gradient history
            z_H = z_H.detach()
            z_L = z_L.detach()
            
            # ONLY these 2 calls get gradients
            z_L, L_ptr_state, L_inj_mem = self.L_level(z_L, z_H + input_emb, L_ptr_state, injection_memory=L_inj_mem, cos_sin=cos_sin, causal=causal)
            z_H, H_ptr_state, H_inj_mem = self.H_level(z_H, z_L, H_ptr_state, injection_memory=H_inj_mem, cos_sin=cos_sin, causal=causal)
        else:
            # Alternative: All calls in last H_cycle get gradients
            for H_step in range(self.H_cycles):
                use_grad = (H_step == self.H_cycles - 1)
                
                for L_step in range(self.L_cycles):
                    if use_grad:
                        z_L, L_ptr_state, L_inj_mem = self.L_level(z_L, z_H + input_emb, L_ptr_state, injection_memory=L_inj_mem, cos_sin=cos_sin, causal=causal)
                    else:
                        with torch.no_grad():
                            z_L, L_ptr_state, L_inj_mem = self.L_level(z_L, z_H + input_emb, L_ptr_state, injection_memory=L_inj_mem, cos_sin=cos_sin, causal=causal)
                        z_L = z_L.detach()
                
                if use_grad:
                    z_H, H_ptr_state, H_inj_mem = self.H_level(z_H, z_L, H_ptr_state, injection_memory=H_inj_mem, cos_sin=cos_sin, causal=causal)
                else:
                    with torch.no_grad():
                        z_H, H_ptr_state, H_inj_mem = self.H_level(z_H, z_L, H_ptr_state, injection_memory=H_inj_mem, cos_sin=cos_sin, causal=causal)
                    z_H = z_H.detach()
        
        # Normalize output
        hidden = self.final_norm(z_H)
        
        # Q-values for halting (for API compatibility)
        q_logits = self.q_head(z_H[:, 0])  # Use first position
        q_halt = q_logits[:, 0]
        q_continue = q_logits[:, 1]
        
        return hidden, q_halt, q_continue, self.H_cycles * self.L_cycles
    
    def _init_controller_state(self, controller, batch_size: int, device: torch.device):
        """Initialize controller state, handling different controller types."""
        if hasattr(controller, 'init_cache'):
            # Transformer controller
            return controller.init_cache()
        elif hasattr(controller, 'init_state'):
            # GRU, LSTM, xLSTM, minGRU controllers
            return controller.init_state(batch_size, device)
        else:
            # Fallback: return None and let controller handle it
            return None
    
    def _init_carry(self, B: int, device: torch.device) -> ACTCarry:
        """Initialize carry state for ACT wrapper."""
        z_H = self.H_init.view(1, 1, -1).expand(B, self.seq_len, -1).clone()
        z_L = self.L_init.view(1, 1, -1).expand(B, self.seq_len, -1).clone()
        L_ptr_state = self._init_controller_state(self.L_level.pointer_controller, B, device)
        H_ptr_state = self._init_controller_state(self.H_level.pointer_controller, B, device)
        return ACTCarry(
            z_H=z_H, z_L=z_L,
            L_ptr_state=L_ptr_state, H_ptr_state=H_ptr_state,
            L_inj_mem=None, H_inj_mem=None,
        )
    
    def _init_carry_diverse(
        self, B: int, M: int, device: torch.device, noise_std: float = 0.1,
    ) -> ACTCarry:
        """
        Initialize carry state for M parallel games with diverse z_H/z_L.
        
        Creates M*B batch entries where each game chunk gets different random
        noise added to the base H_init/L_init. Shared weights, independent states.
        
        Args:
            B: Original batch size
            M: Number of games (hypotheses)
            device: Device
            noise_std: Standard deviation of per-game initialization noise
            
        Returns:
            ACTCarry with batch size M*B, each game chunk differently initialized
        """
        MB = M * B
        
        # Base initialization (same for all games)
        z_H_base = self.H_init.view(1, 1, -1).expand(MB, self.seq_len, -1).clone()
        z_L_base = self.L_init.view(1, 1, -1).expand(MB, self.seq_len, -1).clone()
        
        # Add per-game diversity noise
        # Each game chunk [m*B : (m+1)*B] gets different noise
        if noise_std > 0 and M > 1:
            for m in range(M):
                start = m * B
                end = (m + 1) * B
                z_H_base[start:end] += torch.randn_like(z_H_base[start:end]) * noise_std
                z_L_base[start:end] += torch.randn_like(z_L_base[start:end]) * noise_std
        
        L_ptr_state = self._init_controller_state(self.L_level.pointer_controller, MB, device)
        H_ptr_state = self._init_controller_state(self.H_level.pointer_controller, MB, device)
        
        return ACTCarry(
            z_H=z_H_base, z_L=z_L_base,
            L_ptr_state=L_ptr_state, H_ptr_state=H_ptr_state,
            L_inj_mem=None, H_inj_mem=None,
        )
    
    def _single_act_step(
        self, 
        input_emb: torch.Tensor,
        carry: ACTCarry,
        use_grad: bool = True,
    ) -> Tuple[ACTCarry, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run one ACT step (H_cycles x L_cycles reasoning iterations).
        
        Args:
            input_emb: Scaled input embedding [B, seq_len, d_model]
            carry: Current carry state
            use_grad: Whether to compute gradients for this step
            
        Returns:
            new_carry: Updated carry state (detached for O(1) memory)
            hidden: Hidden state after this step [B, seq_len, d_model]
            q_halt: Q-values for halting [B]
            q_continue: Q-values for continuing [B]
        """
        z_H, z_L = carry.z_H, carry.z_L
        L_ptr_state, H_ptr_state = carry.L_ptr_state, carry.H_ptr_state
        L_inj_mem, H_inj_mem = carry.L_inj_mem, carry.H_inj_mem  # Preserve injection memory across ACT steps
        
        # Randomize step budget during training (anti-shortcut)
        # Forces the model to be "ready" at any depth, preventing last-step-only behavior
        import random
        if self.training and self.randomize_steps:
            L_cycles = random.randint(self.min_L_cycles, self.L_cycles)
        else:
            L_cycles = self.L_cycles
        
        # Get RoPE cos/sin (None if not using RoPE)
        cos_sin = self._get_rope_cos_sin(self.seq_len)
        
        if use_grad:
            # With gradients (only on last step typically)
            for H_step in range(self.H_cycles):
                for L_step in range(L_cycles):
                    # HRM-style: skip last L in last H (will be done with grad)
                    is_last = (H_step == self.H_cycles - 1) and (L_step == L_cycles - 1)
                    if not is_last:
                        with torch.no_grad():
                            z_L, L_ptr_state, L_inj_mem = self.L_level(z_L, z_H + input_emb, L_ptr_state, injection_memory=L_inj_mem, cos_sin=cos_sin)
                    if self.verbose:
                        iter_num = H_step * L_cycles + L_step + 1
                        total = self.H_cycles * L_cycles
                        print(f"\r    H={H_step+1}/{self.H_cycles} L={L_step+1}/{L_cycles} "
                              f"[{iter_num}/{total}]", end="", flush=True)
                
                if H_step < self.H_cycles - 1:
                    with torch.no_grad():
                        z_H, H_ptr_state, H_inj_mem = self.H_level(z_H, z_L, H_ptr_state, injection_memory=H_inj_mem, cos_sin=cos_sin)
            
            # Detach before final calls with grad
            z_H = z_H.detach()
            z_L = z_L.detach()
            
            # Final calls WITH gradients
            z_L, L_ptr_state, L_inj_mem = self.L_level(z_L, z_H + input_emb, L_ptr_state, injection_memory=L_inj_mem, cos_sin=cos_sin)
            z_H, H_ptr_state, H_inj_mem = self.H_level(z_H, z_L, H_ptr_state, injection_memory=H_inj_mem, cos_sin=cos_sin)
        else:
            # Without gradients (eval uses full L_cycles always)
            with torch.no_grad():
                for H_step in range(self.H_cycles):
                    for L_step in range(L_cycles):
                        z_L, L_ptr_state, L_inj_mem = self.L_level(z_L, z_H + input_emb, L_ptr_state, injection_memory=L_inj_mem, cos_sin=cos_sin)
                        if self.verbose:
                            iter_num = H_step * L_cycles + L_step + 1
                            total = self.H_cycles * L_cycles
                            print(f"\r    H={H_step+1}/{self.H_cycles} L={L_step+1}/{L_cycles} "
                                  f"[{iter_num}/{total}]", end="", flush=True)
                    z_H, H_ptr_state, H_inj_mem = self.H_level(z_H, z_L, H_ptr_state, injection_memory=H_inj_mem, cos_sin=cos_sin)
        
        # Normalize and compute Q-values
        hidden = self.final_norm(z_H)
        q_logits = self.q_head(z_H[:, 0])
        q_halt = q_logits[:, 0]
        q_continue = q_logits[:, 1]
        
        # New carry (detached for O(1) memory)
        new_carry = ACTCarry(
            z_H=z_H.detach(),
            z_L=z_L.detach(),
            L_ptr_state=L_ptr_state,
            H_ptr_state=H_ptr_state,
            L_inj_mem=L_inj_mem,
            H_inj_mem=H_inj_mem,
        )
        
        return new_carry, hidden, q_halt, q_continue
    
    def act_forward(
        self,
        input_emb: torch.Tensor,
        return_intermediate: bool = False,
        return_final_carry: bool = False,
    ) -> Dict[str, Any]:
        """
        ACT wrapper forward pass - like HRM's adaptive outer loop.
        
        Runs multiple ACT steps with Q-learning based halting:
        - Training: Adaptive halting when q_halt > q_continue (with exploration)
        - Evaluation: Always runs halt_max_steps
        
        Args:
            input_emb: Scaled input embedding [B, seq_len, d_model]
            return_intermediate: If True, collect hidden states from each ACT step
                for per-step metric analysis (evaluation only).
            return_final_carry: If True, include the final ACTCarry in the result
                for stability probes (evaluation only).
            
        Returns:
            Dict with:
                hidden: Final hidden state [B, seq_len, d_model]
                q_halt: Final Q-values for halting [B]
                q_continue: Final Q-values for continuing [B]
                steps: Number of ACT steps taken
                target_q_continue: Q-learning target (training only)
                intermediate_hiddens: List of hidden states per step (if return_intermediate)
                final_carry: ACTCarry after last step (if return_final_carry)
        """
        B = input_emb.size(0)
        device = input_emb.device
        
        # Initialize carry
        carry = self._init_carry(B, device)
        
        # Track halting state
        steps = torch.zeros(B, dtype=torch.int32, device=device)
        halted = torch.zeros(B, dtype=torch.bool, device=device)
        
        # Storage for final outputs
        final_hidden = None
        final_q_halt = None
        final_q_continue = None
        target_q_continue = None
        
        # Optional intermediate tracking
        intermediate_hiddens = [] if return_intermediate else None
        
        for act_step in range(self.halt_max_steps):
            is_last_step = (act_step == self.halt_max_steps - 1)
            
            # Use gradients only on last step (like HRM)
            use_grad = is_last_step
            
            # Verbose progress
            if self.verbose:
                total_inner = self.H_cycles * self.L_cycles
                print(f"\r  ACT step {act_step + 1}/{self.halt_max_steps} "
                      f"({total_inner} inner iters per step)", end="", flush=True)
            
            # Run one ACT step
            carry, hidden, q_halt, q_continue = self._single_act_step(
                input_emb, carry, use_grad=use_grad
            )
            
            steps = steps + 1
            
            # Store outputs
            final_hidden = hidden
            final_q_halt = q_halt
            final_q_continue = q_continue
            
            # Collect intermediate hidden states for per-step analysis
            if return_intermediate:
                intermediate_hiddens.append(hidden.detach())
            
            # Halting logic (during training, or eval with allow_early_halt_eval)
            if (self.training or self.allow_early_halt_eval) and self.halt_max_steps > 1 and not is_last_step:
                with torch.no_grad():
                    # Halt when q_halt > q_continue
                    should_halt = (q_halt > q_continue)
                    
                    # Exploration: randomly vary min_steps
                    if torch.rand(1).item() < self.halt_exploration_prob:
                        min_steps = torch.randint(2, self.halt_max_steps + 1, (1,), device=device).item()
                        should_halt = should_halt & (act_step + 1 >= min_steps)
                    
                    halted = halted | should_halt
                    
                    # Early exit if all halted (during training)
                    if halted.all():
                        # Compute target Q for Q-learning
                        _, _, next_q_halt, next_q_continue = self._single_act_step(
                            input_emb, carry, use_grad=False
                        )
                        target_q_continue = torch.sigmoid(
                            torch.maximum(next_q_halt, next_q_continue)
                        )
                        break
        
        # Clear verbose line
        if self.verbose:
            print()  # newline after progress
        
        # Compute target Q if not already computed (for Q-learning)
        if self.training and self.halt_max_steps > 1 and target_q_continue is None:
            with torch.no_grad():
                _, _, next_q_halt, next_q_continue = self._single_act_step(
                    input_emb, carry, use_grad=False
                )
                target_q_continue = torch.sigmoid(
                    torch.maximum(next_q_halt, next_q_continue)
                )
        
        result = {
            'hidden': final_hidden,
            'q_halt': final_q_halt,
            'q_continue': final_q_continue,
            'steps': int(steps.float().mean().item()),
            'target_q_continue': target_q_continue,
        }
        if return_intermediate:
            result['intermediate_hiddens'] = intermediate_hiddens
        if return_final_carry:
            result['final_carry'] = carry
        return result
    
    # ========== Async Batching Methods (HRM-style) ==========
    
    def initial_async_carry(self, batch_size: int, device: torch.device) -> PoTAsyncCarry:
        """
        Create initial carry state for async batching.
        
        All samples start as 'halted' so they will be replaced with actual data
        on the first forward call.
        
        Args:
            batch_size: Number of samples in batch
            device: Device to create tensors on
            
        Returns:
            Initial carry with all samples marked as halted
        """
        return PoTAsyncCarry(
            z_H=self.H_init.view(1, 1, -1).expand(batch_size, self.seq_len, -1).clone(),
            z_L=self.L_init.view(1, 1, -1).expand(batch_size, self.seq_len, -1).clone(),
            L_ptr_state=self.L_level.pointer_controller.init_state(batch_size, device),
            H_ptr_state=self.H_level.pointer_controller.init_state(batch_size, device),
            steps=torch.zeros(batch_size, dtype=torch.int32, device=device),
            halted=torch.ones(batch_size, dtype=torch.bool, device=device),  # All halted initially
            current_input=torch.zeros(batch_size, self.seq_len, dtype=torch.long, device=device),
            current_labels=torch.zeros(batch_size, self.seq_len, dtype=torch.long, device=device),
            current_puzzle_ids=torch.zeros(batch_size, dtype=torch.long, device=device),
            L_inj_mem=None,
            H_inj_mem=None,
        )
    
    def _reset_ptr_state(self, old_state, fresh_state, halted_mask: torch.Tensor):
        """
        Reset pointer state for halted samples only.
        
        Handles different state formats:
        - Tensor: use torch.where
        - HRMState (dataclass with z_L, z_H, step): reset each field
        - DepthControllerCache (dataclass with u_list): reset depth history for halted samples
        - Other: return fresh state (fallback)
        """
        if isinstance(old_state, torch.Tensor):
            # Simple tensor state
            mask = halted_mask.view(-1, *([1] * (old_state.dim() - 1)))
            return torch.where(mask, fresh_state, old_state)
        elif isinstance(old_state, HRMState):
            # HRMState dataclass - reset each field for halted samples
            mask_2d = halted_mask.view(-1, 1)  # [B, 1] for broadcasting to [B, d_ctrl]
            new_z_L = torch.where(mask_2d, fresh_state.z_L, old_state.z_L)
            new_z_H = torch.where(mask_2d, fresh_state.z_H, old_state.z_H)
            new_step = torch.where(halted_mask, fresh_state.step, old_state.step)
            return HRMState(z_L=new_z_L, z_H=new_z_H, step=new_step)
        elif isinstance(old_state, LSTMDepthState):
            # LSTMDepthState - reset h, c, step for halted samples
            mask_2d = halted_mask.view(-1, 1)  # [B, 1]
            new_h = torch.where(mask_2d, fresh_state.h, old_state.h)
            new_c = torch.where(mask_2d, fresh_state.c, old_state.c)
            # step is an int, so use fresh for all if any halted (simplification)
            new_step = fresh_state.step if halted_mask.any() else old_state.step
            return LSTMDepthState(h=new_h, c=new_c, step=new_step)
        elif isinstance(old_state, xLSTMDepthState):
            # xLSTMDepthState - reset h, c, n, m for halted samples
            mask_2d = halted_mask.view(-1, 1)  # [B, 1]
            new_h = torch.where(mask_2d, fresh_state.h, old_state.h)
            new_c = torch.where(mask_2d, fresh_state.c, old_state.c)
            new_n = torch.where(mask_2d, fresh_state.n, old_state.n)
            new_m = torch.where(halted_mask, fresh_state.m, old_state.m)
            new_step = fresh_state.step if halted_mask.any() else old_state.step
            return xLSTMDepthState(h=new_h, c=new_c, n=new_n, m=new_m, step=new_step)
        elif isinstance(old_state, DepthControllerCache):
            # DepthControllerCache - for halted samples, we clear their depth history
            # u_list contains tensors of shape [B, d_ctrl]
            # For halted samples, we zero out their entries to effectively reset
            if len(old_state.u_list) == 0:
                return fresh_state  # Already empty
            
            mask_2d = halted_mask.view(-1, 1)  # [B, 1]
            new_u_list = []
            for u in old_state.u_list:
                # Zero out halted samples' entries (they'll rebuild from scratch)
                new_u = torch.where(mask_2d, torch.zeros_like(u), u)
                new_u_list.append(new_u)
            return DepthControllerCache(u_list=new_u_list)
        else:
            # Unknown format - fallback to fresh (logs warning in debug)
            return fresh_state
    
    def reset_async_carry(
        self,
        carry: PoTAsyncCarry,
        halted_mask: torch.Tensor,
        new_input: torch.Tensor,
        new_labels: torch.Tensor,
        new_puzzle_ids: torch.Tensor,
        device: torch.device,
    ) -> PoTAsyncCarry:
        """
        Reset hidden states for halted samples and replace their data.
        
        For samples where halted_mask=True:
        - Reset z_H, z_L to initial values
        - Reset pointer states
        - Reset step counter to 0
        - Replace input/labels/puzzle_ids with new data
        
        Handles partial batches: if new_input has fewer samples than halted,
        only the first N halted samples are replaced, rest stay halted.
        
        Args:
            carry: Current carry state
            halted_mask: [B] bool tensor indicating which samples halted
            new_input: [N, seq_len] new input data (N <= num_halted)
            new_labels: [N, seq_len] new labels (N <= num_halted)
            new_puzzle_ids: [N] new puzzle IDs (N <= num_halted)
            device: Device
            
        Returns:
            Updated carry with halted samples reset
        """
        B = carry.z_H.size(0)
        N = new_input.size(0)  # Number of new samples available
        
        # Get indices of halted samples
        halted_indices = halted_mask.nonzero(as_tuple=True)[0]
        num_halted = halted_indices.size(0)
        
        # Only replace up to N samples (handle partial batches)
        num_to_replace = min(N, num_halted)
        
        # Create mask for samples that will be replaced (subset of halted)
        replace_mask = torch.zeros(B, dtype=torch.bool, device=device)
        if num_to_replace > 0:
            replace_indices = halted_indices[:num_to_replace]
            replace_mask[replace_indices] = True
        
        # Samples that stay halted (halted but not enough new data to replace)
        still_halted_mask = halted_mask & ~replace_mask
        
        # Reset hidden states for replaced samples
        fresh_z_H = self.H_init.view(1, 1, -1).expand(B, self.seq_len, -1)
        fresh_z_L = self.L_init.view(1, 1, -1).expand(B, self.seq_len, -1)
        
        # Expand mask for broadcasting
        mask_expanded = replace_mask.view(B, 1, 1)
        
        new_z_H = torch.where(mask_expanded, fresh_z_H, carry.z_H)
        new_z_L = torch.where(mask_expanded, fresh_z_L, carry.z_L)
        
        # Reset steps for replaced samples
        new_steps = torch.where(replace_mask, torch.zeros_like(carry.steps), carry.steps)
        
        # Replace input/labels/puzzle_ids for replaced samples
        # new_input/new_labels are [N, seq_len], new_puzzle_ids is [N]
        new_current_input = carry.current_input.clone()
        new_current_labels = carry.current_labels.clone()
        new_current_puzzle_ids = carry.current_puzzle_ids.clone()
        if num_to_replace > 0:
            replace_indices = halted_indices[:num_to_replace]
            new_current_input[replace_indices] = new_input[:num_to_replace]
            new_current_labels[replace_indices] = new_labels[:num_to_replace]
            new_current_puzzle_ids[replace_indices] = new_puzzle_ids[:num_to_replace]
        
        # Reset pointer states for replaced samples
        fresh_L_ptr = self.L_level.pointer_controller.init_state(B, device)
        fresh_H_ptr = self.H_level.pointer_controller.init_state(B, device)
        
        # Handle different pointer state formats
        new_L_ptr = self._reset_ptr_state(carry.L_ptr_state, fresh_L_ptr, replace_mask)
        new_H_ptr = self._reset_ptr_state(carry.H_ptr_state, fresh_H_ptr, replace_mask)
        
        # Reset injection memory for replaced samples
        # For tensor memories: zero out halted entries; for None: stay None
        def _reset_inj_mem(old_mem, replace_mask):
            if old_mem is None:
                return None
            if isinstance(old_mem, torch.Tensor):
                # Memory is [B, T, D] - zero out replaced samples
                mask_expanded = replace_mask.view(-1, 1, 1)
                return torch.where(mask_expanded, torch.zeros_like(old_mem), old_mem)
            return None  # Unknown format, reset to None
        
        new_L_inj_mem = _reset_inj_mem(carry.L_inj_mem, replace_mask)
        new_H_inj_mem = _reset_inj_mem(carry.H_inj_mem, replace_mask)
        
        return PoTAsyncCarry(
            z_H=new_z_H,
            z_L=new_z_L,
            L_ptr_state=new_L_ptr,
            H_ptr_state=new_H_ptr,
            steps=new_steps,
            halted=still_halted_mask,  # Samples that are still halted (waiting for data)
            current_input=new_current_input,
            current_labels=new_current_labels,
            current_puzzle_ids=new_current_puzzle_ids,
            L_inj_mem=new_L_inj_mem,
            H_inj_mem=new_H_inj_mem,
        )
    
    def forward_single_step_async(
        self,
        carry: PoTAsyncCarry,
        input_emb: torch.Tensor,
    ) -> Tuple[PoTAsyncCarry, Dict[str, torch.Tensor]]:
        """
        Run one ACT step for async batching (HRM-style).
        
        Each call does H_cycles x L_cycles inner iterations, then:
        - Increments step counter
        - Computes halting decision
        - Returns new carry + outputs
        
        Args:
            carry: Current carry state
            input_emb: Scaled input embedding [B, seq_len, d_model]
            
        Returns:
            new_carry: Updated carry state
            outputs: Dict with logits, q_halt, q_continue, target_q_continue
        """
        B = input_emb.size(0)
        device = input_emb.device
        
        z_H, z_L = carry.z_H, carry.z_L
        L_ptr_state, H_ptr_state = carry.L_ptr_state, carry.H_ptr_state
        L_inj_mem, H_inj_mem = carry.L_inj_mem, carry.H_inj_mem  # Preserve injection memory across ACT steps
        
        # Get RoPE cos/sin (None if not using RoPE)
        cos_sin = self._get_rope_cos_sin(self.seq_len)
        
        # Run H_cycles x L_cycles inner iterations with HRM-style gradients
        # Only the final L and H calls get gradients
        with torch.no_grad():
            for H_step in range(self.H_cycles):
                for L_step in range(self.L_cycles):
                    is_last = (H_step == self.H_cycles - 1) and (L_step == self.L_cycles - 1)
                    if not is_last:
                        z_L, L_ptr_state, L_inj_mem = self.L_level(z_L, z_H + input_emb, L_ptr_state, injection_memory=L_inj_mem, cos_sin=cos_sin)
                    if self.verbose:
                        iter_num = H_step * self.L_cycles + L_step + 1
                        total = self.H_cycles * self.L_cycles
                        print(f"\r    H={H_step+1}/{self.H_cycles} L={L_step+1}/{self.L_cycles} "
                              f"[{iter_num}/{total}]", end="", flush=True)
                
                if H_step < self.H_cycles - 1:
                    z_H, H_ptr_state, H_inj_mem = self.H_level(z_H, z_L, H_ptr_state, injection_memory=H_inj_mem, cos_sin=cos_sin)
        
        # Detach before final calls with grad
        z_H = z_H.detach()
        z_L = z_L.detach()
        
        # Final calls WITH gradients (these are the only ones that backprop)
        z_L, L_ptr_state, L_inj_mem = self.L_level(z_L, z_H + input_emb, L_ptr_state, injection_memory=L_inj_mem, cos_sin=cos_sin)
        z_H, H_ptr_state, H_inj_mem = self.H_level(z_H, z_L, H_ptr_state, injection_memory=H_inj_mem, cos_sin=cos_sin)
        
        if self.verbose:
            print()  # newline after inner progress
        
        # Normalize and compute Q-values
        hidden = self.final_norm(z_H)
        q_logits = self.q_head(z_H[:, 0])
        q_halt_logits = q_logits[:, 0]
        q_continue_logits = q_logits[:, 1]
        
        # Increment step counter
        new_steps = carry.steps + 1
        is_last_step = new_steps >= self.halt_max_steps
        
        # Halting decision
        halted = is_last_step.clone()
        target_q_continue = None
        
        if self.training and self.halt_max_steps > 1:
            with torch.no_grad():
                # Halt when q_halt > q_continue
                halted = halted | (q_halt_logits > q_continue_logits)
                
                # Exploration: random continuation
                explore_mask = torch.rand(B, device=device) < self.halt_exploration_prob
                min_halt_steps = torch.randint(2, self.halt_max_steps + 1, (B,), device=device)
                halted = halted & (~explore_mask | (new_steps >= min_halt_steps))
                
                # Compute target Q for Q-learning (like HRM)
                # Look ahead one step to get target
                with torch.no_grad():
                    temp_z_L = z_L.detach()
                    temp_z_H = z_H.detach()
                    temp_L_ptr = L_ptr_state
                    temp_H_ptr = H_ptr_state
                    
                    # One more reasoning step (without grad)
                    # Preserve injection memory for accurate Q-target estimate
                    temp_L_inj_mem, temp_H_inj_mem = L_inj_mem, H_inj_mem
                    for H_step in range(self.H_cycles):
                        for L_step in range(self.L_cycles):
                            temp_z_L, temp_L_ptr, temp_L_inj_mem = self.L_level(temp_z_L, temp_z_H + input_emb, temp_L_ptr, injection_memory=temp_L_inj_mem, cos_sin=cos_sin)
                        temp_z_H, temp_H_ptr, temp_H_inj_mem = self.H_level(temp_z_H, temp_z_L, temp_H_ptr, injection_memory=temp_H_inj_mem, cos_sin=cos_sin)
                    
                    next_q_logits = self.q_head(temp_z_H[:, 0])
                    next_q_halt = next_q_logits[:, 0]
                    next_q_continue = next_q_logits[:, 1]
                    
                    target_q_continue = torch.sigmoid(
                        torch.where(is_last_step, next_q_halt, torch.maximum(next_q_halt, next_q_continue))
                    )
        
        # Build new carry (detached z_H, z_L for O(1) memory)
        new_carry = PoTAsyncCarry(
            z_H=z_H.detach(),
            z_L=z_L.detach(),
            L_ptr_state=L_ptr_state,
            H_ptr_state=H_ptr_state,
            steps=new_steps,
            halted=halted,
            current_input=carry.current_input,
            current_labels=carry.current_labels,
            current_puzzle_ids=carry.current_puzzle_ids,
            L_inj_mem=L_inj_mem,
            H_inj_mem=H_inj_mem,
        )
        
        outputs = {
            'hidden': hidden,
            'q_halt_logits': q_halt_logits,
            'q_continue_logits': q_continue_logits,
            'target_q_continue': target_q_continue,
        }
        
        return new_carry, outputs

