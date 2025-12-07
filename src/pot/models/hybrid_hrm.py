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


@dataclass
class ACTCarry:
    """Carry state for ACT wrapper."""
    z_H: torch.Tensor  # [B, seq_len, d_model]
    z_L: torch.Tensor  # [B, seq_len, d_model]
    L_ptr_state: Any
    H_ptr_state: Any


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
    ):
        super().__init__()
        
        self.d_model = d_model
        self.seq_len = seq_len
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.hrm_grad_style = hrm_grad_style
        self.halt_max_steps = halt_max_steps
        self.halt_exploration_prob = halt_exploration_prob
        self.allow_early_halt_eval = allow_early_halt_eval
        
        # Embedding scaling factor (CRITICAL: HRM uses this!)
        self.embed_scale = d_model ** 0.5  # sqrt(512) ≈ 22.6
        
        # Initial hidden states as BUFFERS (NOT learned, like HRM)
        # HRM uses trunc_normal with std=1
        self.register_buffer('H_init', torch.randn(d_model) * 1.0)
        self.register_buffer('L_init', torch.randn(d_model) * 1.0)
        
        # Two-timescale reasoning modules with PoT head routing
        self.L_level = ReasoningModule(d_model, n_heads, L_layers, d_ff, dropout, T)
        self.H_level = ReasoningModule(d_model, n_heads, H_layers, d_ff, dropout, T)
        
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
    
    def reasoning_loop(
        self, 
        input_emb: torch.Tensor,
        hrm_grad_style: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Core two-timescale reasoning loop.
        
        Args:
            input_emb: Scaled input embedding [B, seq_len, d_model]
            hrm_grad_style: If True, only last L_level + H_level calls get gradients (HRM-style).
                           If False, all calls in last H_cycle get gradients.
                           If None (default), uses self.hrm_grad_style.
            
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
        
        # Initialize hidden states from BUFFERS (not learned, like HRM)
        z_H = self.H_init.view(1, 1, -1).expand(B, self.seq_len, -1).clone()
        z_L = self.L_init.view(1, 1, -1).expand(B, self.seq_len, -1).clone()
        
        # Initialize pointer controller states
        L_ptr_state = self.L_level.pointer_controller.init_state(B, device)
        H_ptr_state = self.H_level.pointer_controller.init_state(B, device)
        
        if hrm_grad_style:
            # HRM-style: ONLY the very last L_level call and very last H_level call get gradients
            with torch.no_grad():
                for H_step in range(self.H_cycles):
                    for L_step in range(self.L_cycles):
                        # Skip the very last L_step of the very last H_step
                        is_last_L = (H_step == self.H_cycles - 1) and (L_step == self.L_cycles - 1)
                        if not is_last_L:
                            z_L, L_ptr_state = self.L_level(z_L, z_H + input_emb, L_ptr_state)
                    
                    # Skip the very last H_step
                    is_last_H = (H_step == self.H_cycles - 1)
                    if not is_last_H:
                        z_H, H_ptr_state = self.H_level(z_H, z_L, H_ptr_state)
            
            # Detach to cut gradient history
            z_H = z_H.detach()
            z_L = z_L.detach()
            
            # ONLY these 2 calls get gradients
            z_L, L_ptr_state = self.L_level(z_L, z_H + input_emb, L_ptr_state)
            z_H, H_ptr_state = self.H_level(z_H, z_L, H_ptr_state)
        else:
            # Alternative: All calls in last H_cycle get gradients
            for H_step in range(self.H_cycles):
                use_grad = (H_step == self.H_cycles - 1)
                
                for L_step in range(self.L_cycles):
                    if use_grad:
                        z_L, L_ptr_state = self.L_level(z_L, z_H + input_emb, L_ptr_state)
                    else:
                        with torch.no_grad():
                            z_L, L_ptr_state = self.L_level(z_L, z_H + input_emb, L_ptr_state)
                        z_L = z_L.detach()
                
                if use_grad:
                    z_H, H_ptr_state = self.H_level(z_H, z_L, H_ptr_state)
                else:
                    with torch.no_grad():
                        z_H, H_ptr_state = self.H_level(z_H, z_L, H_ptr_state)
                    z_H = z_H.detach()
        
        # Normalize output
        hidden = self.final_norm(z_H)
        
        # Q-values for halting (for API compatibility)
        q_logits = self.q_head(z_H[:, 0])  # Use first position
        q_halt = q_logits[:, 0]
        q_continue = q_logits[:, 1]
        
        return hidden, q_halt, q_continue, self.H_cycles * self.L_cycles
    
    def _init_carry(self, B: int, device: torch.device) -> ACTCarry:
        """Initialize carry state for ACT wrapper."""
        z_H = self.H_init.view(1, 1, -1).expand(B, self.seq_len, -1).clone()
        z_L = self.L_init.view(1, 1, -1).expand(B, self.seq_len, -1).clone()
        L_ptr_state = self.L_level.pointer_controller.init_state(B, device)
        H_ptr_state = self.H_level.pointer_controller.init_state(B, device)
        return ACTCarry(z_H=z_H, z_L=z_L, L_ptr_state=L_ptr_state, H_ptr_state=H_ptr_state)
    
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
        
        if use_grad:
            # With gradients (only on last step typically)
            for H_step in range(self.H_cycles):
                for L_step in range(self.L_cycles):
                    # HRM-style: skip last L in last H (will be done with grad)
                    is_last = (H_step == self.H_cycles - 1) and (L_step == self.L_cycles - 1)
                    if not is_last:
                        with torch.no_grad():
                            z_L, L_ptr_state = self.L_level(z_L, z_H + input_emb, L_ptr_state)
                
                if H_step < self.H_cycles - 1:
                    with torch.no_grad():
                        z_H, H_ptr_state = self.H_level(z_H, z_L, H_ptr_state)
            
            # Detach before final calls with grad
            z_H = z_H.detach()
            z_L = z_L.detach()
            
            # Final calls WITH gradients
            z_L, L_ptr_state = self.L_level(z_L, z_H + input_emb, L_ptr_state)
            z_H, H_ptr_state = self.H_level(z_H, z_L, H_ptr_state)
        else:
            # Without gradients
            with torch.no_grad():
                for H_step in range(self.H_cycles):
                    for L_step in range(self.L_cycles):
                        z_L, L_ptr_state = self.L_level(z_L, z_H + input_emb, L_ptr_state)
                    z_H, H_ptr_state = self.H_level(z_H, z_L, H_ptr_state)
        
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
        )
        
        return new_carry, hidden, q_halt, q_continue
    
    def act_forward(
        self,
        input_emb: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        ACT wrapper forward pass - like HRM's adaptive outer loop.
        
        Runs multiple ACT steps with Q-learning based halting:
        - Training: Adaptive halting when q_halt > q_continue (with exploration)
        - Evaluation: Always runs halt_max_steps
        
        Args:
            input_emb: Scaled input embedding [B, seq_len, d_model]
            
        Returns:
            Dict with:
                hidden: Final hidden state [B, seq_len, d_model]
                q_halt: Final Q-values for halting [B]
                q_continue: Final Q-values for continuing [B]
                steps: Number of ACT steps taken
                target_q_continue: Q-learning target (training only)
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
        
        for act_step in range(self.halt_max_steps):
            is_last_step = (act_step == self.halt_max_steps - 1)
            
            # Use gradients only on last step (like HRM)
            use_grad = is_last_step
            
            # Run one ACT step
            carry, hidden, q_halt, q_continue = self._single_act_step(
                input_emb, carry, use_grad=use_grad
            )
            
            steps = steps + 1
            
            # Store outputs
            final_hidden = hidden
            final_q_halt = q_halt
            final_q_continue = q_continue
            
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
        
        # Compute target Q if not already computed (for Q-learning)
        if self.training and self.halt_max_steps > 1 and target_q_continue is None:
            with torch.no_grad():
                _, _, next_q_halt, next_q_continue = self._single_act_step(
                    input_emb, carry, use_grad=False
                )
                target_q_continue = torch.sigmoid(
                    torch.maximum(next_q_halt, next_q_continue)
                )
        
        return {
            'hidden': final_hidden,
            'q_halt': final_q_halt,
            'q_continue': final_q_continue,
            'steps': int(steps.float().mean().item()),
            'target_q_continue': target_q_continue,
        }
    
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
        )
    
    def _reset_ptr_state(self, old_state, fresh_state, halted_mask: torch.Tensor):
        """
        Reset pointer state for halted samples only.
        
        Handles different state formats:
        - Tensor: use torch.where
        - HRMState (dataclass with z_L, z_H, step): reset each field
        - Other: return fresh state (fallback)
        """
        if isinstance(old_state, torch.Tensor):
            # Simple tensor state
            mask = halted_mask.view(-1, *([1] * (old_state.dim() - 1)))
            return torch.where(mask, fresh_state, old_state)
        elif hasattr(old_state, 'z_L') and hasattr(old_state, 'z_H') and hasattr(old_state, 'step'):
            # HRMState dataclass - reset each field for halted samples
            mask_2d = halted_mask.view(-1, 1)  # [B, 1] for broadcasting to [B, d_ctrl]
            new_z_L = torch.where(mask_2d, fresh_state.z_L, old_state.z_L)
            new_z_H = torch.where(mask_2d, fresh_state.z_H, old_state.z_H)
            new_step = torch.where(halted_mask, fresh_state.step, old_state.step)
            return HRMState(z_L=new_z_L, z_H=new_z_H, step=new_step)
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
        
        # Run H_cycles x L_cycles inner iterations with HRM-style gradients
        # Only the final L and H calls get gradients
        with torch.no_grad():
            for H_step in range(self.H_cycles):
                for L_step in range(self.L_cycles):
                    is_last = (H_step == self.H_cycles - 1) and (L_step == self.L_cycles - 1)
                    if not is_last:
                        z_L, L_ptr_state = self.L_level(z_L, z_H + input_emb, L_ptr_state)
                
                if H_step < self.H_cycles - 1:
                    z_H, H_ptr_state = self.H_level(z_H, z_L, H_ptr_state)
        
        # Detach before final calls with grad
        z_H = z_H.detach()
        z_L = z_L.detach()
        
        # Final calls WITH gradients (these are the only ones that backprop)
        z_L, L_ptr_state = self.L_level(z_L, z_H + input_emb, L_ptr_state)
        z_H, H_ptr_state = self.H_level(z_H, z_L, H_ptr_state)
        
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
                    for H_step in range(self.H_cycles):
                        for L_step in range(self.L_cycles):
                            temp_z_L, temp_L_ptr = self.L_level(temp_z_L, temp_z_H + input_emb, temp_L_ptr)
                        temp_z_H, temp_H_ptr = self.H_level(temp_z_H, temp_z_L, temp_H_ptr)
                    
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
        )
        
        outputs = {
            'hidden': hidden,
            'q_halt_logits': q_halt_logits,
            'q_continue_logits': q_continue_logits,
            'target_q_continue': target_q_continue,
        }
        
        return new_carry, outputs

