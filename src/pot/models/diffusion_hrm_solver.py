"""
Diffusion HRM Solver

A standalone solver module that uses diffusion-based H,L cycles for
two-timescale reasoning. This module is task-agnostic and can be used
for Sudoku, mazes, or any iterative reasoning task.

Key Features:
1. Diffusion-based H,L cycles instead of deterministic GRU updates
2. Learned timing for H-level updates (instead of fixed T)
3. Q-halting for adaptive computation time
4. Compatible with existing training infrastructure

The solver treats reasoning as a hierarchical diffusion process:
- z_L (fast): Rapidly denoises tactical, local decisions
- z_H (slow): Slowly denoises strategic, global structure
- Timing: Learns when to "think slow" vs "think fast"

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pot.core.diffusion_hl_cycles import (
    DiffusionHLCycles,
    DiffusionHLState,
)
from src.pot.models.hrm_layers import RMSNorm


# =============================================================================
# Carry State for Async Batching
# =============================================================================

@dataclass
class DiffusionHRMCarry:
    """
    Carry state for diffusion HRM solver with async batching support.
    
    Enables samples that halt early to be immediately replaced with new puzzles,
    improving GPU utilization during ACT training.
    
    Attributes:
        hl_state: Current DiffusionHLState (z_H, z_L, step counters)
        steps: [B] ACT step counter
        halted: [B] whether each sample has halted
        current_input: [B, S] current input data
        current_labels: [B, S] current target labels
        current_puzzle_ids: [B] current puzzle IDs
    """
    hl_state: DiffusionHLState
    steps: torch.Tensor
    halted: torch.Tensor
    current_input: torch.Tensor
    current_labels: torch.Tensor
    current_puzzle_ids: torch.Tensor
    
    def detach(self) -> 'DiffusionHRMCarry':
        """Detach all tensors from computation graph."""
        return DiffusionHRMCarry(
            hl_state=self.hl_state.detach(),
            steps=self.steps.detach(),
            halted=self.halted.detach(),
            current_input=self.current_input.detach(),
            current_labels=self.current_labels.detach(),
            current_puzzle_ids=self.current_puzzle_ids.detach(),
        )


# =============================================================================
# Main Solver
# =============================================================================

class DiffusionHRMSolver(nn.Module):
    """
    Standalone solver using diffusion-based H,L cycles.
    
    This module combines:
    - Input embedding (with optional puzzle embeddings)
    - DiffusionHLCycles for two-timescale reasoning
    - Output projection for task-specific predictions
    - Q-halting for adaptive computation
    
    The architecture treats iterative reasoning as hierarchical diffusion:
    
        Input → Embed → [Diffusion H,L Cycles] → Output Head → Logits
                              ↑
                        z_H (slow) ⟷ z_L (fast)
                        denoised at different rates
    
    Args:
        vocab_size: Size of input vocabulary
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feedforward dimension (for output projection)
        seq_len: Sequence length (e.g., 81 for Sudoku)
        max_steps: Maximum diffusion steps per ACT step
        T: Base timescale ratio (H updates ~T times slower)
        noise_schedule: Type of noise schedule
        dropout: Dropout probability
        num_puzzles: Number of puzzle embeddings (0 to disable)
        puzzle_emb_dim: Puzzle embedding dimension
        halt_max_steps: Maximum ACT outer steps
        halt_exploration_prob: Exploration probability for Q-learning
        allow_early_halt_eval: Allow early halting during evaluation
        use_sequence_denoiser: Use attention-based sequence denoiser
        learned_timing: Learn when to update H (vs fixed T)
    """
    
    def __init__(
        self,
        vocab_size: int = 10,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        seq_len: int = 81,
        max_steps: int = 32,
        T: int = 4,
        noise_schedule: Literal["linear", "cosine", "sqrt"] = "cosine",
        dropout: float = 0.0,
        num_puzzles: int = 1000,
        puzzle_emb_dim: int = 512,
        halt_max_steps: int = 1,
        halt_exploration_prob: float = 0.1,
        allow_early_halt_eval: bool = False,
        use_sequence_denoiser: bool = True,
        learned_timing: bool = True,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len
        self.max_steps = max_steps
        self.halt_max_steps = halt_max_steps
        self.halt_exploration_prob = halt_exploration_prob
        self.allow_early_halt_eval = allow_early_halt_eval
        
        # Embedding scaling (like HRM)
        self.embed_scale = d_model ** 0.5
        
        # Input embedding
        self.input_embed = nn.Embedding(vocab_size, d_model)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.input_norm = nn.LayerNorm(d_model)
        
        # Optional puzzle embeddings
        self.use_puzzle_emb = num_puzzles > 0
        if self.use_puzzle_emb:
            self.puzzle_emb = nn.Embedding(num_puzzles, puzzle_emb_dim)
            self.puzzle_proj = nn.Linear(puzzle_emb_dim, d_model)
            nn.init.normal_(self.puzzle_emb.weight, std=0.02)
        
        # Diffusion H,L cycles
        self.hl_cycles = DiffusionHLCycles(
            d_model=d_model,
            n_heads=n_heads,
            max_steps=max_steps,
            T=T,
            noise_schedule=noise_schedule,
            dropout=dropout,
            use_sequence_denoiser=use_sequence_denoiser,
            learned_timing=learned_timing,
        )
        
        # Output normalization
        self.final_norm = RMSNorm(d_model)
        
        # Output head
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, vocab_size),
        )
        
        # Q-halting head (for adaptive computation)
        self.q_head = nn.Linear(d_model, 2)
        # HRM-style initialization
        nn.init.zeros_(self.q_head.weight)
        nn.init.constant_(self.q_head.bias, -5.0)
    
    def get_input_embedding(
        self,
        input_seq: torch.Tensor,
        puzzle_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute input embedding with optional puzzle conditioning.
        
        Args:
            input_seq: [B, S] input token IDs
            puzzle_ids: [B] optional puzzle IDs for per-puzzle embeddings
            
        Returns:
            [B, S, d_model] input embeddings with HRM-style scaling
        """
        # Token embedding + position
        x = self.input_embed(input_seq)  # [B, S, d_model]
        x = x + self.pos_embed[:, :x.size(1), :]
        
        # Add puzzle embedding if available
        if self.use_puzzle_emb and puzzle_ids is not None:
            puz_emb = self.puzzle_emb(puzzle_ids)  # [B, puzzle_emb_dim]
            puz_proj = self.puzzle_proj(puz_emb)  # [B, d_model]
            x = x + puz_proj.unsqueeze(1)  # Add to all positions
        
        # HRM-style scaling and normalization
        x = x * self.embed_scale
        x = self.input_norm(x)
        
        return x
    
    def reasoning_loop(
        self,
        input_emb: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, DiffusionHLState]:
        """
        Run diffusion-based reasoning loop.
        
        Args:
            input_emb: [B, S, d_model] input embedding
            n_steps: Number of diffusion steps (default: max_steps)
            
        Returns:
            hidden: [B, S, d_model] final hidden state (with output residual)
            final_state: Final DiffusionHLState
        """
        B, S, D = input_emb.shape
        
        # Run diffusion H,L cycles
        final_state, aux = self.hl_cycles(
            x=input_emb,
            n_steps=n_steps or self.max_steps,
        )
        
        # Use z_H as the output (strategic representation)
        hidden = self.final_norm(final_state.z_H)
        
        # HRM-style output residual: add input back to output
        # This provides a strong gradient path and preserves input information
        hidden = hidden + input_emb
        
        return hidden, final_state
    
    def forward(
        self,
        input_seq: torch.Tensor,
        puzzle_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Forward pass with adaptive halting.
        
        Args:
            input_seq: [B, S] input token IDs
            puzzle_ids: [B] optional puzzle IDs
            
        Returns:
            logits: [B, S, vocab_size] output logits
            q_halt: [B] Q-values for halting
            q_continue: [B] Q-values for continuing
            steps: Number of ACT steps taken
        """
        B = input_seq.size(0)
        device = input_seq.device
        
        # Get input embedding
        input_emb = self.get_input_embedding(input_seq, puzzle_ids)
        
        if self.halt_max_steps == 1:
            # No ACT: single pass
            hidden, final_state = self.reasoning_loop(input_emb)
            
            # Output logits
            logits = self.output_proj(hidden)
            
            # Q-values (for compatibility)
            q_logits = self.q_head(hidden[:, 0])
            q_halt = q_logits[:, 0]
            q_continue = q_logits[:, 1]
            
            return logits, q_halt, q_continue, 1
        else:
            # ACT: multiple passes with Q-learning halting
            return self._act_forward(input_emb)
    
    def _act_forward(
        self,
        input_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        ACT forward pass with adaptive halting.
        
        Args:
            input_emb: [B, S, d_model] input embedding
            
        Returns:
            logits: [B, S, vocab_size] final logits
            q_halt: [B] final Q-halt values
            q_continue: [B] final Q-continue values
            steps: Average number of steps taken
        """
        B, S, D = input_emb.shape
        device = input_emb.device
        
        # Initialize state
        hl_state = self.hl_cycles.init_state(B, S, device)
        
        # Track halting
        steps = torch.zeros(B, dtype=torch.int32, device=device)
        halted = torch.zeros(B, dtype=torch.bool, device=device)
        
        final_hidden = None
        final_q_halt = None
        final_q_continue = None
        
        for act_step in range(self.halt_max_steps):
            is_last_step = act_step == self.halt_max_steps - 1
            
            # Run reasoning loop (with grad only on last step)
            if not is_last_step:
                with torch.no_grad():
                    hidden, hl_state = self.reasoning_loop(input_emb)
                hl_state = hl_state.detach()
            else:
                hidden, hl_state = self.reasoning_loop(input_emb)
            
            # Compute Q-values
            q_logits = self.q_head(hidden[:, 0])
            q_halt = q_logits[:, 0]
            q_continue = q_logits[:, 1]
            
            steps = steps + 1
            
            # Store outputs
            final_hidden = hidden
            final_q_halt = q_halt
            final_q_continue = q_continue
            
            # Halting decision (during training or if allowed in eval)
            if (self.training or self.allow_early_halt_eval) and not is_last_step:
                with torch.no_grad():
                    should_halt = q_halt > q_continue
                    
                    # Exploration
                    if self.training and torch.rand(1).item() < self.halt_exploration_prob:
                        min_steps = torch.randint(2, self.halt_max_steps + 1, (1,), device=device).item()
                        should_halt = should_halt & (act_step + 1 >= min_steps)
                    
                    halted = halted | should_halt
                    
                    if halted.all():
                        break
            
            # Update input with residual from z_H (iterative refinement)
            if not is_last_step:
                input_emb = input_emb + 0.1 * hl_state.z_H.detach()
        
        # Output logits
        logits = self.output_proj(final_hidden)
        
        return logits, final_q_halt, final_q_continue, int(steps.float().mean().item())
    
    # =========================================================================
    # Async Batching Support
    # =========================================================================
    
    def initial_async_carry(
        self,
        batch_size: int,
        device: torch.device,
    ) -> DiffusionHRMCarry:
        """
        Create initial carry state for async batching.
        
        All samples start as 'halted' so they will be replaced with
        actual data on the first forward call.
        
        Args:
            batch_size: Number of samples in batch
            device: Target device
            
        Returns:
            Initial carry with all samples marked as halted
        """
        hl_state = self.hl_cycles.init_state(batch_size, self.seq_len, device)
        
        return DiffusionHRMCarry(
            hl_state=hl_state,
            steps=torch.zeros(batch_size, dtype=torch.int32, device=device),
            halted=torch.ones(batch_size, dtype=torch.bool, device=device),
            current_input=torch.zeros(batch_size, self.seq_len, dtype=torch.long, device=device),
            current_labels=torch.zeros(batch_size, self.seq_len, dtype=torch.long, device=device),
            current_puzzle_ids=torch.zeros(batch_size, dtype=torch.long, device=device),
        )
    
    def reset_async_carry(
        self,
        carry: DiffusionHRMCarry,
        halted_mask: torch.Tensor,
        new_input: torch.Tensor,
        new_labels: torch.Tensor,
        new_puzzle_ids: torch.Tensor,
        device: torch.device,
    ) -> DiffusionHRMCarry:
        """
        Reset hidden states for halted samples and replace their data.
        
        Args:
            carry: Current carry state
            halted_mask: [B] bool tensor indicating which samples halted
            new_input: [N, S] new input data
            new_labels: [N, S] new labels
            new_puzzle_ids: [N] new puzzle IDs
            device: Device
            
        Returns:
            Updated carry with halted samples reset
        """
        B = carry.hl_state.z_H.size(0)
        N = new_input.size(0)
        
        halted_indices = halted_mask.nonzero(as_tuple=True)[0]
        num_halted = halted_indices.size(0)
        num_to_replace = min(N, num_halted)
        
        # Create mask for samples to replace
        replace_mask = torch.zeros(B, dtype=torch.bool, device=device)
        if num_to_replace > 0:
            replace_indices = halted_indices[:num_to_replace]
            replace_mask[replace_indices] = True
        
        still_halted_mask = halted_mask & ~replace_mask
        
        # Get fresh H,L state
        fresh_state = self.hl_cycles.init_state(B, self.seq_len, device)
        
        # Reset z_H, z_L for replaced samples
        mask_expanded = replace_mask.view(B, 1, 1)
        new_z_H = torch.where(mask_expanded, fresh_state.z_H, carry.hl_state.z_H)
        new_z_L = torch.where(mask_expanded, fresh_state.z_L, carry.hl_state.z_L)
        
        new_hl_state = DiffusionHLState(
            z_H=new_z_H,
            z_L=new_z_L,
            h_step=0 if replace_mask.all() else carry.hl_state.h_step,
            l_step=0 if replace_mask.all() else carry.hl_state.l_step,
            level_logits=None,
            cumulative_gate=torch.where(replace_mask, torch.zeros(B, device=device), 
                                        carry.hl_state.cumulative_gate if carry.hl_state.cumulative_gate is not None 
                                        else torch.zeros(B, device=device)),
        )
        
        # Reset step counter
        new_steps = torch.where(replace_mask, torch.zeros_like(carry.steps), carry.steps)
        
        # Replace input/labels/puzzle_ids
        new_current_input = carry.current_input.clone()
        new_current_labels = carry.current_labels.clone()
        new_current_puzzle_ids = carry.current_puzzle_ids.clone()
        
        if num_to_replace > 0:
            replace_indices = halted_indices[:num_to_replace]
            new_current_input[replace_indices] = new_input[:num_to_replace]
            new_current_labels[replace_indices] = new_labels[:num_to_replace]
            new_current_puzzle_ids[replace_indices] = new_puzzle_ids[:num_to_replace]
        
        return DiffusionHRMCarry(
            hl_state=new_hl_state,
            steps=new_steps,
            halted=still_halted_mask,
            current_input=new_current_input,
            current_labels=new_current_labels,
            current_puzzle_ids=new_current_puzzle_ids,
        )
    
    def forward_single_step_async(
        self,
        carry: DiffusionHRMCarry,
        input_emb: torch.Tensor,
    ) -> Tuple[DiffusionHRMCarry, Dict[str, torch.Tensor]]:
        """
        Run one ACT step for async batching.
        
        Args:
            carry: Current carry state
            input_emb: [B, S, d_model] input embedding
            
        Returns:
            new_carry: Updated carry state
            outputs: Dict with hidden, q_halt, q_continue, etc.
        """
        B = input_emb.size(0)
        device = input_emb.device
        
        # Run one complete diffusion loop
        hl_state = carry.hl_state
        
        # Diffusion with gradients only on last steps
        with torch.no_grad():
            for step_idx in range(self.max_steps - 2):
                hl_state, _ = self.hl_cycles.step(input_emb, hl_state)
            hl_state = hl_state.detach()
        
        # Last 2 steps with gradients
        hl_state, _ = self.hl_cycles.step(input_emb, hl_state)
        hl_state, _ = self.hl_cycles.step(input_emb, hl_state)
        
        # Compute outputs with HRM-style output residual
        hidden = self.final_norm(hl_state.z_H)
        hidden = hidden + input_emb  # Output residual
        q_logits = self.q_head(hidden[:, 0])
        q_halt = q_logits[:, 0]
        q_continue = q_logits[:, 1]
        
        # Update step counter
        new_steps = carry.steps + 1
        is_last_step = new_steps >= self.halt_max_steps
        
        # Halting decision
        halted = is_last_step.clone()
        
        if self.training and self.halt_max_steps > 1:
            with torch.no_grad():
                halted = halted | (q_halt > q_continue)
                
                # Exploration
                explore_mask = torch.rand(B, device=device) < self.halt_exploration_prob
                min_halt = torch.randint(2, self.halt_max_steps + 1, (B,), device=device)
                halted = halted & (~explore_mask | (new_steps >= min_halt))
        
        # Build new carry
        new_carry = DiffusionHRMCarry(
            hl_state=hl_state.detach(),
            steps=new_steps,
            halted=halted,
            current_input=carry.current_input,
            current_labels=carry.current_labels,
            current_puzzle_ids=carry.current_puzzle_ids,
        )
        
        # Output logits
        logits = self.output_proj(hidden)
        
        outputs = {
            'hidden': hidden,
            'logits': logits,
            'q_halt_logits': q_halt,
            'q_continue_logits': q_continue,
        }
        
        return new_carry, outputs


# =============================================================================
# Specialized Solvers
# =============================================================================

class DiffusionSudokuSolver(DiffusionHRMSolver):
    """
    Diffusion HRM solver specialized for Sudoku.
    
    Uses Sudoku-specific defaults:
    - 9x9 grid (81 cells)
    - Vocabulary: 0-9 (0=blank, 1-9=digits)
    - Appropriate model dimensions for the task
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        max_steps: int = 32,
        T: int = 4,
        num_puzzles: int = 10000,
        halt_max_steps: int = 8,
        **kwargs,
    ):
        super().__init__(
            vocab_size=10,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_model * 4,
            seq_len=81,
            max_steps=max_steps,
            T=T,
            num_puzzles=num_puzzles,
            puzzle_emb_dim=d_model,
            halt_max_steps=halt_max_steps,
            **kwargs,
        )


class DiffusionMazeSolver(DiffusionHRMSolver):
    """
    Diffusion HRM solver specialized for maze navigation.
    
    Uses maze-specific defaults:
    - Grid size configurable (default 30x30 = 900 cells)
    - Vocabulary: wall, path, start, end, etc.
    """
    
    def __init__(
        self,
        grid_size: int = 30,
        vocab_size: int = 5,  # wall, path, start, end, visited
        d_model: int = 256,
        n_heads: int = 8,
        max_steps: int = 48,
        T: int = 6,
        halt_max_steps: int = 12,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_model * 4,
            seq_len=grid_size * grid_size,
            max_steps=max_steps,
            T=T,
            num_puzzles=0,  # No puzzle embeddings for mazes
            halt_max_steps=halt_max_steps,
            **kwargs,
        )
        
        self.grid_size = grid_size

