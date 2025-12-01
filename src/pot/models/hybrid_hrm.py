"""
HybridHRM - Two-timescale reasoning with PoT head routing.

This module implements the Hybrid PoT-HRM architecture that combines:
- HRM's two-timescale reasoning (H_level slow, L_level fast)
- PoT's pointer-based head routing inside each reasoning module

The architecture uses:
- z_H, z_L: Persistent hidden states
- L_level: Fast reasoning, updates every inner step
- H_level: Slow reasoning, updates every outer step
- Both use PoT head routing for dynamic attention

Key HRM implementation details matched:
1. Embedding scaling by sqrt(d_model) - amplifies signal ~22x
2. H_init/L_init as buffers (NOT learned parameters)
3. Zero-initialized task embeddings
4. Q head initialized to near-zero (bias=-5)
5. No dropout in transformer blocks
6. Post-norm architecture with RMSNorm and SwiGLU
7. Only final iteration gets gradients (all others in no_grad)

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from src.pot.models.reasoning_module import ReasoningModule
from src.pot.models.hrm_layers import RMSNorm


class HybridHRMBase(nn.Module):
    """
    Base class for Hybrid PoT-HRM models.
    
    Provides the core two-timescale reasoning loop with PoT head routing.
    Subclass this for specific tasks (Sudoku, Maze, ARC, etc.).
    
    Architecture:
    - z_H, z_L: Persistent hidden states over seq_len positions
    - L_level: Fast reasoning, updates every inner step
    - H_level: Slow reasoning, updates every outer step
    - Both use PoT head routing for dynamic attention
    
    Args:
        d_model: Hidden dimension size
        n_heads: Number of attention heads
        H_layers: Number of layers in H_level module
        L_layers: Number of layers in L_level module
        d_ff: Feedforward hidden dimension
        seq_len: Sequence length (e.g., 81 for Sudoku)
        H_cycles: Number of outer loop iterations
        L_cycles: Number of inner loop iterations
        dropout: Dropout rate (default 0.0 to match HRM)
        T: HRM period for pointer controller
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
    ):
        super().__init__()
        
        self.d_model = d_model
        self.seq_len = seq_len
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Core two-timescale reasoning loop.
        
        Matches HRM exactly: ONLY the very last L_level call and 
        very last H_level call get gradients.
        
        Args:
            input_emb: Scaled input embedding [B, seq_len, d_model]
            
        Returns:
            hidden: Final hidden state [B, seq_len, d_model]
            q_halt: Q-values for halting [B]
            q_continue: Q-values for continuing [B]
            steps: Total number of reasoning steps
        """
        B = input_emb.size(0)
        device = input_emb.device
        
        # Initialize hidden states from BUFFERS (not learned, like HRM)
        z_H = self.H_init.view(1, 1, -1).expand(B, self.seq_len, -1).clone()
        z_L = self.L_init.view(1, 1, -1).expand(B, self.seq_len, -1).clone()
        
        # Initialize pointer controller states
        L_ptr_state = self.L_level.pointer_controller.init_state(B, device)
        H_ptr_state = self.H_level.pointer_controller.init_state(B, device)
        
        # ALL iterations in no_grad EXCEPT the very last L and H calls
        # This matches HRM exactly
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
        
        # ONLY these 2 calls get gradients (matching HRM exactly)
        z_L, L_ptr_state = self.L_level(z_L, z_H + input_emb, L_ptr_state)
        z_H, H_ptr_state = self.H_level(z_H, z_L, H_ptr_state)
        
        # Normalize output
        hidden = self.final_norm(z_H)
        
        # Q-values for halting (for API compatibility)
        q_logits = self.q_head(z_H[:, 0])  # Use first position
        q_halt = q_logits[:, 0]
        q_continue = q_logits[:, 1]
        
        return hidden, q_halt, q_continue, self.H_cycles * self.L_cycles

