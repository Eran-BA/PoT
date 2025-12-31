"""
ARC Solver Models.

Solver architectures for the Abstraction and Reasoning Corpus (ARC).
Based on the Sudoku solver architecture, adapted for variable-size grids.

Key differences from Sudoku:
- Larger sequence length: 900 (30x30) vs 81 (9x9)
- Larger vocab: 12 (PAD + EOS + 0-9) vs 10 (0-9)
- Variable input/output sizes within the 30x30 grid

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.pot.models.hybrid_hrm import HybridHRMBase
from src.pot.models.puzzle_embedding import PuzzleEmbedding


# ARC constants
ARC_VOCAB_SIZE = 12  # PAD(0) + EOS(1) + digits(2-11)
ARC_SEQ_LEN = 900    # 30 * 30


class HybridPoHARCSolver(HybridHRMBase):
    """
    Hybrid PoT-HRM ARC solver with ACT wrapper.
    
    Extends HybridHRMBase with ARC-specific embeddings and output.
    Designed for abstract reasoning puzzles with variable-size grids.
    
    Architecture (from base class):
    - z_H, z_L: Persistent hidden states over 900 grid positions
    - L_level: Fast reasoning (inner loop)
    - H_level: Slow reasoning (outer loop)
    - ACT: Adaptive outer steps via Q-learning
    
    Args:
        vocab_size: Vocabulary size (12 for ARC)
        d_model: Hidden dimension
        n_heads: Number of attention heads
        H_layers: Layers in H_level module
        L_layers: Layers in L_level module
        d_ff: Feedforward dimension
        dropout: Dropout rate
        H_cycles: Fixed outer loop iterations per ACT step
        L_cycles: Fixed inner loop iterations per H_cycle
        T: HRM period for pointer controller
        num_puzzles: Number of puzzle embeddings (1 for shared)
        puzzle_emb_dim: Puzzle embedding dimension
        hrm_grad_style: If True, only last L+H get gradients
        halt_max_steps: Maximum ACT outer steps
        halt_exploration_prob: Exploration probability for Q-learning
        controller_type: Depth controller type
        controller_kwargs: Additional controller kwargs
        injection_mode: Feature injection mode
        injection_kwargs: Feature injection kwargs
    """
    
    def __init__(
        self,
        vocab_size: int = ARC_VOCAB_SIZE,
        d_model: int = 512,
        n_heads: int = 8,
        H_layers: int = 2,
        L_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.0,
        H_cycles: int = 2,
        L_cycles: int = 8,
        T: int = 4,
        num_puzzles: int = 1,
        puzzle_emb_dim: int = 512,
        hrm_grad_style: bool = False,
        halt_max_steps: int = 1,
        halt_exploration_prob: float = 0.1,
        allow_early_halt_eval: bool = False,
        controller_type: str = "gru",
        controller_kwargs: dict = None,
        injection_mode: str = "none",
        injection_kwargs: dict = None,
        use_rope: bool = True,  # Use RoPE by default (like HRM)
        rope_base: float = 10000.0,
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            H_layers=H_layers,
            L_layers=L_layers,
            d_ff=d_ff,
            seq_len=ARC_SEQ_LEN,  # 900 for ARC
            H_cycles=H_cycles,
            L_cycles=L_cycles,
            dropout=dropout,
            T=T,
            hrm_grad_style=hrm_grad_style,
            halt_max_steps=halt_max_steps,
            halt_exploration_prob=halt_exploration_prob,
            allow_early_halt_eval=allow_early_halt_eval,
            controller_type=controller_type,
            controller_kwargs=controller_kwargs,
            injection_mode=injection_mode,
            injection_kwargs=injection_kwargs,
            use_rope=use_rope,  # Pass to base class
            rope_base=rope_base,
        )
        
        self.vocab_size = vocab_size
        embed_init_std = 1.0 / self.embed_scale
        self._use_rope = use_rope
        
        # ARC-specific: Input embedding (vocab 12)
        self.input_embed = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.input_embed.weight, mean=0, std=embed_init_std)
        
        # ARC-specific: Puzzle embedding (ZERO initialized like HRM)
        self.puzzle_emb = PuzzleEmbedding(num_puzzles, puzzle_emb_dim, init_std=0.0)
        self.puzzle_emb_proj = (
            nn.Linear(puzzle_emb_dim, d_model) 
            if puzzle_emb_dim != d_model 
            else nn.Identity()
        )
        
        # Position embeddings: RoPE or fallback to sinusoidal
        # RoPE is applied inside attention (Q/K rotation), not added to input
        # Only create sinusoidal embeddings if NOT using RoPE
        if not use_rope:
            # ARC-specific: FIXED sinusoidal position embeddings (like original Transformer)
            # Using 2D row/col sinusoidal embeddings for spatial reasoning
            self.use_2d_pos = True
            
            # Create fixed sinusoidal embeddings
            pos_embed_1d = self._create_sinusoidal_embeddings(self.seq_len, d_model)
            self.register_buffer('pos_embed', pos_embed_1d.unsqueeze(0))  # [1, 900, d]
            
            if self.use_2d_pos:
                row_embed = self._create_sinusoidal_embeddings(30, d_model)
                col_embed = self._create_sinusoidal_embeddings(30, d_model)
                self.register_buffer('row_embed', row_embed.unsqueeze(0))  # [1, 30, d]
                self.register_buffer('col_embed', col_embed.unsqueeze(0))  # [1, 30, d]
        else:
            self.use_2d_pos = False
        
        # ARC-specific: Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    @staticmethod
    def _create_sinusoidal_embeddings(seq_len: int, d_model: int) -> torch.Tensor:
        """
        Create fixed sinusoidal position embeddings (Vaswani et al., 2017).
        
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        Args:
            seq_len: Sequence length
            d_model: Embedding dimension
            
        Returns:
            [seq_len, d_model] tensor of position embeddings
        """
        position = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def _compute_input_embedding(
        self, 
        input_seq: torch.Tensor, 
        puzzle_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute scaled input embedding.
        
        When using RoPE: position information is injected via Q/K rotation in attention
        When not using RoPE: adds fixed sinusoidal 2D position embeddings
        """
        B = input_seq.size(0)
        
        # Input embedding
        input_emb = self.input_embed(input_seq)
        
        # Add position embedding (only if NOT using RoPE)
        # RoPE applies positional information inside attention via Q/K rotation
        if not self._use_rope:
            if self.use_2d_pos:
                # Combine 1D, row, and column embeddings
                row_pos = self.row_embed.repeat(1, 30, 1)  # [1, 30, d] -> [1, 900, d]
                col_pos = self.col_embed.repeat_interleave(30, dim=1)  # [1, 30, d] -> [1, 900, d]
                pos_emb = self.pos_embed + row_pos + col_pos
            else:
                pos_emb = self.pos_embed
            input_emb = input_emb + pos_emb
        
        # Add puzzle embedding (broadcast to all positions)
        max_puzzle_id = self.puzzle_emb.num_puzzles - 1
        valid_mask = puzzle_ids <= max_puzzle_id
        clamped_ids = puzzle_ids.clamp(0, max_puzzle_id)
        puzzle_emb = self.puzzle_emb(clamped_ids)
        puzzle_emb = puzzle_emb * valid_mask.unsqueeze(-1).float()
        puzzle_emb = self.puzzle_emb_proj(puzzle_emb)
        input_emb = input_emb + puzzle_emb.unsqueeze(1)
        
        # Scale embeddings by sqrt(d_model) like HRM
        return self.embed_scale * input_emb
    
    def forward(self, input_seq: torch.Tensor, puzzle_ids: torch.Tensor):
        """
        Forward pass for ARC solving.
        
        Args:
            input_seq: [B, 900] - flattened 30x30 grid
            puzzle_ids: [B] - puzzle identifiers
            
        Returns:
            logits: [B, 900, vocab_size]
            q_halt, q_continue: Halting Q-values
            steps: Number of reasoning steps
            target_q_continue: Q-learning target (if ACT enabled)
        """
        input_emb = self._compute_input_embedding(input_seq, puzzle_ids)
        
        if self.halt_max_steps > 1:
            # Use ACT wrapper
            act_out = self.act_forward(input_emb)
            hidden = act_out['hidden']
            q_halt = act_out['q_halt']
            q_continue = act_out['q_continue']
            steps = act_out['steps']
            target_q_continue = act_out.get('target_q_continue')
            
            logits = self.output_proj(hidden)
            return logits, q_halt, q_continue, steps, target_q_continue
        else:
            hidden, q_halt, q_continue, steps = self.reasoning_loop(input_emb)
            logits = self.output_proj(hidden)
            return logits, q_halt, q_continue, steps
    
    # Async batching methods (HRM-style)
    def create_async_carry(self, batch_size: int, device: torch.device):
        """Create initial async carry for HRM-style batching."""
        return self.initial_async_carry(batch_size, device)
    
    def async_forward(self, carry, new_batch: dict):
        """Async-compatible forward for HRM-style batching."""
        device = carry.z_H.device
        
        new_input = new_batch['input'].to(device)
        new_labels = new_batch['label'].to(device)
        new_puzzle_ids = new_batch['puzzle_id'].to(device)
        
        carry = self.reset_async_carry(
            carry, carry.halted, new_input, new_labels, new_puzzle_ids, device
        )
        
        input_emb = self._compute_input_embedding(
            carry.current_input, carry.current_puzzle_ids
        )
        
        new_carry, outputs = self.forward_single_step_async(carry, input_emb)
        
        logits = self.output_proj(outputs['hidden'])
        outputs['logits'] = logits
        outputs['labels'] = new_carry.current_labels
        
        return new_carry, outputs


class BaselineARCSolver(nn.Module):
    """
    Standard Transformer baseline for ARC.
    
    Simple encoder-only transformer without iterative refinement.
    """
    
    def __init__(
        self,
        vocab_size: int = ARC_VOCAB_SIZE,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.seq_len = ARC_SEQ_LEN
        
        self.input_embed = nn.Embedding(vocab_size, d_model)
        
        # Fixed sinusoidal position embeddings (like HRM)
        pos_embed_1d = self._create_sinusoidal_embeddings(self.seq_len, d_model)
        self.register_buffer('pos_embed', pos_embed_1d.unsqueeze(0))  # [1, 900, d]
        
        # 2D position embeddings (fixed sinusoidal)
        row_embed = self._create_sinusoidal_embeddings(30, d_model)
        col_embed = self._create_sinusoidal_embeddings(30, d_model)
        self.register_buffer('row_embed', row_embed.unsqueeze(0))  # [1, 30, d]
        self.register_buffer('col_embed', col_embed.unsqueeze(0))  # [1, 30, d]
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_seq: torch.Tensor, puzzle_ids: torch.Tensor = None):
        """Forward pass."""
        # Combine position embeddings
        row_pos = self.row_embed.repeat(1, 30, 1)
        col_pos = self.col_embed.repeat_interleave(30, dim=1)
        pos_emb = self.pos_embed + row_pos + col_pos
        
        x = self.input_embed(input_seq) + pos_emb
        x = self.encoder(x)
        x = self.final_norm(x)
        logits = self.output_proj(x)
        
        return logits, None, None, 1
    
    @staticmethod
    def _create_sinusoidal_embeddings(seq_len: int, d_model: int) -> torch.Tensor:
        """Create fixed sinusoidal position embeddings."""
        position = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe

