"""
Sudoku Solver Models.

Three solver architectures for Sudoku:
1. PoHSudokuSolver - Full PoH with latent tokens and Q-halting
2. HybridPoHHRMSolver - Two-timescale reasoning (closest to HRM paper)
3. BaselineSudokuSolver - Standard transformer baseline

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.pot.core.hrm_controller import HRMPointerController, HRMState
from src.pot.core.controller_factory import create_controller
from src.pot.models.hybrid_hrm import HybridHRMBase
from src.pot.models.puzzle_embedding import PuzzleEmbedding
from src.pot.models.adaptive_halting import QHaltingController
from src.pot.models.hrm_layers import RMSNorm, SwiGLU


class PoHSudokuSolver(nn.Module):
    """
    PoH-based Sudoku solver with HRM-style components.
    
    Architecture:
    - Puzzle embeddings for per-instance specialization
    - HRM pointer controller for hierarchical routing
    - Q-halting for adaptive computation
    - Post-norm transformer layers with SwiGLU
    - Latent tokens for working memory
    
    Args:
        vocab_size: Vocabulary size (0=blank, 1-9=digits)
        d_model: Hidden dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feedforward dimension
        dropout: Dropout rate
        R: Refinement iterations
        T: HRM outer period
        num_puzzles: Number of puzzle embeddings
        puzzle_emb_dim: Puzzle embedding dimension
        max_halting_steps: Maximum halting steps
        latent_len: Number of latent tokens
        latent_k: Inner latent update iterations
        controller_type: Type of depth controller ("gru", "lstm", "xlstm", "mingru", "transformer", "pot_transformer")
        controller_kwargs: Additional kwargs for controller creation
    """
    
    def __init__(
        self,
        vocab_size: int = 10,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.1,
        R: int = 8,
        T: int = 4,
        num_puzzles: int = 1000,
        puzzle_emb_dim: int = 512,
        max_halting_steps: int = 16,
        latent_len: int = 16,
        latent_k: int = 3,
        controller_type: str = "gru",  # NEW: controller type
        controller_kwargs: dict = None,  # NEW: controller kwargs
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.R = R
        self.n_layers = n_layers
        self.seq_len = 81  # 9x9 Sudoku
        self.controller_type = controller_type
        
        # Puzzle embeddings
        self.puzzle_emb = PuzzleEmbedding(num_puzzles, puzzle_emb_dim, init_std=0.02)
        self.puzzle_emb_len = (puzzle_emb_dim + d_model - 1) // d_model
        
        # Latent tokens (TRM-style)
        self.latent_len = latent_len
        self.latent_k = latent_k
        self.latent_init = nn.Parameter(torch.randn(1, latent_len, d_model) * 0.02)
        
        # Q-halting
        self.q_halt_controller = QHaltingController(d_model, max_steps=max_halting_steps)
        self.max_halting_steps = max_halting_steps
        
        # Input embedding
        self.input_embed = nn.Embedding(vocab_size, d_model)
        
        # Position embedding (seq + puzzle + latent)
        max_len = self.seq_len + self.puzzle_emb_len + latent_len
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.pre_norm = nn.LayerNorm(d_model)
        
        # Depth controller (supports multiple types)
        ctrl_kwargs = controller_kwargs or {}
        self.hrm_controller = create_controller(
            controller_type=controller_type,
            d_model=d_model,
            n_heads=n_heads,
            T=T,
            dropout=dropout,
            **ctrl_kwargs,
        )
        
        # Transformer layers (Post-norm + SwiGLU)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            SwiGLU(d_model, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm1_layers = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.norm2_layers = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
        
        # Latent cross-attention
        self.latent_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.latent_ffn = SwiGLU(d_model, d_ff, dropout)
        self.latent_norm1 = RMSNorm(d_model)
        self.latent_norm2 = RMSNorm(d_model)
        self.latent_drop = nn.Dropout(dropout)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def _init_controller_state(self, batch_size: int, device):
        """Initialize controller state, handling different controller types."""
        if hasattr(self.hrm_controller, 'init_cache'):
            return self.hrm_controller.init_cache()
        elif hasattr(self.hrm_controller, 'init_state'):
            return self.hrm_controller.init_state(batch_size, device)
        return None
    
    def _detach_controller_state(self, state):
        """Detach controller state for O(1) memory."""
        if state is None:
            return None
        
        # Handle HRMState (GRU controller)
        if hasattr(state, 'z_L') and hasattr(state, 'z_H'):
            return HRMState(
                z_L=state.z_L.detach(),
                z_H=state.z_H.detach(),
                step=state.step
            )
        
        # Handle LSTMDepthState
        if hasattr(state, 'h') and hasattr(state, 'c') and hasattr(state, 'step'):
            from src.pot.core.lstm_controllers import LSTMDepthState
            return LSTMDepthState(
                h=state.h.detach(),
                c=state.c.detach(),
                step=state.step
            )
        
        # Handle xLSTMDepthState
        if hasattr(state, 'n') and hasattr(state, 'm'):
            from src.pot.core.lstm_controllers import xLSTMDepthState
            return xLSTMDepthState(
                h=state.h.detach(),
                c=state.c.detach(),
                n=state.n.detach(),
                m=state.m.detach(),
                step=state.step
            )
        
        # Handle DepthControllerCache (transformer)
        if hasattr(state, 'u_list'):
            from src.pot.core.depth_transformer_controller import DepthControllerCache
            return DepthControllerCache(
                u_list=[u.detach() for u in state.u_list]
            )
        
        # Handle simple tensor state (minGRU)
        if isinstance(state, torch.Tensor):
            return state.detach()
        
        # Fallback
        return state
    
    def _encode_once(self, x: torch.Tensor, hrm_state=None, depth_step: int = 0):
        """Single encoding pass with routing."""
        B, T, D = x.shape
        n_heads = self.attn_layers[0].num_heads
        
        # Get routing weights (handle different controller APIs)
        if self.controller_type == "gru":
            route_weights, hrm_state, _ = self.hrm_controller(x, state=hrm_state)
        elif self.controller_type in ("transformer", "pot_transformer"):
            route_weights, hrm_state, _ = self.hrm_controller.step(
                x, t=depth_step, cache=hrm_state
            )
        else:
            route_weights, hrm_state, _ = self.hrm_controller.step(x, state=hrm_state)
        
        # Handle different output shapes
        if route_weights.dim() == 2:
            # [B, H] -> scale and expand
            route_weights_scaled = route_weights * n_heads
            route_exp = route_weights_scaled.unsqueeze(1).unsqueeze(-1)
        else:
            # [B, S, H] -> per-token routing
            route_weights_scaled = route_weights * n_heads
            route_exp = route_weights_scaled.unsqueeze(-1)
        
        # Apply transformer layers with head routing
        for attn, ffn, norm1, norm2, drop in zip(
            self.attn_layers, self.ffn_layers, 
            self.norm1_layers, self.norm2_layers, self.dropout_layers
        ):
            # Attention with head routing
            attn_out, _ = attn(x, x, x, need_weights=False)
            d_head = D // attn.num_heads
            attn_out_heads = attn_out.view(B, T, attn.num_heads, d_head)
            attn_out_routed = (attn_out_heads * route_exp).view(B, T, D)
            x = norm1(x + drop(attn_out_routed))
            
            # FFN
            x = norm2(x + drop(ffn(x)))
        
        return x, hrm_state
    
    def forward(self, input_seq: torch.Tensor, puzzle_ids: torch.Tensor):
        """
        Forward pass with iterative refinement.
        
        Args:
            input_seq: (B, 81) - flattened Sudoku grid
            puzzle_ids: (B,) - puzzle identifiers
        
        Returns:
            logits: (B, 81, vocab_size)
            q_halt, q_continue: halting Q-values
            actual_steps: number of refinement steps
        """
        B = input_seq.size(0)
        device = input_seq.device
        
        # Puzzle embedding (clamp IDs to valid range)
        max_puzzle_id = self.puzzle_emb.num_puzzles - 1
        valid_mask = puzzle_ids <= max_puzzle_id
        clamped_ids = puzzle_ids.clamp(0, max_puzzle_id)
        puzzle_emb = self.puzzle_emb(clamped_ids)
        puzzle_emb = puzzle_emb * valid_mask.unsqueeze(-1).float()
        
        pad_size = self.puzzle_emb_len * self.d_model - puzzle_emb.size(-1)
        if pad_size > 0:
            puzzle_emb = F.pad(puzzle_emb, (0, pad_size))
        puzzle_emb = puzzle_emb.view(B, self.puzzle_emb_len, self.d_model)
        
        # Input embedding
        x_grid_input = self.input_embed(input_seq)
        
        # Latent tokens
        latent_cur = self.latent_init.expand(B, -1, -1)
        
        # Persistent hidden state (HRM-style)
        x_grid_hidden = torch.zeros_like(x_grid_input)
        
        # Initialize controller state
        hrm_state = self._init_controller_state(B, device)
        actual_steps = self.max_halting_steps
        x_out = None
        
        # Iterative refinement
        for step in range(1, self.max_halting_steps + 1):
            # INPUT INJECTION: add original input to hidden state
            x_grid_step = x_grid_hidden + x_grid_input
            
            # Inner latent updates
            for _ in range(self.latent_k):
                ctx = torch.cat([puzzle_emb, x_grid_step], dim=1)
                lat_attn, _ = self.latent_attn(latent_cur, ctx, ctx, need_weights=False)
                latent_cur = self.latent_norm1(latent_cur + self.latent_drop(lat_attn))
                latent_cur = self.latent_norm2(latent_cur + self.latent_drop(self.latent_ffn(latent_cur)))
            
            # Build sequence
            x_step = torch.cat([puzzle_emb, latent_cur, x_grid_step], dim=1)
            x_step = x_step + self.pos_embed[:, :x_step.size(1), :]
            x_step = self.pre_norm(x_step)
            
            # Encode (pass depth_step for transformer controller)
            x_out, hrm_state = self._encode_once(x_step, hrm_state, depth_step=step - 1)
            
            # Check halting
            q_halt, q_continue = self.q_halt_controller(x_out)
            should_halt = self.q_halt_controller.should_halt(q_halt, q_continue, step, self.training)
            
            if should_halt.all():
                actual_steps = step
                break
            
            # Update hidden state for next iteration
            x_grid_hidden = x_out[:, self.puzzle_emb_len + self.latent_len:, :]
            
            # Detach for O(1) memory
            latent_cur = latent_cur.detach()
            x_out = x_out.detach()
            x_grid_hidden = x_grid_hidden.detach()
            hrm_state = self._detach_controller_state(hrm_state)
        
        # Extract output (remove puzzle + latent tokens)
        x = x_out[:, self.puzzle_emb_len + self.latent_len:, :]
        
        # Residual connection to input
        x = x + x_grid_input
        
        logits = self.output_proj(x)
        
        return logits, q_halt, q_continue, actual_steps


class HybridPoHHRMSolver(HybridHRMBase):
    """
    Hybrid PoT-HRM Sudoku solver with ACT wrapper.
    
    Extends HybridHRMBase with Sudoku-specific embeddings and output.
    This is the closest architecture to the original HRM paper.
    
    Architecture (from base class):
    - z_H, z_L: Persistent hidden states over 81 grid positions
    - L_level: Fast reasoning, updates every inner step (fixed L_cycles)
    - H_level: Slow reasoning, updates every outer step (fixed H_cycles)
    - Both use PoT head routing for dynamic attention
    - ACT: Adaptive outer steps via Q-learning (like HRM)
    
    Args:
        vocab_size: Vocabulary size (0=blank, 1-9=digits)
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
        hrm_grad_style: If True, only last L+H get gradients.
        halt_max_steps: Maximum ACT outer steps (1 = no ACT, like original)
        halt_exploration_prob: Exploration probability for Q-learning
        allow_early_halt_eval: If True, enable Q-learning based early halting during eval
        controller_type: Type of depth controller ("gru", "lstm", "xlstm", "mingru", "transformer", "pot_transformer", "swin", "mamba", "diffusion")
        controller_kwargs: Additional kwargs for controller creation
        injection_mode: Feature injection mode ("none", "broadcast", "film", "depth_token", "cross_attn")
        injection_kwargs: Additional kwargs for FeatureInjector
    """
    
    def __init__(
        self,
        vocab_size: int = 10,
        d_model: int = 512,
        n_heads: int = 8,
        H_layers: int = 2,
        L_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.0,
        H_cycles: int = 2,
        L_cycles: int = 8,
        T: int = 4,
        num_puzzles: int = 1,  # Single shared embedding like HRM
        puzzle_emb_dim: int = 512,
        hrm_grad_style: bool = False,  # Default: all calls in last H_cycle get gradients
        halt_max_steps: int = 1,  # 1 = no ACT (original behavior), >1 = ACT enabled
        halt_exploration_prob: float = 0.1,
        allow_early_halt_eval: bool = False,  # Enable early halting during eval
        controller_type: str = "gru",  # NEW: controller type
        controller_kwargs: dict = None,  # NEW: controller kwargs
        injection_mode: str = "none",  # Feature injection mode
        injection_kwargs: dict = None,  # Feature injection kwargs
    ):
        # Initialize base class with ACT parameters
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            H_layers=H_layers,
            L_layers=L_layers,
            d_ff=d_ff,
            seq_len=81,  # Sudoku grid
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
        )
        
        self.vocab_size = vocab_size
        embed_init_std = 1.0 / self.embed_scale
        
        # Sudoku-specific: Input embedding
        self.input_embed = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.input_embed.weight, mean=0, std=embed_init_std)
        
        # Sudoku-specific: Puzzle embedding (ZERO initialized like HRM!)
        self.puzzle_emb = PuzzleEmbedding(num_puzzles, puzzle_emb_dim, init_std=0.0)
        self.puzzle_emb_proj = nn.Linear(puzzle_emb_dim, d_model) if puzzle_emb_dim != d_model else nn.Identity()
        
        # Sudoku-specific: Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, d_model) * embed_init_std)
        
        # Sudoku-specific: Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def _compute_input_embedding(self, input_seq: torch.Tensor, puzzle_ids: torch.Tensor) -> torch.Tensor:
        """Compute scaled input embedding."""
        # Compute input embedding
        input_emb = self.input_embed(input_seq) + self.pos_embed
        
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
        Forward pass for Sudoku solving.
        
        Uses ACT wrapper if halt_max_steps > 1, otherwise uses simple reasoning_loop.
        
        Returns:
            logits: Output logits [B, 81, vocab_size]
            q_halt: Q-values for halting [B]
            q_continue: Q-values for continuing [B]
            steps: Number of reasoning steps
            target_q_continue: Q-learning target (only if ACT enabled and training)
        """
        input_emb = self._compute_input_embedding(input_seq, puzzle_ids)
        
        if self.halt_max_steps > 1:
            # Use ACT wrapper (like HRM)
            act_out = self.act_forward(input_emb)
            hidden = act_out['hidden']
            q_halt = act_out['q_halt']
            q_continue = act_out['q_continue']
            steps = act_out['steps']
            target_q_continue = act_out.get('target_q_continue')
            
            # Output projection
            logits = self.output_proj(hidden)
            
            return logits, q_halt, q_continue, steps, target_q_continue
        else:
            # Original behavior (no ACT)
            hidden, q_halt, q_continue, steps = self.reasoning_loop(input_emb)
            logits = self.output_proj(hidden)
            return logits, q_halt, q_continue, steps
    
    # ========== Async Batching Methods (HRM-style) ==========
    
    def create_async_carry(self, batch_size: int, device: torch.device):
        """
        Create initial async carry for HRM-style async batching.
        
        All samples start as 'halted' so they will receive data on first forward call.
        
        Args:
            batch_size: Number of samples in batch
            device: Device to create tensors on
            
        Returns:
            Initial PoTAsyncCarry with all samples marked as halted
        """
        return self.initial_async_carry(batch_size, device)
    
    def async_forward(
        self,
        carry,  # PoTAsyncCarry
        new_batch: dict,  # {'input': [B, 81], 'label': [B, 81], 'puzzle_id': [B]}
    ):
        """
        Async-compatible forward for HRM-style batching.
        
        This method:
        1. Replaces halted samples' data with new batch data
        2. Computes input embeddings
        3. Runs one ACT step (H_cycles x L_cycles inner iterations)
        4. Returns new carry + outputs
        
        Args:
            carry: Current PoTAsyncCarry state
            new_batch: Dict with 'input', 'label', 'puzzle_id' for new samples
            
        Returns:
            new_carry: Updated carry state
            outputs: Dict with 'logits', 'q_halt_logits', 'q_continue_logits', 
                    'target_q_continue', 'labels'
        """
        device = carry.z_H.device
        
        # Get new data from batch
        new_input = new_batch['input'].to(device)
        new_labels = new_batch['label'].to(device)
        new_puzzle_ids = new_batch['puzzle_id'].to(device)
        
        # Reset carry for halted samples, replacing their data (including puzzle_ids)
        carry = self.reset_async_carry(
            carry, 
            carry.halted,
            new_input,
            new_labels,
            new_puzzle_ids,
            device,
        )
        
        # Compute input embedding for current samples
        # Use carry.current_input and carry.current_puzzle_ids which now have correct data
        input_emb = self._compute_input_embedding(carry.current_input, carry.current_puzzle_ids)
        
        # Run one ACT step
        new_carry, outputs = self.forward_single_step_async(carry, input_emb)
        
        # Compute logits from hidden state
        logits = self.output_proj(outputs['hidden'])
        
        # Add logits and labels to outputs
        outputs['logits'] = logits
        outputs['labels'] = new_carry.current_labels
        
        return new_carry, outputs


class BaselineSudokuSolver(nn.Module):
    """
    Standard Transformer baseline for Sudoku.
    
    Simple encoder-only transformer without iterative refinement.
    Used for comparison with PoH/HRM architectures.
    
    Args:
        vocab_size: Vocabulary size
        d_model: Hidden dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feedforward dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        vocab_size: int = 10,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.seq_len = 81
        
        self.input_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_seq: torch.Tensor, puzzle_ids: torch.Tensor = None):
        """Forward pass."""
        x = self.input_embed(input_seq) + self.pos_embed
        x = self.encoder(x)
        x = self.final_norm(x)
        logits = self.output_proj(x)
        return logits, None, None, 1

