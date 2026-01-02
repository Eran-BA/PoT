"""
Sokoban Solver Models.

PoT-based models for Sokoban puzzle solving:
1. PoTSokobanSolver - PoT with CausalDepthTransformerRouter + Conv encoder
2. BaselineSokobanSolver - Standard CNN policy baseline

The model predicts action logits given current board state.
Uses convolutional encoder for grid input, then PoT refinement on flattened tokens.

State representation:
- 10x10 board with 7 tile types (one-hot encoding: [B, 10, 10, 7])
- Output: action logits [B, 4] and value [B]

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any, List

from src.pot.core.controller_factory import create_controller
from src.pot.core.depth_transformer_controller import DepthControllerCache


# =============================================================================
# Constants
# =============================================================================

NUM_TILE_TYPES = 7
BOARD_HEIGHT = 10
BOARD_WIDTH = 10
NUM_ACTIONS = 4


# =============================================================================
# Conv Encoder for Grid Input
# =============================================================================

class SokobanConvEncoder(nn.Module):
    """
    Convolutional encoder for Sokoban grid input.
    
    Processes 10x10x7 one-hot board into spatial features.
    
    Args:
        d_model: Output feature dimension
        n_filters: Number of conv filters per layer
        n_layers: Number of conv layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_filters: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Build conv stack
        layers = []
        in_channels = NUM_TILE_TYPES
        
        for i in range(n_layers):
            out_channels = n_filters * (2 ** min(i, 2))  # 64, 128, 256...
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.GELU())
            layers.append(nn.Dropout2d(dropout))
            in_channels = out_channels
        
        self.conv_stack = nn.Sequential(*layers)
        
        # Project to d_model
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode board to spatial features.
        
        Args:
            x: [B, H, W, 7] one-hot board (channels last)
        
        Returns:
            [B, H, W, d_model] spatial features
        """
        # Convert to channels first: [B, 7, H, W]
        x = x.permute(0, 3, 1, 2)
        
        # Apply conv stack
        x = self.conv_stack(x)
        
        # Project to d_model
        x = self.proj(x)  # [B, d_model, H, W]
        
        # Convert back to channels last: [B, H, W, d_model]
        x = x.permute(0, 2, 3, 1)
        
        return x


# =============================================================================
# PoT Sokoban Solver
# =============================================================================

class PoTSokobanSolver(nn.Module):
    """
    PoT-based Sokoban solver with convolutional encoder and transformer refinement.
    
    Architecture:
        1. Conv encoder: 10x10x7 -> 10x10xD
        2. Flatten to 100 tokens: [B, 100, D]
        3. PoT refinement with CausalDepthTransformerRouter: R iterations
        4. Action head: pool -> 4 logits
        5. Value head: pool -> 1 scalar
    
    Args:
        d_model: Hidden dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feedforward dimension
        dropout: Dropout rate
        R: Number of refinement iterations
        conv_layers: Number of conv layers in encoder
        conv_filters: Number of filters per conv layer
        controller_type: Type of depth controller
        controller_kwargs: Additional kwargs for controller
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        R: int = 4,
        conv_layers: int = 3,
        conv_filters: int = 64,
        controller_type: str = "transformer",
        controller_kwargs: Optional[Dict] = None,
        max_depth: int = None,
        # Dynamic board size (default to constants for backward compat)
        board_height: int = None,
        board_width: int = None,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.R = R
        self.n_heads = n_heads
        self.board_height = board_height or BOARD_HEIGHT
        self.board_width = board_width or BOARD_WIDTH
        self.seq_len = self.board_height * self.board_width
        
        # Conv encoder
        self.conv_encoder = SokobanConvEncoder(
            d_model=d_model,
            n_filters=conv_filters,
            n_layers=conv_layers,
            dropout=dropout,
        )
        
        # Positional embedding for flattened grid
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, d_model) * 0.02)
        
        # Transformer refinement layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # PoT depth controller
        # max_depth defaults to max(R, 16) if not specified
        effective_max_depth = max_depth if max_depth is not None else max(R, 16)
        ctrl_kwargs = controller_kwargs or {}
        self.controller = create_controller(
            controller_type=controller_type,
            d_model=d_model,
            n_heads=n_heads,
            max_depth=effective_max_depth,
            **ctrl_kwargs,
        )
        
        # Layer norm for refinement
        self.refine_norm = nn.LayerNorm(d_model)
        
        # Output heads
        self.action_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, NUM_ACTIONS),
        )
        
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        
        # Q-head for ACT halt/continue prediction (same as HybridPoHHRMSolver)
        self.q_head = nn.Linear(d_model, 2)  # [q_halt, q_continue]
        
    def forward(
        self,
        board: torch.Tensor,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass for action prediction.
        
        Args:
            board: [B, H, W, 7] one-hot encoded board
            return_aux: If True, return auxiliary info
        
        Returns:
            action_logits: [B, 4] action logits
            value: [B] state value estimate
            q_halt: [B] Q-value for halting (ACT)
            q_continue: [B] Q-value for continuing (ACT)
            aux: Dict with auxiliary metrics (alphas, entropies, etc.)
        """
        B = board.shape[0]
        
        # Encode board
        spatial = self.conv_encoder(board)  # [B, H, W, d_model]
        
        # Flatten to tokens
        x = spatial.view(B, self.seq_len, self.d_model)  # [B, 100, d_model]
        x = x + self.pos_embed
        
        # Initial encoding
        x = self.encoder(x)
        
        # PoT refinement iterations
        cache = None
        alphas = []
        entropies = []
        
        for t in range(self.R):
            # Get routing weights
            alpha_t, cache, step_aux = self.controller.step(x, t, cache, return_aux=True)
            alphas.append(alpha_t)
            if 'entropy' in step_aux:
                entropies.append(step_aux['entropy'])
            
            # Apply head-weighted attention (simplified: just run encoder again)
            # In full implementation, would route through different heads
            x = self.encoder(x)
            x = self.refine_norm(x)
        
        # Pool sequence for output heads
        x_pool = x.mean(dim=1)  # [B, d_model]
        
        # Action and value predictions
        action_logits = self.action_head(x_pool)  # [B, 4]
        value = self.value_head(x_pool).squeeze(-1)  # [B]
        
        # Q-values for ACT halt/continue (same pattern as HybridPoHHRMSolver)
        q_logits = self.q_head(x_pool)  # [B, 2]
        q_halt = q_logits[:, 0]  # [B]
        q_continue = q_logits[:, 1]  # [B]
        
        aux = {}
        if return_aux:
            aux['alphas'] = alphas
            aux['entropies'] = entropies if entropies else None
            aux['num_steps'] = self.R
        
        return action_logits, value, q_halt, q_continue, aux
    
    def predict_action(
        self,
        board: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Predict action from board state.
        
        Args:
            board: [B, H, W, 7] one-hot encoded board
            legal_mask: [B, 4] mask of legal actions (1=legal, 0=illegal)
            temperature: Softmax temperature for sampling
        
        Returns:
            [B] predicted actions
        """
        action_logits, _, _, _, _ = self.forward(board, return_aux=False)
        
        # Apply legal mask if provided
        if legal_mask is not None:
            # Set illegal actions to very negative
            action_logits = action_logits.masked_fill(~legal_mask.bool(), float('-inf'))
        
        # Sample or argmax
        if temperature > 0:
            probs = F.softmax(action_logits / temperature, dim=-1)
            action = torch.multinomial(probs, 1).squeeze(-1)
        else:
            action = action_logits.argmax(dim=-1)
        
        return action
    
    def get_action_probs(
        self,
        board: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get action probabilities.
        
        Args:
            board: [B, H, W, 7] one-hot encoded board
            legal_mask: [B, 4] mask of legal actions
        
        Returns:
            [B, 4] action probabilities
        """
        action_logits, _, _, _, _ = self.forward(board, return_aux=False)
        
        if legal_mask is not None:
            action_logits = action_logits.masked_fill(~legal_mask.bool(), float('-inf'))
        
        return F.softmax(action_logits, dim=-1)


# =============================================================================
# Baseline CNN Solver
# =============================================================================

class BaselineSokobanSolver(nn.Module):
    """
    Standard CNN baseline for Sokoban (no iterative refinement).
    
    Simple conv net that directly predicts action logits and value.
    
    Args:
        n_filters: Number of conv filters per layer
        n_layers: Number of conv layers
        d_hidden: Hidden dimension after conv
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        n_filters: int = 64,
        n_layers: int = 4,
        d_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Build conv stack
        layers = []
        in_channels = NUM_TILE_TYPES
        
        for i in range(n_layers):
            out_channels = n_filters * (2 ** min(i, 2))
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.GELU())
            layers.append(nn.Dropout2d(dropout))
            in_channels = out_channels
        
        self.conv_stack = nn.Sequential(*layers)
        
        # Global average pooling + FC
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Output heads
        self.action_head = nn.Linear(d_hidden, NUM_ACTIONS)
        self.value_head = nn.Linear(d_hidden, 1)
        
    def forward(
        self,
        board: torch.Tensor,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass.
        
        Args:
            board: [B, H, W, 7] one-hot encoded board
            return_aux: If True, return auxiliary info
        
        Returns:
            action_logits: [B, 4] action logits
            value: [B] state value estimate
            aux: Dict with auxiliary metrics
        """
        # Convert to channels first
        x = board.permute(0, 3, 1, 2)  # [B, 7, H, W]
        
        # Conv stack
        x = self.conv_stack(x)  # [B, C, H, W]
        
        # Pool and FC
        x = self.pool(x).squeeze(-1).squeeze(-1)  # [B, C]
        x = self.fc(x)  # [B, d_hidden]
        
        # Output heads
        action_logits = self.action_head(x)  # [B, 4]
        value = self.value_head(x).squeeze(-1)  # [B]
        
        aux = {'num_steps': 1} if return_aux else {}
        
        return action_logits, value, aux
    
    def predict_action(
        self,
        board: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Predict action from board state."""
        action_logits, _, _ = self.forward(board, return_aux=False)
        
        if legal_mask is not None:
            action_logits = action_logits.masked_fill(~legal_mask.bool(), float('-inf'))
        
        if temperature > 0:
            probs = F.softmax(action_logits / temperature, dim=-1)
            action = torch.multinomial(probs, 1).squeeze(-1)
        else:
            action = action_logits.argmax(dim=-1)
        
        return action
    
    def get_action_probs(
        self,
        board: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get action probabilities."""
        action_logits, _, _ = self.forward(board, return_aux=False)
        
        if legal_mask is not None:
            action_logits = action_logits.masked_fill(~legal_mask.bool(), float('-inf'))
        
        return F.softmax(action_logits, dim=-1)


# =============================================================================
# Hybrid PoT Sokoban Solver (HybridHRMBase)
# =============================================================================

from src.pot.models.hybrid_hrm import HybridHRMBase
from src.pot.models.hrm_layers import RMSNorm, SwiGLU


class HybridPoTSokobanSolver(HybridHRMBase):
    """
    Hybrid PoT Sokoban solver with two-timescale reasoning (aligned with Sudoku).
    
    Uses the HybridHRMBase architecture for iterative refinement:
    - L_level: Fast reasoning, updates every inner step
    - H_level: Slow reasoning, updates every outer step
    - Both use PoT head routing via CausalDepthTransformerRouter
    - ACT: Adaptive computation time with Q-learning
    
    Architecture:
        1. Conv encoder: 10x10x7 -> 100 tokens of d_model
        2. HybridHRM two-timescale refinement
        3. Action head: pool -> 4 logits
        4. Value head: pool -> 1 scalar
    
    Args:
        d_model: Hidden dimension
        n_heads: Number of attention heads
        H_layers: Layers in H_level module
        L_layers: Layers in L_level module
        d_ff: Feedforward dimension
        dropout: Dropout rate
        H_cycles: Fixed outer loop iterations per ACT step
        L_cycles: Fixed inner loop iterations per H_cycle
        T: HRM period for pointer controller
        conv_layers: Number of conv layers in encoder
        conv_filters: Number of filters per conv layer
        controller_type: Type of depth controller
        controller_kwargs: Additional kwargs for controller
        hrm_grad_style: If True, only last L+H get gradients
        halt_max_steps: Maximum ACT outer steps (1 = no ACT)
        halt_exploration_prob: Exploration probability for Q-learning
        allow_early_halt_eval: Enable early halting during eval
        injection_mode: Feature injection mode
        injection_kwargs: Additional kwargs for FeatureInjector
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        H_layers: int = 2,
        L_layers: int = 2,
        d_ff: int = 1024,
        dropout: float = 0.0,
        H_cycles: int = 2,
        L_cycles: int = 8,
        T: int = 4,
        conv_layers: int = 3,
        conv_filters: int = 64,
        controller_type: str = "transformer",
        controller_kwargs: dict = None,
        # Aligned with Sudoku's HybridPoHHRMSolver
        hrm_grad_style: bool = False,
        halt_max_steps: int = 1,
        halt_exploration_prob: float = 0.1,
        allow_early_halt_eval: bool = False,
        injection_mode: str = "none",
        injection_kwargs: dict = None,
        # Dynamic board size (default to constants for backward compat)
        board_height: int = None,
        board_width: int = None,
    ):
        # Sequence length from board size (dynamic or default)
        self.board_height = board_height or BOARD_HEIGHT
        self.board_width = board_width or BOARD_WIDTH
        seq_len = self.board_height * self.board_width
        
        # Initialize base class with ALL parameters (aligned with Sudoku)
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            H_layers=H_layers,
            L_layers=L_layers,
            d_ff=d_ff,
            seq_len=seq_len,
            H_cycles=H_cycles,
            L_cycles=L_cycles,
            dropout=dropout,
            T=T,
            halt_max_steps=halt_max_steps,
            controller_type=controller_type,
            controller_kwargs=controller_kwargs,
            hrm_grad_style=hrm_grad_style,
            halt_exploration_prob=halt_exploration_prob,
            allow_early_halt_eval=allow_early_halt_eval,
            injection_mode=injection_mode,
            injection_kwargs=injection_kwargs,
        )
        
        self.d_model = d_model
        embed_init_std = 1.0 / self.embed_scale
        
        # Conv encoder for board input
        self.conv_encoder = SokobanConvEncoder(
            d_model=d_model,
            n_filters=conv_filters,
            n_layers=conv_layers,
            dropout=dropout,
        )
        
        # Position embedding for 100 tokens
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * embed_init_std)
        
        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, NUM_ACTIONS),
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
    
    def _compute_input_embedding(self, board: torch.Tensor) -> torch.Tensor:
        """
        Compute scaled input embedding from board.
        
        Args:
            board: [B, H, W, 7] one-hot encoded board
        
        Returns:
            [B, 100, d_model] input embeddings
        """
        # Conv encode: [B, H, W, 7] -> [B, H, W, d_model]
        spatial = self.conv_encoder(board)
        
        # Flatten to tokens: [B, H*W, d_model]
        B = spatial.size(0)
        tokens = spatial.view(B, -1, self.d_model)
        
        # Add position embedding
        tokens = tokens + self.pos_embed
        
        # Scale embeddings by sqrt(d_model) like HRM
        return self.embed_scale * tokens
    
    def forward(
        self,
        board: torch.Tensor,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Forward pass for action prediction.
        
        Args:
            board: [B, H, W, 7] one-hot encoded board
            return_aux: Whether to return auxiliary info
        
        Returns:
            action_logits: [B, 4] action logits
            value: [B] state value estimate
            q_halt: [B] Q-value for halting
            q_continue: [B] Q-value for continuing
            aux: Optional dict with auxiliary outputs
        """
        input_emb = self._compute_input_embedding(board)
        
        if self.halt_max_steps > 1:
            # Use ACT wrapper (like Sudoku HybridPoHHRMSolver)
            act_out = self.act_forward(input_emb)
            hidden = act_out['hidden']
            q_halt = act_out['q_halt']
            q_continue = act_out['q_continue']
            steps = act_out['steps']
            target_q_continue = act_out.get('target_q_continue')  # ACT Q-learning target
        else:
            # Simple reasoning loop
            hidden, q_halt, q_continue, steps = self.reasoning_loop(input_emb)
            target_q_continue = None
        
        # Pool over sequence
        pooled = hidden.mean(dim=1)  # [B, d_model]
        
        # Action and value heads
        action_logits = self.action_head(pooled)  # [B, 4]
        value = self.value_head(pooled).squeeze(-1)  # [B]
        
        aux = {'steps': steps, 'target_q_continue': target_q_continue} if return_aux else None
        return action_logits, value, q_halt, q_continue, aux
    
    def get_action_probs(
        self,
        board: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get action probabilities."""
        action_logits, _, _, _, _ = self.forward(board, return_aux=False)
        
        if legal_mask is not None:
            action_logits = action_logits.masked_fill(~legal_mask.bool(), float('-inf'))
        
        return F.softmax(action_logits, dim=-1)
    
    # ========== Async Batching Methods (HRM-style, identical to Sudoku) ==========
    
    def create_async_carry(self, batch_size: int, device: torch.device):
        """
        Create initial async carry for HRM-style async batching.
        
        Sokoban-specific: uses [B, H, W, 7] for current_input and [B] for labels.
        
        All samples start as 'halted' so they will receive data on first forward call.
        """
        from src.pot.models.hybrid_hrm import PoTAsyncCarry
        
        return PoTAsyncCarry(
            z_H=self.H_init.view(1, 1, -1).expand(batch_size, self.seq_len, -1).clone(),
            z_L=self.L_init.view(1, 1, -1).expand(batch_size, self.seq_len, -1).clone(),
            L_ptr_state=self.L_level.pointer_controller.init_state(batch_size, device),
            H_ptr_state=self.H_level.pointer_controller.init_state(batch_size, device),
            steps=torch.zeros(batch_size, dtype=torch.int32, device=device),
            halted=torch.ones(batch_size, dtype=torch.bool, device=device),
            # Sokoban: [B, H, W, 7] for board input
            current_input=torch.zeros(batch_size, self.board_height, self.board_width, NUM_TILE_TYPES, device=device),
            # Sokoban: [B] for action labels (not [B, seq_len])
            current_labels=torch.zeros(batch_size, dtype=torch.long, device=device),
            current_puzzle_ids=torch.zeros(batch_size, dtype=torch.long, device=device),
        )
    
    def reset_async_carry_sokoban(
        self,
        carry,  # PoTAsyncCarry
        halted_mask: torch.Tensor,
        new_input: torch.Tensor,  # [N, H, W, 7]
        new_labels: torch.Tensor,  # [N]
        device: torch.device,
    ):
        """
        Reset hidden states for halted samples and replace their data.
        Sokoban-specific: handles [B, H, W, 7] input shape.
        """
        from src.pot.models.hybrid_hrm import PoTAsyncCarry
        
        B = carry.z_H.size(0)
        replace_mask = halted_mask
        num_to_replace = min(replace_mask.sum().item(), new_input.size(0))
        
        # Get indices of samples to replace
        halted_indices = torch.where(replace_mask)[0]
        
        # Samples that won't get data remain halted
        still_halted_mask = replace_mask.clone()
        if num_to_replace > 0:
            still_halted_mask[halted_indices[:num_to_replace]] = False
        
        # Reset hidden states for replaced samples
        mask_expanded = replace_mask.view(B, 1, 1).expand_as(carry.z_H)
        fresh_z_H = self.H_init.view(1, 1, -1).expand(B, self.seq_len, -1)
        fresh_z_L = self.L_init.view(1, 1, -1).expand(B, self.seq_len, -1)
        
        new_z_H = torch.where(mask_expanded, fresh_z_H, carry.z_H)
        new_z_L = torch.where(mask_expanded, fresh_z_L, carry.z_L)
        
        # Reset steps
        new_steps = torch.where(replace_mask, torch.zeros_like(carry.steps), carry.steps)
        
        # Replace input/labels
        new_current_input = carry.current_input.clone()
        new_current_labels = carry.current_labels.clone()
        if num_to_replace > 0:
            replace_indices = halted_indices[:num_to_replace]
            new_current_input[replace_indices] = new_input[:num_to_replace]
            new_current_labels[replace_indices] = new_labels[:num_to_replace]
        
        # Reset pointer states
        fresh_L_ptr = self.L_level.pointer_controller.init_state(B, device)
        fresh_H_ptr = self.H_level.pointer_controller.init_state(B, device)
        new_L_ptr = self._reset_ptr_state(carry.L_ptr_state, fresh_L_ptr, replace_mask)
        new_H_ptr = self._reset_ptr_state(carry.H_ptr_state, fresh_H_ptr, replace_mask)
        
        return PoTAsyncCarry(
            z_H=new_z_H,
            z_L=new_z_L,
            L_ptr_state=new_L_ptr,
            H_ptr_state=new_H_ptr,
            steps=new_steps,
            halted=still_halted_mask,
            current_input=new_current_input,
            current_labels=new_current_labels,
            current_puzzle_ids=carry.current_puzzle_ids,  # Keep unchanged
        )
    
    def async_forward(
        self,
        carry,  # PoTAsyncCarry
        new_batch: dict,  # {'input': [B, H, W, 7], 'label': [B]}
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
            new_batch: Dict with 'input' (board) and 'label' (action) for new samples
            
        Returns:
            new_carry: Updated carry state
            outputs: Dict with 'logits', 'q_halt_logits', 'q_continue_logits', 
                    'target_q_continue', 'labels'
        """
        device = carry.z_H.device
        
        # Get new data from batch
        new_input = new_batch['input'].to(device)
        new_labels = new_batch['label'].to(device)
        
        # Reset carry for halted samples (Sokoban-specific reset)
        carry = self.reset_async_carry_sokoban(
            carry, 
            carry.halted,
            new_input,
            new_labels,
            device,
        )
        
        # Compute input embedding for current samples
        input_emb = self._compute_input_embedding(carry.current_input)
        
        # Run one ACT step
        new_carry, outputs = self.forward_single_step_async(carry, input_emb)
        
        # Pool over sequence for action prediction
        hidden = outputs['hidden']
        pooled = hidden.mean(dim=1)  # [B, d_model]
        
        # Compute action logits (not full grid logits like Sudoku)
        action_logits = self.action_head(pooled)  # [B, 4]
        
        # Add logits and labels to outputs
        outputs['logits'] = action_logits
        outputs['labels'] = new_carry.current_labels
        
        return new_carry, outputs


# =============================================================================
# Actor-Critic Wrapper for PPO
# =============================================================================

class SokobanActorCritic(nn.Module):
    """
    Combined Actor-Critic for PPO training.
    
    Wraps either PoT or Baseline solver to provide:
    - Actor: action distribution
    - Critic: value estimate
    - Q-halt/Q-continue: ACT halting predictions
    
    Args:
        model: PoTSokobanSolver or BaselineSokobanSolver
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    
    def forward(
        self,
        board: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action distribution, value, and Q-values.
        
        Args:
            board: [B, H, W, 7] one-hot encoded board
            legal_mask: [B, 4] mask of legal actions
        
        Returns:
            action_probs: [B, 4] action probabilities
            action_logits: [B, 4] raw logits
            value: [B] state value estimate
            q_halt: [B] Q-value for halting
            q_continue: [B] Q-value for continuing
        """
        # Handle both PoT (5 returns) and Baseline (3 returns) models
        outputs = self.model(board, return_aux=False)
        if len(outputs) == 5:
            action_logits, value, q_halt, q_continue, _ = outputs
        else:
            # Baseline model doesn't have q_halt/q_continue
            action_logits, value, _ = outputs
            q_halt = torch.zeros(board.size(0), device=board.device)
            q_continue = torch.zeros(board.size(0), device=board.device)
        
        if legal_mask is not None:
            action_logits = action_logits.masked_fill(~legal_mask.bool(), float('-inf'))
        
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_probs, action_logits, value, q_halt, q_continue
    
    def get_value(self, board: torch.Tensor) -> torch.Tensor:
        """Get value estimate only."""
        outputs = self.model(board, return_aux=False)
        if len(outputs) == 5:
            _, value, _, _, _ = outputs
        else:
            _, value, _ = outputs
        return value
    
    def evaluate_actions(
        self,
        board: torch.Tensor,
        actions: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Args:
            board: [B, H, W, 7] one-hot encoded board
            actions: [B] actions taken
            legal_mask: [B, 4] mask of legal actions
        
        Returns:
            log_probs: [B] log probabilities of actions
            entropy: [B] entropy of action distribution
            value: [B] state value estimate
            q_halt: [B] Q-value for halting
            q_continue: [B] Q-value for continuing
        """
        action_probs, action_logits, value, q_halt, q_continue = self.forward(board, legal_mask)
        
        # Compute log probs for taken actions
        dist = torch.distributions.Categorical(probs=action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy, value, q_halt, q_continue

