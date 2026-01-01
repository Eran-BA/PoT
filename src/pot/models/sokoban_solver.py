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
    ):
        super().__init__()
        
        self.d_model = d_model
        self.R = R
        self.n_heads = n_heads
        self.seq_len = BOARD_HEIGHT * BOARD_WIDTH  # 100 tokens
        
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
        
    def forward(
        self,
        board: torch.Tensor,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass for action prediction.
        
        Args:
            board: [B, H, W, 7] one-hot encoded board
            return_aux: If True, return auxiliary info
        
        Returns:
            action_logits: [B, 4] action logits
            value: [B] state value estimate
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
        
        aux = {}
        if return_aux:
            aux['alphas'] = alphas
            aux['entropies'] = entropies if entropies else None
            aux['num_steps'] = self.R
        
        return action_logits, value, aux
    
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
        action_logits, _, _ = self.forward(board, return_aux=False)
        
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
        action_logits, _, _ = self.forward(board, return_aux=False)
        
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
# Actor-Critic Wrapper for PPO
# =============================================================================

class SokobanActorCritic(nn.Module):
    """
    Combined Actor-Critic for PPO training.
    
    Wraps either PoT or Baseline solver to provide:
    - Actor: action distribution
    - Critic: value estimate
    
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action distribution and value.
        
        Args:
            board: [B, H, W, 7] one-hot encoded board
            legal_mask: [B, 4] mask of legal actions
        
        Returns:
            action_probs: [B, 4] action probabilities
            action_logits: [B, 4] raw logits
            value: [B] state value estimate
        """
        action_logits, value, _ = self.model(board, return_aux=False)
        
        if legal_mask is not None:
            action_logits = action_logits.masked_fill(~legal_mask.bool(), float('-inf'))
        
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_probs, action_logits, value
    
    def get_value(self, board: torch.Tensor) -> torch.Tensor:
        """Get value estimate only."""
        _, value, _ = self.model(board, return_aux=False)
        return value
    
    def evaluate_actions(
        self,
        board: torch.Tensor,
        actions: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        """
        action_probs, action_logits, value = self.forward(board, legal_mask)
        
        # Compute log probs for taken actions
        dist = torch.distributions.Categorical(probs=action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy, value

