"""
Blocksworld Critic Network for PPO Training.

This module provides a value function estimator V(state, goal) for PPO training.
The critic estimates the expected cumulative reward from a given state toward
a goal state, used to compute advantages for policy gradient updates.

Architecture:
- Shares embedding structure with actor (SimplePoTBlocksworldSolver)
- Simpler transformer encoder (no iterative refinement needed)
- Outputs scalar value estimate

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any


class BlocksworldCritic(nn.Module):
    """
    Value function network for Blocksworld PPO.
    
    Estimates V(state, goal) - the expected return from the current state
    toward the goal state. Used to compute advantages in PPO.
    
    Architecture:
    - State embedding (matches actor for potential weight sharing)
    - Goal embedding (goal-conditioned value)
    - Simple transformer encoder
    - Pooling and value head
    
    Args:
        num_blocks: Maximum number of blocks (N)
        d_model: Hidden dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feedforward dimension
        dropout: Dropout rate
        pool_method: How to pool sequence for value ('mean', 'first', 'last')
    """
    
    def __init__(
        self,
        num_blocks: int = 8,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        pool_method: str = 'mean',
    ):
        super().__init__()
        
        self.num_blocks = num_blocks
        self.vocab_size = num_blocks + 2  # table + blocks + holding
        self.d_model = d_model
        self.pool_method = pool_method
        
        # State embedding
        self.state_embed = nn.Embedding(self.vocab_size, d_model)
        
        # Block identity embedding
        self.block_embed = nn.Embedding(num_blocks, d_model)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_blocks, d_model) * 0.02)
        
        # Goal embedding
        self.goal_embed = nn.Embedding(self.vocab_size, d_model)
        self.goal_proj = nn.Linear(d_model, d_model)
        
        # State-goal difference encoding
        self.diff_proj = nn.Linear(d_model, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Final normalization
        self.final_norm = nn.LayerNorm(d_model)
        
        # Value head: pool -> hidden -> value
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate value for (state, goal) pair.
        
        Args:
            state: [B, N] current block positions
            goal: [B, N] goal block positions
        
        Returns:
            value: [B] estimated value for each sample
        """
        B, N = state.shape
        device = state.device
        
        # Block indices
        block_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        
        # Embed state
        state_emb = self.state_embed(state)  # [B, N, d_model]
        block_emb = self.block_embed(block_ids)  # [B, N, d_model]
        pos_emb = self.pos_embed[:, :N]  # [1, N, d_model]
        
        x = state_emb + block_emb + pos_emb
        
        # Embed goal and project
        goal_emb = self.goal_embed(goal)  # [B, N, d_model]
        goal_proj = self.goal_proj(goal_emb)
        
        # Combine with goal information
        x = x + 0.5 * goal_proj
        
        # Add difference information (helps with distance estimation)
        # Where state differs from goal
        diff_mask = (state != goal).float().unsqueeze(-1)  # [B, N, 1]
        diff_emb = self.diff_proj(goal_emb - state_emb) * diff_mask
        x = x + 0.3 * diff_emb
        
        # Encode
        x = self.encoder(x)  # [B, N, d_model]
        x = self.final_norm(x)
        
        # Pool to get single representation
        if self.pool_method == 'mean':
            pooled = x.mean(dim=1)  # [B, d_model]
        elif self.pool_method == 'first':
            pooled = x[:, 0]
        elif self.pool_method == 'last':
            pooled = x[:, -1]
        else:
            pooled = x.mean(dim=1)
        
        # Value prediction
        value = self.value_head(pooled).squeeze(-1)  # [B]
        
        return value


class BlocksworldActorCritic(nn.Module):
    """
    Combined Actor-Critic network for Blocksworld PPO.
    
    Combines:
    - Actor: SimplePoTBlocksworldSolver (predicts next state)
    - Critic: BlocksworldCritic (estimates value)
    
    Can optionally share the embedding layers between actor and critic.
    
    Args:
        actor: The actor network (SimplePoTBlocksworldSolver)
        critic: The critic network (BlocksworldCritic)
        share_embeddings: Whether to share embeddings between actor/critic
    """
    
    def __init__(
        self,
        actor: nn.Module,
        critic: BlocksworldCritic,
        share_embeddings: bool = False,
    ):
        super().__init__()
        
        self.actor = actor
        self.critic = critic
        self.share_embeddings = share_embeddings
        
        if share_embeddings:
            # Share embedding weights
            self.critic.state_embed = self.actor.state_embed
            self.critic.block_embed = self.actor.block_embed
            self.critic.goal_embed = self.actor.goal_embed
    
    def forward(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass for both actor and critic.
        
        Args:
            state: [B, N] current state
            goal: [B, N] goal state
        
        Returns:
            logits: [B, N, vocab_size] action logits from actor
            value: [B] value estimates from critic
            aux: Dictionary with auxiliary outputs
        """
        # Actor forward
        logits, aux = self.actor(state, goal)
        
        # Critic forward
        value = self.critic(state, goal)
        
        return logits, value, aux
    
    def get_action_and_value(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probabilities, entropy, and value.
        
        Used during PPO rollouts and updates.
        
        Args:
            state: [B, N] current state
            goal: [B, N] goal state
            action: [B, N] optional actions to evaluate (for PPO update)
        
        Returns:
            action: [B, N] sampled or provided actions
            log_prob: [B] log probabilities of actions
            entropy: [B] entropy of action distribution
            value: [B] value estimates
        """
        # Get logits and value
        logits, value, _ = self.forward(state, goal)
        
        # Create categorical distribution over positions for each block
        # logits: [B, N, vocab_size]
        probs = F.softmax(logits, dim=-1)  # [B, N, vocab_size]
        
        # Sample or use provided action
        if action is None:
            # Sample action for each block position
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()  # [B, N]
        
        # Compute log probabilities
        # log_prob for each position, then sum over blocks
        log_probs = torch.log(probs + 1e-8)  # [B, N, vocab_size]
        action_log_probs = torch.gather(
            log_probs, dim=-1, index=action.unsqueeze(-1)
        ).squeeze(-1)  # [B, N]
        total_log_prob = action_log_probs.sum(dim=-1)  # [B]
        
        # Compute entropy for each position, then mean over blocks
        entropy = -(probs * log_probs).sum(dim=-1).mean(dim=-1)  # [B]
        
        return action, total_log_prob, entropy, value


def create_actor_critic(
    num_blocks: int = 8,
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 1024,
    dropout: float = 0.1,
    R: int = 4,
    controller_type: str = 'transformer',
    share_embeddings: bool = False,
) -> BlocksworldActorCritic:
    """
    Factory function to create Actor-Critic for Blocksworld.
    
    Args:
        num_blocks: Maximum number of blocks
        d_model: Hidden dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers (for actor)
        d_ff: Feedforward dimension
        dropout: Dropout rate
        R: Number of refinement iterations for actor
        controller_type: Controller type for actor
        share_embeddings: Whether to share embeddings
    
    Returns:
        BlocksworldActorCritic instance
    """
    from src.pot.models.blocksworld_solver import SimplePoTBlocksworldSolver
    
    # Create actor
    actor = SimplePoTBlocksworldSolver(
        num_blocks=num_blocks,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        R=R,
        controller_type=controller_type,
        goal_conditioned=True,
    )
    
    # Create critic (simpler architecture)
    critic = BlocksworldCritic(
        num_blocks=num_blocks,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=max(1, n_layers // 2),  # Critic is simpler
        d_ff=d_ff // 2,
        dropout=dropout,
    )
    
    return BlocksworldActorCritic(actor, critic, share_embeddings)

