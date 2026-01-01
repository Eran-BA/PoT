"""
PPO Trainer for Blocksworld.

Implements Proximal Policy Optimization (PPO) for training the PoT Blocksworld
solver with contrastive good/bad trajectory learning.

Key features:
- Clipped PPO objective for stable policy updates
- Generalized Advantage Estimation (GAE)
- Entropy bonus for exploration
- Value function estimation with critic network
- Good trajectory reward (+1) / Bad trajectory reward (-1)

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    # PPO hyperparameters
    clip_epsilon: float = 0.2
    clip_value: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # GAE parameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Training
    ppo_epochs: int = 4
    minibatch_size: int = 64
    normalize_advantages: bool = True
    
    # Target KL for early stopping
    target_kl: Optional[float] = 0.01


@dataclass
class RolloutBuffer:
    """Buffer to store rollout data for PPO updates."""
    states: List[torch.Tensor]
    goals: List[torch.Tensor]
    actions: List[torch.Tensor]
    log_probs: List[torch.Tensor]
    values: List[torch.Tensor]
    rewards: List[torch.Tensor]
    dones: List[torch.Tensor]
    is_valid: List[torch.Tensor]  # Track good vs bad trajectories
    
    def __init__(self):
        self.states = []
        self.goals = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.is_valid = []
    
    def add(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        is_valid: torch.Tensor,
    ):
        """Add a step to the buffer."""
        self.states.append(state)
        self.goals.append(goal)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.is_valid.append(is_valid)
    
    def clear(self):
        """Clear all stored data."""
        self.__init__()
    
    def get_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Convert lists to tensors."""
        return {
            'states': torch.stack(self.states).to(device),
            'goals': torch.stack(self.goals).to(device),
            'actions': torch.stack(self.actions).to(device),
            'log_probs': torch.stack(self.log_probs).to(device),
            'values': torch.stack(self.values).to(device),
            'rewards': torch.stack(self.rewards).to(device),
            'dones': torch.stack(self.dones).to(device),
            'is_valid': torch.stack(self.is_valid).to(device),
        }
    
    def __len__(self):
        return len(self.states)


class BlocksworldPPOTrainer:
    """
    PPO Trainer for Blocksworld using good/bad trajectory contrastive learning.
    
    Training flow:
    1. Sample batch of (state, goal, is_valid, reward) from PPO dataset
    2. Actor predicts next state (all positions at once)
    3. Compute reward: +1 for valid trajectories, -1 for invalid
    4. Update using PPO clipped objective
    
    Args:
        actor_critic: Combined actor-critic network
        config: PPO configuration
        lr_actor: Learning rate for actor
        lr_critic: Learning rate for critic
        device: Device to use (cpu/cuda/mps)
    """
    
    def __init__(
        self,
        actor_critic: nn.Module,
        config: PPOConfig = None,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        device: str = 'cpu',
    ):
        self.actor_critic = actor_critic.to(device)
        self.config = config or PPOConfig()
        self.device = torch.device(device)
        
        # Separate optimizers for actor and critic
        actor_params = list(self.actor_critic.actor.parameters())
        critic_params = list(self.actor_critic.critic.parameters())
        
        self.optimizer = torch.optim.AdamW([
            {'params': actor_params, 'lr': lr_actor},
            {'params': critic_params, 'lr': lr_critic},
        ], weight_decay=0.01)
        
        # Metrics tracking
        self.training_stats = defaultdict(list)
    
    def compute_reward(
        self,
        predicted_state: torch.Tensor,
        target_goal: torch.Tensor,
        is_valid: torch.Tensor,
        num_blocks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute reward for predicted state.
        
        Reward structure:
        - Invalid trajectory: -1.0 (regardless of prediction)
        - Valid + matches goal: +1.0
        - Valid + partial match: distance-based (0 to 1)
        
        Args:
            predicted_state: [B, N] predicted block positions
            target_goal: [B, N] target goal state
            is_valid: [B] whether trajectory is valid
            num_blocks: [B] number of blocks per sample (for proper distance calc)
        
        Returns:
            reward: [B] reward for each sample
        """
        B, N = predicted_state.shape
        
        # Base reward from trajectory validity
        reward = torch.zeros(B, device=predicted_state.device)
        
        # Invalid trajectories get -1
        invalid_mask = ~is_valid
        reward[invalid_mask] = -1.0
        
        # Valid trajectories: reward based on distance to goal
        valid_mask = is_valid
        if valid_mask.any():
            # Compute slot accuracy (percentage of matching positions)
            matches = (predicted_state == target_goal).float()  # [B, N]
            
            if num_blocks is not None:
                # Mask out padded positions
                block_mask = torch.arange(N, device=predicted_state.device).unsqueeze(0) < num_blocks.unsqueeze(1)
                matches = matches * block_mask.float()
                slot_acc = matches.sum(dim=1) / num_blocks.float().clamp(min=1)
            else:
                slot_acc = matches.mean(dim=1)  # [B]
            
            # Reward for valid: +1 if perfect, scaled otherwise
            # Range: [0, 1] for valid trajectories
            reward[valid_mask] = slot_acc[valid_mask]
        
        return reward
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: [T, B] rewards at each step
            values: [T, B] value estimates at each step
            dones: [T, B] done flags
            next_value: [B] value estimate for next state
        
        Returns:
            advantages: [T, B] GAE advantages
            returns: [T, B] discounted returns
        """
        T, B = rewards.shape
        gamma = self.config.gamma
        gae_lambda = self.config.gae_lambda
        
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - dones[t].float()
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t].float()
                next_val = values[t + 1]
            
            delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values
        
        return advantages, returns
    
    def ppo_update(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Perform PPO update on a batch.
        
        Args:
            states: [B, N] states
            goals: [B, N] goals
            actions: [B, N] actions (predicted states)
            old_log_probs: [B] log probs from rollout
            advantages: [B] computed advantages
            returns: [B] computed returns
        
        Returns:
            Dictionary of loss metrics
        """
        # Get current policy outputs
        action, log_prob, entropy, value = self.actor_critic.get_action_and_value(
            states, goals, action=actions
        )
        
        # Normalize advantages
        if self.config.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO clipped objective
        ratio = torch.exp(log_prob - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss (clipped)
        value_loss = F.mse_loss(value, returns)
        
        # Entropy bonus
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = (
            policy_loss 
            + self.config.value_coef * value_loss 
            + self.config.entropy_coef * entropy_loss
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), 
                self.config.max_grad_norm
            )
        
        self.optimizer.step()
        
        # Compute approximate KL divergence
        with torch.no_grad():
            approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item(),
            'total_loss': total_loss.item(),
            'approx_kl': approx_kl,
            'clip_fraction': ((ratio - 1).abs() > self.config.clip_epsilon).float().mean().item(),
        }
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Perform a single training step on a batch.
        
        For Blocksworld PPO, we treat each (state, goal) pair as a single-step
        episode where the action is the predicted next state.
        
        Args:
            batch: Dictionary containing:
                - init_state: [B, N] initial states
                - goal_state: [B, N] goal states
                - is_valid: [B] validity flags
                - reward: [B] trajectory rewards (+1/-1)
                - num_blocks: [B] number of blocks
        
        Returns:
            Dictionary of metrics
        """
        states = batch['init_state'].to(self.device)
        goals = batch['goal_state'].to(self.device)
        is_valid = batch['is_valid'].to(self.device)
        target_reward = batch['reward'].to(self.device)
        num_blocks = batch.get('num_blocks', None)
        if num_blocks is not None:
            num_blocks = num_blocks.to(self.device)
        
        B = states.shape[0]
        
        # Forward pass: get action, log_prob, entropy, value
        with torch.no_grad():
            action, old_log_prob, _, old_value = self.actor_critic.get_action_and_value(
                states, goals
            )
        
        # Compute reward based on prediction quality and validity
        reward = self.compute_reward(action, goals, is_valid, num_blocks)
        
        # For single-step episodes, advantages = reward - value
        # (no future rewards to consider)
        advantages = reward - old_value.detach()
        returns = reward
        
        # PPO epochs
        all_metrics = defaultdict(list)
        
        for epoch in range(self.config.ppo_epochs):
            # Shuffle indices
            indices = torch.randperm(B, device=self.device)
            
            for start in range(0, B, self.config.minibatch_size):
                end = min(start + self.config.minibatch_size, B)
                mb_indices = indices[start:end]
                
                metrics = self.ppo_update(
                    states[mb_indices],
                    goals[mb_indices],
                    action[mb_indices],
                    old_log_prob[mb_indices],
                    advantages[mb_indices],
                    returns[mb_indices],
                )
                
                for k, v in metrics.items():
                    all_metrics[k].append(v)
                
                # Early stopping on KL
                if (self.config.target_kl is not None 
                    and metrics['approx_kl'] > 1.5 * self.config.target_kl):
                    break
        
        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        
        # Add additional metrics
        avg_metrics['mean_reward'] = reward.mean().item()
        avg_metrics['good_reward'] = reward[is_valid].mean().item() if is_valid.any() else 0
        avg_metrics['bad_reward'] = reward[~is_valid].mean().item() if (~is_valid).any() else 0
        
        # Track prediction accuracy
        with torch.no_grad():
            predicted = action
            slot_acc = (predicted == goals).float().mean().item()
            exact_match = (predicted == goals).all(dim=1).float().mean().item()
            avg_metrics['slot_accuracy'] = slot_acc
            avg_metrics['exact_match'] = exact_match
        
        return avg_metrics
    
    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int = 0,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader for training data
            epoch: Current epoch number
        
        Returns:
            Dictionary of average metrics for the epoch
        """
        self.actor_critic.train()
        
        epoch_metrics = defaultdict(list)
        
        for batch in dataloader:
            metrics = self.train_step(batch)
            for k, v in metrics.items():
                epoch_metrics[k].append(v)
        
        # Average over epoch
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        
        # Store in training stats
        for k, v in avg_metrics.items():
            self.training_stats[k].append(v)
        
        return avg_metrics
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate on validation/test set.
        
        Args:
            dataloader: DataLoader for evaluation data
        
        Returns:
            Dictionary of metrics
        """
        self.actor_critic.eval()
        
        all_metrics = defaultdict(list)
        
        for batch in dataloader:
            states = batch['init_state'].to(self.device)
            goals = batch['goal_state'].to(self.device)
            is_valid = batch.get('is_valid')
            if is_valid is not None:
                is_valid = is_valid.to(self.device)
            num_blocks = batch.get('num_blocks')
            if num_blocks is not None:
                num_blocks = num_blocks.to(self.device)
            
            # Get predictions
            logits, value, _ = self.actor_critic(states, goals)
            predicted = logits.argmax(dim=-1)
            
            # Compute metrics
            B, N = states.shape
            
            # Slot accuracy
            if num_blocks is not None:
                mask = torch.arange(N, device=states.device).unsqueeze(0) < num_blocks.unsqueeze(1)
                matches = (predicted == goals) & mask
                slot_acc = matches.sum(dim=1).float() / num_blocks.float().clamp(min=1)
            else:
                slot_acc = (predicted == goals).float().mean(dim=1)
            
            all_metrics['slot_accuracy'].append(slot_acc.mean().item())
            
            # Exact match
            exact = (predicted == goals).all(dim=1).float()
            all_metrics['exact_match'].append(exact.mean().item())
            
            # Value estimate
            all_metrics['mean_value'].append(value.mean().item())
        
        return {k: np.mean(v) for k, v in all_metrics.items()}
    
    def get_training_stats(self) -> Dict[str, List[float]]:
        """Get all training statistics."""
        return dict(self.training_stats)

