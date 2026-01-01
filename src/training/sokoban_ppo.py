"""
Sokoban PPO Training.

PPO fine-tuning for Sokoban solver:
- Environment: Custom Sokoban gym-style environment
- Reward shaping: +1 solved, -0.01 per step, -0.2 deadlock
- GAE for advantage estimation
- Clipped objective

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm

from src.data.sokoban import (
    SokobanDataset,
    load_boxoban_levels,
    board_to_onehot,
)
from src.data.sokoban_rules import (
    legal_actions,
    step,
    is_solved,
    is_deadlock,
    get_legal_action_list,
)
from src.pot.models.sokoban_solver import (
    PoTSokobanSolver,
    BaselineSokobanSolver,
    SokobanActorCritic,
)


# =============================================================================
# Config
# =============================================================================

@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    # Environment
    max_episode_steps: int = 200
    
    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Training
    n_envs: int = 8
    n_steps: int = 128  # Steps per rollout
    ppo_epochs: int = 4
    batch_size: int = 64
    learning_rate: float = 3e-4
    total_timesteps: int = 1_000_000
    
    # Evaluation
    eval_interval: int = 10_000
    eval_episodes: int = 100
    
    # Rewards
    reward_solve: float = 1.0
    reward_step: float = -0.01
    reward_deadlock: float = -0.2
    reward_push: float = 0.0  # Optional: bonus for pushing onto target


# =============================================================================
# Sokoban Environment
# =============================================================================

class SokobanEnv:
    """
    Simple Sokoban environment for PPO training.
    
    Args:
        levels: List of initial board states
        config: PPO configuration
        seed: Random seed
    """
    
    def __init__(
        self,
        levels: List[np.ndarray],
        config: PPOConfig,
        seed: int = 42,
    ):
        self.levels = levels
        self.config = config
        self.rng = np.random.default_rng(seed)
        
        self.board = None
        self.steps = 0
        self.level_idx = 0
        
    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset to a random level."""
        self.level_idx = self.rng.integers(0, len(self.levels))
        self.board = self.levels[self.level_idx].copy()
        self.steps = 0
        
        obs = board_to_onehot(self.board)
        info = {
            'legal_mask': legal_actions(self.board),
            'level_idx': self.level_idx,
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        
        Returns:
            obs: Next observation (one-hot board)
            reward: Reward signal
            terminated: True if episode ended (solved or deadlock)
            truncated: True if max steps reached
            info: Additional info
        """
        # Apply action
        new_board, was_legal = step(self.board, action)
        self.board = new_board
        self.steps += 1
        
        # Compute reward
        reward = self.config.reward_step
        
        # Check terminal conditions
        terminated = False
        truncated = False
        
        if is_solved(self.board):
            reward += self.config.reward_solve
            terminated = True
        elif is_deadlock(self.board):
            reward += self.config.reward_deadlock
            terminated = True
        elif self.steps >= self.config.max_episode_steps:
            truncated = True
        
        obs = board_to_onehot(self.board)
        info = {
            'legal_mask': legal_actions(self.board),
            'solved': is_solved(self.board),
            'deadlock': is_deadlock(self.board),
            'steps': self.steps,
        }
        
        return obs, reward, terminated, truncated, info


class VectorizedSokobanEnv:
    """
    Vectorized Sokoban environment for parallel rollouts.
    
    Args:
        levels: List of initial board states
        n_envs: Number of parallel environments
        config: PPO configuration
        seed: Random seed
    """
    
    def __init__(
        self,
        levels: List[np.ndarray],
        n_envs: int,
        config: PPOConfig,
        seed: int = 42,
    ):
        self.envs = [
            SokobanEnv(levels, config, seed=seed + i)
            for i in range(n_envs)
        ]
        self.n_envs = n_envs
        
    def reset(self) -> Tuple[np.ndarray, List[Dict]]:
        """Reset all environments."""
        obs_list = []
        info_list = []
        
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            info_list.append(info)
        
        return np.stack(obs_list), info_list
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all environments."""
        obs_list = []
        reward_list = []
        terminated_list = []
        truncated_list = []
        info_list = []
        
        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            
            # Auto-reset on episode end
            if terminated or truncated:
                obs, reset_info = env.reset()
                info['terminal_observation'] = board_to_onehot(env.board)
                info.update(reset_info)
            
            obs_list.append(obs)
            reward_list.append(reward)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            info_list.append(info)
        
        return (
            np.stack(obs_list),
            np.array(reward_list),
            np.array(terminated_list),
            np.array(truncated_list),
            info_list,
        )


# =============================================================================
# Rollout Buffer
# =============================================================================

class RolloutBuffer:
    """
    Buffer for storing rollout experiences.
    """
    
    def __init__(
        self,
        buffer_size: int,
        n_envs: int,
        obs_shape: Tuple[int, ...],
        device: torch.device,
    ):
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.device = device
        
        # Storage
        self.observations = np.zeros((buffer_size, n_envs, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, n_envs), dtype=np.int64)
        self.rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.values = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.legal_masks = np.zeros((buffer_size, n_envs, 4), dtype=np.float32)
        # Track if episode ended with solving (for q_halt target)
        self.solved = np.zeros((buffer_size, n_envs), dtype=np.float32)
        
        self.pos = 0
        self.full = False
        
        # Computed after rollout
        self.advantages = None
        self.returns = None
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray,
        legal_mask: np.ndarray,
        solved: np.ndarray = None,
    ):
        """Add a transition to the buffer."""
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.legal_masks[self.pos] = legal_mask
        if solved is not None:
            self.solved[self.pos] = solved
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
    
    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ):
        """Compute GAE advantages and returns."""
        self.advantages = np.zeros_like(self.rewards)
        last_gae = 0
        
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        
        self.returns = self.advantages + self.values
    
    def get_samples(self, batch_size: int):
        """Generate random mini-batches for training."""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            
            # Convert flat indices to (timestep, env) pairs
            timesteps = batch_indices // self.n_envs
            env_ids = batch_indices % self.n_envs
            
            yield {
                'observations': torch.tensor(
                    self.observations[timesteps, env_ids], device=self.device
                ),
                'actions': torch.tensor(
                    self.actions[timesteps, env_ids], device=self.device
                ),
                'old_values': torch.tensor(
                    self.values[timesteps, env_ids], device=self.device
                ),
                'old_log_probs': torch.tensor(
                    self.log_probs[timesteps, env_ids], device=self.device
                ),
                'advantages': torch.tensor(
                    self.advantages[timesteps, env_ids], device=self.device
                ),
                'returns': torch.tensor(
                    self.returns[timesteps, env_ids], device=self.device
                ),
                'legal_masks': torch.tensor(
                    self.legal_masks[timesteps, env_ids], device=self.device
                ),
                'solved': torch.tensor(
                    self.solved[timesteps, env_ids], device=self.device
                ),
            }
    
    def reset(self):
        """Reset buffer for next rollout."""
        self.pos = 0
        self.full = False


# =============================================================================
# PPO Trainer
# =============================================================================

class SokobanPPOTrainer:
    """
    PPO trainer for Sokoban.
    
    Args:
        model: Actor-critic model
        config: PPO configuration
        device: Torch device
    """
    
    def __init__(
        self,
        model: SokobanActorCritic,
        config: PPOConfig,
        device: torch.device,
    ):
        self.model = model
        self.config = config
        self.device = device
        
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
        )
    
    def collect_rollouts(
        self,
        env: VectorizedSokobanEnv,
        buffer: RolloutBuffer,
        n_steps: int,
    ) -> Dict[str, float]:
        """
        Collect rollout experiences.
        
        Returns:
            Dictionary with rollout statistics
        """
        self.model.eval()
        
        obs, infos = env.reset()
        
        episode_rewards = []
        episode_lengths = []
        current_rewards = np.zeros(env.n_envs)
        current_lengths = np.zeros(env.n_envs)
        
        for _ in range(n_steps):
            # Get legal masks
            legal_masks = np.stack([info['legal_mask'] for info in infos])
            
            # Get action from policy
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                legal_mask_tensor = torch.tensor(legal_masks, dtype=torch.float32, device=self.device)
                
                action_probs, action_logits, values, _, _ = self.model(obs_tensor, legal_mask_tensor)
                
                # Sample action
                dist = torch.distributions.Categorical(probs=action_probs)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                
                actions_np = actions.cpu().numpy()
                values_np = values.cpu().numpy()
                log_probs_np = log_probs.cpu().numpy()
            
            # Store in buffer
            buffer.add(
                obs=obs,
                action=actions_np,
                reward=np.zeros(env.n_envs),  # Will be filled by step
                done=np.zeros(env.n_envs),
                value=values_np,
                log_prob=log_probs_np,
                legal_mask=legal_masks,
            )
            
            # Step environment
            next_obs, rewards, terminated, truncated, infos = env.step(actions_np)
            
            # Update buffer with actual rewards and dones
            buffer.rewards[buffer.pos - 1] = rewards
            buffer.dones[buffer.pos - 1] = np.logical_or(terminated, truncated).astype(np.float32)
            # Track if episode ended with solving (positive reward = solved)
            buffer.solved[buffer.pos - 1] = (rewards > 0).astype(np.float32)
            
            # Track episode stats
            current_rewards += rewards
            current_lengths += 1
            
            for i in range(env.n_envs):
                if terminated[i] or truncated[i]:
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])
                    current_rewards[i] = 0
                    current_lengths[i] = 0
            
            obs = next_obs
        
        # Compute last values for GAE
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            _, _, last_values, _, _ = self.model(obs_tensor)
            last_values_np = last_values.cpu().numpy()
        
        buffer.compute_returns_and_advantages(
            last_values_np,
            self.config.gamma,
            self.config.gae_lambda,
        )
        
        return {
            'mean_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
            'mean_length': np.mean(episode_lengths) if episode_lengths else 0.0,
            'n_episodes': len(episode_rewards),
        }
    
    def train_step(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Perform PPO training on collected rollouts.
        
        Returns:
            Dictionary with training statistics
        """
        self.model.train()
        
        # Normalize advantages
        advantages = buffer.advantages.flatten()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        buffer.advantages = advantages.reshape(buffer.advantages.shape)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0
        
        total_q_halt_loss = 0.0
        
        for _ in range(self.config.ppo_epochs):
            for batch in buffer.get_samples(self.config.batch_size):
                obs = batch['observations']
                actions = batch['actions']
                old_log_probs = batch['old_log_probs']
                advantages = batch['advantages']
                returns = batch['returns']
                legal_masks = batch['legal_masks']
                solved = batch['solved']  # Target for q_halt
                
                # Forward pass - now returns 5 values
                log_probs, entropy, values, q_halt, q_continue = self.model.evaluate_actions(
                    obs, actions, legal_masks
                )
                
                # Policy loss (clipped)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Q-halt loss: predict if episode leads to solving (same as Sudoku)
                q_halt_loss = F.binary_cross_entropy_with_logits(q_halt, solved)
                
                # Total loss (same 0.5 weight for q_halt as Sudoku)
                loss = (
                    policy_loss
                    + self.config.value_loss_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                    + 0.5 * q_halt_loss
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_q_halt_loss += q_halt_loss.item()
                n_updates += 1
        
        buffer.reset()
        
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'q_halt_loss': total_q_halt_loss / n_updates,
        }


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_solve_rate(
    model: nn.Module,
    levels: List[np.ndarray],
    device: torch.device,
    max_steps: int = 200,
) -> Dict[str, float]:
    """
    Evaluate model solve rate on a set of levels.
    
    Args:
        model: Trained model
        levels: List of levels to evaluate
        device: Torch device
        max_steps: Maximum steps per level
    
    Returns:
        Dictionary with solve_rate, deadlock_rate, avg_steps
    """
    model.eval()
    
    solved = 0
    deadlocked = 0
    total_steps = []
    
    with torch.no_grad():
        for level in levels:
            board = level.copy()
            steps = 0
            
            while steps < max_steps:
                if is_solved(board):
                    solved += 1
                    total_steps.append(steps)
                    break
                
                if is_deadlock(board):
                    deadlocked += 1
                    break
                
                # Get model prediction
                board_onehot = board_to_onehot(board)
                obs = torch.from_numpy(board_onehot).float().unsqueeze(0).to(device)
                
                # Get action
                if hasattr(model, 'get_action'):
                    action, _, _, _ = model.get_action(obs)
                    action = action.item()
                else:
                    logits = model(obs)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    # Mask illegal actions
                    legal = legal_actions(board)
                    logits_np = logits.cpu().numpy()[0]
                    logits_np[~legal] = float('-inf')
                    action = int(np.argmax(logits_np))
                
                # Execute action
                board, _ = step(board, action)
                steps += 1
            
            if steps >= max_steps:
                # Didn't solve or deadlock
                pass
    
    n_levels = len(levels)
    return {
        'solve_rate': solved / n_levels if n_levels > 0 else 0.0,
        'deadlock_rate': deadlocked / n_levels if n_levels > 0 else 0.0,
        'avg_steps': np.mean(total_steps) if total_steps else 0.0,
    }


# =============================================================================
# Main Training Function
# =============================================================================

def train_ppo(
    model: nn.Module,
    train_levels: List[np.ndarray],
    val_levels: List[np.ndarray],
    config: PPOConfig,
    device: torch.device,
    save_dir: str = "experiments/results/sokoban_ppo",
    verbose: bool = True,
    wandb_log: bool = False,
) -> Dict[str, Any]:
    """
    Train Sokoban solver with PPO.
    
    Args:
        model: PoT or Baseline model
        train_levels: Training levels
        val_levels: Validation levels
        config: PPO configuration
        device: Torch device
        save_dir: Directory to save results
        verbose: If True, print progress
        wandb_log: If True, log to Weights & Biases
    
    Returns:
        Dictionary with training results
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Wrap model in actor-critic
    actor_critic = SokobanActorCritic(model).to(device)
    
    # Create environments
    train_env = VectorizedSokobanEnv(
        train_levels, config.n_envs, config, seed=42
    )
    
    # Create buffer
    obs_shape = (10, 10, 7)  # One-hot board shape
    buffer = RolloutBuffer(
        config.n_steps, config.n_envs, obs_shape, device
    )
    
    # Trainer
    trainer = SokobanPPOTrainer(actor_critic, config, device)
    
    # Training loop
    n_updates = config.total_timesteps // (config.n_steps * config.n_envs)
    
    history = {
        'mean_reward': [],
        'mean_length': [],
        'policy_loss': [],
        'value_loss': [],
        'entropy': [],
    }
    
    best_reward = float('-inf')
    
    for update in tqdm(range(n_updates), desc="PPO Training", disable=not verbose):
        # Collect rollouts
        rollout_stats = trainer.collect_rollouts(train_env, buffer, config.n_steps)
        
        # Train
        train_stats = trainer.train_step(buffer)
        
        # Log
        history['mean_reward'].append(rollout_stats['mean_reward'])
        history['mean_length'].append(rollout_stats['mean_length'])
        history['policy_loss'].append(train_stats['policy_loss'])
        history['value_loss'].append(train_stats['value_loss'])
        history['entropy'].append(train_stats['entropy'])
        
        # Progress
        timestep = (update + 1) * config.n_steps * config.n_envs
        
        if verbose and (update + 1) % 10 == 0:
            print(f"Update {update+1}/{n_updates} ({timestep:,} steps) - "
                  f"Reward: {rollout_stats['mean_reward']:.3f}, "
                  f"Policy Loss: {train_stats['policy_loss']:.4f}, "
                  f"Value Loss: {train_stats['value_loss']:.4f}")
        
        # Log to W&B
        if wandb_log:
            import wandb
            wandb.log({
                'train/mean_reward': rollout_stats['mean_reward'],
                'train/mean_length': rollout_stats['mean_length'],
                'train/policy_loss': train_stats['policy_loss'],
                'train/value_loss': train_stats['value_loss'],
                'train/entropy': train_stats['entropy'],
                'train/q_halt_loss': train_stats.get('q_halt_loss', 0),
                'timestep': timestep,
            })
        
        # Evaluate
        if timestep >= config.eval_interval and timestep % config.eval_interval < config.n_steps * config.n_envs:
            eval_stats = evaluate_solve_rate(
                model, val_levels[:config.eval_episodes], device,
                max_steps=config.max_episode_steps,
            )
            
            if verbose:
                print(f"  [Eval] Solve rate: {eval_stats['solve_rate']:.2%}, "
                      f"Deadlock rate: {eval_stats['deadlock_rate']:.2%}")
            
            # Log eval to W&B
            if wandb_log:
                import wandb
                wandb.log({
                    'eval/solve_rate': eval_stats['solve_rate'],
                    'eval/deadlock_rate': eval_stats['deadlock_rate'],
                    'eval/avg_steps': eval_stats['avg_steps'],
                    'timestep': timestep,
                })
            
            # Save best
            if rollout_stats['mean_reward'] > best_reward:
                best_reward = rollout_stats['mean_reward']
                torch.save({
                    'timestep': timestep,
                    'model_state_dict': model.state_dict(),
                    'mean_reward': best_reward,
                    'eval_stats': eval_stats,
                }, save_path / 'best_model.pt')
    
    # Save final
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'config': config,
    }, save_path / 'final_model.pt')
    
    return {
        'history': history,
        'best_reward': best_reward,
        'model': model,
    }

