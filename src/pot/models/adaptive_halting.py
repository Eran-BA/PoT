"""
Q-Learning Based Adaptive Halting
==================================

Adaptive Computation Time (ACT) using Q-learning.
Model learns when to stop refining based on correctness signals.

Based on HRM's Q-halting mechanism from the paper.

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class QHaltingController(nn.Module):
    """
    Q-learning based adaptive computation time (ACT).
    
    The model learns to predict:
    - q_halt: "Should I stop now?" (trained on sequence correctness)
    - q_continue: "Should I keep going?" (trained via Q-learning bootstrap)
    
    Training uses two losses:
    1. BCE on q_halt with correctness labels (supervised)
    2. BCE on q_continue with bootstrap targets (Q-learning)
    
    During inference, halts when q_halt > q_continue or max_steps reached.
    
    Args:
        d_model: Model hidden dimension
        max_steps: Maximum computation steps (default: 16, HRM value)
        exploration_prob: Probability of random exploration (default: 0.1)
    
    Example:
        >>> q_halt = QHaltingController(d_model=256, max_steps=16)
        >>> hidden = torch.randn(8, 900, 256)  # [B, L, D]
        >>> q_halt_logit, q_continue_logit = q_halt(hidden)  # [B], [B]
        >>> should_stop = q_halt.should_halt(q_halt_logit, q_continue_logit, step=3, is_training=True)
    """
    
    def __init__(
        self, 
        d_model: int, 
        max_steps: int = 16,
        exploration_prob: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_steps = max_steps
        self.exploration_prob = exploration_prob
        
        # Q-head: outputs [q_halt, q_continue]
        self.q_head = nn.Linear(d_model, 2)
        
        # Init q_halt to -5 (HRM approach: start conservative, avoid early halting)
        # This encourages model to compute longer initially
        with torch.no_grad():
            nn.init.zeros_(self.q_head.weight)
            nn.init.constant_(self.q_head.bias, -5.0)
    
    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Q-values for halting decision.
        
        Args:
            hidden: [B, L, d_model] hidden states from transformer
        
        Returns:
            q_halt: [B] Q-value for "halt now"
            q_continue: [B] Q-value for "continue"
        """
        # Use first token ([CLS] style) for decision
        # In HRM, they use the first token of z_H (high-level state)
        q_logits = self.q_head(hidden[:, 0])  # [B, 2]
        
        q_halt = q_logits[:, 0]      # [B]
        q_continue = q_logits[:, 1]  # [B]
        
        return q_halt, q_continue
    
    def should_halt(
        self, 
        q_halt: torch.Tensor, 
        q_continue: torch.Tensor, 
        step: int, 
        is_training: bool
    ) -> torch.Tensor:
        """
        Decide whether to halt computation.
        
        During training:
        - Halt when q_halt > q_continue (learned policy)
        - Minimum 2 steps (allow some computation)
        - 10% exploration: random min_steps
        - Always halt at max_steps
        
        During inference:
        - Use learned policy (halt when q_halt > q_continue)
        
        Args:
            q_halt: [B] Q-values for halting
            q_continue: [B] Q-values for continuing
            step: Current computation step (1-indexed)
            is_training: Whether in training mode
        
        Returns:
            halt: [B] boolean tensor, True = should halt
        """
        batch_size = q_halt.size(0)
        device = q_halt.device
        
        # At max steps, always halt
        if step >= self.max_steps:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        
        # Minimum 2 steps (allow some computation)
        if step < 2:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Halt when q_halt > q_continue (learned policy)
        halt = (q_halt > q_continue)
        
        # Exploration during training: randomly force early/late halting
        if is_training and torch.rand(1).item() < self.exploration_prob:
            # Random min_steps in [2, max_steps]
            min_steps = torch.randint(2, self.max_steps + 1, (1,), device=device).item()
            # Force halt if past min_steps (encourages exploration of shorter paths)
            halt = halt | (step >= min_steps)
        
        return halt
    
    def extra_repr(self) -> str:
        return (f'd_model={self.d_model}, max_steps={self.max_steps}, '
                f'exploration_prob={self.exploration_prob}')

