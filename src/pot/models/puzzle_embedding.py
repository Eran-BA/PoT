"""
Generic Puzzle Embedding Module
================================

Per-instance learned embeddings for task specialization.
Each puzzle instance (maze, sorting problem, etc.) gets a unique learned embedding.

Based on HRM's CastedSparseEmbedding approach but made generic and simplified.

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import torch
import torch.nn as nn


class PuzzleEmbedding(nn.Module):
    """
    Per-instance learned embeddings for task specialization.
    
    Each puzzle (maze, sorting instance, etc.) gets a unique embedding vector
    that the model can use to specialize its behavior for that specific instance.
    
    Key design decisions:
    - Zero initialization: Let model learn from scratch (HRM approach)
    - Generic: Works across any task with instance-specific structure
    - Separate optimizer: Use different lr/weight_decay (puzzle_emb_lr)
    
    Args:
        num_puzzles: Number of unique puzzle instances
        emb_dim: Dimension of embedding vector
        init_std: Standard deviation for initialization (0.0 = zero init)
    
    Example:
        >>> puzzle_emb = PuzzleEmbedding(num_puzzles=1000, emb_dim=256)
        >>> puzzle_ids = torch.tensor([0, 42, 137])  # Batch of 3
        >>> emb = puzzle_emb(puzzle_ids)  # [3, 256]
    """
    
    def __init__(self, num_puzzles: int, emb_dim: int, init_std: float = 0.0):
        super().__init__()
        
        self.num_puzzles = num_puzzles
        self.emb_dim = emb_dim
        
        # Embedding lookup table
        self.embeddings = nn.Embedding(num_puzzles, emb_dim)
        
        # Zero init (HRM approach: let model learn puzzle-specific features)
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=init_std)
    
    def forward(self, puzzle_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings for puzzle IDs.
        
        Args:
            puzzle_ids: [B] or [B, 1] tensor of puzzle indices
        
        Returns:
            embeddings: [B, emb_dim] tensor of puzzle embeddings
        """
        if puzzle_ids.dim() > 1:
            puzzle_ids = puzzle_ids.squeeze(-1)
        
        return self.embeddings(puzzle_ids)
    
    def extra_repr(self) -> str:
        return f'num_puzzles={self.num_puzzles}, emb_dim={self.emb_dim}'

