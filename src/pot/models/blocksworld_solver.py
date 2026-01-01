"""
Blocksworld Solver Models.

Two solver architectures for Blocksworld planning:
1. HybridPoTBlocksworldSolver - PoT with CausalDepthTransformerRouter (main model)
2. BaselineBlocksworldSolver - Standard transformer baseline

The model predicts next state given current state (transition model).
Can also be used for goal-conditioned planning.

State representation:
- For N blocks: state is [pos_block_0, pos_block_1, ..., pos_block_{N-1}]
- Each pos_block_i in {0=table, 1..N = on_block_j, N+1 = holding}

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any

from src.pot.core.controller_factory import create_controller
from src.pot.models.hybrid_hrm import HybridHRMBase
from src.pot.models.hrm_layers import RMSNorm, SwiGLU


class HybridPoTBlocksworldSolver(HybridHRMBase):
    """
    PoT-based Blocksworld transition model with CausalDepthTransformerRouter.
    
    Uses the two-timescale HybridHRMBase architecture for iterative refinement:
    - L_level: Fast reasoning, updates every inner step
    - H_level: Slow reasoning, updates every outer step
    - Both use PoT head routing via CausalDepthTransformerRouter
    
    Input: current state [B, N] where N = num_blocks
    Output: next state logits [B, N, vocab_size]
    
    Args:
        num_blocks: Maximum number of blocks (N)
        d_model: Hidden dimension
        n_heads: Number of attention heads
        H_layers: Number of layers in H_level module
        L_layers: Number of layers in L_level module
        d_ff: Feedforward dimension
        dropout: Dropout rate
        H_cycles: Number of H_level cycles per ACT step
        L_cycles: Number of L_level cycles per H_cycle
        T: HRM period for pointer controller
        controller_type: Type of depth controller
        controller_kwargs: Additional kwargs for controller creation
        goal_conditioned: If True, use goal state as additional input
    """
    
    def __init__(
        self,
        num_blocks: int = 8,
        d_model: int = 256,
        n_heads: int = 4,
        H_layers: int = 2,
        L_layers: int = 2,
        d_ff: int = 1024,
        dropout: float = 0.0,
        H_cycles: int = 2,
        L_cycles: int = 8,
        T: int = 4,
        halt_max_steps: int = 1,
        controller_type: str = "transformer",  # CausalDepthTransformerRouter
        controller_kwargs: dict = None,
        goal_conditioned: bool = False,
    ):
        # Sequence length = number of blocks
        seq_len = num_blocks
        
        # Vocab size: table + blocks + holding
        vocab_size = num_blocks + 2
        
        # Initialize base class
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
        )
        
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.goal_conditioned = goal_conditioned
        
        embed_init_std = 1.0 / self.embed_scale
        
        # State embedding: position of each block
        self.state_embed = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.state_embed.weight, mean=0, std=embed_init_std)
        
        # Block identity embedding: which block slot this is (A, B, C, ...)
        self.block_embed = nn.Embedding(num_blocks, d_model)
        nn.init.normal_(self.block_embed.weight, mean=0, std=embed_init_std)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * embed_init_std)
        
        # Goal embedding (optional)
        if goal_conditioned:
            self.goal_embed = nn.Embedding(vocab_size, d_model)
            nn.init.normal_(self.goal_embed.weight, mean=0, std=embed_init_std)
            self.goal_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def _compute_input_embedding(
        self,
        state: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute scaled input embedding.
        
        Args:
            state: [B, N] current block positions
            goal: [B, N] optional goal state for goal-conditioned planning
        
        Returns:
            [B, N, d_model] scaled input embeddings
        """
        B, N = state.shape
        device = state.device
        
        # Block indices for identity embedding
        block_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        
        # Embed current state + block identity + position
        state_emb = self.state_embed(state)          # [B, N, d_model]
        block_emb = self.block_embed(block_ids)      # [B, N, d_model]
        pos_emb = self.pos_embed[:, :N]              # [1, N, d_model]
        
        input_emb = state_emb + block_emb + pos_emb
        
        # Add goal embedding if goal-conditioned
        if self.goal_conditioned and goal is not None:
            goal_emb = self.goal_embed(goal)
            goal_proj = self.goal_proj(goal_emb)
            input_emb = input_emb + 0.5 * goal_proj  # Weighted addition
        
        # Scale embeddings by sqrt(d_model) like HRM
        return self.embed_scale * input_emb
    
    def forward(
        self,
        state: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], int]:
        """
        Forward pass for next-state prediction.
        
        Args:
            state: [B, N] current block positions
            goal: [B, N] optional goal state
        
        Returns:
            logits: [B, N, vocab_size] next state logits
            q_halt: Q-values for halting (if ACT enabled)
            q_continue: Q-values for continuing (if ACT enabled)
            steps: Number of reasoning steps
        """
        input_emb = self._compute_input_embedding(state, goal)
        
        if self.halt_max_steps > 1:
            # Use ACT wrapper
            act_out = self.act_forward(input_emb)
            hidden = act_out['hidden']
            q_halt = act_out['q_halt']
            q_continue = act_out['q_continue']
            steps = act_out['steps']
            
            logits = self.output_proj(hidden)
            return logits, q_halt, q_continue, steps
        else:
            # Simple reasoning loop
            hidden, q_halt, q_continue, steps = self.reasoning_loop(input_emb)
            logits = self.output_proj(hidden)
            return logits, q_halt, q_continue, steps
    
    def predict_next_state(
        self,
        state: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict next state (greedy argmax).
        
        Args:
            state: [B, N] current block positions
            goal: [B, N] optional goal state
        
        Returns:
            [B, N] predicted next state
        """
        logits, _, _, _ = self.forward(state, goal)
        return logits.argmax(dim=-1)
    
    def rollout(
        self,
        init_state: torch.Tensor,
        num_steps: int,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Autoregressive rollout for multi-step planning.
        
        Args:
            init_state: [B, N] initial state
            num_steps: number of planning steps
            goal: [B, N] optional goal state
        
        Returns:
            trajectory: [B, num_steps+1, N] state sequence
        """
        states = [init_state]
        state = init_state
        
        with torch.no_grad():
            for _ in range(num_steps):
                next_state = self.predict_next_state(state, goal)
                states.append(next_state)
                state = next_state
        
        return torch.stack(states, dim=1)


class BaselineBlocksworldSolver(nn.Module):
    """
    Standard Transformer baseline for Blocksworld.
    
    Simple encoder-only transformer without iterative refinement.
    Used for comparison with PoT architectures.
    
    Args:
        num_blocks: Maximum number of blocks
        d_model: Hidden dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feedforward dimension
        dropout: Dropout rate
        goal_conditioned: If True, use goal state as additional input
    """
    
    def __init__(
        self,
        num_blocks: int = 8,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        goal_conditioned: bool = False,
    ):
        super().__init__()
        
        self.num_blocks = num_blocks
        self.vocab_size = num_blocks + 2  # table + blocks + holding
        self.seq_len = num_blocks
        self.goal_conditioned = goal_conditioned
        
        # Embeddings
        self.state_embed = nn.Embedding(self.vocab_size, d_model)
        self.block_embed = nn.Embedding(num_blocks, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, num_blocks, d_model) * 0.02)
        
        if goal_conditioned:
            self.goal_embed = nn.Embedding(self.vocab_size, d_model)
            self.goal_proj = nn.Linear(d_model, d_model)
        
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
        
        # Output
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, self.vocab_size)
    
    def forward(
        self,
        state: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None, None, int]:
        """
        Forward pass.
        
        Args:
            state: [B, N] current block positions
            goal: [B, N] optional goal state
        
        Returns:
            logits: [B, N, vocab_size] next state logits
            q_halt: None (no ACT)
            q_continue: None (no ACT)
            steps: 1 (single pass)
        """
        B, N = state.shape
        device = state.device
        
        # Block indices
        block_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        
        # Embed
        x = self.state_embed(state) + self.block_embed(block_ids) + self.pos_embed[:, :N]
        
        # Add goal if goal-conditioned
        if self.goal_conditioned and goal is not None:
            goal_emb = self.goal_embed(goal)
            goal_proj = self.goal_proj(goal_emb)
            x = x + 0.5 * goal_proj
        
        # Encode
        x = self.encoder(x)
        x = self.final_norm(x)
        
        # Project to output
        logits = self.output_proj(x)
        
        return logits, None, None, 1
    
    def predict_next_state(
        self,
        state: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict next state (greedy argmax)."""
        logits, _, _, _ = self.forward(state, goal)
        return logits.argmax(dim=-1)
    
    def rollout(
        self,
        init_state: torch.Tensor,
        num_steps: int,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressive rollout for multi-step planning."""
        states = [init_state]
        state = init_state
        
        with torch.no_grad():
            for _ in range(num_steps):
                next_state = self.predict_next_state(state, goal)
                states.append(next_state)
                state = next_state
        
        return torch.stack(states, dim=1)


class SimplePoTBlocksworldSolver(nn.Module):
    """
    Simplified PoT Blocksworld solver with explicit inner cycles.
    
    Uses CausalDepthTransformerRouter directly for explicit control over
    the number of refinement iterations. Simpler than HybridHRMBase but
    still demonstrates the PoT concept.
    
    Args:
        num_blocks: Maximum number of blocks
        d_model: Hidden dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feedforward dimension
        dropout: Dropout rate
        R: Number of refinement iterations (inner cycles)
        controller_type: Type of controller ("transformer" or "pot_transformer")
        goal_conditioned: If True, use goal state as additional input
    """
    
    def __init__(
        self,
        num_blocks: int = 8,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 1024,
        dropout: float = 0.1,
        R: int = 8,
        controller_type: str = "transformer",
        goal_conditioned: bool = False,
    ):
        super().__init__()
        
        self.num_blocks = num_blocks
        self.vocab_size = num_blocks + 2  # table + blocks + holding
        self.seq_len = num_blocks
        self.d_model = d_model
        self.n_heads = n_heads
        self.R = R
        self.goal_conditioned = goal_conditioned
        
        # Embeddings
        self.state_embed = nn.Embedding(self.vocab_size, d_model)
        self.block_embed = nn.Embedding(num_blocks, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, num_blocks, d_model) * 0.02)
        
        if goal_conditioned:
            self.goal_embed = nn.Embedding(self.vocab_size, d_model)
            self.goal_proj = nn.Linear(d_model, d_model)
        
        # Pre-norm
        self.pre_norm = nn.LayerNorm(d_model)
        
        # Depth controller
        self.controller = create_controller(
            controller_type=controller_type,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            max_depth=R * 2,  # Allow for longer depths if needed
        )
        
        # Transformer layers
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
        
        # Output
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, self.vocab_size)
    
    def _encode_step(
        self,
        x: torch.Tensor,
        route_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single encoding step with head routing.
        
        Args:
            x: [B, N, d_model] hidden state
            route_weights: [B, N, H] or [B, H] routing weights
        
        Returns:
            [B, N, d_model] updated hidden state
        """
        B, T, D = x.shape
        
        # Handle different route_weights shapes
        if route_weights.dim() == 2:
            # [B, H] -> expand to [B, N, H, 1]
            route_exp = route_weights.unsqueeze(1).unsqueeze(-1)  # [B, 1, H, 1]
            route_exp = route_exp.expand(B, T, -1, -1)  # [B, N, H, 1]
        else:
            # [B, N, H] -> [B, N, H, 1]
            route_exp = route_weights.unsqueeze(-1)
        
        # Scale routing weights
        route_exp = route_exp * self.n_heads
        
        # Apply transformer layers
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
        
        return x
    
    def forward(
        self,
        state: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with iterative refinement.
        
        Args:
            state: [B, N] current block positions
            goal: [B, N] optional goal state
        
        Returns:
            logits: [B, N, vocab_size] next state logits
            aux: Dict with alphas and other diagnostics
        """
        B, N = state.shape
        device = state.device
        
        # Block indices
        block_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        
        # Embed
        x = self.state_embed(state) + self.block_embed(block_ids) + self.pos_embed[:, :N]
        
        # Add goal if goal-conditioned
        if self.goal_conditioned and goal is not None:
            goal_emb = self.goal_embed(goal)
            goal_proj = self.goal_proj(goal_emb)
            x = x + 0.5 * goal_proj
        
        x = self.pre_norm(x)
        
        # Iterative refinement with depth controller
        ctrl_state = None
        alphas = []
        
        for t in range(self.R):
            # Get routing weights
            alpha, ctrl_state, aux = self.controller.step(x, t=t, cache=ctrl_state)
            alphas.append(alpha)
            
            # Apply one encoding step with routing
            x = self._encode_step(x, alpha)
        
        # Final output
        x = self.final_norm(x)
        logits = self.output_proj(x)
        
        return logits, {"alphas": alphas}
    
    def predict_next_state(
        self,
        state: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict next state (greedy argmax)."""
        logits, _ = self.forward(state, goal)
        return logits.argmax(dim=-1)
    
    def rollout(
        self,
        init_state: torch.Tensor,
        num_steps: int,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressive rollout for multi-step planning."""
        states = [init_state]
        state = init_state
        
        with torch.no_grad():
            for _ in range(num_steps):
                next_state = self.predict_next_state(state, goal)
                states.append(next_state)
                state = next_state
        
        return torch.stack(states, dim=1)

