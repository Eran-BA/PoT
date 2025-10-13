"""
Connect Four A/B Test: PoH-HRM vs Baseline Transformer

Tests the ability of models to play Connect Four at a strategic level.
The task requires multi-step planning, board evaluation, and tactical reasoning -
ideal for hierarchical reasoning with the HRM controller.

Author: Eran Ben Artzy
Date: October 2025
"""

import os
import sys
import time
import csv
from datetime import datetime
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Add parent directory to path for imports - robust for Colab
repo_root = None
try:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except (NameError, OSError):
    cwd = os.getcwd()
    if os.path.exists(os.path.join(cwd, 'src', 'pot')):
        repo_root = cwd
    elif os.path.basename(cwd) == 'experiments':
        repo_root = os.path.dirname(cwd)
if repo_root:
    sys.path.insert(0, repo_root)

from src.pot.core.hrm_controller import HRMPointerController, HRMState

# Try to import transformers for BERT, fallback if not available
try:
    from transformers import BertModel, BertConfig
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("Warning: transformers library not available. BERT baseline will be skipped.")
    print("Install with: pip install transformers")


# ========== Connect Four Game Engine ==========

class ConnectFour:
    """Connect Four game engine."""
    
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.reset()
    
    def reset(self):
        """Reset the game board."""
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1
        self.winner = None
        self.game_over = False
        return self.board.copy()
    
    def is_valid_move(self, col):
        """Check if a column is available for a move."""
        return 0 <= col < self.cols and self.board[0, col] == 0
    
    def get_valid_moves(self):
        """Return list of valid column indices."""
        return [col for col in range(self.cols) if self.is_valid_move(col)]
    
    def make_move(self, col):
        """Drop a piece in the specified column."""
        if not self.is_valid_move(col):
            return False
        
        # Find the lowest empty row
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, col] == 0:
                self.board[row, col] = self.current_player
                break
        
        # Check for win
        if self._check_win(row, col):
            self.winner = self.current_player
            self.game_over = True
        elif len(self.get_valid_moves()) == 0:
            self.game_over = True  # Draw
        else:
            # Switch player
            self.current_player = 3 - self.current_player  # Toggle between 1 and 2
        
        return True
    
    def _check_win(self, row, col):
        """Check if the last move resulted in a win."""
        player = self.board[row, col]
        
        # Check horizontal
        count = 1
        # Check left
        for c in range(col - 1, -1, -1):
            if self.board[row, c] == player:
                count += 1
            else:
                break
        # Check right
        for c in range(col + 1, self.cols):
            if self.board[row, c] == player:
                count += 1
            else:
                break
        if count >= 4:
            return True
        
        # Check vertical
        count = 1
        # Check down
        for r in range(row + 1, self.rows):
            if self.board[r, col] == player:
                count += 1
            else:
                break
        if count >= 4:
            return True
        
        # Check diagonal (top-left to bottom-right)
        count = 1
        # Check up-left
        r, c = row - 1, col - 1
        while r >= 0 and c >= 0:
            if self.board[r, c] == player:
                count += 1
                r -= 1
                c -= 1
            else:
                break
        # Check down-right
        r, c = row + 1, col + 1
        while r < self.rows and c < self.cols:
            if self.board[r, c] == player:
                count += 1
                r += 1
                c += 1
            else:
                break
        if count >= 4:
            return True
        
        # Check anti-diagonal (top-right to bottom-left)
        count = 1
        # Check up-right
        r, c = row - 1, col + 1
        while r >= 0 and c < self.cols:
            if self.board[r, c] == player:
                count += 1
                r -= 1
                c += 1
            else:
                break
        # Check down-left
        r, c = row + 1, col - 1
        while r < self.rows and c >= 0:
            if self.board[r, c] == player:
                count += 1
                r += 1
                c -= 1
            else:
                break
        if count >= 4:
            return True
        
        return False
    
    def get_board_state(self):
        """Return current board state."""
        return self.board.copy()


def minimax(game: ConnectFour, depth: int, alpha: float, beta: float, 
            maximizing: bool, player: int) -> Tuple[int, Optional[int]]:
    """Minimax with alpha-beta pruning to find optimal move."""
    valid_moves = game.get_valid_moves()
    
    # Terminal states
    if game.game_over:
        if game.winner == player:
            return 1000000, None
        elif game.winner == 3 - player:
            return -1000000, None
        else:
            return 0, None  # Draw
    
    if depth == 0 or not valid_moves:
        return evaluate_board(game.board, player), None
    
    if maximizing:
        max_eval = float('-inf')
        best_move = valid_moves[0]
        
        for col in valid_moves:
            # Make move
            board_copy = game.board.copy()
            current_player = game.current_player
            winner = game.winner
            game_over = game.game_over
            
            game.make_move(col)
            eval_score, _ = minimax(game, depth - 1, alpha, beta, False, player)
            
            # Undo move
            game.board = board_copy
            game.current_player = current_player
            game.winner = winner
            game.game_over = game_over
            
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = col
            
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        
        return max_eval, best_move
    else:
        min_eval = float('inf')
        best_move = valid_moves[0]
        
        for col in valid_moves:
            # Make move
            board_copy = game.board.copy()
            current_player = game.current_player
            winner = game.winner
            game_over = game.game_over
            
            game.make_move(col)
            eval_score, _ = minimax(game, depth - 1, alpha, beta, True, player)
            
            # Undo move
            game.board = board_copy
            game.current_player = current_player
            game.winner = winner
            game.game_over = game_over
            
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = col
            
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        
        return min_eval, best_move


def evaluate_board(board: np.ndarray, player: int) -> int:
    """Heuristic evaluation of board state."""
    score = 0
    rows, cols = board.shape
    opponent = 3 - player
    
    # Check all possible windows of 4
    # Horizontal
    for row in range(rows):
        for col in range(cols - 3):
            window = board[row, col:col+4]
            score += evaluate_window(window, player, opponent)
    
    # Vertical
    for col in range(cols):
        for row in range(rows - 3):
            window = board[row:row+4, col]
            score += evaluate_window(window, player, opponent)
    
    # Positive diagonal
    for row in range(rows - 3):
        for col in range(cols - 3):
            window = [board[row+i, col+i] for i in range(4)]
            score += evaluate_window(window, player, opponent)
    
    # Negative diagonal
    for row in range(3, rows):
        for col in range(cols - 3):
            window = [board[row-i, col+i] for i in range(4)]
            score += evaluate_window(window, player, opponent)
    
    return score


def evaluate_window(window, player: int, opponent: int) -> int:
    """Evaluate a window of 4 positions."""
    score = 0
    window = np.array(window)
    
    player_count = np.sum(window == player)
    opponent_count = np.sum(window == opponent)
    empty_count = np.sum(window == 0)
    
    if player_count == 4:
        score += 100
    elif player_count == 3 and empty_count == 1:
        score += 5
    elif player_count == 2 and empty_count == 2:
        score += 2
    
    if opponent_count == 3 and empty_count == 1:
        score -= 4  # Block opponent
    
    return score


def generate_training_data(num_games: int, depth: int = 3, seed: int = 42):
    """Generate training data using minimax."""
    np.random.seed(seed)
    data = []
    
    for game_idx in tqdm(range(num_games), desc="Generating games"):
        game = ConnectFour()
        game_states = []
        game_moves = []
        
        while not game.game_over:
            board_state = game.get_board_state()
            valid_moves = game.get_valid_moves()
            
            # Get optimal move using minimax
            _, best_move = minimax(game, depth, float('-inf'), float('inf'), 
                                   True, game.current_player)
            
            if best_move is None:
                best_move = np.random.choice(valid_moves)
            
            game_states.append(board_state.copy())
            game_moves.append(best_move)
            
            game.make_move(best_move)
        
        # Store game
        for state, move in zip(game_states, game_moves):
            data.append({
                'board': state,
                'move': move,
                'winner': game.winner if game.winner else 0  # 0 for draw
            })
    
    return data


# ========== Models ==========

class PoHBlock(nn.Module):
    """PoH transformer block with HRM controller."""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, max_iters: int, T: int = 4
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_iters = max_iters
        self.T = T

        # HRM controller for routing
        self.controller = HRMPointerController(
            d_model=d_model,
            n_heads=n_heads,
            T=T,
            topk=None,
            temperature_init=2.0,
            temperature_min=0.7,
            entropy_reg=1e-3,
            use_layernorm=True,
            dropout=0.1
        )

        # Per-head attention
        self.q_proj = nn.ModuleList(
            [nn.Linear(d_model, d_model // n_heads) for _ in range(n_heads)]
        )
        self.k_proj = nn.ModuleList(
            [nn.Linear(d_model, d_model // n_heads) for _ in range(n_heads)]
        )
        self.v_proj = nn.ModuleList(
            [nn.Linear(d_model, d_model // n_heads) for _ in range(n_heads)]
        )
        self.out_proj = nn.Linear(d_model, d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), 
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, z):
        B = z.size(0)
        device = z.device
        
        # Initialize HRM state
        hrm_state = self.controller.init_state(B, device)
        
        for iter_idx in range(self.max_iters):
            # HRM controller routing
            alphas, hrm_state, aux = self.controller(
                x=z,
                state=hrm_state,
                return_aux=True
            )
            
            # Per-head attention
            head_outs = []
            for h_idx in range(self.n_heads):
                q = self.q_proj[h_idx](z)
                k = self.k_proj[h_idx](z)
                v = self.v_proj[h_idx](z)

                scores = torch.einsum("btd,bsd->bts", q, k) / (
                    (self.d_model // self.n_heads) ** 0.5
                )
                attn = F.softmax(scores, dim=-1)
                out = torch.einsum("bts,bsd->btd", attn, v)
                head_outs.append(out)

            # Concat heads and project
            head_outs_concat = torch.cat(head_outs, dim=-1)
            attn_out = self.out_proj(head_outs_concat)

            # Residual + norm
            z = self.ln1(z + attn_out)
            z_refined = z

            # FFN
            z = self.ln2(z + self.ffn(z))

        return z_refined


class ConnectFourModel(nn.Module):
    """Connect Four move prediction model."""
    
    def __init__(
        self,
        rows: int = 6,
        cols: int = 7,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 1024,
        max_inner_iters: int = 3,
        T: int = 4,
        use_poh: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.d_model = d_model
        self.use_poh = use_poh
        
        # Embed board positions (0=empty, 1=player1, 2=player2)
        self.cell_embed = nn.Embedding(3, d_model)
        self.pos_embed = nn.Embedding(rows * cols, d_model)
        
        # Encoder
        if use_poh:
            self.encoder = PoHBlock(d_model, n_heads, d_ff, max_inner_iters, T=T)
        else:
            self.encoder_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.encoder_ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model)
            )
            self.encoder_ln1 = nn.LayerNorm(d_model)
            self.encoder_ln2 = nn.LayerNorm(d_model)
        
        # Policy head (predict move probabilities)
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, cols)
        )
        
        # Value head (predict win probability)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1),
            nn.Tanh()
        )
        
    def forward(self, board):
        """
        Args:
            board: (B, rows, cols) tensor with values 0, 1, 2
        Returns:
            policy: (B, cols) move logits
            value: (B, 1) win probability
        """
        B, rows, cols = board.shape
        
        # Flatten board
        board_flat = board.view(B, rows * cols)  # (B, 42)
        
        # Embed cells
        cell_emb = self.cell_embed(board_flat.long())  # (B, 42, d_model)
        
        # Add position embeddings
        positions = torch.arange(rows * cols, device=board.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embed(positions)
        
        x = cell_emb + pos_emb
        
        # Encode
        if self.use_poh:
            encoded = self.encoder(x)
        else:
            # Standard transformer encoder
            attn_out, _ = self.encoder_attn(x, x, x)
            x = self.encoder_ln1(x + attn_out)
            ffn_out = self.encoder_ffn(x)
            encoded = self.encoder_ln2(x + ffn_out)
        
        # Pool to get board representation
        board_repr = encoded.mean(dim=1)  # (B, d_model)
        
        # Predict policy and value
        policy = self.policy_head(board_repr)  # (B, cols)
        value = self.value_head(board_repr)    # (B, 1)
        
        return policy, value


class BERTConnectFourModel(nn.Module):
    """BERT-based Connect Four model for comparison."""
    
    def __init__(
        self,
        rows: int = 6,
        cols: int = 7,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 1024,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if not BERT_AVAILABLE:
            raise ImportError("transformers library required for BERT baseline")
        
        self.rows = rows
        self.cols = cols
        self.d_model = d_model
        
        # Embed board positions
        self.cell_embed = nn.Embedding(3, d_model)
        self.pos_embed = nn.Embedding(rows * cols, d_model)
        
        # BERT encoder configuration (configurable layers for parameter parity)
        bert_config = BertConfig(
            hidden_size=d_model,
            num_hidden_layers=num_layers,
            num_attention_heads=n_heads,
            intermediate_size=d_ff,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        self.bert = BertModel(bert_config)
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, cols)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1),
            nn.Tanh()
        )
    
    def forward(self, board):
        """Forward pass using BERT encoder."""
        B, rows, cols = board.shape
        
        # Flatten board
        board_flat = board.view(B, rows * cols)
        
        # Embed cells
        cell_emb = self.cell_embed(board_flat.long())
        
        # Add position embeddings
        positions = torch.arange(rows * cols, device=board.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embed(positions)
        
        x = cell_emb + pos_emb
        
        # BERT encoding
        bert_output = self.bert(inputs_embeds=x)
        encoded = bert_output.last_hidden_state
        
        # Pool to get board representation
        board_repr = encoded.mean(dim=1)
        
        # Predict policy and value
        policy = self.policy_head(board_repr)
        value = self.value_head(board_repr)
        
        return policy, value


# ========== Training & Evaluation ==========

def train_and_evaluate(
    model, model_name, train_data, test_data,
    epochs=100, lr=1e-3, device='cpu'
):
    """Train and evaluate a Connect Four model."""
    print(f"\nTraining {model_name}...")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    start_time = time.time()
    best_accuracy = 0
    
    pbar = tqdm(range(epochs), desc=model_name)
    for epoch in pbar:
        model.train()
        total_policy_loss = 0
        total_value_loss = 0
        correct = 0
        total = 0
        
        # Training
        for batch_idx in range(0, len(train_data), 32):
            batch = train_data[batch_idx:batch_idx + 32]
            
            # Prepare batch
            boards = torch.tensor(np.stack([d['board'] for d in batch]), device=device, dtype=torch.float32)
            moves = torch.tensor([d['move'] for d in batch], device=device, dtype=torch.long)
            winners = torch.tensor([d['winner'] for d in batch], device=device, dtype=torch.float32)
            
            # Normalize winners: 1 → 1.0, 2 → -1.0, 0 → 0.0
            values = torch.where(winners == 1, torch.ones_like(winners), 
                               torch.where(winners == 2, -torch.ones_like(winners), 
                                         torch.zeros_like(winners))).unsqueeze(1)
            
            # Forward
            policy, value = model(boards)
            
            # Losses
            policy_loss = policy_criterion(policy, moves)
            value_loss = value_criterion(value, values)
            loss = policy_loss + value_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            
            # Accuracy
            pred_moves = policy.argmax(dim=-1)
            correct += (pred_moves == moves).sum().item()
            total += len(moves)
        
        train_accuracy = correct / total if total > 0 else 0
        avg_policy_loss = total_policy_loss / max(len(train_data) // 32, 1)
        avg_value_loss = total_value_loss / max(len(train_data) // 32, 1)
        
        # Evaluation every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_correct = 0
                test_total = 0
                
                for batch_idx in range(0, len(test_data), 32):
                    batch = test_data[batch_idx:batch_idx + 32]
                    boards = torch.tensor(np.stack([d['board'] for d in batch]), device=device, dtype=torch.float32)
                    moves = torch.tensor([d['move'] for d in batch], device=device, dtype=torch.long)
                    
                    policy, _ = model(boards)
                    pred_moves = policy.argmax(dim=-1)
                    test_correct += (pred_moves == moves).sum().item()
                    test_total += len(moves)
                
                test_accuracy = test_correct / test_total if test_total > 0 else 0
                
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                
                pbar.set_postfix({
                    'p_loss': f'{avg_policy_loss:.3f}',
                    'v_loss': f'{avg_value_loss:.3f}',
                    'train_acc': f'{train_accuracy:.2%}',
                    'test_acc': f'{test_accuracy:.2%}'
                })
        else:
            pbar.set_postfix({
                'p_loss': f'{avg_policy_loss:.3f}',
                'v_loss': f'{avg_value_loss:.3f}',
                'train_acc': f'{train_accuracy:.2%}'
            })
    
    training_time = (time.time() - start_time) / 60
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_correct = 0
        test_total = 0
        
        for batch_idx in range(0, len(test_data), 32):
            batch = test_data[batch_idx:batch_idx + 32]
            boards = torch.tensor(np.stack([d['board'] for d in batch]), device=device, dtype=torch.float32)
            moves = torch.tensor([d['move'] for d in batch], device=device, dtype=torch.long)
            
            policy, _ = model(boards)
            pred_moves = policy.argmax(dim=-1)
            test_correct += (pred_moves == moves).sum().item()
            test_total += len(moves)
        
        final_accuracy = test_correct / test_total if test_total > 0 else 0
    
    return {
        'best_accuracy': best_accuracy,
        'final_accuracy': final_accuracy,
        'time_min': training_time,
        'params_M': sum(p.numel() for p in model.parameters()) / 1e6,
    }


# ========== Main A/B Test ==========

def run_ab_test(train_games=500, test_games=100, minimax_depth=3, R=4, T=4, n_heads=4,
                epochs=100, seed=42):
    """Run A/B test for Connect Four."""
    
    print(f"\n{'='*80}")
    print(f"A/B Test: Connect Four")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Training games: {train_games:,}")
    print(f"  Test games: {test_games:,}")
    print(f"  Minimax depth: {minimax_depth}")
    print(f"  PoH n_heads: {n_heads}")
    print(f"  PoH R (refinement steps): {R}")
    print(f"  PoH T (HRM outer loop period): {T}")
    print(f"  Epochs: {epochs}")
    print(f"  Seed: {seed}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    # Generate data
    print(f"\nGenerating training data...")
    train_data = generate_training_data(train_games, depth=minimax_depth, seed=seed)
    test_data = generate_training_data(test_games, depth=minimax_depth, seed=seed+10000)
    
    print(f"  Train positions: {len(train_data):,}")
    print(f"  Test positions: {len(test_data):,}")
    
    # Build models
    print(f"\nBuilding models...")
    
    # Configuration for parameter parity
    d_model = 256
    d_ff = 1024
    
    # Baseline: single-pass transformer
    baseline = ConnectFourModel(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        max_inner_iters=1,
        T=1,
        use_poh=False
    ).to(device)
    
    # Count baseline parameters
    baseline_params = sum(p.numel() for p in baseline.parameters())
    print(f"  Baseline parameters: {baseline_params / 1e6:.2f}M")
    
    # Create PoH-HRM first to get target parameter count
    poh = ConnectFourModel(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        max_inner_iters=R,
        T=T,
        use_poh=True
    ).to(device)
    
    poh_params = sum(p.numel() for p in poh.parameters())
    print(f"  PoH-HRM parameters: {poh_params / 1e6:.2f}M")
    
    # BERT baseline (if available) - dynamically adjust layers for parameter parity with PoH
    bert = None
    bert_results = None
    if BERT_AVAILABLE:
        try:
            # Try different layer counts to match PoH parameters
            for attempt_layers in [3, 2, 4, 1]:
                bert_test = BERTConnectFourModel(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    num_layers=attempt_layers
                ).to(device)
                
                bert_params = sum(p.numel() for p in bert_test.parameters())
                parity_pct = abs(bert_params - poh_params) / poh_params * 100
                
                if parity_pct < 15:  # Accept if within 15%
                    bert = bert_test
                    print(f"  BERT parameters: {bert_params / 1e6:.2f}M (layers={attempt_layers}, parity={parity_pct:.1f}%)")
                    break
                else:
                    del bert_test
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            if bert is None:
                print(f"  Warning: Could not achieve parameter parity (<15%) for BERT")
                # Use smallest BERT anyway for comparison
                bert = BERTConnectFourModel(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    num_layers=1
                ).to(device)
                bert_params = sum(p.numel() for p in bert.parameters())
                parity_pct = abs(bert_params - poh_params) / poh_params * 100
                print(f"  BERT parameters: {bert_params / 1e6:.2f}M (layers=1, parity={parity_pct:.1f}%) [best effort]")
                
        except Exception as e:
            print(f"  Warning: Could not create BERT model: {e}")
            bert = None
    
    # Train baseline
    baseline_results = train_and_evaluate(
        baseline, "Baseline", train_data, test_data,
        epochs=epochs, device=device
    )
    
    # Train BERT (if available)
    if bert is not None:
        bert_results = train_and_evaluate(
            bert, "BERT", train_data, test_data,
            epochs=epochs, device=device
        )
    
    # Train PoH
    poh_results = train_and_evaluate(
        poh, f"PoH-HRM (R={R}, T={T})", train_data, test_data,
        epochs=epochs, device=device
    )
    
    # Results
    print(f"\n{'='*80}")
    print("📊 RESULTS")
    print(f"{'='*80}")
    
    print(f"\n📚 Baseline (Standard Transformer)")
    print(f"  Parameters: {baseline_results['params_M']:.2f}M")
    print(f"  Best Accuracy: {baseline_results['best_accuracy']:.2%}")
    print(f"  Final Accuracy: {baseline_results['final_accuracy']:.2%}")
    print(f"  Training time: {baseline_results['time_min']:.2f} min")
    
    if bert_results is not None:
        print(f"\n🤖 BERT Baseline")
        print(f"  Parameters: {bert_results['params_M']:.2f}M")
        print(f"  Best Accuracy: {bert_results['best_accuracy']:.2%}")
        print(f"  Final Accuracy: {bert_results['final_accuracy']:.2%}")
        print(f"  Training time: {bert_results['time_min']:.2f} min")
    
    print(f"\n🔬 PoH with HRM (R={R}, T={T}, n_heads={n_heads})")
    print(f"  Parameters: {poh_results['params_M']:.2f}M")
    print(f"  Best Accuracy: {poh_results['best_accuracy']:.2%}")
    print(f"  Final Accuracy: {poh_results['final_accuracy']:.2%}")
    print(f"  Training time: {poh_results['time_min']:.2f} min")
    
    # Comparison
    delta_acc = poh_results['final_accuracy'] - baseline_results['final_accuracy']
    delta_pct = (delta_acc / baseline_results['final_accuracy'] * 100) if baseline_results['final_accuracy'] > 0 else 0
    
    print(f"\n{'='*80}")
    print("📈 COMPARISON")
    print(f"{'='*80}")
    print(f"Accuracy delta: {delta_acc:+.2%} ({delta_pct:+.1f}%)")
    
    if delta_acc > 0.05:
        winner = "PoH-HRM"
        print(f"🏆 Winner: {winner} by {delta_acc:.2%}")
    elif delta_acc < -0.05:
        winner = "Baseline"
        print(f"🏆 Winner: {winner} by {-delta_acc:.2%}")
    else:
        winner = "Tie"
        print(f"⚖️  TIE (difference < 5%)")
    
    results = {
        'baseline': baseline_results,
        'poh': poh_results,
        'delta_accuracy': delta_acc,
        'winner': winner
    }
    
    if bert_results is not None:
        results['bert'] = bert_results
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Connect Four A/B Test')
    parser.add_argument('--train-games', type=int, default=500, help='Training games')
    parser.add_argument('--test-games', type=int, default=100, help='Test games')
    parser.add_argument('--minimax-depth', type=int, default=3, help='Minimax search depth')
    parser.add_argument('--R', type=int, default=4, help='PoH refinement steps (default: 4)')
    parser.add_argument('--T', type=int, default=4, help='HRM outer loop period (default: 4)')
    parser.add_argument('--n-heads', type=int, default=4, help='Number of heads (default: 4)')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save-dir', type=str, default='experiments/results/connect_four_ab',
                        help='Save directory')
    
    args = parser.parse_args()
    
    print("="*80)
    print("CONNECT FOUR A/B TEST")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  n_heads = {args.n_heads}")
    print(f"  R = {args.R} (refinement steps)")
    print(f"  T = {args.T} (HRM outer loop period)")
    print("="*80)
    
    # Run test
    result = run_ab_test(
        train_games=args.train_games,
        test_games=args.test_games,
        minimax_depth=args.minimax_depth,
        R=args.R,
        T=args.T,
        n_heads=args.n_heads,
        epochs=args.epochs,
        seed=args.seed
    )
    
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    results_file = os.path.join(args.save_dir, f"connect_four_R{args.R}_T{args.T}_nheads{args.n_heads}.csv")
    
    with open(results_file, 'w', newline='') as f:
        fieldnames = [
            'timestamp', 'model', 'R', 'T', 'n_heads',
            'best_accuracy', 'final_accuracy',
            'time_min', 'params_M', 'delta_accuracy', 'winner'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Baseline row
        writer.writerow({
            'timestamp': timestamp,
            'model': 'Baseline',
            'R': '-',
            'T': '-',
            'n_heads': args.n_heads,
            'best_accuracy': f"{result['baseline']['best_accuracy']:.4f}",
            'final_accuracy': f"{result['baseline']['final_accuracy']:.4f}",
            'time_min': f"{result['baseline']['time_min']:.2f}",
            'params_M': f"{result['baseline']['params_M']:.2f}",
            'delta_accuracy': "0.0000",
            'winner': ''
        })
        
        # PoH row
        writer.writerow({
            'timestamp': timestamp,
            'model': 'PoH-HRM',
            'R': args.R,
            'T': args.T,
            'n_heads': args.n_heads,
            'best_accuracy': f"{result['poh']['best_accuracy']:.4f}",
            'final_accuracy': f"{result['poh']['final_accuracy']:.4f}",
            'time_min': f"{result['poh']['time_min']:.2f}",
            'params_M': f"{result['poh']['params_M']:.2f}",
            'delta_accuracy': f"{result['delta_accuracy']:+.4f}",
            'winner': result['winner']
        })
    
    print(f"\n✅ Results saved to: {results_file}")
    print("="*80)


if __name__ == "__main__":
    main()

