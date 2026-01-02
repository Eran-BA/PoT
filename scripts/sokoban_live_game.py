#!/usr/bin/env python3
"""
Sokoban Live Game - Watch the PoT (Pointer-over-Heads) model play!

This script loads a trained model and visualizes it solving Sokoban puzzles
in real-time with ASCII rendering.

Usage:
    python scripts/sokoban_live_game.py --checkpoint path/to/model.pt
    python scripts/sokoban_live_game.py --difficulty simple --n-games 5
    python scripts/sokoban_live_game.py --interactive  # Manual play mode

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import argparse
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.sokoban import (
    TILE_WALL, TILE_FLOOR, TILE_PLAYER, TILE_BOX, TILE_TARGET,
    TILE_BOX_ON_TARGET, TILE_PLAYER_ON_TARGET, NUM_TILE_TYPES,
)
from src.data.sokoban_rules import step as sokoban_step, is_solved, is_deadlock


# =============================================================================
# ASCII Rendering (using paper's symbols)
# =============================================================================

# Paper's symbols: # Wall | _ Floor | O Target | X Box | P You | √ = Box on Target | S = You on Target
TILE_TO_CHAR = {
    TILE_WALL: '█',           # Wall (solid block)
    TILE_FLOOR: ' ',          # Floor (empty)
    TILE_PLAYER: '☺',         # Player
    TILE_BOX: '□',            # Box
    TILE_TARGET: '·',         # Target (goal)
    TILE_BOX_ON_TARGET: '■',  # Box on target (success!)
    TILE_PLAYER_ON_TARGET: '☻',  # Player on target
}

# Alternative ASCII-only mode
TILE_TO_ASCII = {
    TILE_WALL: '#',
    TILE_FLOOR: '_',
    TILE_PLAYER: 'P',
    TILE_BOX: 'X',
    TILE_TARGET: 'O',
    TILE_BOX_ON_TARGET: '*',
    TILE_PLAYER_ON_TARGET: 'S',
}

ACTION_NAMES = ['↑ UP', '↓ DOWN', '← LEFT', '→ RIGHT']
ACTION_ARROWS = ['↑', '↓', '←', '→']


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def render_board(
    board: np.ndarray, 
    step: int = 0, 
    action: Optional[str] = None,
    status: str = "",
    use_ascii: bool = False,
) -> str:
    """Render board as ASCII art."""
    tile_map = TILE_TO_ASCII if use_ascii else TILE_TO_CHAR
    
    H, W = board.shape
    lines = []
    
    # Header
    lines.append("")
    lines.append(f"  {'═' * (W * 2 + 4)}")
    header = f"  Step {step}"
    if action:
        header += f"  │  Action: {action}"
    lines.append(f"  {header}")
    lines.append(f"  {'═' * (W * 2 + 4)}")
    
    # Board
    for row in board:
        line = '  ║ '
        for cell in row:
            char = tile_map.get(cell, '?')
            line += char + ' '
        line += '║'
        lines.append(line)
    
    lines.append(f"  {'═' * (W * 2 + 4)}")
    
    # Status
    if status:
        lines.append(f"  {status}")
    
    # Legend
    lines.append("")
    if use_ascii:
        lines.append("  Legend: # Wall  _ Floor  P Player  X Box  O Target  * Box✓  S Player on Target")
    else:
        lines.append("  Legend: █ Wall    ☺ Player  □ Box  · Target  ■ Box✓  ☻ Player on Target")
    
    return '\n'.join(lines)


def board_to_onehot(board: np.ndarray) -> torch.Tensor:
    """Convert integer board to one-hot encoding."""
    H, W = board.shape
    onehot = np.zeros((H, W, NUM_TILE_TYPES), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            onehot[i, j, board[i, j]] = 1.0
    return torch.from_numpy(onehot)


def get_model_action(
    model: torch.nn.Module,
    board: np.ndarray,
    device: torch.device,
) -> Tuple[int, np.ndarray]:
    """Get model's predicted action and probabilities."""
    model.eval()
    
    # Convert to tensor
    board_tensor = board_to_onehot(board).unsqueeze(0).to(device)
    
    with torch.no_grad():
        model_out = model(board_tensor, return_aux=False)
        logits = model_out[0]  # [1, 4]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        action = logits.argmax(dim=-1).item()
    
    return action, probs


def play_game(
    model: torch.nn.Module,
    board: np.ndarray,
    device: torch.device,
    max_steps: int = 100,
    delay: float = 0.5,
    use_ascii: bool = False,
    verbose: bool = True,
) -> Tuple[bool, int, list]:
    """
    Play a game with the model.
    
    Returns:
        (solved, steps_taken, action_history)
    """
    current_board = board.copy()
    action_history = []
    
    for step in range(max_steps):
        # Check if solved
        if is_solved(current_board):
            if verbose:
                clear_screen()
                status = "🎉 SOLVED! 🎉"
                print(render_board(current_board, step, None, status, use_ascii))
            return True, step, action_history
        
        # Check if deadlock
        if is_deadlock(current_board):
            if verbose:
                clear_screen()
                status = "💀 DEADLOCK - Game Over"
                print(render_board(current_board, step, None, status, use_ascii))
            return False, step, action_history
        
        # Get model prediction
        action, probs = get_model_action(model, current_board, device)
        action_name = ACTION_NAMES[action]
        
        # Display
        if verbose:
            clear_screen()
            prob_str = ' '.join([f"{ACTION_ARROWS[i]}:{p:.0%}" for i, p in enumerate(probs)])
            status = f"Probs: {prob_str}"
            print(render_board(current_board, step, action_name, status, use_ascii))
            time.sleep(delay)
        
        # Execute action (actions are 0-indexed, sokoban_step expects 1-indexed)
        next_board, moved = sokoban_step(current_board, action + 1)
        action_history.append(action)
        
        if not moved:
            if verbose:
                print(f"  ⚠️ Invalid move! Continuing...")
            # Don't count invalid moves as progress
            continue
        
        current_board = next_board
    
    # Max steps exceeded
    if verbose:
        clear_screen()
        status = "⏰ Max steps exceeded"
        print(render_board(current_board, max_steps, None, status, use_ascii))
    
    return False, max_steps, action_history


def interactive_mode(
    board: np.ndarray,
    use_ascii: bool = False,
):
    """Interactive manual play mode."""
    current_board = board.copy()
    step = 0
    action_history = []
    
    key_to_action = {
        'w': 0, 'W': 0,  # Up
        's': 1, 'S': 1,  # Down
        'a': 2, 'A': 2,  # Left
        'd': 3, 'D': 3,  # Right
    }
    
    print("\n🎮 INTERACTIVE MODE")
    print("Controls: W=Up, S=Down, A=Left, D=Right, Q=Quit, R=Reset")
    print()
    
    while True:
        clear_screen()
        
        if is_solved(current_board):
            status = "🎉 SOLVED! Press R to reset, Q to quit"
        elif is_deadlock(current_board):
            status = "💀 DEADLOCK! Press R to reset, Q to quit"
        else:
            status = "Controls: W↑ S↓ A← D→ | Q=Quit R=Reset"
        
        print(render_board(current_board, step, None, status, use_ascii))
        
        try:
            key = input("\n  Your move: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if key.lower() == 'q':
            break
        elif key.lower() == 'r':
            current_board = board.copy()
            step = 0
            action_history = []
            continue
        elif key in key_to_action:
            action = key_to_action[key]
            next_board, moved = sokoban_step(current_board, action + 1)
            if moved:
                current_board = next_board
                step += 1
                action_history.append(action)
            else:
                print("  ⚠️ Can't move there!")
                time.sleep(0.3)
    
    print(f"\n  Thanks for playing! Steps: {step}")
    return action_history


def load_model(checkpoint_path: str, device: torch.device):
    """Load a trained model from checkpoint."""
    from src.pot.models.sokoban_solver import HybridPoTSokobanSolver
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    config = checkpoint.get('config', {})
    
    # Determine board size from config or default
    board_height = config.get('board_height', 10)
    board_width = config.get('board_width', 10)
    
    # Create model
    model = HybridPoTSokobanSolver(
        board_height=board_height,
        board_width=board_width,
        d_model=config.get('d_model', 256),
        d_ff=config.get('d_ff', 512),
        n_heads=config.get('n_heads', 4),
        H_layers=config.get('H_layers', 2),
        L_layers=config.get('L_layers', 2),
        H_cycles=config.get('H_cycles', 2),
        L_cycles=config.get('L_cycles', 6),
        T=config.get('T', 4),
        halt_max_steps=config.get('halt_max_steps', 4),
        controller_type=config.get('controller_type', 'transformer'),
        controller_kwargs=config.get('controller_kwargs', {'d_ctrl': 128, 'max_depth': 128}),
        injection_mode=config.get('injection_mode', 'broadcast'),
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  Board size: {board_height}x{board_width}")
    
    return model, (board_height, board_width)


def generate_puzzle(difficulty: str = 'simple', seed: Optional[int] = None) -> np.ndarray:
    """Generate a puzzle using gym-sokoban."""
    from src.data.sokoban_generator import SokobanGenerator
    
    generator = SokobanGenerator(difficulty=difficulty, seed=seed)
    samples = generator.generate_dataset(1, verbose=False)
    
    if samples:
        return samples[0]['board']
    else:
        raise RuntimeError(f"Failed to generate {difficulty} puzzle")


def main():
    parser = argparse.ArgumentParser(
        description="Sokoban Live Game - Watch PoT (Pointer-over-Heads) model play!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/sokoban_live_game.py --checkpoint checkpoints/best_model.pt
  python scripts/sokoban_live_game.py --difficulty simple --n-games 5
  python scripts/sokoban_live_game.py --interactive --difficulty simple
        """
    )
    
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--difficulty', type=str, default='simple',
                        choices=['simple', 'larger', 'two_boxes', 'complex'],
                        help='Puzzle difficulty (default: simple)')
    parser.add_argument('--n-games', type=int, default=3,
                        help='Number of games to play (default: 3)')
    parser.add_argument('--max-steps', type=int, default=50,
                        help='Max steps per game (default: 50)')
    parser.add_argument('--delay', type=float, default=0.3,
                        help='Delay between moves in seconds (default: 0.3)')
    parser.add_argument('--ascii', action='store_true',
                        help='Use ASCII-only characters')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive manual play mode')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for puzzle generation')
    parser.add_argument('--no-clear', action='store_true',
                        help='Don\'t clear screen between moves')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🎮 Sokoban Live Game - PoT (Pointer-over-Heads)")
    print(f"   Device: {device}")
    print(f"   Difficulty: {args.difficulty}")
    print()
    
    # Interactive mode
    if args.interactive:
        print("Generating puzzle...")
        board = generate_puzzle(args.difficulty, args.seed)
        interactive_mode(board, use_ascii=args.ascii)
        return
    
    # Model mode
    if args.checkpoint is None:
        print("⚠️ No checkpoint provided. Generating a puzzle for manual inspection.")
        print("   Use --checkpoint to load a trained model.")
        print("   Use --interactive for manual play mode.")
        print()
        board = generate_puzzle(args.difficulty, args.seed)
        print(render_board(board, 0, None, "Generated puzzle", args.ascii))
        return
    
    # Load model
    model, board_size = load_model(args.checkpoint, device)
    
    # Play games
    results = []
    
    for game_idx in range(args.n_games):
        print(f"\n{'=' * 40}")
        print(f"  GAME {game_idx + 1}/{args.n_games}")
        print(f"{'=' * 40}")
        
        # Generate puzzle
        seed = args.seed + game_idx if args.seed else None
        board = generate_puzzle(args.difficulty, seed)
        
        # Pad if needed
        H, W = board.shape
        if H < board_size[0] or W < board_size[1]:
            # Pad with walls
            padded = np.zeros(board_size, dtype=board.dtype)
            pad_h = (board_size[0] - H) // 2
            pad_w = (board_size[1] - W) // 2
            padded[pad_h:pad_h+H, pad_w:pad_w+W] = board
            board = padded
        
        # Play
        solved, steps, history = play_game(
            model, board, device,
            max_steps=args.max_steps,
            delay=args.delay,
            use_ascii=args.ascii,
        )
        
        results.append({
            'solved': solved,
            'steps': steps,
            'history': history,
        })
        
        time.sleep(1)
    
    # Summary
    print(f"\n{'=' * 40}")
    print("  SUMMARY")
    print(f"{'=' * 40}")
    
    n_solved = sum(r['solved'] for r in results)
    avg_steps = np.mean([r['steps'] for r in results])
    
    print(f"  Games: {args.n_games}")
    print(f"  Solved: {n_solved}/{args.n_games} ({100*n_solved/args.n_games:.0f}%)")
    print(f"  Avg Steps: {avg_steps:.1f}")
    print()


if __name__ == '__main__':
    main()

