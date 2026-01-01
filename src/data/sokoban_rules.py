"""
Sokoban Rules Implementation.

Pure functions for Sokoban game mechanics:
- legal_actions: Check which moves are valid
- step: Apply an action to get next state
- is_solved: Check if puzzle is solved
- is_deadlock: Detect deadlock states

These correspond to "constraint checking" in the Sudoku analogy.

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any

from .sokoban import (
    TILE_WALL, TILE_FLOOR, TILE_PLAYER, TILE_BOX, TILE_TARGET,
    TILE_BOX_ON_TARGET, TILE_PLAYER_ON_TARGET, NUM_TILE_TYPES,
    BOARD_HEIGHT, BOARD_WIDTH,
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, NUM_ACTIONS,
    ACTION_DELTAS, ACTION_NAMES,
    get_board_masks,
)


# =============================================================================
# Core Rule Functions
# =============================================================================

def is_walkable(tile: int) -> bool:
    """Check if a tile can be walked on (not wall, not box)."""
    return tile in (TILE_FLOOR, TILE_TARGET, TILE_PLAYER, TILE_PLAYER_ON_TARGET)


def is_pushable_dest(tile: int) -> bool:
    """Check if a tile can receive a pushed box."""
    return tile in (TILE_FLOOR, TILE_TARGET)


def is_box(tile: int) -> bool:
    """Check if a tile contains a box."""
    return tile in (TILE_BOX, TILE_BOX_ON_TARGET)


def is_target(tile: int) -> bool:
    """Check if a tile is a target (goal) position."""
    return tile in (TILE_TARGET, TILE_BOX_ON_TARGET, TILE_PLAYER_ON_TARGET)


def in_bounds(row: int, col: int, height: int = BOARD_HEIGHT, width: int = BOARD_WIDTH) -> bool:
    """Check if position is within board bounds."""
    return 0 <= row < height and 0 <= col < width


def get_player_pos(board: np.ndarray) -> Optional[Tuple[int, int]]:
    """Get player position from board."""
    player_mask = (board == TILE_PLAYER) | (board == TILE_PLAYER_ON_TARGET)
    positions = np.argwhere(player_mask)
    if len(positions) == 0:
        return None
    return tuple(positions[0])


def count_boxes_on_targets(board: np.ndarray) -> int:
    """Count how many boxes are on target positions."""
    return int(np.sum(board == TILE_BOX_ON_TARGET))


def count_targets(board: np.ndarray) -> int:
    """Count total number of target positions."""
    return int(np.sum(
        (board == TILE_TARGET) | 
        (board == TILE_BOX_ON_TARGET) | 
        (board == TILE_PLAYER_ON_TARGET)
    ))


# =============================================================================
# Legal Actions
# =============================================================================

def is_action_legal(board: np.ndarray, action: int, player_pos: Optional[Tuple[int, int]] = None) -> bool:
    """
    Check if an action is legal from the current state.
    
    Args:
        board: Current board state [H, W]
        action: Action to check (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        player_pos: Optional player position (computed if not provided)
    
    Returns:
        True if action is legal
    """
    if player_pos is None:
        player_pos = get_player_pos(board)
    
    if player_pos is None:
        return False
    
    pr, pc = player_pos
    dr, dc = ACTION_DELTAS[action]
    
    # Destination position
    dest_r, dest_c = pr + dr, pc + dc
    
    # Check bounds
    if not in_bounds(dest_r, dest_c):
        return False
    
    dest_tile = board[dest_r, dest_c]
    
    # Can walk onto floor or target
    if dest_tile in (TILE_FLOOR, TILE_TARGET):
        return True
    
    # Can push box if destination beyond is empty
    if is_box(dest_tile):
        # Position beyond the box
        beyond_r, beyond_c = dest_r + dr, dest_c + dc
        
        # Check bounds for beyond position
        if not in_bounds(beyond_r, beyond_c):
            return False
        
        beyond_tile = board[beyond_r, beyond_c]
        
        # Can only push if beyond is floor or target
        if is_pushable_dest(beyond_tile):
            return True
    
    return False


def legal_actions(board: np.ndarray) -> np.ndarray:
    """
    Get mask of legal actions from current state.
    
    Args:
        board: Current board state [H, W]
    
    Returns:
        np.ndarray of shape [4] with 1 for legal actions, 0 for illegal
    """
    player_pos = get_player_pos(board)
    
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    
    for action in range(NUM_ACTIONS):
        if is_action_legal(board, action, player_pos):
            mask[action] = 1.0
    
    return mask


def get_legal_action_list(board: np.ndarray) -> List[int]:
    """
    Get list of legal actions from current state.
    
    Args:
        board: Current board state [H, W]
    
    Returns:
        List of legal action indices
    """
    mask = legal_actions(board)
    return [a for a in range(NUM_ACTIONS) if mask[a] > 0]


# =============================================================================
# State Transition
# =============================================================================

def step(board: np.ndarray, action: int) -> Tuple[np.ndarray, bool]:
    """
    Apply an action and return the new state.
    
    Args:
        board: Current board state [H, W]
        action: Action to apply (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
    
    Returns:
        Tuple of (new_board, action_was_legal)
        If action was illegal, returns a copy of original board
    """
    player_pos = get_player_pos(board)
    
    if player_pos is None or not is_action_legal(board, action, player_pos):
        return board.copy(), False
    
    # Create new board
    new_board = board.copy()
    
    pr, pc = player_pos
    dr, dc = ACTION_DELTAS[action]
    dest_r, dest_c = pr + dr, pc + dc
    
    dest_tile = new_board[dest_r, dest_c]
    
    # Handle pushing box
    pushed_box = False
    if is_box(dest_tile):
        beyond_r, beyond_c = dest_r + dr, dest_c + dc
        beyond_tile = new_board[beyond_r, beyond_c]
        
        # Move box to beyond position
        if beyond_tile == TILE_TARGET:
            new_board[beyond_r, beyond_c] = TILE_BOX_ON_TARGET
        else:  # FLOOR
            new_board[beyond_r, beyond_c] = TILE_BOX
        
        # Clear box from dest (reveal target if any)
        if dest_tile == TILE_BOX_ON_TARGET:
            new_board[dest_r, dest_c] = TILE_TARGET
        else:  # TILE_BOX
            new_board[dest_r, dest_c] = TILE_FLOOR
        
        pushed_box = True
    
    # Move player to destination
    current_tile = new_board[pr, pc]
    
    # Clear player from current position
    if current_tile == TILE_PLAYER_ON_TARGET:
        new_board[pr, pc] = TILE_TARGET
    else:  # TILE_PLAYER
        new_board[pr, pc] = TILE_FLOOR
    
    # Place player at destination
    dest_tile_after_box = new_board[dest_r, dest_c]
    if dest_tile_after_box == TILE_TARGET:
        new_board[dest_r, dest_c] = TILE_PLAYER_ON_TARGET
    else:  # FLOOR
        new_board[dest_r, dest_c] = TILE_PLAYER
    
    return new_board, True


# =============================================================================
# Solved Check
# =============================================================================

def is_solved(board: np.ndarray) -> bool:
    """
    Check if the puzzle is solved.
    
    Solved when all boxes are on target positions.
    
    Args:
        board: Current board state [H, W]
    
    Returns:
        True if solved
    """
    # Count boxes on targets
    boxes_on_targets = count_boxes_on_targets(board)
    
    # Count total targets (including covered ones)
    total_targets = count_targets(board)
    
    # Solved if all targets have boxes (4 for Boxoban medium)
    # Note: TILE_BOX_ON_TARGET counts as both box-on-target and target
    return boxes_on_targets == total_targets


# =============================================================================
# Deadlock Detection
# =============================================================================

def is_corner_deadlock(board: np.ndarray, row: int, col: int) -> bool:
    """
    Check if a box at (row, col) is in a corner deadlock.
    
    Corner deadlock: Box is stuck in corner formed by walls/edges,
    and the box is NOT on a target.
    
    Args:
        board: Current board state
        row, col: Position to check
    
    Returns:
        True if deadlock
    """
    tile = board[row, col]
    
    # Only check boxes not on targets
    if tile != TILE_BOX:
        return False
    
    H, W = board.shape
    
    # Check all four corner configurations
    # Up-Left corner
    up_blocked = (row == 0) or (board[row - 1, col] == TILE_WALL)
    left_blocked = (col == 0) or (board[row, col - 1] == TILE_WALL)
    if up_blocked and left_blocked:
        return True
    
    # Up-Right corner
    right_blocked = (col == W - 1) or (board[row, col + 1] == TILE_WALL)
    if up_blocked and right_blocked:
        return True
    
    # Down-Left corner
    down_blocked = (row == H - 1) or (board[row + 1, col] == TILE_WALL)
    if down_blocked and left_blocked:
        return True
    
    # Down-Right corner
    if down_blocked and right_blocked:
        return True
    
    return False


def is_wall_deadlock(board: np.ndarray, row: int, col: int) -> bool:
    """
    Check if a box at (row, col) is in a wall deadlock.
    
    Wall deadlock: Box is against a wall with no target along that wall.
    
    Args:
        board: Current board state
        row, col: Position to check
    
    Returns:
        True if deadlock
    """
    tile = board[row, col]
    
    # Only check boxes not on targets
    if tile != TILE_BOX:
        return False
    
    H, W = board.shape
    
    # Check if box is against a wall edge
    # For simplicity, we only check if box is on edge with no escape
    
    # Top wall
    if row == 0:
        # Check if there's any target in this row
        has_target_in_row = any(
            board[row, c] in (TILE_TARGET, TILE_PLAYER_ON_TARGET)
            for c in range(W)
        )
        if not has_target_in_row:
            return True
    
    # Bottom wall
    if row == H - 1:
        has_target_in_row = any(
            board[row, c] in (TILE_TARGET, TILE_PLAYER_ON_TARGET)
            for c in range(W)
        )
        if not has_target_in_row:
            return True
    
    # Left wall
    if col == 0:
        has_target_in_col = any(
            board[r, col] in (TILE_TARGET, TILE_PLAYER_ON_TARGET)
            for r in range(H)
        )
        if not has_target_in_col:
            return True
    
    # Right wall
    if col == W - 1:
        has_target_in_col = any(
            board[r, col] in (TILE_TARGET, TILE_PLAYER_ON_TARGET)
            for r in range(H)
        )
        if not has_target_in_col:
            return True
    
    return False


def is_freeze_deadlock(board: np.ndarray, row: int, col: int) -> bool:
    """
    Check if a box at (row, col) is part of a 2x2 freeze deadlock.
    
    2x2 freeze: Four adjacent cells form a frozen group
    (boxes + walls with no targets).
    
    Args:
        board: Current board state
        row, col: Position to check
    
    Returns:
        True if deadlock
    """
    tile = board[row, col]
    
    # Only check boxes not on targets
    if tile != TILE_BOX:
        return False
    
    H, W = board.shape
    
    def is_obstacle(r: int, c: int) -> bool:
        """Check if position is wall or box (not on target)."""
        if not in_bounds(r, c, H, W):
            return True  # Out of bounds = wall
        t = board[r, c]
        return t == TILE_WALL or t == TILE_BOX
    
    def has_target_in_2x2(r: int, c: int) -> bool:
        """Check if 2x2 block starting at (r,c) has any target."""
        for dr in range(2):
            for dc in range(2):
                rr, cc = r + dr, c + dc
                if in_bounds(rr, cc, H, W):
                    if is_target(board[rr, cc]):
                        return True
        return False
    
    # Check four 2x2 configurations containing this cell
    # Top-left corner
    if row > 0 and col > 0:
        if (is_obstacle(row - 1, col - 1) and
            is_obstacle(row - 1, col) and
            is_obstacle(row, col - 1)):
            if not has_target_in_2x2(row - 1, col - 1):
                return True
    
    # Top-right corner
    if row > 0 and col < W - 1:
        if (is_obstacle(row - 1, col) and
            is_obstacle(row - 1, col + 1) and
            is_obstacle(row, col + 1)):
            if not has_target_in_2x2(row - 1, col):
                return True
    
    # Bottom-left corner
    if row < H - 1 and col > 0:
        if (is_obstacle(row, col - 1) and
            is_obstacle(row + 1, col - 1) and
            is_obstacle(row + 1, col)):
            if not has_target_in_2x2(row, col - 1):
                return True
    
    # Bottom-right corner
    if row < H - 1 and col < W - 1:
        if (is_obstacle(row, col + 1) and
            is_obstacle(row + 1, col) and
            is_obstacle(row + 1, col + 1)):
            if not has_target_in_2x2(row, col):
                return True
    
    return False


def is_deadlock(board: np.ndarray) -> bool:
    """
    Check if the board is in a deadlock state.
    
    Deadlock occurs when it's impossible to solve the puzzle.
    
    Args:
        board: Current board state [H, W]
    
    Returns:
        True if deadlock detected
    """
    H, W = board.shape
    
    # Find all boxes
    box_positions = np.argwhere(board == TILE_BOX)
    
    for pos in box_positions:
        row, col = pos
        
        # Check corner deadlock
        if is_corner_deadlock(board, row, col):
            return True
        
        # Check 2x2 freeze deadlock
        if is_freeze_deadlock(board, row, col):
            return True
    
    return False


def get_deadlock_info(board: np.ndarray) -> Dict[str, Any]:
    """
    Get detailed deadlock information.
    
    Args:
        board: Current board state
    
    Returns:
        Dictionary with deadlock details
    """
    H, W = board.shape
    
    corner_deadlocks = []
    freeze_deadlocks = []
    
    box_positions = np.argwhere(board == TILE_BOX)
    
    for pos in box_positions:
        row, col = pos
        if is_corner_deadlock(board, row, col):
            corner_deadlocks.append((row, col))
        if is_freeze_deadlock(board, row, col):
            freeze_deadlocks.append((row, col))
    
    return {
        'is_deadlock': len(corner_deadlocks) > 0 or len(freeze_deadlocks) > 0,
        'corner_deadlocks': corner_deadlocks,
        'freeze_deadlocks': freeze_deadlocks,
        'total_deadlocked_boxes': len(set(corner_deadlocks + freeze_deadlocks)),
    }


# =============================================================================
# Utility Functions
# =============================================================================

def simulate_trajectory(
    board: np.ndarray,
    actions: List[int],
) -> Tuple[List[np.ndarray], List[bool], int]:
    """
    Simulate a sequence of actions.
    
    Args:
        board: Initial board state
        actions: List of actions to apply
    
    Returns:
        Tuple of (states, action_legal_flags, final_step_count)
    """
    states = [board.copy()]
    legal_flags = []
    
    current = board.copy()
    for action in actions:
        next_board, legal = step(current, action)
        states.append(next_board)
        legal_flags.append(legal)
        current = next_board
        
        # Stop if solved or deadlock
        if is_solved(current) or is_deadlock(current):
            break
    
    return states, legal_flags, len(states) - 1


def random_walk(
    board: np.ndarray,
    num_steps: int,
    seed: Optional[int] = None,
    avoid_undo: bool = True,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Perform a random walk from a board state.
    
    Args:
        board: Initial board state
        num_steps: Maximum number of steps
        seed: Random seed
        avoid_undo: If True, avoid immediately undoing previous action
    
    Returns:
        Tuple of (visited_states, actions_taken)
    """
    rng = np.random.default_rng(seed)
    
    # Opposite actions for undo avoidance
    opposite = {
        ACTION_UP: ACTION_DOWN,
        ACTION_DOWN: ACTION_UP,
        ACTION_LEFT: ACTION_RIGHT,
        ACTION_RIGHT: ACTION_LEFT,
    }
    
    states = [board.copy()]
    actions_taken = []
    
    current = board.copy()
    prev_action = None
    
    for _ in range(num_steps):
        legal = get_legal_action_list(current)
        
        if not legal:
            break
        
        # Optionally avoid undoing previous action
        if avoid_undo and prev_action is not None:
            undo_action = opposite.get(prev_action)
            if undo_action in legal and len(legal) > 1:
                legal = [a for a in legal if a != undo_action]
        
        # Random choice
        action = rng.choice(legal)
        next_board, _ = step(current, action)
        
        states.append(next_board)
        actions_taken.append(action)
        
        current = next_board
        prev_action = action
        
        # Stop if solved or deadlock
        if is_solved(current) or is_deadlock(current):
            break
    
    return states, actions_taken

