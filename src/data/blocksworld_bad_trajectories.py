"""
Bad Trajectory Generator for Blocksworld PPO Training.

This module generates invalid/bad trajectories for contrastive PPO training.
Each bad trajectory is guaranteed to contain at least one invalid transition,
ensuring that NO valid sub-trajectory can be extracted from it.

Three methods are combined:
1. Invalid Physics: Move blocks with something on top (impossible in Blocksworld)
2. Teleportation: Skip intermediate states (impossible single-step transitions)
3. Corruption: Insert random states mid-trajectory (discontinuity)

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import random


@dataclass
class BadTrajectoryInfo:
    """Information about a generated bad trajectory."""
    trajectory: List[np.ndarray]  # Sequence of states
    init_state: np.ndarray
    goal_state: np.ndarray  # The (invalid) goal
    num_blocks: int
    method: str  # 'invalid_physics', 'teleportation', 'corruption'
    invalid_transitions: List[int]  # Indices of invalid transitions


def is_valid_single_move(
    state_from: np.ndarray,
    state_to: np.ndarray,
    num_blocks: int = None,
) -> bool:
    """
    Check if state_to can be reached from state_from in exactly one action.
    
    Valid single moves in Blocksworld (4-operator):
    1. pick-up: block goes from table (0) to holding (N+1)
    2. put-down: block goes from holding (N+1) to table (0)
    3. stack: block goes from holding (N+1) to on-top-of-another (1-N)
    4. unstack: block goes from on-top-of-another (1-N) to holding (N+1)
    
    Only ONE block can change position per action.
    
    Args:
        state_from: Source state vector [N]
        state_to: Target state vector [N]
        num_blocks: Number of blocks (N), defaults to len(state_from)
    
    Returns:
        True if the transition is valid (single action)
    """
    # Infer N from state length if not provided
    N = num_blocks if num_blocks is not None else len(state_from)
    
    # Use actual state length (may be shorter than N due to padding)
    actual_len = min(len(state_from), len(state_to), N)
    
    HOLDING = N + 1
    
    # Find which blocks changed
    changed_blocks = []
    for i in range(actual_len):
        if state_from[i] != state_to[i]:
            changed_blocks.append(i)
    
    # Exactly one block must change per action
    if len(changed_blocks) != 1:
        return False
    
    block_idx = changed_blocks[0]
    pos_from = state_from[block_idx]
    pos_to = state_to[block_idx]
    
    # Check if the block was clear (nothing on top) in the source state
    # A block is clear if no other block has position = block_idx + 1
    def is_clear(state, block_idx):
        for i in range(min(len(state), actual_len)):
            if i != block_idx and state[i] == block_idx + 1:
                return False
        return True
    
    # Check if hand is empty
    def hand_empty(state):
        return not any(state[i] == HOLDING for i in range(min(len(state), actual_len)))
    
    # Validate based on action type
    # pick-up: table (0) -> holding (N+1), block must be clear, hand empty
    if pos_from == 0 and pos_to == HOLDING:
        return is_clear(state_from, block_idx) and hand_empty(state_from)
    
    # put-down: holding (N+1) -> table (0)
    if pos_from == HOLDING and pos_to == 0:
        return True  # If holding, can always put down
    
    # stack: holding (N+1) -> on block (1-N), target block must be clear
    if pos_from == HOLDING and 1 <= pos_to <= N:
        target_block = pos_to - 1
        return is_clear(state_from, target_block)
    
    # unstack: on block (1-N) -> holding (N+1), block must be clear, hand empty
    if 1 <= pos_from <= N and pos_to == HOLDING:
        return is_clear(state_from, block_idx) and hand_empty(state_from)
    
    # Any other transition is invalid
    return False


def count_invalid_transitions(
    trajectory: List[np.ndarray],
    num_blocks: int = None,
) -> List[int]:
    """
    Count and locate all invalid transitions in a trajectory.
    
    Args:
        trajectory: List of state vectors
        num_blocks: Number of blocks (defaults to length of first state)
    
    Returns:
        List of indices where trajectory[i] -> trajectory[i+1] is invalid
    """
    if not trajectory:
        return []
    
    # Infer num_blocks from trajectory if not provided
    if num_blocks is None:
        num_blocks = len(trajectory[0])
    
    invalid_indices = []
    for i in range(len(trajectory) - 1):
        if not is_valid_single_move(trajectory[i], trajectory[i+1], num_blocks):
            invalid_indices.append(i)
    return invalid_indices


def validate_bad_trajectory(
    trajectory: List[np.ndarray],
    num_blocks: int,
) -> bool:
    """
    Validate that a trajectory has at least one invalid transition.
    
    This ensures no valid sub-trajectory can be extracted.
    
    Args:
        trajectory: The trajectory to validate
        num_blocks: Number of blocks
    
    Returns:
        True if the trajectory contains at least one invalid move
    """
    invalid_indices = count_invalid_transitions(trajectory, num_blocks)
    return len(invalid_indices) > 0


class InvalidPhysicsGenerator:
    """
    Generate trajectories with invalid physics.
    
    These trajectories attempt to move blocks that have something on top,
    which violates Blocksworld preconditions.
    """
    
    def __init__(self, num_blocks: int, seed: Optional[int] = None):
        self.num_blocks = num_blocks
        self.rng = np.random.RandomState(seed)
    
    def generate(
        self,
        init_state: np.ndarray,
        num_steps: int = 3,
    ) -> BadTrajectoryInfo:
        """
        Generate a trajectory with invalid physics moves.
        
        Strategy: Find a block with something on top and try to move it.
        
        Args:
            init_state: Starting state
            num_steps: Number of steps in trajectory
        
        Returns:
            BadTrajectoryInfo with the invalid trajectory
        """
        # Use actual state length, not max blocks
        actual_N = len(init_state)
        N = actual_N
        HOLDING = N + 1
        
        trajectory = [init_state.copy()]
        current_state = init_state.copy()
        
        for step in range(num_steps):
            next_state = current_state.copy()
            
            # Find blocks with something on top (not clear)
            not_clear = []
            for i in range(N):
                for j in range(N):
                    if j != i and current_state[j] == i + 1:  # j is on top of i
                        not_clear.append(i)
                        break
            
            if not_clear and step == num_steps // 2:
                # Make an invalid move: try to pick up a blocked block
                blocked_block = self.rng.choice(not_clear)
                # Illegally move it to holding (this violates physics)
                next_state[blocked_block] = HOLDING
            else:
                # Make a random (possibly valid) move
                block_to_move = self.rng.randint(0, N)
                new_pos = self.rng.randint(0, HOLDING + 1)
                next_state[block_to_move] = new_pos
            
            trajectory.append(next_state)
            current_state = next_state
        
        # Verify trajectory has invalid transitions
        invalid_indices = count_invalid_transitions(trajectory, N)
        if not invalid_indices:
            # Force an invalid transition by random teleportation
            mid = len(trajectory) // 2
            trajectory[mid] = self._random_state(N)
            invalid_indices = count_invalid_transitions(trajectory, N)
        
        return BadTrajectoryInfo(
            trajectory=trajectory,
            init_state=init_state.copy(),
            goal_state=trajectory[-1].copy(),
            num_blocks=N,
            method='invalid_physics',
            invalid_transitions=invalid_indices,
        )
    
    def _random_state(self, N: int = None) -> np.ndarray:
        """Generate a random (but structurally valid) state."""
        if N is None:
            N = self.num_blocks
        state = np.zeros(N, dtype=np.int64)
        for i in range(N):
            # Random position: table, on another block, or holding (rarely)
            if self.rng.rand() < 0.1:
                state[i] = N + 1  # Holding
            elif self.rng.rand() < 0.5:
                state[i] = 0  # Table
            else:
                state[i] = self.rng.randint(1, N + 1)  # On some block
        return state


class TeleportationGenerator:
    """
    Generate trajectories with teleportation (impossible jumps).
    
    These trajectories skip intermediate states, creating transitions
    that would require multiple actions in a single step.
    """
    
    def __init__(self, num_blocks: int, seed: Optional[int] = None):
        self.num_blocks = num_blocks
        self.rng = np.random.RandomState(seed)
    
    def generate_from_good(
        self,
        good_trajectory: List[np.ndarray],
        skip_ratio: float = 0.5,
    ) -> BadTrajectoryInfo:
        """
        Create a bad trajectory by skipping states from a good one.
        
        Args:
            good_trajectory: A valid trajectory from FastDownward
            skip_ratio: Fraction of states to skip (0.5 = skip half)
        
        Returns:
            BadTrajectoryInfo with the teleportation trajectory
        """
        # Use actual state size from trajectory
        N = len(good_trajectory[0]) if good_trajectory else self.num_blocks
        
        if len(good_trajectory) < 3:
            # Too short to skip, corrupt instead
            return self._random_teleport(good_trajectory[0])
        
        # Select which indices to keep
        num_to_keep = max(2, int(len(good_trajectory) * (1 - skip_ratio)))
        
        # Always keep first and last
        indices = [0, len(good_trajectory) - 1]
        
        # Add some intermediate indices
        middle_indices = list(range(1, len(good_trajectory) - 1))
        self.rng.shuffle(middle_indices)
        
        for idx in middle_indices[:num_to_keep - 2]:
            indices.append(idx)
        
        indices = sorted(indices)
        
        # Create the skipped trajectory
        trajectory = [good_trajectory[i].copy() for i in indices]
        
        # Verify it has invalid transitions (skips should cause this)
        invalid_indices = count_invalid_transitions(trajectory, N)
        
        if not invalid_indices:
            # If somehow still valid, insert a random state
            mid = len(trajectory) // 2
            trajectory.insert(mid, self._random_state(N))
            invalid_indices = count_invalid_transitions(trajectory, N)
        
        return BadTrajectoryInfo(
            trajectory=trajectory,
            init_state=trajectory[0].copy(),
            goal_state=trajectory[-1].copy(),
            num_blocks=N,
            method='teleportation',
            invalid_transitions=invalid_indices,
        )
    
    def _random_teleport(self, init_state: np.ndarray) -> BadTrajectoryInfo:
        """Generate a random teleportation trajectory from init."""
        N = len(init_state)
        
        trajectory = [init_state.copy()]
        
        # Create states with multiple block changes (impossible in one step)
        current = init_state.copy()
        for _ in range(3):
            next_state = current.copy()
            # Change 2+ blocks at once (impossible)
            num_changes = self.rng.randint(2, N + 1)
            blocks_to_change = self.rng.choice(N, min(num_changes, N), replace=False)
            for block in blocks_to_change:
                next_state[block] = self.rng.randint(0, N + 2)
            trajectory.append(next_state)
            current = next_state
        
        invalid_indices = count_invalid_transitions(trajectory, N)
        
        return BadTrajectoryInfo(
            trajectory=trajectory,
            init_state=init_state.copy(),
            goal_state=trajectory[-1].copy(),
            num_blocks=N,
            method='teleportation',
            invalid_transitions=invalid_indices,
        )
    
    def _random_state(self, N: int = None) -> np.ndarray:
        """Generate a random state."""
        if N is None:
            N = self.num_blocks
        return self.rng.randint(0, N + 2, size=N)


class CorruptionGenerator:
    """
    Generate trajectories by corrupting valid ones.
    
    Insert random states mid-trajectory to create discontinuities.
    """
    
    def __init__(self, num_blocks: int, seed: Optional[int] = None):
        self.num_blocks = num_blocks
        self.rng = np.random.RandomState(seed)
    
    def generate_from_good(
        self,
        good_trajectory: List[np.ndarray],
        num_corruptions: int = 1,
    ) -> BadTrajectoryInfo:
        """
        Corrupt a good trajectory by inserting random states.
        
        Args:
            good_trajectory: A valid trajectory
            num_corruptions: Number of random states to insert
        
        Returns:
            BadTrajectoryInfo with the corrupted trajectory
        """
        N = len(good_trajectory[0]) if good_trajectory else self.num_blocks
        
        trajectory = [s.copy() for s in good_trajectory]
        
        for _ in range(num_corruptions):
            if len(trajectory) < 2:
                break
            
            # Insert a random state at a random position (not first or last)
            insert_pos = self.rng.randint(1, len(trajectory))
            random_state = self._random_state(N)
            trajectory.insert(insert_pos, random_state)
        
        invalid_indices = count_invalid_transitions(trajectory, N)
        
        # Ensure at least one invalid transition
        if not invalid_indices:
            # Swap two adjacent states
            if len(trajectory) >= 3:
                swap_pos = self.rng.randint(1, len(trajectory) - 1)
                trajectory[swap_pos], trajectory[swap_pos - 1] = \
                    trajectory[swap_pos - 1], trajectory[swap_pos]
                invalid_indices = count_invalid_transitions(trajectory, N)
        
        return BadTrajectoryInfo(
            trajectory=trajectory,
            init_state=trajectory[0].copy(),
            goal_state=trajectory[-1].copy(),
            num_blocks=N,
            method='corruption',
            invalid_transitions=invalid_indices,
        )
    
    def _random_state(self, N: int = None) -> np.ndarray:
        """Generate a random state."""
        if N is None:
            N = self.num_blocks
        state = np.zeros(N, dtype=np.int64)
        for i in range(N):
            if self.rng.rand() < 0.6:
                state[i] = 0  # On table
            elif self.rng.rand() < 0.5:
                state[i] = self.rng.randint(1, N + 1)  # On block
            else:
                state[i] = N + 1  # Holding
        return state


class BadTrajectoryGenerator:
    """
    Combined bad trajectory generator using multiple methods.
    
    Generates invalid trajectories that are guaranteed to contain
    at least one invalid transition, preventing any valid
    sub-trajectory extraction.
    
    Args:
        num_blocks: Number of blocks in Blocksworld
        seed: Random seed for reproducibility
        method_weights: Relative weights for each method
    """
    
    def __init__(
        self,
        num_blocks: int,
        seed: Optional[int] = None,
        method_weights: Optional[Dict[str, float]] = None,
    ):
        self.num_blocks = num_blocks
        self.rng = np.random.RandomState(seed)
        
        # Default weights
        self.method_weights = method_weights or {
            'invalid_physics': 0.3,
            'teleportation': 0.4,
            'corruption': 0.3,
        }
        
        # Normalize weights
        total = sum(self.method_weights.values())
        self.method_weights = {k: v / total for k, v in self.method_weights.items()}
        
        # Initialize generators
        self.invalid_physics = InvalidPhysicsGenerator(num_blocks, seed)
        self.teleportation = TeleportationGenerator(num_blocks, seed)
        self.corruption = CorruptionGenerator(num_blocks, seed)
    
    def generate(
        self,
        init_state: np.ndarray,
        good_trajectory: Optional[List[np.ndarray]] = None,
        method: Optional[str] = None,
    ) -> BadTrajectoryInfo:
        """
        Generate a bad trajectory.
        
        Args:
            init_state: Initial state
            good_trajectory: Optional good trajectory to corrupt/skip
            method: Force a specific method, or None for random selection
        
        Returns:
            BadTrajectoryInfo with the bad trajectory
        """
        if method is None:
            # Random selection based on weights
            methods = list(self.method_weights.keys())
            weights = list(self.method_weights.values())
            method = self.rng.choice(methods, p=weights)
        
        if method == 'invalid_physics':
            return self.invalid_physics.generate(init_state)
        
        elif method == 'teleportation':
            if good_trajectory is not None and len(good_trajectory) >= 3:
                return self.teleportation.generate_from_good(good_trajectory)
            else:
                return self.teleportation._random_teleport(init_state)
        
        elif method == 'corruption':
            if good_trajectory is not None:
                return self.corruption.generate_from_good(good_trajectory)
            else:
                # Create a short random trajectory and corrupt it
                fake_traj = [init_state.copy()]
                current = init_state.copy()
                for _ in range(3):
                    next_s = current.copy()
                    next_s[self.rng.randint(0, self.num_blocks)] = \
                        self.rng.randint(0, self.num_blocks + 2)
                    fake_traj.append(next_s)
                    current = next_s
                return self.corruption.generate_from_good(fake_traj)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def generate_batch(
        self,
        good_trajectories: List[Tuple[np.ndarray, List[np.ndarray]]],
        ratio: float = 1.0,
    ) -> List[BadTrajectoryInfo]:
        """
        Generate a batch of bad trajectories.
        
        Args:
            good_trajectories: List of (init_state, trajectory) tuples
            ratio: Ratio of bad to good trajectories (1.0 = same count)
        
        Returns:
            List of BadTrajectoryInfo objects
        """
        num_bad = int(len(good_trajectories) * ratio)
        bad_trajectories = []
        
        for i in range(num_bad):
            # Use a good trajectory as base (cycling if needed)
            idx = i % len(good_trajectories)
            init_state, good_traj = good_trajectories[idx]
            
            bad_info = self.generate(init_state, good_traj)
            
            # Validate it has invalid transitions
            if not bad_info.invalid_transitions:
                # Retry with different method
                for method in ['teleportation', 'corruption', 'invalid_physics']:
                    bad_info = self.generate(init_state, good_traj, method=method)
                    if bad_info.invalid_transitions:
                        break
            
            if bad_info.invalid_transitions:
                bad_trajectories.append(bad_info)
        
        return bad_trajectories
    
    def generate_from_init_states(
        self,
        init_states: List[np.ndarray],
        num_per_state: int = 1,
    ) -> List[BadTrajectoryInfo]:
        """
        Generate bad trajectories from initial states only.
        
        Args:
            init_states: List of initial states
            num_per_state: Number of bad trajectories per state
        
        Returns:
            List of BadTrajectoryInfo objects
        """
        bad_trajectories = []
        
        for init_state in init_states:
            for _ in range(num_per_state):
                bad_info = self.generate(init_state)
                if bad_info.invalid_transitions:
                    bad_trajectories.append(bad_info)
        
        return bad_trajectories


def convert_bad_trajectory_to_dict(bad_info: BadTrajectoryInfo) -> Dict[str, Any]:
    """
    Convert BadTrajectoryInfo to a dictionary for dataset use.
    
    Args:
        bad_info: BadTrajectoryInfo object
    
    Returns:
        Dictionary with trajectory information
    """
    return {
        'trajectory': [s.copy() for s in bad_info.trajectory],
        'init_state': bad_info.init_state.copy(),
        'goal_state': bad_info.goal_state.copy(),
        'num_blocks': bad_info.num_blocks,
        'method': bad_info.method,
        'is_valid': False,
        'reward': -1.0,
        'plan_length': len(bad_info.trajectory) - 1,
    }

