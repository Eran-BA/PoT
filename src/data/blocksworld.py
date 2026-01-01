"""
Blocksworld Dataset for PlanBench.

Loads the PlanBench Blocksworld dataset from HuggingFace and converts
PDDL-style states to discrete vectors for neural network training.

State representation:
- For N blocks: state is [pos_block_0, pos_block_1, ..., pos_block_{N-1}]
- Each pos_block_i in {0=table, 1..N = on_block_j}

Task types:
- Transition prediction: (s_t, a_t) -> s_{t+1}
- Goal-conditioned planning: (s_0, s_goal) -> plan

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import re
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import Dataset
from tqdm import tqdm


# =============================================================================
# PDDL Parsing Utilities
# =============================================================================

def parse_pddl_state(state_str: str, block_names: List[str]) -> np.ndarray:
    """
    Parse a PDDL state string into a discrete vector.
    
    Args:
        state_str: PDDL-style state description, e.g.:
            "(on a b) (on-table c) (clear a) (clear c) (holding d)"
            or "(on a b) (ontable c) (clear a) (clear c)"
        block_names: List of block names in canonical order, e.g. ['a', 'b', 'c', 'd']
    
    Returns:
        np.ndarray of shape [N] where N = len(block_names)
        Each element is the position:
            0 = on table
            1..N = on top of block (index+1)
            -1 = being held (if holding predicate present)
    """
    N = len(block_names)
    name_to_idx = {name.lower(): i for i, name in enumerate(block_names)}
    
    # Initialize: unknown position (-1 will be resolved later)
    positions = np.zeros(N, dtype=np.int64)
    
    # Parse predicates
    state_lower = state_str.lower()
    
    # Pattern for (on X Y) - X is on top of Y
    on_pattern = r'\(on\s+(\w+)\s+(\w+)\)'
    for match in re.finditer(on_pattern, state_lower):
        top_block = match.group(1)
        bottom_block = match.group(2)
        if top_block in name_to_idx and bottom_block in name_to_idx:
            top_idx = name_to_idx[top_block]
            bottom_idx = name_to_idx[bottom_block]
            # Position = bottom_block index + 1 (0 is table)
            positions[top_idx] = bottom_idx + 1
    
    # Pattern for (on-table X) or (ontable X) - X is on the table
    ontable_pattern = r'\(on-?table\s+(\w+)\)'
    for match in re.finditer(ontable_pattern, state_lower):
        block = match.group(1)
        if block in name_to_idx:
            positions[name_to_idx[block]] = 0
    
    # Pattern for (holding X) - X is being held
    holding_pattern = r'\(holding\s+(\w+)\)'
    for match in re.finditer(holding_pattern, state_lower):
        block = match.group(1)
        if block in name_to_idx:
            # Use N+1 as "holding" state
            positions[name_to_idx[block]] = N + 1
    
    return positions


def state_to_pddl(state: np.ndarray, block_names: List[str]) -> str:
    """
    Convert a state vector back to PDDL string (for debugging).
    
    Args:
        state: Position vector [N]
        block_names: Block names in order
    
    Returns:
        PDDL-style string
    """
    N = len(block_names)
    predicates = []
    
    for i, pos in enumerate(state):
        block = block_names[i]
        if pos == 0:
            predicates.append(f"(on-table {block})")
        elif pos == N + 1:
            predicates.append(f"(holding {block})")
        else:
            bottom_block = block_names[pos - 1]
            predicates.append(f"(on {block} {bottom_block})")
    
    return " ".join(predicates)


def parse_action(action_str: str, block_names: List[str]) -> Tuple[int, Optional[int], Optional[int]]:
    """
    Parse a Blocksworld action.
    
    Actions:
    - pick-up X: pick block X from table (action_type=0)
    - put-down X: put held block X on table (action_type=1)
    - stack X Y: put X on top of Y (action_type=2)
    - unstack X Y: pick up X from Y (action_type=3)
    
    Args:
        action_str: Action string, e.g. "(pick-up a)" or "(stack a b)"
        block_names: List of block names
    
    Returns:
        Tuple of (action_type, block1_idx, block2_idx)
        block2_idx is None for pick-up/put-down
    """
    name_to_idx = {name.lower(): i for i, name in enumerate(block_names)}
    action_lower = action_str.lower().strip()
    
    # Pick-up
    match = re.match(r'\(pick-up\s+(\w+)\)', action_lower)
    if match:
        block = match.group(1)
        return (0, name_to_idx.get(block, 0), None)
    
    # Put-down
    match = re.match(r'\(put-down\s+(\w+)\)', action_lower)
    if match:
        block = match.group(1)
        return (1, name_to_idx.get(block, 0), None)
    
    # Stack
    match = re.match(r'\(stack\s+(\w+)\s+(\w+)\)', action_lower)
    if match:
        block1, block2 = match.group(1), match.group(2)
        return (2, name_to_idx.get(block1, 0), name_to_idx.get(block2, 0))
    
    # Unstack
    match = re.match(r'\(unstack\s+(\w+)\s+(\w+)\)', action_lower)
    if match:
        block1, block2 = match.group(1), match.group(2)
        return (3, name_to_idx.get(block1, 0), name_to_idx.get(block2, 0))
    
    # Unknown action
    return (-1, None, None)


def extract_block_names(state_str: str) -> List[str]:
    """
    Extract block names from a PDDL state string.
    
    Args:
        state_str: PDDL state string
    
    Returns:
        Sorted list of unique block names
    """
    # Find all block names in predicates
    pattern = r'\((?:on|on-table|ontable|clear|holding|arm-empty)\s+(\w+)(?:\s+(\w+))?\)'
    blocks = set()
    
    for match in re.finditer(pattern, state_str.lower()):
        if match.group(1):
            blocks.add(match.group(1))
        if match.group(2):
            blocks.add(match.group(2))
    
    return sorted(blocks)


# =============================================================================
# Dataset Classes
# =============================================================================

class BlocksworldDataset(Dataset):
    """
    Blocksworld dataset with sub-trajectory augmentation for training.
    
    For training (augment=True):
        - Loads full trajectories
        - Extracts ALL C(n+1, 2) sub-trajectories as (init, goal) pairs
        - Shuffles globally (like DQN experience replay)
        - Re-shuffles each epoch via on_epoch_end()
    
    For test/val (augment=False):
        - Uses original (init, goal) pairs without augmentation
        - No data leakage from training trajectories
    
    Args:
        data_dir: Path to dataset directory
        split: 'train', 'val', or 'test'
        max_blocks: Maximum number of blocks (filters larger problems)
        mode: 'transition' for single-step, 'goal' for (s_init, s_goal) pairs
        max_plan_length: Filter to plans with at most this many steps (None = no filter)
        augment: If True, extract sub-trajectories (default: True for train, False otherwise)
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        max_blocks: int = 8,
        mode: str = 'goal',  # 'transition' or 'goal'
        max_plan_length: Optional[int] = None,
        augment: Optional[bool] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_blocks = max_blocks
        self.mode = mode
        self.max_plan_length = max_plan_length
        
        # Default: augment only for training
        self.augment = augment if augment is not None else (split == 'train')
        
        split_dir = self.data_dir / split
        
        # Check if trajectories exist for augmentation
        traj_file = split_dir / 'trajectories.npz'
        has_trajectories = traj_file.exists()
        
        if self.augment and has_trajectories:
            # Load trajectories and extract sub-trajectories
            self._load_with_augmentation(split_dir)
        else:
            if self.augment and not has_trajectories:
                print(f"  Warning: No trajectories available, augmentation disabled")
                self.augment = False
            # Load original data without augmentation
            self._load_original(split_dir)
        
        # Epoch indices for shuffling (DQN-style experience replay)
        self._epoch_indices = np.arange(len(self._samples))
        if self.augment:
            np.random.shuffle(self._epoch_indices)
        
        aug_str = "SUB-TRAJECTORY AUGMENTATION" if self.augment else "NO AUGMENTATION"
        print(f"[{split}] Loaded {len(self)} samples (mode={mode}, max_blocks={max_blocks})")
        print(f"  Augmentation: {aug_str}")
    
    def _load_with_augmentation(self, split_dir: Path):
        """Load trajectories and extract all C(n+1,2) sub-trajectories."""
        traj_file = split_dir / 'trajectories.npz'
        
        if not traj_file.exists():
            raise FileNotFoundError(
                f"Trajectory data not found at {traj_file}. "
                f"Run with --download flag to fetch from HuggingFace."
            )
        
        data = np.load(traj_file, allow_pickle=True)
        trajectories = data['trajectories']  # [N_traj, max_len, max_blocks]
        num_blocks_arr = data['num_blocks']  # [N_traj]
        
        # Filter by max_blocks
        mask = num_blocks_arr <= self.max_blocks
        trajectories = trajectories[mask]
        num_blocks_arr = num_blocks_arr[mask]
        
        # Extract ALL sub-trajectories: C(n+1, 2) pairs per trajectory
        self._samples = []
        
        for traj_idx, traj in enumerate(trajectories):
            num_blocks = num_blocks_arr[traj_idx]
            
            # Find actual trajectory length (non-zero states)
            # Assume state is zero-padded
            traj_len = 0
            for t in range(len(traj)):
                if np.any(traj[t, :num_blocks] != 0) or t == 0:
                    traj_len = t + 1
                else:
                    # Check if this looks like a valid state (could be all on table)
                    # A state where all blocks are on table is valid
                    if t > 0 and np.all(traj[t] == 0) and np.all(traj[t-1] == 0):
                        break
                    traj_len = t + 1
            
            # Actually, let's find the real length by looking for consecutive zeros
            # Better approach: count non-duplicated states
            traj_len = len(traj)
            for t in range(1, len(traj)):
                if np.array_equal(traj[t], traj[t-1]) and np.all(traj[t] == 0):
                    traj_len = t
                    break
            
            if traj_len < 2:
                continue
            
            # Extract all C(traj_len, 2) sub-trajectories
            # Each (i, j) pair where i < j gives a sub-trajectory from state i to state j
            for i in range(traj_len):
                for j in range(i + 1, traj_len):
                    plan_length = j - i
                    
                    # Filter by max_plan_length if specified
                    if self.max_plan_length is not None and plan_length > self.max_plan_length:
                        continue
                    
                    self._samples.append({
                        'init_state': traj[i, :self.max_blocks].copy(),
                        'goal_state': traj[j, :self.max_blocks].copy(),
                        'plan_length': plan_length,
                        'num_blocks': num_blocks,
                    })
        
        n_original = len(trajectories)
        print(f"  Extracted {len(self._samples)} sub-trajectories from {n_original} original trajectories")
    
    def _load_original(self, split_dir: Path):
        """Load original data without augmentation."""
        if self.mode == 'transition':
            data_file = split_dir / 'transitions.npz'
        else:
            data_file = split_dir / 'goals.npz'
        
        if not data_file.exists():
            raise FileNotFoundError(
                f"Dataset not found at {data_file}. "
                f"Run with --download flag to fetch from HuggingFace."
            )
        
        data = np.load(data_file, allow_pickle=True)
        
        if self.mode == 'transition':
            states = data['states']
            actions = data['actions']
            next_states = data['next_states']
            num_blocks_arr = data['num_blocks']
            
            # Filter by max_blocks
            mask = num_blocks_arr <= self.max_blocks
            
            self._samples = []
            for i in np.where(mask)[0]:
                self._samples.append({
                    'state': states[i],
                    'action': actions[i],
                    'next_state': next_states[i],
                    'num_blocks': num_blocks_arr[i],
                })
        else:
            init_states = data['init_states']
            goal_states = data['goal_states']
            plan_lengths = data['plan_lengths']
            num_blocks_arr = data['num_blocks']
            
            # Filter by max_blocks and max_plan_length
            mask = num_blocks_arr <= self.max_blocks
            if self.max_plan_length is not None:
                mask = mask & (plan_lengths <= self.max_plan_length)
            
            self._samples = []
            for i in np.where(mask)[0]:
                self._samples.append({
                    'init_state': init_states[i],
                    'goal_state': goal_states[i],
                    'plan_length': plan_lengths[i],
                    'num_blocks': num_blocks_arr[i],
                })
    
    def __len__(self):
        return len(self._epoch_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = self._epoch_indices[idx]
        sample = self._samples[real_idx]
        
        if self.mode == 'transition' and not self.augment:
            return {
                'state': torch.LongTensor(sample['state']),
                'action': torch.LongTensor(sample['action']),
                'next_state': torch.LongTensor(sample['next_state']),
                'num_blocks': torch.tensor(sample['num_blocks'], dtype=torch.long),
            }
        else:
            return {
                'init_state': torch.LongTensor(sample['init_state']),
                'goal_state': torch.LongTensor(sample['goal_state']),
                'plan_length': torch.tensor(sample['plan_length'], dtype=torch.long),
                'num_blocks': torch.tensor(sample['num_blocks'], dtype=torch.long),
            }
    
    def on_epoch_end(self):
        """Shuffle for next epoch (DQN-style experience replay)."""
        if self.augment:
            self._epoch_indices = np.random.permutation(self._epoch_indices)
    
    @property
    def vocab_size(self) -> int:
        """Vocabulary size: table + N blocks + holding."""
        return self.max_blocks + 2
    
    @property
    def seq_len(self) -> int:
        """Sequence length = max number of blocks."""
        return self.max_blocks


class BlocksworldTrajectoryDataset(Dataset):
    """
    Blocksworld dataset with full trajectories for rollout training.
    
    Each sample is a full (s_0, a_0, s_1, a_1, ..., s_T) trajectory.
    Useful for training with teacher forcing on full plans.
    
    Args:
        data_dir: Path to dataset directory
        split: 'train', 'val', or 'test'
        max_blocks: Maximum number of blocks
        max_plan_length: Maximum trajectory length (pads shorter, truncates longer)
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        max_blocks: int = 8,
        max_plan_length: int = 10,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_blocks = max_blocks
        self.max_plan_length = max_plan_length
        
        # Load trajectory data
        split_dir = self.data_dir / split
        data_file = split_dir / 'trajectories.npz'
        
        if not data_file.exists():
            raise FileNotFoundError(
                f"Trajectory data not found at {data_file}. "
                f"Run with --download flag to fetch from HuggingFace."
            )
        
        data = np.load(data_file, allow_pickle=True)
        
        self.trajectories = data['trajectories']  # List of state sequences
        self.actions_seq = data['actions']        # List of action sequences
        self.num_blocks_arr = data['num_blocks']  # [N]
        
        # Filter by max_blocks
        mask = self.num_blocks_arr <= max_blocks
        self.trajectories = self.trajectories[mask]
        self.actions_seq = self.actions_seq[mask]
        self.num_blocks_arr = self.num_blocks_arr[mask]
        
        print(f"[{split}] Loaded {len(self)} trajectories (max_len={max_plan_length})")
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj = self.trajectories[idx]
        actions = self.actions_seq[idx]
        num_blocks = self.num_blocks_arr[idx]
        
        T = len(traj)
        
        # Pad/truncate to max_plan_length + 1 states
        padded_states = np.zeros((self.max_plan_length + 1, self.max_blocks), dtype=np.int64)
        padded_actions = np.zeros((self.max_plan_length, 3), dtype=np.int64)
        mask = np.zeros(self.max_plan_length + 1, dtype=np.bool_)
        
        # Fill in actual trajectory
        actual_len = min(T, self.max_plan_length + 1)
        padded_states[:actual_len] = traj[:actual_len, :self.max_blocks]
        mask[:actual_len] = True
        
        if len(actions) > 0:
            actual_action_len = min(len(actions), self.max_plan_length)
            padded_actions[:actual_action_len] = actions[:actual_action_len]
        
        return {
            'states': torch.LongTensor(padded_states),      # [T+1, N]
            'actions': torch.LongTensor(padded_actions),    # [T, 3]
            'mask': torch.BoolTensor(mask),                 # [T+1]
            'trajectory_length': torch.tensor(actual_len - 1, dtype=torch.long),
            'num_blocks': torch.tensor(num_blocks, dtype=torch.long),
        }


# =============================================================================
# Download and Preprocessing
# =============================================================================

def download_blocksworld_dataset(
    output_dir: str,
    max_blocks: int = 8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    generate_trajectories: bool = False,
    fd_path: str = "fast-downward.py",
    fd_timeout: int = 30,
) -> None:
    """
    Download and preprocess Blocksworld dataset from HuggingFace PlanBench.
    
    Creates:
    - goals.npz: (init_state, goal_state) pairs
    - trajectories.npz: full state-action sequences (if generate_trajectories=True)
    
    Args:
        output_dir: Directory to save the dataset
        max_blocks: Maximum number of blocks to include
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        generate_trajectories: If True, use FastDownward to generate plans/trajectories
        fd_path: Path to FastDownward executable
        fd_timeout: Per-problem timeout in seconds
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Please install the 'datasets' library: pip install datasets"
        )
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Downloading PlanBench Blocksworld from HuggingFace...")
    print(f"  Max blocks: {max_blocks}")
    
    # Load the dataset from HuggingFace
    dataset = load_dataset("chiayewken/blocksworld", split="train")
    
    # Process all examples - extract (init, goal) pairs
    all_goals = []
    all_block_names = []  # Store block names for trajectory generation
    
    for example in tqdm(dataset, desc="Processing examples"):
        try:
            result = _process_planbench_example(example, max_blocks)
            if result is not None:
                all_goals.append(result)
                # Also store block names for potential trajectory generation
                all_block_names.append(result.get('block_names', []))
        except Exception as e:
            # Skip malformed examples
            continue
    
    print(f"  Extracted {len(all_goals)} goal pairs (init → goal)")
    
    if len(all_goals) == 0:
        raise ValueError("No valid examples found in dataset. Check max_blocks parameter.")
    
    # Split into train/val/test
    n_total = len(all_goals)
    n_test = max(1, int(n_total * test_ratio))
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_test - n_val
    
    # Shuffle
    np.random.seed(42)  # For reproducibility
    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    # Optional: Generate trajectories using FastDownward
    all_trajectories = None
    if generate_trajectories:
        all_trajectories = _generate_trajectories_with_fd(
            all_goals, all_block_names, max_blocks, fd_path, fd_timeout
        )
    
    # Save each split
    for split, split_idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        split_dir = output_path / split
        split_dir.mkdir(exist_ok=True)
        
        # Save goals
        split_goals = [all_goals[i] for i in split_idx]
        _save_goals(split_dir, split_goals, max_blocks)
        
        # Save trajectories if generated
        has_trajectories = False
        if all_trajectories is not None:
            split_trajs = [all_trajectories[i] for i in split_idx if all_trajectories[i] is not None]
            if split_trajs:
                _save_trajectories(split_dir, split_trajs, max_blocks)
                has_trajectories = True
                print(f"  {split}: {len(split_idx)} problems, {len(split_trajs)} trajectories")
            else:
                print(f"  {split}: {len(split_idx)} problems, no trajectories")
        else:
            print(f"  {split}: {len(split_idx)} problems")
        
        # Save metadata
        with open(split_dir / 'metadata.json', 'w') as f:
            json.dump({
                'num_samples': len(split_idx),
                'max_blocks': max_blocks,
                'vocab_size': max_blocks + 2,  # table + blocks + holding
                'has_trajectories': has_trajectories,
            }, f)
    
    print(f"Dataset saved to {output_dir}")


def _generate_trajectories_with_fd(
    all_goals: List[Dict],
    all_block_names: List[List[str]],
    max_blocks: int,
    fd_path: str,
    fd_timeout: int,
) -> List[Optional[Dict]]:
    """
    Generate trajectories for all problems using FastDownward.
    
    Returns a list of trajectory dicts (or None for failed problems).
    """
    from .blocksworld_planner import TrajectoryGenerator
    
    generator = TrajectoryGenerator(
        fd_path=fd_path,
        timeout=fd_timeout,
        max_blocks=max_blocks,
    )
    
    if not generator.is_available():
        print("  WARNING: FastDownward not available, skipping trajectory generation")
        print(f"           Install from: https://github.com/aibasel/downward")
        return None
    
    print("  Generating trajectories with FastDownward...")
    
    trajectories = []
    for i, goal in enumerate(tqdm(all_goals, desc="Solving with FD")):
        # Create block names list (padded)
        block_names = all_block_names[i] if i < len(all_block_names) else []
        if not block_names:
            # Reconstruct from num_blocks
            num_blocks = goal['num_blocks']
            block_names = [chr(ord('a') + j) for j in range(num_blocks)]
        
        # Pad to max_blocks
        while len(block_names) < max_blocks:
            block_names.append(f'_pad_{len(block_names)}')
        
        result = generator.generate_trajectory(
            goal['init_state'],
            goal['goal_state'],
            block_names,
        )
        trajectories.append(result)
    
    stats = generator.get_stats()
    print(f"  Solved {stats['solved']}/{stats['solved'] + stats['failed']} problems")
    print(f"  Average plan length: {stats['avg_plan_length']:.1f} actions")
    
    return trajectories


def _parse_pddl_instance(instance_str: str) -> Tuple[List[str], str, str]:
    """
    Parse a PDDL problem instance.
    
    Args:
        instance_str: Raw PDDL instance string
    
    Returns:
        Tuple of (block_names, init_predicates, goal_predicates)
    """
    # Extract objects (block names)
    objects_match = re.search(r'\(:objects\s+([^)]+)\)', instance_str, re.IGNORECASE)
    if objects_match:
        objects_str = objects_match.group(1).strip()
        block_names = objects_str.split()
    else:
        block_names = []
    
    # Extract init predicates
    init_match = re.search(r'\(:init\s+(.*?)\)\s*\(:goal', instance_str, re.DOTALL | re.IGNORECASE)
    if init_match:
        init_str = init_match.group(1).strip()
    else:
        init_str = ""
    
    # Extract goal predicates
    goal_match = re.search(r'\(:goal\s*\(and\s+(.*?)\)\s*\)\s*\)', instance_str, re.DOTALL | re.IGNORECASE)
    if goal_match:
        goal_str = goal_match.group(1).strip()
    else:
        # Try simpler goal format
        goal_match = re.search(r'\(:goal\s+(.*?)\)\s*\)', instance_str, re.DOTALL | re.IGNORECASE)
        goal_str = goal_match.group(1).strip() if goal_match else ""
    
    return block_names, init_str, goal_str


def _process_planbench_example(
    example: Dict[str, Any],
    max_blocks: int,
) -> Optional[Dict]:
    """
    Process a single PlanBench PDDL example.
    
    The dataset has 'domain' and 'instance' fields with raw PDDL.
    This function extracts (init_state, goal_state) pairs.
    
    Returns:
        Dict with init_state, goal_state, num_blocks or None if invalid
    """
    instance_str = example.get('instance', '')
    if not instance_str:
        return None
    
    # Parse PDDL instance
    block_names, init_str, goal_str = _parse_pddl_instance(instance_str)
    
    if not block_names or not init_str:
        return None
    
    num_blocks = len(block_names)
    if num_blocks > max_blocks or num_blocks == 0:
        return None
    
    # Pad block names to max_blocks
    padded_block_names = block_names.copy()
    while len(padded_block_names) < max_blocks:
        padded_block_names.append(f'_pad_{len(padded_block_names)}')
    
    # Parse initial and goal states
    init_vec = parse_pddl_state(init_str, padded_block_names)
    goal_vec = parse_pddl_state(goal_str, padded_block_names)
    
    return {
        'init_state': init_vec,
        'goal_state': goal_vec,
        'num_blocks': num_blocks,
        'block_names': padded_block_names,  # For trajectory generation
    }


def _simulate_action(
    state: np.ndarray,
    action: Tuple[int, Optional[int], Optional[int]],
    num_blocks: int,
) -> np.ndarray:
    """
    Simulate a Blocksworld action on a state.
    
    This is a simplified simulation that may not catch all edge cases.
    """
    next_state = state.copy()
    action_type, block1, block2 = action
    
    if action_type == 0:  # pick-up: block1 from table -> holding
        if block1 is not None and block1 < len(next_state):
            next_state[block1] = num_blocks + 1  # holding state
    
    elif action_type == 1:  # put-down: holding block1 -> table
        if block1 is not None and block1 < len(next_state):
            next_state[block1] = 0  # on table
    
    elif action_type == 2:  # stack: put block1 on block2
        if block1 is not None and block2 is not None:
            if block1 < len(next_state) and block2 < len(next_state):
                next_state[block1] = block2 + 1  # on block2
    
    elif action_type == 3:  # unstack: pick block1 from block2
        if block1 is not None and block1 < len(next_state):
            next_state[block1] = num_blocks + 1  # holding state
    
    return next_state


def _save_transitions(split_dir: Path, transitions: List[Dict], max_blocks: int):
    """Save transition data to npz file."""
    states = np.zeros((len(transitions), max_blocks), dtype=np.int64)
    actions = np.zeros((len(transitions), 3), dtype=np.int64)
    next_states = np.zeros((len(transitions), max_blocks), dtype=np.int64)
    num_blocks = np.zeros(len(transitions), dtype=np.int64)
    
    for i, t in enumerate(transitions):
        states[i, :len(t['state'])] = t['state'][:max_blocks]
        actions[i] = t['action'][:3] if len(t['action']) >= 3 else t['action'] + [0] * (3 - len(t['action']))
        next_states[i, :len(t['next_state'])] = t['next_state'][:max_blocks]
        num_blocks[i] = t['num_blocks']
    
    np.savez(
        split_dir / 'transitions.npz',
        states=states,
        actions=actions,
        next_states=next_states,
        num_blocks=num_blocks,
    )


def _save_goals(split_dir: Path, goals: List[Dict], max_blocks: int):
    """Save goal data to npz file."""
    init_states = np.zeros((len(goals), max_blocks), dtype=np.int64)
    goal_states = np.zeros((len(goals), max_blocks), dtype=np.int64)
    plan_lengths = np.zeros(len(goals), dtype=np.int64)
    num_blocks = np.zeros(len(goals), dtype=np.int64)
    
    for i, g in enumerate(goals):
        init_states[i, :len(g['init_state'])] = g['init_state'][:max_blocks]
        goal_states[i, :len(g['goal_state'])] = g['goal_state'][:max_blocks]
        plan_lengths[i] = g.get('plan_length', 0)  # May not be available
        num_blocks[i] = g['num_blocks']
    
    np.savez(
        split_dir / 'goals.npz',
        init_states=init_states,
        goal_states=goal_states,
        plan_lengths=plan_lengths,
        num_blocks=num_blocks,
    )


def _save_trajectories(split_dir: Path, trajectories: List[Dict], max_blocks: int):
    """Save trajectory data to npz file."""
    # Handle different trajectory formats (legacy 'states' or new 'trajectory')
    def get_states(t):
        if 'states' in t:
            return t['states']
        elif 'trajectory' in t:
            return t['trajectory']
        else:
            return []
    
    # Find max trajectory length
    max_len = max(len(get_states(t)) for t in trajectories)
    
    all_trajs = []
    all_actions = []
    plan_lengths = np.zeros(len(trajectories), dtype=np.int64)
    num_blocks = np.zeros(len(trajectories), dtype=np.int64)
    
    for i, t in enumerate(trajectories):
        states = get_states(t)
        
        # Convert states to padded array
        states_arr = np.zeros((max_len, max_blocks), dtype=np.int64)
        for j, s in enumerate(states):
            if isinstance(s, np.ndarray):
                states_arr[j, :min(len(s), max_blocks)] = s[:max_blocks]
            else:
                states_arr[j, :min(len(s), max_blocks)] = np.array(s)[:max_blocks]
        all_trajs.append(states_arr)
        
        # Convert actions - handle PlanAction objects or raw lists
        actions_arr = np.zeros((max_len - 1, 3), dtype=np.int64)
        actions = t.get('actions', [])
        for j, a in enumerate(actions):
            if j < max_len - 1:
                if hasattr(a, 'name'):
                    # PlanAction object - encode as (action_type, arg1_idx, arg2_idx)
                    action_types = {'pick-up': 0, 'put-down': 1, 'stack': 2, 'unstack': 3}
                    action_type = action_types.get(a.name, 0)
                    arg1 = ord(a.args[0][0]) - ord('a') if a.args else 0
                    arg2 = ord(a.args[1][0]) - ord('a') if len(a.args) > 1 else 0
                    actions_arr[j] = [action_type, arg1, arg2]
                elif isinstance(a, (list, tuple)):
                    actions_arr[j] = list(a[:3]) + [0] * (3 - len(a))
                else:
                    actions_arr[j] = [0, 0, 0]
        all_actions.append(actions_arr)
        
        plan_lengths[i] = t.get('plan_length', len(states) - 1)
        num_blocks[i] = t.get('num_blocks', max_blocks)
    
    np.savez(
        split_dir / 'trajectories.npz',
        trajectories=np.array(all_trajs),
        actions=np.array(all_actions),
        plan_lengths=plan_lengths,
        num_blocks=num_blocks,
    )


# =============================================================================
# PPO Dataset: Good (Augmented) + Bad (No Augmentation) Trajectories
# =============================================================================

class BlocksworldPPODataset(Dataset):
    """
    Dataset for PPO training with good and bad trajectories.
    
    Good trajectories:
    - Loaded from FastDownward solutions
    - Augmented with C(n+1, 2) sub-trajectory extraction
    - Labeled with is_valid=True, reward=+1
    
    Bad trajectories:
    - Generated using combined methods (physics/teleport/corruption)
    - NO augmentation (full trajectories only)
    - Each guaranteed to have at least one invalid transition
    - Labeled with is_valid=False, reward=-1
    
    Args:
        data_dir: Directory containing blocksworld data
        split: 'train', 'val', or 'test'
        max_blocks: Maximum number of blocks
        good_bad_ratio: Ratio of bad to good samples (1.0 = equal)
        max_plan_length: Maximum plan length to include
        seed: Random seed for bad trajectory generation
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        max_blocks: int = 8,
        good_bad_ratio: float = 1.0,
        max_plan_length: Optional[int] = None,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_blocks = max_blocks
        self.good_bad_ratio = good_bad_ratio
        self.max_plan_length = max_plan_length
        self.seed = seed
        
        # Load and prepare samples
        self._samples = []
        self._load_samples()
        
        # Shuffle indices for iteration
        self._epoch_indices = np.arange(len(self._samples))
        np.random.shuffle(self._epoch_indices)
        
    def _load_samples(self):
        """Load good and bad trajectory samples."""
        split_dir = self.data_dir / self.split
        
        # Check for trajectory file
        traj_file = split_dir / 'trajectories.npz'
        if not traj_file.exists():
            raise FileNotFoundError(
                f"No trajectories found at {traj_file}. "
                "Run with --generate-trajectories first."
            )
        
        # Load good trajectories
        good_samples = self._load_good_trajectories(split_dir)
        print(f"[{self.split}] Loaded {len(good_samples)} good samples (with augmentation)")
        
        # Generate bad trajectories (only for training)
        if self.split == 'train' and self.good_bad_ratio > 0:
            bad_samples = self._generate_bad_trajectories(split_dir)
            print(f"[{self.split}] Generated {len(bad_samples)} bad samples (no augmentation)")
        else:
            bad_samples = []
            print(f"[{self.split}] No bad trajectories (eval mode)")
        
        # Combine samples
        self._samples = good_samples + bad_samples
        
        # Shuffle combined samples
        np.random.shuffle(self._samples)
        
        print(f"[{self.split}] Total: {len(self._samples)} PPO samples")
    
    def _load_good_trajectories(self, split_dir: Path) -> List[Dict]:
        """Load good trajectories with sub-trajectory augmentation."""
        traj_file = split_dir / 'trajectories.npz'
        data = np.load(traj_file)
        
        trajectories = data['trajectories']  # [num_traj, max_len, max_blocks]
        plan_lengths = data['plan_lengths']
        num_blocks_arr = data['num_blocks']
        
        samples = []
        
        for traj_idx in range(len(trajectories)):
            traj = trajectories[traj_idx]
            traj_len = plan_lengths[traj_idx] + 1  # +1 for final state
            num_blocks = num_blocks_arr[traj_idx]
            
            if traj_len < 2:
                continue
            
            # C(n+1, 2) sub-trajectory augmentation
            for i in range(traj_len):
                for j in range(i + 1, traj_len):
                    plan_length = j - i
                    
                    if self.max_plan_length and plan_length > self.max_plan_length:
                        continue
                    
                    samples.append({
                        'init_state': traj[i, :self.max_blocks].copy(),
                        'goal_state': traj[j, :self.max_blocks].copy(),
                        'plan_length': plan_length,
                        'num_blocks': num_blocks,
                        'is_valid': True,
                        'reward': 1.0,
                    })
        
        return samples
    
    def _generate_bad_trajectories(self, split_dir: Path) -> List[Dict]:
        """Generate bad trajectories from good ones."""
        from src.data.blocksworld_bad_trajectories import (
            BadTrajectoryGenerator,
            convert_bad_trajectory_to_dict,
        )
        
        traj_file = split_dir / 'trajectories.npz'
        data = np.load(traj_file)
        
        trajectories = data['trajectories']
        plan_lengths = data['plan_lengths']
        num_blocks_arr = data['num_blocks']
        
        # Initialize bad trajectory generator
        generator = BadTrajectoryGenerator(
            num_blocks=self.max_blocks,
            seed=self.seed,
        )
        
        # Prepare good trajectories for reference
        good_trajs = []
        for traj_idx in range(len(trajectories)):
            traj = trajectories[traj_idx]
            traj_len = plan_lengths[traj_idx] + 1
            num_blocks = num_blocks_arr[traj_idx]
            
            if traj_len < 2:
                continue
            
            # Extract actual trajectory (remove padding)
            actual_traj = [traj[t, :num_blocks].copy() for t in range(traj_len)]
            good_trajs.append((actual_traj[0], actual_traj))
        
        # Calculate how many bad samples we need
        # (matching the number of AUGMENTED good samples, not original trajectories)
        num_augmented_good = len(self._samples) if hasattr(self, '_samples') else 0
        if num_augmented_good == 0:
            # Count what we'd get from augmentation
            for traj_idx in range(len(trajectories)):
                traj_len = plan_lengths[traj_idx] + 1
                if traj_len >= 2:
                    # C(n, 2) = n*(n-1)/2 sub-trajectories
                    num_augmented_good += (traj_len * (traj_len - 1)) // 2
        
        num_bad_needed = int(num_augmented_good * self.good_bad_ratio)
        
        # Generate bad trajectories (full trajectories, no augmentation)
        bad_samples = []
        attempts = 0
        max_attempts = num_bad_needed * 3
        
        while len(bad_samples) < num_bad_needed and attempts < max_attempts:
            attempts += 1
            
            # Pick a random good trajectory to base bad one on
            idx = np.random.randint(len(good_trajs))
            init_state, good_traj = good_trajs[idx]
            
            # Generate bad trajectory
            bad_info = generator.generate(init_state, good_traj)
            
            if bad_info.invalid_transitions:
                # Convert to sample dict (full trajectory, no sub-sampling)
                sample = {
                    'init_state': np.zeros(self.max_blocks, dtype=np.int64),
                    'goal_state': np.zeros(self.max_blocks, dtype=np.int64),
                    'plan_length': len(bad_info.trajectory) - 1,
                    'num_blocks': bad_info.num_blocks,
                    'is_valid': False,
                    'reward': -1.0,
                }
                
                # Copy states with proper padding
                nb = min(len(bad_info.init_state), self.max_blocks)
                sample['init_state'][:nb] = bad_info.init_state[:nb]
                sample['goal_state'][:nb] = bad_info.goal_state[:nb]
                
                bad_samples.append(sample)
        
        return bad_samples
    
    def __len__(self) -> int:
        return len(self._samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample."""
        actual_idx = self._epoch_indices[idx]
        sample = self._samples[actual_idx]
        
        return {
            'init_state': torch.tensor(sample['init_state'], dtype=torch.long),
            'goal_state': torch.tensor(sample['goal_state'], dtype=torch.long),
            'plan_length': torch.tensor(sample['plan_length'], dtype=torch.long),
            'num_blocks': torch.tensor(sample['num_blocks'], dtype=torch.long),
            'is_valid': torch.tensor(sample['is_valid'], dtype=torch.bool),
            'reward': torch.tensor(sample['reward'], dtype=torch.float32),
        }
    
    def on_epoch_end(self):
        """Shuffle samples for next epoch."""
        np.random.shuffle(self._epoch_indices)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        num_good = sum(1 for s in self._samples if s['is_valid'])
        num_bad = sum(1 for s in self._samples if not s['is_valid'])
        
        return {
            'total': len(self._samples),
            'good': num_good,
            'bad': num_bad,
            'good_ratio': num_good / len(self._samples) if self._samples else 0,
            'bad_ratio': num_bad / len(self._samples) if self._samples else 0,
        }


