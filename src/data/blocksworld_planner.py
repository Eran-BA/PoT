"""
FastDownward Planner Integration for Blocksworld.

This module provides a wrapper around the FastDownward classical planner
to generate optimal plans for Blocksworld problems. These plans can then
be used to create trajectories for training neural network planners.

The generated trajectories enable:
- Sub-trajectory extraction: C(N+1, 2) training samples per problem
- Curriculum learning: easier (shorter) sub-goals before full plans
- Data augmentation: more training signal from limited problems

Requirements:
    FastDownward must be installed separately:
    ```
    git clone https://github.com/aibasel/downward.git
    cd downward && ./build.py
    ```

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import os
import re
import shutil
import tempfile
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


# =============================================================================
# Blocksworld Domain PDDL (4-operator version)
# =============================================================================

BLOCKSWORLD_DOMAIN_PDDL = """
(define (domain blocksworld-4ops)
  (:requirements :strips)
  (:predicates 
    (clear ?x)
    (ontable ?x)
    (handempty)
    (holding ?x)
    (on ?x ?y))

  (:action pick-up
   :parameters (?x)
   :precondition (and (clear ?x) (ontable ?x) (handempty))
   :effect (and (not (ontable ?x))
                (not (clear ?x))
                (not (handempty))
                (holding ?x)))

  (:action put-down
   :parameters (?x)
   :precondition (holding ?x)
   :effect (and (not (holding ?x))
                (clear ?x)
                (handempty)
                (ontable ?x)))

  (:action stack
   :parameters (?x ?y)
   :precondition (and (holding ?x) (clear ?y))
   :effect (and (not (holding ?x))
                (not (clear ?y))
                (clear ?x)
                (handempty)
                (on ?x ?y)))

  (:action unstack
   :parameters (?x ?y)
   :precondition (and (on ?x ?y) (clear ?x) (handempty))
   :effect (and (holding ?x)
                (clear ?y)
                (not (clear ?x))
                (not (handempty))
                (not (on ?x ?y)))))
"""


@dataclass
class PlanAction:
    """Represents a single action in a plan."""
    name: str  # pick-up, put-down, stack, unstack
    args: List[str]  # block names
    
    def __repr__(self):
        return f"({self.name} {' '.join(self.args)})"


@dataclass
class BlocksworldState:
    """Represents a Blocksworld state."""
    positions: np.ndarray  # [N] - position of each block
    block_names: List[str]
    holding: Optional[str] = None  # Block being held, if any
    
    def to_pddl_predicates(self) -> str:
        """Convert state to PDDL predicate list."""
        predicates = []
        N = len(self.block_names)
        
        for i, pos in enumerate(self.positions):
            block = self.block_names[i]
            if pos == 0:
                predicates.append(f"(ontable {block})")
            elif pos == N + 1:
                # Holding this block
                predicates.append(f"(holding {block})")
            elif 1 <= pos <= N:
                bottom_block = self.block_names[pos - 1]
                predicates.append(f"(on {block} {bottom_block})")
        
        # Add clear predicates
        blocks_with_something_on_top = set()
        for i, pos in enumerate(self.positions):
            if 1 <= pos <= N:
                blocks_with_something_on_top.add(pos - 1)
        
        for i, block in enumerate(self.block_names):
            if i not in blocks_with_something_on_top and self.positions[i] != N + 1:
                predicates.append(f"(clear {block})")
        
        # Handempty
        if self.holding is None and not any(p == N + 1 for p in self.positions):
            predicates.append("(handempty)")
        
        return "\n".join(predicates)


class FastDownwardPlanner:
    """
    Wrapper for the FastDownward classical planner.
    
    FastDownward is a state-of-the-art domain-independent classical planner.
    This wrapper handles:
    - PDDL file generation
    - Running the planner with configurable timeout
    - Parsing the solution
    - Converting plans to state trajectories
    
    Args:
        fd_path: Path to fast-downward.py script
        timeout: Per-problem timeout in seconds
        search_algorithm: FastDownward search algorithm (default: A* with LM-cut)
    """
    
    def __init__(
        self,
        fd_path: str = "fast-downward.py",
        timeout: int = 30,
        search_algorithm: str = "astar(lmcut())",
    ):
        self.fd_path = fd_path
        self.timeout = timeout
        self.search_algorithm = search_algorithm
        self._validated = False
    
    def is_available(self) -> bool:
        """Check if FastDownward is available."""
        try:
            # Try to find the executable
            if os.path.isfile(self.fd_path):
                return True
            
            # Check if it's in PATH
            result = shutil.which("fast-downward.py")
            if result:
                self.fd_path = result
                return True
            
            # Check common locations
            common_paths = [
                os.path.expanduser("~/downward/fast-downward.py"),
                "/opt/downward/fast-downward.py",
                "/usr/local/bin/fast-downward.py",
            ]
            for path in common_paths:
                if os.path.isfile(path):
                    self.fd_path = path
                    return True
            
            return False
        except Exception:
            return False
    
    def validate(self) -> bool:
        """Validate that FastDownward works correctly."""
        if self._validated:
            return True
        
        if not self.is_available():
            return False
        
        # Try a simple problem
        try:
            problem = """
(define (problem test)
  (:domain blocksworld-4ops)
  (:objects a b)
  (:init (ontable a) (ontable b) (clear a) (clear b) (handempty))
  (:goal (on a b)))
"""
            plan = self.solve(BLOCKSWORLD_DOMAIN_PDDL, problem)
            self._validated = plan is not None
            return self._validated
        except Exception:
            return False
    
    def solve(
        self,
        domain_pddl: str,
        problem_pddl: str,
    ) -> Optional[List[PlanAction]]:
        """
        Solve a PDDL planning problem using FastDownward.
        
        Args:
            domain_pddl: Domain PDDL string
            problem_pddl: Problem PDDL string
        
        Returns:
            List of PlanAction objects, or None if no solution found
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            domain_file = os.path.join(tmpdir, "domain.pddl")
            problem_file = os.path.join(tmpdir, "problem.pddl")
            sas_file = os.path.join(tmpdir, "output.sas")
            plan_file = os.path.join(tmpdir, "sas_plan")
            
            # Write PDDL files
            with open(domain_file, 'w') as f:
                f.write(domain_pddl)
            with open(problem_file, 'w') as f:
                f.write(problem_pddl)
            
            # Run FastDownward
            cmd = [
                "python3", self.fd_path,
                "--plan-file", plan_file,
                domain_file,
                problem_file,
                "--search", self.search_algorithm,
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmpdir,
                )
            except subprocess.TimeoutExpired:
                return None
            except FileNotFoundError:
                raise RuntimeError(
                    f"FastDownward not found at {self.fd_path}. "
                    "Install it from https://github.com/aibasel/downward"
                )
            
            # Check for solution
            if not os.path.exists(plan_file):
                # Check for numbered plan files (sas_plan.1, sas_plan.2, etc.)
                for i in range(1, 10):
                    alt_plan = f"{plan_file}.{i}"
                    if os.path.exists(alt_plan):
                        plan_file = alt_plan
                        break
                else:
                    return None
            
            # Parse plan
            return self._parse_plan_file(plan_file)
    
    def _parse_plan_file(self, plan_file: str) -> List[PlanAction]:
        """Parse a FastDownward plan file."""
        actions = []
        
        with open(plan_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(';'):
                    continue
                
                # Parse action: (action-name arg1 arg2 ...)
                match = re.match(r'\(([a-zA-Z_-]+)\s*(.*)\)', line)
                if match:
                    action_name = match.group(1)
                    args_str = match.group(2).strip()
                    args = args_str.split() if args_str else []
                    actions.append(PlanAction(name=action_name, args=args))
        
        return actions
    
    def plan_to_trajectory(
        self,
        init_state: np.ndarray,
        plan: List[PlanAction],
        block_names: List[str],
    ) -> List[np.ndarray]:
        """
        Execute a plan to generate a state trajectory.
        
        Args:
            init_state: Initial state vector [N]
            plan: List of actions
            block_names: Block names in order
        
        Returns:
            List of state vectors [s0, s1, ..., sN]
        """
        trajectory = [init_state.copy()]
        current_state = init_state.copy()
        N = len(block_names)
        name_to_idx = {name.lower(): i for i, name in enumerate(block_names)}
        
        for action in plan:
            next_state = self._apply_action(
                current_state, action, name_to_idx, N
            )
            trajectory.append(next_state)
            current_state = next_state
        
        return trajectory
    
    def _apply_action(
        self,
        state: np.ndarray,
        action: PlanAction,
        name_to_idx: Dict[str, int],
        N: int,
    ) -> np.ndarray:
        """Apply an action to a state and return the new state."""
        next_state = state.copy()
        
        if action.name == "pick-up":
            # Pick up block from table
            block = action.args[0].lower()
            if block in name_to_idx:
                idx = name_to_idx[block]
                next_state[idx] = N + 1  # Now holding
        
        elif action.name == "put-down":
            # Put block on table
            block = action.args[0].lower()
            if block in name_to_idx:
                idx = name_to_idx[block]
                next_state[idx] = 0  # On table
        
        elif action.name == "stack":
            # Stack block X on block Y
            block_x = action.args[0].lower()
            block_y = action.args[1].lower()
            if block_x in name_to_idx and block_y in name_to_idx:
                idx_x = name_to_idx[block_x]
                idx_y = name_to_idx[block_y]
                next_state[idx_x] = idx_y + 1  # On top of Y
        
        elif action.name == "unstack":
            # Unstack block X from block Y
            block_x = action.args[0].lower()
            if block_x in name_to_idx:
                idx_x = name_to_idx[block_x]
                next_state[idx_x] = N + 1  # Now holding
        
        return next_state


def generate_problem_pddl(
    block_names: List[str],
    init_state: np.ndarray,
    goal_state: np.ndarray,
) -> str:
    """
    Generate PDDL problem string from state vectors.
    
    Args:
        block_names: List of block names
        init_state: Initial state vector
        goal_state: Goal state vector
    
    Returns:
        PDDL problem string
    """
    N = len(block_names)
    
    # Objects
    objects_str = " ".join(block_names)
    
    # Init predicates
    init_preds = ["(handempty)"]
    blocks_with_something_on_top = set()
    
    for i, pos in enumerate(init_state):
        block = block_names[i]
        if pos == 0:
            init_preds.append(f"(ontable {block})")
        elif 1 <= pos <= N:
            bottom_idx = pos - 1
            bottom_block = block_names[bottom_idx]
            init_preds.append(f"(on {block} {bottom_block})")
            blocks_with_something_on_top.add(bottom_idx)
        elif pos == N + 1:
            init_preds.append(f"(holding {block})")
    
    # Clear predicates for init
    for i, block in enumerate(block_names):
        if i not in blocks_with_something_on_top and init_state[i] != N + 1:
            init_preds.append(f"(clear {block})")
    
    # Goal predicates (only structural, not clear/handempty)
    goal_preds = []
    for i, pos in enumerate(goal_state):
        block = block_names[i]
        if pos == 0:
            goal_preds.append(f"(ontable {block})")
        elif 1 <= pos <= N:
            bottom_idx = pos - 1
            bottom_block = block_names[bottom_idx]
            goal_preds.append(f"(on {block} {bottom_block})")
    
    init_str = "\n".join(init_preds)
    goal_str = "\n".join(goal_preds)
    
    return f"""
(define (problem blocksworld-problem)
  (:domain blocksworld-4ops)
  (:objects {objects_str})
  (:init
    {init_str}
  )
  (:goal (and
    {goal_str}
  )))
"""


def extract_sub_trajectories(
    trajectory: List[np.ndarray],
    actions: Optional[List[PlanAction]] = None,
) -> List[Dict[str, Any]]:
    """
    Extract all C(N+1, 2) contiguous sub-trajectories from a trajectory.
    
    For a trajectory of N+1 states (N actions), this generates:
    - N*(N+1)/2 sub-trajectories
    - Each sub-trajectory is a (init, goal, intermediate_states) tuple
    
    Args:
        trajectory: List of state vectors [s0, s1, ..., sN]
        actions: Optional list of actions (for metadata)
    
    Returns:
        List of sub-trajectory dicts
    """
    sub_trajectories = []
    N = len(trajectory)
    
    for i in range(N):
        for j in range(i + 1, N):
            sub_traj = {
                'init_state': trajectory[i].copy(),
                'goal_state': trajectory[j].copy(),
                'intermediate_states': [s.copy() for s in trajectory[i:j+1]],
                'length': j - i,
                'start_idx': i,
                'end_idx': j,
            }
            
            # Include action slice if available
            if actions is not None:
                sub_traj['actions'] = actions[i:j]
            
            sub_trajectories.append(sub_traj)
    
    return sub_trajectories


class TrajectoryGenerator:
    """
    Generate trajectories for Blocksworld problems using FastDownward.
    
    This class processes (init, goal) pairs and produces full trajectories
    that can be used for training via sub-trajectory extraction.
    
    Args:
        fd_path: Path to FastDownward
        timeout: Per-problem timeout
        max_blocks: Maximum number of blocks to handle
    """
    
    def __init__(
        self,
        fd_path: str = "fast-downward.py",
        timeout: int = 30,
        max_blocks: int = 10,
    ):
        self.planner = FastDownwardPlanner(fd_path=fd_path, timeout=timeout)
        self.max_blocks = max_blocks
        self.stats = {
            'solved': 0,
            'failed': 0,
            'timeout': 0,
            'total_actions': 0,
        }
    
    def is_available(self) -> bool:
        """Check if planner is available."""
        return self.planner.is_available()
    
    def generate_trajectory(
        self,
        init_state: np.ndarray,
        goal_state: np.ndarray,
        block_names: List[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a trajectory from init to goal state.
        
        Args:
            init_state: Initial state vector
            goal_state: Goal state vector  
            block_names: Block names in order
        
        Returns:
            Dict with trajectory, actions, etc. or None if failed
        """
        # Generate PDDL problem
        problem_pddl = generate_problem_pddl(block_names, init_state, goal_state)
        
        # Solve
        plan = self.planner.solve(BLOCKSWORLD_DOMAIN_PDDL, problem_pddl)
        
        if plan is None:
            self.stats['failed'] += 1
            return None
        
        # Convert to trajectory
        trajectory = self.planner.plan_to_trajectory(
            init_state, plan, block_names
        )
        
        self.stats['solved'] += 1
        self.stats['total_actions'] += len(plan)
        
        return {
            'trajectory': trajectory,
            'actions': plan,
            'plan_length': len(plan),
            'block_names': block_names,
            'num_blocks': len([b for b in block_names if not b.startswith('_pad_')]),
        }
    
    def generate_all_trajectories(
        self,
        problems: List[Dict[str, Any]],
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate trajectories for a list of problems.
        
        Args:
            problems: List of dicts with 'init_state', 'goal_state', 'block_names'
            show_progress: Show progress bar
        
        Returns:
            List of trajectory dicts (only successfully solved problems)
        """
        from tqdm import tqdm
        
        trajectories = []
        iterator = tqdm(problems, desc="Generating trajectories") if show_progress else problems
        
        for problem in iterator:
            result = self.generate_trajectory(
                problem['init_state'],
                problem['goal_state'],
                problem['block_names'],
            )
            if result is not None:
                result['original_problem'] = problem
                trajectories.append(result)
        
        return trajectories
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        total = self.stats['solved'] + self.stats['failed']
        return {
            **self.stats,
            'success_rate': self.stats['solved'] / max(total, 1),
            'avg_plan_length': self.stats['total_actions'] / max(self.stats['solved'], 1),
        }

