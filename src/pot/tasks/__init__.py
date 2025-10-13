"""
Task adapters for PoT architecture.

Each task provides:
- Dataset loading
- Task-specific head/loss
- Metrics computation
- Unified interface for training

Available tasks:
- sorting: Partial observability sorting
- dependency: Dependency parsing
"""

from .base import TaskAdapter
from .sorting import SortingTask
from .dependency import DependencyParsingTask

__all__ = [
    "TaskAdapter",
    "SortingTask",
    "DependencyParsingTask",
]

