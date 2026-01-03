"""
Base task adapter interface.

All tasks implement this interface for unified training.

Author: Eran Ben Artzy
Year: 2025
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import torch
from torch.utils.data import Dataset


class TaskAdapter(ABC):
    """
    Abstract base class for task adapters.

    Each task must implement:
    - prepare_data(): Return train/val/test datasets
    - build_model(): Construct task-specific model
    - compute_loss(): Task-specific loss computation
    - compute_metrics(): Task-specific evaluation metrics
    """

    @abstractmethod
    def prepare_data(self, config: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Prepare train/val/test datasets.

        Args:
            config: Task configuration dict

        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        pass

    @abstractmethod
    def build_model(self, config: Dict[str, Any]) -> torch.nn.Module:
        """
        Build task-specific model.

        Args:
            config: Model configuration dict

        Returns:
            PyTorch model
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        model_output: Any,
        batch: Dict[str, torch.Tensor],
        config: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Compute task-specific loss.

        Args:
            model_output: Model predictions
            batch: Batch data dict
            config: Training configuration

        Returns:
            Scalar loss tensor
        """
        pass

    @abstractmethod
    def compute_metrics(
        self,
        model_output: Any,
        batch: Dict[str, torch.Tensor],
        config: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute task-specific metrics.

        Args:
            model_output: Model predictions
            batch: Batch data dict
            config: Evaluation configuration

        Returns:
            Dict of metric_name -> value
        """
        pass

    @abstractmethod
    def collate_fn(self, batch: list) -> Dict[str, torch.Tensor]:
        """
        Collate batch samples into tensors.

        Args:
            batch: List of samples

        Returns:
            Batched dict
        """
        pass

