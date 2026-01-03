"""
Dependency parsing task adapter for PoT.

Task: Universal Dependency parsing (head prediction).

Author: Eran Ben Artzy
Year: 2025
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Dict, Any, Tuple, List

from .base import TaskAdapter
from ..core.metrics import compute_uas


class DependencyParsingTask(TaskAdapter):
    """Dependency parsing task adapter."""

    def prepare_data(self, config: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dataset]:
        """Prepare UD datasets."""
        # Import from existing ud_pointer_parser.py
        raise NotImplementedError("Use existing load_ud_en_ewt() function")

    def build_model(self, config: Dict[str, Any]) -> nn.Module:
        """Build dependency parser."""
        raise NotImplementedError("Use existing UDPointerParser model")

    def compute_loss(
        self,
        model_output: Any,
        batch: Dict[str, torch.Tensor],
        config: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute parsing loss."""
        loss, metrics = model_output  # Assuming model returns (loss, metrics)
        return loss

    def compute_metrics(
        self,
        model_output: Any,
        batch: Dict[str, torch.Tensor],
        config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute UAS metric."""
        loss, metrics = model_output
        return metrics  # Should include 'uas'

    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch for dependency parsing."""
        raise NotImplementedError("Use existing collate_batch() function")

