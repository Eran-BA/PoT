"""
Sorting task adapter for PoT.

Task: Sort partially observable arrays.

Author: Eran Ben Artzy
Year: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict, Any, Tuple, List

from .base import TaskAdapter
from ..core.metrics import compute_mask_aware_kendall_tau


class PartialSortDataset(Dataset):
    """Dataset for partial observability sorting."""

    def __init__(self, num_samples: int, array_len: int, mask_rate: float = 0.5):
        self.num_samples = num_samples
        self.array_len = array_len
        self.mask_rate = mask_rate

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random array
        arr = torch.randperm(self.array_len).float()
        
        # Create observability mask
        obs_mask = torch.rand(self.array_len) > self.mask_rate
        obs_mask[0] = True  # Ensure at least one observable
        
        # Masked array (unobserved = 0)
        arr_masked = arr * obs_mask.float()
        
        # Target: argsort (pointer to sorted position)
        targets = torch.argsort(arr)
        
        return {
            'array': arr_masked,
            'targets': targets,
            'obs_mask': obs_mask,
            'arr_true': arr
        }


class SortingTask(TaskAdapter):
    """Sorting task adapter."""

    def prepare_data(self, config: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dataset]:
        """Prepare sorting datasets."""
        train_ds = PartialSortDataset(
            num_samples=config.get('train_samples', 10000),
            array_len=config.get('array_len', 12),
            mask_rate=config.get('mask_rate', 0.5)
        )
        val_ds = PartialSortDataset(
            num_samples=config.get('val_samples', 1000),
            array_len=config.get('array_len', 12),
            mask_rate=config.get('mask_rate', 0.5)
        )
        test_ds = PartialSortDataset(
            num_samples=config.get('test_samples', 1000),
            array_len=config.get('array_len', 12),
            mask_rate=config.get('mask_rate', 0.5)
        )
        return train_ds, val_ds, test_ds

    def build_model(self, config: Dict[str, Any]) -> nn.Module:
        """Build sorting model (placeholder - use existing model)."""
        # Import from existing experiments
        # For now, return a placeholder
        raise NotImplementedError("Use existing sort_pointer_improved.py model")

    def compute_loss(
        self,
        model_output: Any,
        batch: Dict[str, torch.Tensor],
        config: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute sorting loss (cross-entropy)."""
        logits, loss = model_output  # Assuming model returns (logits, loss)
        return loss

    def compute_metrics(
        self,
        model_output: Any,
        batch: Dict[str, torch.Tensor],
        config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute Kendall-τ metric."""
        logits, _ = model_output
        B, N, _ = logits.shape
        
        # Predicted scores (argmax or soft scores)
        pred_scores = logits.softmax(dim=-1).argmax(dim=-1).float()
        true_scores = batch['arr_true']
        obs_mask = batch['obs_mask']
        
        tau = compute_mask_aware_kendall_tau(true_scores, pred_scores, obs_mask)
        
        return {'kendall_tau': tau.item()}

    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch."""
        return {
            'array': torch.stack([b['array'] for b in batch]),
            'targets': torch.stack([b['targets'] for b in batch]),
            'obs_mask': torch.stack([b['obs_mask'] for b in batch]),
            'arr_true': torch.stack([b['arr_true'] for b in batch]),
        }

