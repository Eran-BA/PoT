"""
Inner-loop logger for analyzing iterative refinement dynamics.

Logs per-inner-step metrics:
- Loss per iteration
- Gradient norms
- Attention entropy
- Halting rates (if ACT enabled)
- Optional UAS probe
- Forward pass timing

Zero external dependencies beyond stdlib.

Author: Eran Ben Artzy
Year: 2025
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import math
import time
from typing import Dict, Any, List, Optional


@dataclass
class InnerStepRow:
    """Single row of inner-loop telemetry."""
    run_id: str
    epoch: int
    global_step: int
    inner_step: int
    batch_size: int
    loss: float
    grad_norm: float
    attn_entropy_mean: Optional[float] = None
    halted_frac: Optional[float] = None
    uas_probe: Optional[float] = None
    ms_forward: Optional[float] = None


class InnerLoopLogger:
    """
    CSV logger for inner-loop dynamics.
    
    Usage:
        with InnerLoopLogger("results/run1/innerloop.csv") as logger:
            for step in range(100):
                row = InnerStepRow(...)
                logger.log(row)
    
    The CSV can be tailed during training (-f) since we flush after each write.
    """
    
    def __init__(self, out_csv: str):
        self.out = Path(out_csv)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.writer = None
        self.fh = None
    
    def __enter__(self):
        self.fh = self.out.open("w", newline="")
        self.writer = None
        return self
    
    def __exit__(self, exc_type, exc, tb):
        if self.fh:
            self.fh.close()
    
    def log(self, row: InnerStepRow):
        """Write one row to CSV."""
        if self.writer is None:
            fieldnames = list(asdict(row).keys())
            self.writer = csv.DictWriter(self.fh, fieldnames=fieldnames)
            self.writer.writeheader()
        self.writer.writerow(asdict(row))
        self.fh.flush()  # Safe to tail -f during training


def grad_global_norm(model) -> float:
    """Compute global gradient norm across all parameters."""
    import torch
    
    tot = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            tot += float(g.norm(2).item() ** 2)
    return math.sqrt(tot)


def maybe_probe_uas(model, probe_batch, decode_and_score_fn) -> Optional[float]:
    """
    Quick UAS probe on a tiny batch (fast sanity check during training).
    
    Args:
        model: The model
        probe_batch: Small validation batch
        decode_and_score_fn: Function that takes (logits, batch) -> UAS score
    
    Returns:
        UAS score or None if probe_batch is None
    """
    import torch
    
    if probe_batch is None:
        return None
    
    model.eval()
    with torch.no_grad():
        # Assuming model returns (logits, loss, inner_stats) when return_inner_stats=False
        output = model(probe_batch, max_inner_iters=1, return_inner_stats=False)
        logits = output[0] if isinstance(output, tuple) else output
        uas = decode_and_score_fn(logits, probe_batch)
    model.train()
    
    return float(uas)

