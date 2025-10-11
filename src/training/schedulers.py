"""
Learning rate schedulers for training.

Provides common LR scheduling strategies including linear warmup.

Functions:
    get_linear_schedule_with_warmup: Linear warmup followed by linear decay

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int
) -> LambdaLR:
    """Create learning rate scheduler with linear warmup and linear decay.
    
    Learning rate increases linearly from 0 to the base LR during warmup,
    then decreases linearly back to 0 over the remaining training steps.
    
    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of steps for linear warmup
        num_training_steps: Total number of training steps
        
    Returns:
        LambdaLR scheduler to use with optimizer.step()
        
    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        >>> num_epochs = 10
        >>> steps_per_epoch = len(train_data) // batch_size
        >>> total_steps = num_epochs * steps_per_epoch
        >>> warmup_steps = int(0.1 * total_steps)  # 10% warmup
        >>> scheduler = get_linear_schedule_with_warmup(
        ...     optimizer, warmup_steps, total_steps
        ... )
        >>> 
        >>> # Training loop
        >>> for epoch in range(num_epochs):
        ...     for batch in train_loader:
        ...         loss = model(batch)
        ...         loss.backward()
        ...         optimizer.step()
        ...         scheduler.step()  # Update LR after each step
        ...         optimizer.zero_grad()
        
    Note:
        The scheduler should be called after optimizer.step() at each
        training step, not once per epoch.
    """
    def lr_lambda(current_step: int) -> float:
        """Compute LR multiplier for current step."""
        if current_step < num_warmup_steps:
            # Linear warmup: 0 -> 1.0
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Linear decay: 1.0 -> 0
        return max(
            0.0,
            float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda)

