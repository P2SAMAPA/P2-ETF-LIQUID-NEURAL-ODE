"""scheduler.py — Cosine annealing LR scheduler with linear warmup."""
from __future__ import annotations

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(
    optimizer:     Optimizer,
    total_epochs:  int,
    warmup_epochs: int = 5,
) -> LambdaLR:
    """Cosine annealing with linear warmup.

    Args:
        optimizer:     Wrapped optimiser.
        total_epochs:  Total training epochs.
        warmup_epochs: Number of linear warmup epochs.

    Returns:
        LambdaLR scheduler — call scheduler.step() each epoch.
    """
    import math

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
