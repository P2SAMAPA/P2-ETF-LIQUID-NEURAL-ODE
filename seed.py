"""seed.py — Reproducibility seed setter."""

import random

import numpy as np
import torch


def set_seed(n: int = 42) -> None:
    """Set Python, NumPy, and PyTorch seeds for full reproducibility."""
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(n)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
