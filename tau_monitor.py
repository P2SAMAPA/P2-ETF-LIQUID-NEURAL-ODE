"""tau_monitor.py — Monitor the τ distribution for regime detection.

The distribution of per-neuron time constants is a natural soft regime
label:
    fast_frac  : fraction of neurons with τ < FAST_THRESHOLD (day-trader mode)
    slow_frac  : fraction of neurons with τ > SLOW_THRESHOLD (position-trade mode)
    mixed_frac : remainder

Output format is compatible with HHMM-REGIME engine labels.
"""

from __future__ import annotations

import torch

FAST_THRESHOLD = 0.3  # days — τ below this → "fast" neuron
SLOW_THRESHOLD = 5.0  # days — τ above this → "slow" neuron


def compute_regime_labels(
    tau_dist: torch.Tensor,  # (batch, hidden_dim) or (hidden_dim,)
    fast_threshold: float = FAST_THRESHOLD,
    slow_threshold: float = SLOW_THRESHOLD,
) -> dict[str, torch.Tensor]:
    """Classify neurons by time constant and return soft regime fractions.

    Returns:
        dict with keys: fast_frac, slow_frac, mixed_frac, tau_mean, tau_log_mean
        All tensors have shape (batch,) or scalar if input was 1D.
    """
    if tau_dist.dim() == 1:
        tau_dist = tau_dist.unsqueeze(0)

    fast_frac = (tau_dist < fast_threshold).float().mean(dim=-1)
    slow_frac = (tau_dist > slow_threshold).float().mean(dim=-1)
    mixed_frac = 1.0 - fast_frac - slow_frac
    tau_mean = tau_dist.mean(dim=-1)
    tau_log_mean = tau_dist.log().mean(dim=-1)

    return {
        "fast_frac": fast_frac,
        "slow_frac": slow_frac,
        "mixed_frac": mixed_frac,
        "tau_mean": tau_mean,
        "tau_log_mean": tau_log_mean,
    }


def regime_label_str(fast_frac: float, slow_frac: float) -> str:
    """Return a human-readable regime string for logging."""
    if fast_frac > 0.5:
        return "FAST (volatile)"
    if slow_frac > 0.5:
        return "SLOW (trending)"
    return "MIXED"
