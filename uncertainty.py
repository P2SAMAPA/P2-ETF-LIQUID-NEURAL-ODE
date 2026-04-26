"""uncertainty.py — MC Dropout uncertainty quantification.

Runs N stochastic forward passes with dropout enabled to produce
per-ETF mean scores and credible intervals.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def uncertainty_pass(
    model: nn.Module,
    x: torch.Tensor,
    delta_t: torch.Tensor,
    n_passes: int = 50,
    ci_level: float = 0.95,
) -> dict[str, torch.Tensor]:
    """Run MC Dropout inference.

    Args:
        model:    LTCModel with dropout layers.
        x:        (B, W, D) input tensor.
        delta_t:  (B, W) elapsed-days tensor.
        n_passes: Number of stochastic forward passes.
        ci_level: Credible interval level (e.g. 0.95).

    Returns:
        dict with keys:
            mean:     (B, N_etf) mean score across passes.
            std:      (B, N_etf) standard deviation.
            ci_lower: (B, N_etf) lower credible interval bound.
            ci_upper: (B, N_etf) upper credible interval bound.
    """
    model.train()  # Enable dropout
    with torch.no_grad():
        samples = torch.stack(
            [model(x, delta_t)[0] for _ in range(n_passes)],
            dim=0,
        )  # (n_passes, B, N_etf)

    alpha = (1.0 - ci_level) / 2.0
    mean = samples.mean(dim=0)
    std = samples.std(dim=0)
    ci_lower = samples.quantile(alpha, dim=0)
    ci_upper = samples.quantile(1.0 - alpha, dim=0)

    return {"mean": mean, "std": std, "ci_lower": ci_lower, "ci_upper": ci_upper}
