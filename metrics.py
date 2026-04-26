"""metrics.py — Standalone performance metrics used by trainer and backtester."""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr


def sharpe(
    returns: np.ndarray | torch.Tensor,
    annualise: bool = True,
    eps: float = 1e-8,
) -> float:
    """Annualised Sharpe ratio (252 trading days)."""
    r = _to_numpy(returns).ravel()
    mu, sigma = r.mean(), r.std()
    s = mu / (sigma + eps)
    return float(s * np.sqrt(252) if annualise else s)


def ic(
    predicted: np.ndarray | torch.Tensor,
    actual: np.ndarray | torch.Tensor,
) -> float:
    """Information Coefficient — Spearman rank correlation of predicted vs actual."""
    p = _to_numpy(predicted).ravel()
    a = _to_numpy(actual).ravel()
    corr, _ = spearmanr(p, a)
    return float(corr)


def hit_rate(
    predicted: np.ndarray | torch.Tensor,
    actual: np.ndarray | torch.Tensor,
) -> float:
    """Fraction of days where predicted sign matches actual sign."""
    p = np.sign(_to_numpy(predicted).ravel())
    a = np.sign(_to_numpy(actual).ravel())
    return float((p == a).mean())


def max_drawdown(equity_curve: np.ndarray | torch.Tensor) -> float:
    """Maximum drawdown of a cumulative equity curve."""
    ec = _to_numpy(equity_curve).ravel()
    peak = np.maximum.accumulate(ec)
    dd = (ec - peak) / (peak + 1e-8)
    return float(dd.min())


def turnover(weights: np.ndarray, annualise: bool = True) -> float:
    """Average daily one-way portfolio turnover.

    Args:
        weights: (T, N) weight matrix.
    """
    w = _to_numpy(weights)
    daily_to = np.abs(np.diff(w, axis=0)).sum(axis=1).mean()
    return float(daily_to * 252 if annualise else daily_to)


def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
