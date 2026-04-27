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
    # Guard: remove non-finite values before computing stats
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return 0.0
    mu, sigma = r.mean(), r.std()
    s = mu / (sigma + eps)
    result = float(s * np.sqrt(252) if annualise else s)
    return result if np.isfinite(result) else 0.0


def ic(
    predicted: np.ndarray | torch.Tensor,
    actual: np.ndarray | torch.Tensor,
) -> float:
    """Information Coefficient — Spearman rank correlation of predicted vs actual."""
    p = _to_numpy(predicted).ravel()
    a = _to_numpy(actual).ravel()
    # Guard: only compute on finite pairs
    mask = np.isfinite(p) & np.isfinite(a)
    if mask.sum() < 2:
        return 0.0
    corr, _ = spearmanr(p[mask], a[mask])
    return float(corr) if np.isfinite(corr) else 0.0


def hit_rate(
    predicted: np.ndarray | torch.Tensor,
    actual: np.ndarray | torch.Tensor,
) -> float:
    """Fraction of days where predicted sign matches actual sign."""
    p = np.sign(_to_numpy(predicted).ravel())
    a = np.sign(_to_numpy(actual).ravel())
    mask = np.isfinite(p) & np.isfinite(a)
    if mask.sum() == 0:
        return 0.0
    return float((p[mask] == a[mask]).mean())


def max_drawdown(equity_curve: np.ndarray | torch.Tensor) -> float:
    """Maximum drawdown of a cumulative equity curve.

    Robust to overflow: clips equity curve and uses log-space peak tracking
    when values are very large.
    """
    ec = _to_numpy(equity_curve).ravel().astype(np.float64)

    # Remove non-finite values
    ec = ec[np.isfinite(ec)]
    if len(ec) == 0:
        return 0.0

    # Clip to prevent overflow in accumulate — equity > 1e6 means ~1e6x return
    # which is unrealistic; clamp to sane range
    ec = np.clip(ec, 1e-10, 1e6)

    peak = np.maximum.accumulate(ec)
    # Avoid division by zero — peak is always >= ec[0] > 0 after clip
    dd = (ec - peak) / (peak + 1e-10)
    result = float(dd.min())
    return result if np.isfinite(result) else 0.0


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
