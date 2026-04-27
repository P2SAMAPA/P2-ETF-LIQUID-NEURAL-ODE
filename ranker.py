"""ranker.py — Cross-sectional z-score ranker.

Converts raw model scores + regime adjustment into a final
cross-sectional z-score ranking compatible with ENSEMBLE-META.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def zscore_rank(
    scores: np.ndarray | torch.Tensor,
    eps: float = 1e-6,  # FIX: raised from 1e-8 — prevents collapse when std~0 (e.g. 7-ticker FI)
) -> np.ndarray:
    """Cross-sectional z-score of scores across ETFs.

    Args:
        scores: (N_etf,) array of raw or regime-adjusted scores for one day.

    Returns:
        (N_etf,) z-scored array with mean~0, std~1.
        If all scores are identical, returns zeros rather than NaN.
    """
    s = np.asarray(scores, dtype=float).ravel()

    # Replace any non-finite values before computing stats
    s = np.where(np.isfinite(s), s, 0.0)

    mu = s.mean()
    std = s.std()
    return (s - mu) / max(std, eps)


def build_daily_ranking(
    scores: np.ndarray,  # (N_etf,)
    tickers: list[str],
    date: pd.Timestamp,
    ci_lower: np.ndarray | None = None,
    ci_upper: np.ndarray | None = None,
    tau_mean: float = float("nan"),
    fast_frac: float = float("nan"),
    universe: str = "combined",
) -> pd.DataFrame:
    """Build a one-row-per-ETF results DataFrame for a single day.

    Output schema matches P2SAMAPA/p2-etf-liquid-neural-ode-results.
    """
    z = zscore_rank(scores)

    df = pd.DataFrame(
        {
            "date": date,
            "ticker": tickers,
            "score_raw": scores,
            "score_adj": z,
            "ci_lower": ci_lower if ci_lower is not None else np.nan,
            "ci_upper": ci_upper if ci_upper is not None else np.nan,
            "tau_mean": tau_mean if np.isfinite(tau_mean) else 0.0,
            "fast_frac": fast_frac if np.isfinite(fast_frac) else 0.0,
            "universe": universe,
        }
    )
    df["rank"] = df["score_adj"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("rank").reset_index(drop=True)
