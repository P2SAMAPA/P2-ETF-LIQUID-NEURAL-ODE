"""regime_score.py — Apply regime-adaptive adjustment to raw model scores.

score_adj = score_raw × (1 + alpha_amp × fast_frac)

Higher fast_frac (volatile market) amplifies signal conviction.
Output format compatible with HHMM-REGIME and ENSEMBLE-META engines.
"""

from __future__ import annotations

import torch

DEFAULT_ALPHA_AMP = 0.3


def apply_regime_adjustment(
    scores: torch.Tensor,  # (batch, n_etf) or (n_etf,)
    fast_frac: torch.Tensor,  # (batch,) or scalar
    alpha_amp: float = DEFAULT_ALPHA_AMP,
) -> torch.Tensor:
    """Multiply raw scores by a regime amplification factor.

    Args:
        scores:    Raw model scores.
        fast_frac: Fraction of fast neurons (from tau_monitor).
        alpha_amp: Amplification coefficient (default 0.3).

    Returns:
        score_adj: Regime-adjusted scores (same shape as scores).
    """
    if scores.dim() == 1:
        factor = 1.0 + alpha_amp * fast_frac
        return scores * factor

    # (batch, n_etf): broadcast fast_frac over etf dimension
    factor = (1.0 + alpha_amp * fast_frac).unsqueeze(-1)  # (batch, 1)
    return scores * factor
