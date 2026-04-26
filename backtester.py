"""backtester.py — Walk-forward backtest over the test period (2023–2026)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from logging_utils import get_logger
from metrics import hit_rate, ic, max_drawdown, sharpe
from ranker import zscore_rank
from regime_score import apply_regime_adjustment
from tau_monitor import compute_regime_labels
from uncertainty import uncertainty_pass

log = get_logger(__name__)


def run_backtest(
    model: nn.Module,
    test_loader: DataLoader,
    tickers: list[str],
    dates: list[pd.Timestamp],
    device: torch.device,
    mc_passes: int = 50,
    alpha_amp: float = 0.3,
    output_dir: str = "results",
) -> dict:
    """Walk-forward backtest.

    Returns a metrics dict and saves equity_curve.csv + metrics.json.
    """
    model.eval()
    all_scores, all_actuals, all_tau, all_dates = [], [], [], []

    sample_idx = 0
    with torch.no_grad():
        for x, dt, y in test_loader:
            x, dt, y = x.to(device), dt.to(device), y.to(device)
            unc = uncertainty_pass(model, x, dt, n_passes=mc_passes)
            scores = unc["mean"]
            tau_dist = model(x, dt)[1]

            regime = compute_regime_labels(tau_dist)
            scores_adj = apply_regime_adjustment(scores, regime["fast_frac"], alpha_amp)

            all_scores.append(scores_adj.cpu().numpy())
            all_actuals.append(y.cpu().numpy())
            all_tau.append(regime["tau_mean"].cpu().numpy())
            n = x.size(0)
            all_dates.extend(dates[sample_idx : sample_idx + n])
            sample_idx += n

    scores_arr = np.vstack(all_scores)  # (T, N_etf)
    actuals_arr = np.vstack(all_actuals)  # (T, N_etf)

    # Long-short portfolio returns
    z_scores = np.apply_along_axis(zscore_rank, 1, scores_arr)
    z_sum = z_scores.sum(axis=1, keepdims=True)
    z_sum = np.where(np.abs(z_sum) < 1e-8, 1.0, z_sum)  # avoid divide-by-zero
    weights = np.clip(z_scores / z_sum, -0.2, 0.2)
    port_r = (weights * actuals_arr).sum(axis=1)
    equity = np.cumprod(1 + port_r)

    flat_ic = np.mean([ic(scores_arr[t], actuals_arr[t]) for t in range(len(port_r))])

    results = {
        "sharpe": sharpe(port_r),
        "max_drawdown": max_drawdown(equity),
        "ic": flat_ic,
        "hit_rate": hit_rate(scores_arr.ravel(), actuals_arr.ravel()),
        "n_days": len(port_r),
    }

    log.info(
        "Backtest results: Sharpe=%.3f  MaxDD=%.3f  IC=%.3f  HitRate=%.3f",
        results["sharpe"],
        results["max_drawdown"],
        results["ic"],
        results["hit_rate"],
    )

    # Save outputs
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    eq_df = pd.DataFrame(
        {"date": [str(d.date()) for d in all_dates], "equity": equity, "return": port_r}
    )
    eq_df.to_csv(out / "equity_curve.csv", index=False)

    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    log.info("Backtest outputs saved → %s", out)
    return results
