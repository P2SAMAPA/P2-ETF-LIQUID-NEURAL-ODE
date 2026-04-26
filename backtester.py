"""backtester.py — Walk-forward backtest over the test period (2023–2026)."""

from __future__ import annotations

import json
import os
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
    universe: str = "combined",
) -> dict:
    """Walk-forward backtest.

    Returns a metrics dict and saves:
        - equity_curve.csv
        - metrics.json
        - scores.csv  ← NEW: per-ETF daily scores for HF dataset + Streamlit app
    Also pushes scores to HuggingFace dataset if HF_TOKEN is set.
    """
    model.eval()
    all_scores, all_scores_raw, all_ci_lower, all_ci_upper = [], [], [], []
    all_actuals, all_tau, all_fast_frac, all_dates = [], [], [], []
    sample_idx = 0

    with torch.no_grad():
        for x, dt, y in test_loader:
            x, dt, y = x.to(device), dt.to(device), y.to(device)

            unc = uncertainty_pass(model, x, dt, n_passes=mc_passes)
            scores_raw = unc["mean"]  # raw motor output
            ci_lower = unc["ci_lower"]
            ci_upper = unc["ci_upper"]

            tau_dist = model(x, dt)[1]
            regime = compute_regime_labels(tau_dist)
            scores_adj = apply_regime_adjustment(scores_raw, regime["fast_frac"], alpha_amp)

            all_scores_raw.append(scores_raw.cpu().numpy())
            all_scores.append(scores_adj.cpu().numpy())
            all_ci_lower.append(ci_lower.cpu().numpy())
            all_ci_upper.append(ci_upper.cpu().numpy())
            all_actuals.append(y.cpu().numpy())
            # Ensure tau_mean and fast_frac are (batch_size,) arrays — broadcast scalar if needed
            n = x.size(0)
            tau_val = np.atleast_1d(regime["tau_mean"].cpu().numpy())
            ff_val = np.atleast_1d(regime["fast_frac"].cpu().numpy())
            if tau_val.shape[0] == 1:
                tau_val = np.repeat(tau_val, n)
            if ff_val.shape[0] == 1:
                ff_val = np.repeat(ff_val, n)
            all_tau.append(tau_val)
            all_fast_frac.append(ff_val)

            all_dates.extend(dates[sample_idx : sample_idx + n])
            sample_idx += n

    scores_arr = np.vstack(all_scores)  # (T, N_etf)
    scores_raw_arr = np.vstack(all_scores_raw)
    ci_lower_arr = np.vstack(all_ci_lower)
    ci_upper_arr = np.vstack(all_ci_upper)
    actuals_arr = np.vstack(all_actuals)  # (T, N_etf)
    tau_arr = np.concatenate(all_tau)  # (T,)
    fast_frac_arr = np.concatenate(all_fast_frac)  # (T,)

    # ── Long-short portfolio returns ──────────────────────────────────────────
    z_scores = np.apply_along_axis(zscore_rank, 1, scores_arr)
    z_sum = z_scores.sum(axis=1, keepdims=True)
    z_sum = np.where(np.abs(z_sum) < 1e-8, 1.0, z_sum)
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

    # ── Save outputs ──────────────────────────────────────────────────────────
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Equity curve (existing)
    eq_df = pd.DataFrame(
        {"date": [str(d.date()) for d in all_dates], "equity": equity, "return": port_r}
    )
    eq_df.to_csv(out / "equity_curve.csv", index=False)

    # 2. Metrics (existing)
    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # 3. Per-ETF daily scores ← NEW
    n_days = len(all_dates)
    rows = []
    for t in range(n_days):
        date_str = str(all_dates[t].date())
        z_row = z_scores[t]  # cross-sectional z-scores for ranking
        ranks = pd.Series(z_row).rank(ascending=False, method="min").astype(int).tolist()
        for i, ticker in enumerate(tickers):
            rows.append(
                {
                    "date": date_str,
                    "ticker": ticker,
                    "score_raw": float(scores_raw_arr[t, i]),
                    "score_adj": float(scores_arr[t, i]),
                    "ci_lower": float(ci_lower_arr[t, i]),
                    "ci_upper": float(ci_upper_arr[t, i]),
                    "tau_mean": float(tau_arr[t]),
                    "fast_frac": float(fast_frac_arr[t]),
                    "rank": ranks[i],
                    "universe": universe,
                }
            )

    scores_df = pd.DataFrame(rows)
    scores_csv = out / "scores.csv"
    scores_df.to_csv(scores_csv, index=False)
    log.info("Scores saved → %s  (%d rows)", scores_csv, len(scores_df))

    # 4. Push scores to HuggingFace dataset ← NEW
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            from publisher import push_results

            push_results(scores_df, token=hf_token)
            log.info("Scores pushed to HuggingFace dataset ✅")
        except Exception as exc:
            log.warning("HF push failed (non-fatal): %s", exc)
    else:
        log.info("HF_TOKEN not set — skipping HF push (scores saved locally only)")

    log.info("Backtest outputs saved → %s", out)
    return results
