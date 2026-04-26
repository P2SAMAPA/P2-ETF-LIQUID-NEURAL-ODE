"""
score_writer.py — Call this at the end of evaluate.py to write results/scores.csv
in the exact schema the Streamlit app and GitHub Actions workflow expect.

Usage inside evaluate.py:
    from score_writer import write_scores
    write_scores(
        dates=dates,          # list/array of datetime-like, one per row
        tickers=tickers,      # list of str, one per row
        score_raw=raw_arr,    # np.ndarray (N,)
        score_adj=adj_arr,    # np.ndarray (N,)
        ci_lower=ci_lo,       # np.ndarray (N,)
        ci_upper=ci_hi,       # np.ndarray (N,)
        tau_mean=tau_arr,     # np.ndarray (N,)
        fast_frac=ff_arr,     # np.ndarray (N,)
        universe="fi",        # "fi" | "equity" | "combined"
        out_dir="results",
    )
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

FI_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]


def write_scores(
    dates,
    tickers,
    score_raw,
    score_adj,
    ci_lower,
    ci_upper,
    tau_mean,
    fast_frac,
    universe: str,
    out_dir: str = "results",
) -> pd.DataFrame:
    """
    Build the scores DataFrame and write to {out_dir}/scores.csv.

    All array arguments must be the same length N (one row per ETF per date).
    Returns the DataFrame for further use.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates).strftime("%Y-%m-%d"),
            "ticker": list(tickers),
            "score_raw": np.asarray(score_raw, dtype=float),
            "score_adj": np.asarray(score_adj, dtype=float),
            "ci_lower": np.asarray(ci_lower, dtype=float),
            "ci_upper": np.asarray(ci_upper, dtype=float),
            "tau_mean": np.asarray(tau_mean, dtype=float),
            "fast_frac": np.asarray(fast_frac, dtype=float),
            "universe": universe,
        }
    )

    # Compute cross-sectional rank per date (1 = best score_adj)
    df["rank"] = df.groupby("date")["score_adj"].rank(ascending=False, method="min").astype(int)

    out_path = os.path.join(out_dir, "scores.csv")
    df.to_csv(out_path, index=False)
    print(f"[score_writer] Wrote {len(df)} rows → {out_path}")
    return df
