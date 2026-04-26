# ruff: noqa: I001
"""test_ranker.py — Tests for cross-sectional z-score ranker."""

import numpy as np
import pandas as pd

from ranker import build_daily_ranking, zscore_rank

N_ETF = 7
TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]


def test_zscore_mean_near_zero():
    rng = np.random.default_rng(0)
    scores = rng.normal(0, 1, N_ETF)
    z = zscore_rank(scores)
    assert abs(z.mean()) < 1e-10


def test_zscore_std_near_one():
    rng = np.random.default_rng(1)
    scores = rng.normal(0, 1, N_ETF)
    z = zscore_rank(scores)
    assert abs(z.std() - 1.0) < 1e-6


def test_zscore_no_nans():
    scores = np.array([0.1, -0.2, 0.3, 0.0, -0.1, 0.5, -0.4])
    z = zscore_rank(scores)
    assert not np.isnan(z).any()


def test_build_daily_ranking_shape():
    scores = np.random.default_rng(42).normal(0, 1, N_ETF)
    df = build_daily_ranking(scores, TICKERS, pd.Timestamp("2024-01-02"))
    assert len(df) == N_ETF
    assert set(df.columns) >= {"date", "ticker", "score_adj", "rank"}


def test_build_daily_ranking_sorted():
    scores = np.arange(N_ETF, dtype=float)
    df = build_daily_ranking(scores, TICKERS, pd.Timestamp("2024-01-02"))
    assert df["rank"].iloc[0] == 1
    assert list(df["rank"]) == sorted(df["rank"].tolist())


def test_build_daily_ranking_unique_ranks():
    scores = np.random.default_rng(7).normal(0, 1, N_ETF)
    df = build_daily_ranking(scores, TICKERS, pd.Timestamp("2024-01-02"))
    assert df["rank"].nunique() == N_ETF
