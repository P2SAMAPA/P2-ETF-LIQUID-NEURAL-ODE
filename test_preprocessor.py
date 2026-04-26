# ruff: noqa: I001
"""test_preprocessor.py — Tests for MODWT, z-score, and Δt encoding."""

import numpy as np
import pandas as pd

from preprocessor import compute_delta_t, modwt_denoise, preprocess, rolling_zscore


def make_series(n=100, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(rng.normal(0, 0.02, n), index=dates, name="r")


def test_modwt_reduces_variance():
    s = make_series()
    d = modwt_denoise(s)
    assert pd.Series(d, index=s.index).var() <= s.var() * 1.01


def test_rolling_zscore_mean_near_zero():
    df = pd.DataFrame({"a": make_series(), "b": make_series(seed=1)})
    z = rolling_zscore(df, window=20)
    assert abs(z["a"].iloc[20:].mean()) < 0.2


def test_rolling_zscore_std_near_one():
    df = pd.DataFrame({"a": make_series()})
    z = rolling_zscore(df, window=20)
    std = z["a"].iloc[20:].std()
    assert 0.8 < std < 1.2


def test_delta_t_all_positive():
    dates = pd.bdate_range("2020-01-01", periods=60)
    dt = compute_delta_t(dates)
    assert (dt > 0).all()


def test_delta_t_weekend_gap():
    """A Monday should have dt ≥ 3 (Fri→Mon = 3 calendar days)."""
    dates = pd.bdate_range("2020-01-01", periods=10)
    dt = compute_delta_t(dates)
    assert dt.max() >= 3


def test_preprocess_no_nans(synthetic_returns, synthetic_macro):
    features, delta_t = preprocess(synthetic_returns, synthetic_macro, z_window=10)
    assert not features.isna().any().any()
    assert not delta_t.isna().any()


def test_preprocess_shape(synthetic_returns, synthetic_macro):
    features, delta_t = preprocess(synthetic_returns, synthetic_macro, z_window=10)
    n_cols = len(synthetic_returns.columns) + len(synthetic_macro.columns)
    assert features.shape[1] == n_cols
    assert len(delta_t) == len(features)
