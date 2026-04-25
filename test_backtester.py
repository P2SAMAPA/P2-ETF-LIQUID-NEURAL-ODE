"""test_backtester.py — Smoke tests for the walk-forward backtester."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


N_DAYS  = 60
N_ETF   = 7
HIDDEN  = 8
TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]


def _make_loader(n=N_DAYS, batch=8):
    rng = np.random.default_rng(0)
    x   = torch.tensor(rng.normal(0, 1,    (n, 10, N_ETF + 6)), dtype=torch.float32)
    dt  = torch.ones(n, 10, dtype=torch.float32)
    y   = torch.tensor(rng.normal(0, 0.01, (n, N_ETF)),          dtype=torch.float32)
    return DataLoader(TensorDataset(x, dt, y), batch_size=batch, shuffle=False)


def test_backtest_returns_required_keys(tmp_path, tiny_model):
    from backtester import run_backtest
    loader = _make_loader()
    dates  = [pd.Timestamp("2024-01-01") + pd.Timedelta(days=i) for i in range(N_DAYS)]
    result = run_backtest(
        tiny_model, loader, TICKERS, dates,
        device=torch.device("cpu"),
        mc_passes=2,
        output_dir=str(tmp_path),
    )
    for key in ("sharpe", "max_drawdown", "ic", "hit_rate", "n_days"):
        assert key in result, f"Missing key: {key}"


def test_backtest_sharpe_is_finite(tmp_path, tiny_model):
    from backtester import run_backtest
    loader = _make_loader()
    dates  = [pd.Timestamp("2024-01-01") + pd.Timedelta(days=i) for i in range(N_DAYS)]
    result = run_backtest(tiny_model, loader, TICKERS, dates,
                          device=torch.device("cpu"), mc_passes=2,
                          output_dir=str(tmp_path))
    assert np.isfinite(result["sharpe"])


def test_backtest_equity_curve_saved(tmp_path, tiny_model):
    from backtester import run_backtest
    loader = _make_loader()
    dates  = [pd.Timestamp("2024-01-01") + pd.Timedelta(days=i) for i in range(N_DAYS)]
    run_backtest(tiny_model, loader, TICKERS, dates,
                 device=torch.device("cpu"), mc_passes=2,
                 output_dir=str(tmp_path))
    assert (tmp_path / "equity_curve.csv").exists()
    assert (tmp_path / "metrics.json").exists()
