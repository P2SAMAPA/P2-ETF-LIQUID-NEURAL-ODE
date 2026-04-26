# ruff: noqa: I001
"""conftest.py — pytest fixtures for LTC engine tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from config import (
    DataConfig,
    LTCConfig,
    ModelConfig,
    ODEConfig,
    OutputConfig,
    ScoringConfig,
    TrainingConfig,
)
from ltc_model import LTCModel  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
N_DAYS = 80
N_ETF = 7  # FI universe size
N_MACRO = 6
INPUT_DIM = N_ETF + N_MACRO
HIDDEN = 8  # tiny for speed
WINDOW = 10


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_returns(rng) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=N_DAYS)
    tickers = [
        "TLT_log_return",
        "VCIT_log_return",
        "LQD_log_return",
        "HYG_log_return",
        "VNQ_log_return",
        "GLD_log_return",
        "SLV_log_return",
    ]
    data = rng.normal(0, 0.01, (N_DAYS, N_ETF))
    return pd.DataFrame(data, index=dates, columns=tickers)


@pytest.fixture
def synthetic_macro(rng) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=N_DAYS)
    cols = ["VIX", "DXY", "T10Y2Y", "HY_SPREADS", "WTI", "DTB3"]
    data = rng.normal(0, 1, (N_DAYS, N_MACRO))
    return pd.DataFrame(data, index=dates, columns=cols)


@pytest.fixture
def mock_config() -> LTCConfig:
    return LTCConfig(
        model=ModelConfig(
            n_neurons=HIDDEN,
            n_sensory=4,
            n_inter=2,
            n_command=2,
            input_dim=INPUT_DIM,
            dropout=0.1,
            tau_min=0.1,
            tau_max=5.0,
            sparsity=0.5,
        ),
        training=TrainingConfig(
            lr=1e-3, epochs=2, patience=5, batch_size=4, warmup_epochs=1, seed=42
        ),
        ode=ODEConfig(method="euler", rtol=1e-2, atol=1e-3, adjoint=False),
        data=DataConfig(window=WINDOW, train_end="2020-06-30", val_end="2020-09-30"),
        scoring=ScoringConfig(alpha_amp=0.3, mc_passes=3, ci_level=0.95),
        output=OutputConfig(checkpoint_dir="/tmp/ckpt_test", results_dir="/tmp/res_test"),
    )


@pytest.fixture
def tiny_model() -> LTCModel:
    return LTCModel(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN,
        n_etf=N_ETF,
        tau_min=0.1,
        tau_max=5.0,
        dropout=0.1,
        use_closed_form=True,  # fast for tests
    )


@pytest.fixture
def batch_tensors(rng):
    batch, win, dim = 4, WINDOW, INPUT_DIM
    x = torch.tensor(rng.normal(0, 1, (batch, win, dim)), dtype=torch.float32)
    dt = torch.ones(batch, win, dtype=torch.float32)
    y = torch.tensor(rng.normal(0, 0.01, (batch, N_ETF)), dtype=torch.float32)
    return x, dt, y
