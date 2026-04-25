"""test_integration.py — End-to-end integration test.

10-day synthetic data → 2 training epochs → score → assert output shape.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from config import LTCConfig
from dataset import LTCDataset
from loss import SharpeLoss
from ltc_model import LTCModel
from preprocessor import preprocess
from ranker import zscore_rank
from tau_monitor import compute_regime_labels

N_DAYS    = 40
N_ETF     = 7
N_MACRO   = 6
INPUT_DIM = N_ETF + N_MACRO
WINDOW    = 5
HIDDEN    = 8


def _make_data():
    rng   = np.random.default_rng(99)
    dates = pd.bdate_range("2021-01-01", periods=N_DAYS)
    ret_cols   = [f"ETF{i}_log_return" for i in range(N_ETF)]
    macro_cols = [f"MAC{i}"            for i in range(N_MACRO)]
    returns_df = pd.DataFrame(rng.normal(0, 0.01, (N_DAYS, N_ETF)),  index=dates, columns=ret_cols)
    macro_df   = pd.DataFrame(rng.normal(0, 1,    (N_DAYS, N_MACRO)), index=dates, columns=macro_cols)
    return returns_df, macro_df


def test_e2e_train_and_score():
    returns_df, macro_df = _make_data()
    features, delta_t   = preprocess(returns_df, macro_df, z_window=5, denoise=False)

    ds     = LTCDataset(features, returns_df, delta_t, window=WINDOW)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    model     = LTCModel(INPUT_DIM, HIDDEN, N_ETF, use_closed_form=True, dropout=0.0)
    criterion = SharpeLoss(lam=0.1)
    optim     = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 2 training epochs
    model.train()
    for _ in range(2):
        for x, dt, y in loader:
            optim.zero_grad()
            scores, _ = model(x, dt)
            loss, _   = criterion(scores, y)
            loss.backward()
            optim.step()

    # Score one batch
    model.eval()
    with torch.no_grad():
        x, dt, y = next(iter(loader))
        scores, tau_dist = model(x, dt)

    assert scores.shape  == (x.size(0), N_ETF),   f"Bad score shape: {scores.shape}"
    assert not torch.isnan(scores).any(),           "NaN in scores"

    regime = compute_regime_labels(tau_dist)
    assert torch.isfinite(regime["fast_frac"]).all()
    assert torch.isfinite(regime["tau_mean"]).all()

    z = zscore_rank(scores[0].numpy())
    assert abs(z.mean()) < 1e-9
    assert not np.isnan(z).any()


def test_e2e_output_n_etf():
    """Ensure output dimension matches universe size."""
    returns_df, macro_df = _make_data()
    features, delta_t   = preprocess(returns_df, macro_df, z_window=5, denoise=False)
    ds = LTCDataset(features, returns_df, delta_t, window=WINDOW)
    x, dt, _ = ds[0]
    model = LTCModel(INPUT_DIM, HIDDEN, N_ETF, use_closed_form=True)
    model.eval()
    with torch.no_grad():
        scores, _ = model(x.unsqueeze(0), dt.unsqueeze(0))
    assert scores.shape[-1] == N_ETF
