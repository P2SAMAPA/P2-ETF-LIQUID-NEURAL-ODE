"""dataset.py — PyTorch Dataset for rolling-window LTC training.

Each sample is a (x, delta_t, y) tuple:
  x        : (window, input_dim) float32 feature tensor
  delta_t  : (window,) float32 elapsed-days tensor
  y        : (n_etf,) float32 next-day returns (targets)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class LTCDataset(Dataset):
    """Rolling-window dataset for LTC training.

    Args:
        features:  (T × D) normalised feature DataFrame.
        returns:   (T × N) raw log-return DataFrame (targets).
        delta_t:   (T,) elapsed calendar days Series.
        window:    Lookback window length.
    """

    def __init__(
        self,
        features: pd.DataFrame,
        returns:  pd.DataFrame,
        delta_t:  pd.Series,
        window:   int = 63,
    ) -> None:
        self.features = features.values.astype(np.float32)
        self.returns  = returns.values.astype(np.float32)
        self.delta_t  = delta_t.values.astype(np.float32)
        self.window   = window
        # Align: target is the day *after* the window ends
        self.n_samples = len(features) - window

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x   = torch.from_numpy(self.features[idx : idx + self.window])       # (W, D)
        dt  = torch.from_numpy(self.delta_t[idx : idx + self.window])        # (W,)
        y   = torch.from_numpy(self.returns[idx + self.window])              # (N,)
        return x, dt, y
