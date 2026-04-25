"""splitter.py — Walk-forward train / val / test split with DataLoaders."""
from __future__ import annotations

import pandas as pd
from torch.utils.data import DataLoader

from dataset import LTCDataset
from logging_utils import get_logger

log = get_logger(__name__)


def make_dataloaders(
    features:   pd.DataFrame,
    returns:    pd.DataFrame,
    delta_t:    pd.Series,
    window:     int = 63,
    batch_size: int = 32,
    train_end:  str = "2019-12-31",
    val_end:    str = "2022-12-31",
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train / val / test DataLoaders using walk-forward splits.

    Split:
        train : start → train_end
        val   : train_end+1 → val_end
        test  : val_end+1  → end

    Args:
        features:   (T × D) preprocessed feature DataFrame.
        returns:    (T × N) raw return DataFrame (targets).
        delta_t:    (T,) elapsed-days Series.
        window:     Lookback window.
        batch_size: Mini-batch size.
        train_end:  Last date of training set.
        val_end:    Last date of validation set.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    idx = features.index

    train_mask = idx <= train_end
    val_mask   = (idx > train_end) & (idx <= val_end)
    test_mask  = idx > val_end

    def _split(mask: pd.Series) -> LTCDataset:
        return LTCDataset(
            features=features.loc[mask],
            returns=returns.loc[mask],
            delta_t=delta_t.loc[mask],
            window=window,
        )

    train_ds = _split(train_mask)
    val_ds   = _split(val_mask)
    test_ds  = _split(test_mask)

    log.info("Split sizes — train: %d  val: %d  test: %d",
             len(train_ds), len(val_ds), len(test_ds))

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)
    return train_loader, val_loader, test_loader
