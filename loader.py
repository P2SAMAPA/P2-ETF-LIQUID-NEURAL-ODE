"""loader.py — Load master_data.parquet from HuggingFace Hub.

Source: P2SAMAPA/fi-etf-macro-signal-master-data
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from logging_utils import get_logger

log = get_logger(__name__)

HF_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
PARQUET_FILE = "master_data.parquet"

FI_TICKERS     = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_TICKERS = ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
                   "XLY", "XLP", "XLU", "GDX", "XME", "IWM"]
ALL_TICKERS    = FI_TICKERS + EQUITY_TICKERS

MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "HY_SPREADS", "WTI", "DTB3"]

UNIVERSE_MAP = {
    "fi":       FI_TICKERS,
    "equity":   EQUITY_TICKERS,
    "combined": ALL_TICKERS,
}


def load_master_data(
    hf_repo: str = HF_REPO,
    parquet_file: str = PARQUET_FILE,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Download and return master_data.parquet as a DataFrame.

    The parquet is cached locally after first download.
    """
    log.info("Loading master data from %s / %s", hf_repo, parquet_file)
    local_path = hf_hub_download(
        repo_id=hf_repo,
        filename=parquet_file,
        repo_type="dataset",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    df = pd.read_parquet(local_path)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    log.info("Loaded %d rows × %d cols  [%s → %s]",
             len(df), len(df.columns), df.index[0].date(), df.index[-1].date())
    return df


def get_universe_data(
    df: pd.DataFrame,
    universe: str = "combined",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (returns_df, macro_df) for a given universe.

    Args:
        df: Full master DataFrame.
        universe: 'fi', 'equity', or 'combined'.

    Returns:
        returns_df: log-return columns for universe tickers.
        macro_df:   macro feature columns.
    """
    tickers = UNIVERSE_MAP[universe]
    ret_cols = [f"{t}_log_return" for t in tickers if f"{t}_log_return" in df.columns]
    macro_cols = [c for c in MACRO_COLS if c in df.columns]

    if not ret_cols:
        raise ValueError(f"No return columns found for universe '{universe}'")

    returns_df = df[ret_cols].copy()
    macro_df   = df[macro_cols].copy()

    # Align index — drop rows where all returns are NaN
    valid = returns_df.dropna(how="all").index
    returns_df = returns_df.loc[valid]
    macro_df   = macro_df.loc[valid]

    log.info("Universe '%s': %d tickers, %d macro features, %d trading days",
             universe, len(ret_cols), len(macro_cols), len(returns_df))
    return returns_df, macro_df
