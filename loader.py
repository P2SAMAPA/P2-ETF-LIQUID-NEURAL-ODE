"""loader.py — Load master_data.parquet from HuggingFace Hub.

Source: P2SAMAPA/fi-etf-macro-signal-master-data

Columns are plain ticker names (e.g. "TLT", "SPY") for returns,
plus macro columns: VIX, DXY, T10Y2Y, TBILL_3M, IG_SPREAD, HY_SPREAD.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from logging_utils import get_logger

log = get_logger(__name__)

HF_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
PARQUET_FILE = "master_data.parquet"

FI_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLY", "XLP", "XLU", "GDX", "XME", "IWM",
]
EXTENDED_TICKERS = ["IWF", "XSD", "XBI", "XLB", "XLRE"]
ALL_TICKERS = FI_TICKERS + EQUITY_TICKERS + EXTENDED_TICKERS

UNIVERSE_MAP = {
    "fi": FI_TICKERS,
    "equity": EQUITY_TICKERS,
    "extended": EXTENDED_TICKERS,
    "combined": ALL_TICKERS,
}

# Exact macro column names as they appear in master_data.parquet
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]

# Columns to drop — parquet artefact and benchmark tickers (not return targets)
_DROP_COLS = ["__index_level_0__"]
_BENCHMARK_COLS = ["AGG"]  # benchmark only — exclude from return targets


def load_master_data(
    hf_repo: str = HF_REPO,
    parquet_file: str = PARQUET_FILE,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Download and return master_data.parquet as a DataFrame."""
    log.info("Loading master data from %s / %s", hf_repo, parquet_file)
    local_path = hf_hub_download(
        repo_id=hf_repo,
        filename=parquet_file,
        repo_type="dataset",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    df = pd.read_parquet(local_path)

    # Drop artefact columns
    df.drop(columns=[c for c in _DROP_COLS if c in df.columns], inplace=True)

    # Normalise index to DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    log.info(
        "Loaded %d rows x %d cols  [%s -> %s]",
        len(df), len(df.columns), df.index[0].date(), df.index[-1].date(),
    )
    log.info("Columns: %s", list(df.columns))
    return df


def get_universe_data(
    df: pd.DataFrame,
    universe: str = "combined",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (returns_df, macro_df) for a given universe.

    Returns columns are plain ticker names. Macro columns are exact names
    from MACRO_COLS. Any ticker not present in the parquet is skipped with
    a warning so partial universes still work.

    Args:
        df: Full master DataFrame from load_master_data().
        universe: 'fi', 'equity', or 'combined'.

    Returns:
        returns_df: Ticker return columns for the universe.
        macro_df:   Macro feature columns.
    """
    if universe not in UNIVERSE_MAP:
        raise ValueError(f"Unknown universe '{universe}'. Choose from {list(UNIVERSE_MAP)}")

    tickers = UNIVERSE_MAP[universe]

    # Return cols — plain ticker names
    ret_cols = [t for t in tickers if t in df.columns and t not in _BENCHMARK_COLS]
    missing = [t for t in tickers if t not in df.columns]
    if missing:
        log.warning("Tickers not found in parquet (skipped): %s", missing)

    # Macro cols — exact names
    macro_cols = [c for c in MACRO_COLS if c in df.columns]
    missing_macro = [c for c in MACRO_COLS if c not in df.columns]
    if missing_macro:
        log.warning("Macro cols not found in parquet (skipped): %s", missing_macro)

    if not ret_cols:
        log.error(
            "No return columns found for universe '%s'. "
            "Available columns: %s",
            universe, list(df.columns),
        )
        raise ValueError(
            f"No return columns found for universe '{universe}'. "
            "Check logs for available column names."
        )

    returns_df = df[ret_cols].copy()
    macro_df = df[macro_cols].copy() if macro_cols else pd.DataFrame(index=df.index)

    # Drop rows where all returns are NaN
    valid = returns_df.dropna(how="all").index
    returns_df = returns_df.loc[valid]
    macro_df = macro_df.loc[valid].ffill()

    log.info(
        "Universe '%s': %d tickers, %d macro cols, %d trading days",
        universe, len(ret_cols), len(macro_cols), len(returns_df),
    )
    return returns_df, macro_df
