"""publisher.py — Push daily scores to HuggingFace Dataset.

Target: P2SAMAPA/p2-etf-liquid-neural-ode-results
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset

from logging_utils import get_logger

log = get_logger(__name__)

HF_RESULTS_REPO = "P2SAMAPA/p2-etf-liquid-neural-ode-results"


def push_results(
    results_df:  pd.DataFrame,
    hf_repo:     str = HF_RESULTS_REPO,
    token:       str | None = None,
) -> None:
    """Append new results to the HF Dataset.

    The dataset stores all historical daily scores. Each call appends
    new rows without overwriting existing data.

    Args:
        results_df: DataFrame with columns matching the results schema.
        hf_repo:    HF dataset repository ID.
        token:      HF write token (reads HF_TOKEN env var if None).
    """
    import os
    hf_token = token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("Set HF_TOKEN environment variable or pass token=")

    log.info("Pushing %d rows to %s", len(results_df), hf_repo)

    # Convert date column to string for parquet compatibility
    df = results_df.copy()
    if pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    new_ds = Dataset.from_pandas(df, preserve_index=False)

    try:
        existing = load_dataset(hf_repo, split="train", token=hf_token)
        combined = Dataset.from_pandas(
            pd.concat([existing.to_pandas(), df], ignore_index=True),
            preserve_index=False,
        )
    except Exception:
        log.warning("No existing dataset found — creating fresh.")
        combined = new_ds

    combined.push_to_hub(hf_repo, token=hf_token)
    log.info("Results pushed successfully → %s", hf_repo)


def save_results_locally(
    results_df: pd.DataFrame,
    output_dir: str = "results",
) -> Path:
    """Save results as parquet locally (fallback / debug)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    min_date = results_df["date"].astype(str).min()
    max_date = results_df["date"].astype(str).max()
    path = out / f"scores_{min_date}_{max_date}.parquet"
    results_df.to_parquet(path, index=False)
    log.info("Results saved locally → %s", path)
    return path
