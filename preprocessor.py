"""preprocessor.py — MODWT denoising, z-score normalisation, Δt encoding.

Δt (elapsed calendar days) is a first-class feature that drives the
adaptive time constants τ(x,h) inside the LTC cell.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pywt

from logging_utils import get_logger

log = get_logger(__name__)

WAVELET = "db4"
WV_LEVEL = 4
Z_WINDOW = 63  # rolling z-score window (trading days)


# ── Wavelet denoising ─────────────────────────────────────────────────────────


def modwt_denoise(series: pd.Series, wavelet: str = WAVELET, level: int = WV_LEVEL) -> pd.Series:
    """Apply MODWT (à-trous) denoising via soft-thresholding.

    Returns a denoised series with the same index as input.
    Level is automatically capped to the max supported by the data length.
    """
    values = series.ffill().fillna(0.0).values.astype(float)
    # Cap level to what PyWavelets supports for this data length
    max_level = pywt.swt_max_level(len(values))
    actual_level = min(level, max_level)
    if actual_level < 1:
        return series.copy()
    coeffs = pywt.swt(values, wavelet=wavelet, level=actual_level, trim_approx=False)
    denoised_coeffs = []
    for i, (approx, detail) in enumerate(coeffs):
        if i < len(coeffs) - 1:
            threshold = np.median(np.abs(detail)) / 0.6745 * np.sqrt(2 * np.log(len(detail)))
            detail = pywt.threshold(detail, threshold, mode="soft")
        denoised_coeffs.append((approx, detail))
    reconstructed = pywt.iswt(denoised_coeffs, wavelet=wavelet)
    return pd.Series(reconstructed, index=series.index, name=series.name)


def denoise_dataframe(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Apply MODWT denoising to every column of a DataFrame."""
    return df.apply(lambda col: modwt_denoise(col, **kwargs))


# ── Rolling z-score normalisation ────────────────────────────────────────────


def rolling_zscore(df: pd.DataFrame, window: int = Z_WINDOW) -> pd.DataFrame:
    """Rolling z-score normalisation (mean and std over past `window` days)."""
    mu = df.rolling(window, min_periods=max(1, window // 2)).mean()
    std = df.rolling(window, min_periods=max(1, window // 2)).std().replace(0, 1e-8)
    return (df - mu) / std


# ── Δt encoding ──────────────────────────────────────────────────────────────


def compute_delta_t(index: pd.DatetimeIndex) -> pd.Series:
    """Compute elapsed calendar days between consecutive observations.

    Δt is a first-class LTC input — it drives the adaptive time constants.
    Missing / holiday gaps naturally produce larger Δt values.
    """
    dates = pd.Series(index, index=index)
    delta = dates.diff().dt.days.fillna(1.0).clip(lower=1.0)
    return delta.rename("delta_t")


# ── Master preprocessor ───────────────────────────────────────────────────────


def preprocess(
    returns_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    z_window: int = Z_WINDOW,
    denoise: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Full preprocessing pipeline.

    Steps:
        1. MODWT denoising on returns and macro.
        2. Rolling z-score normalisation.
        3. Δt encoding.
        4. Forward-fill then zero-fill any remaining NaNs.
        5. Concatenate into a single feature DataFrame.

    Args:
        returns_df: Raw log-return DataFrame.
        macro_df:   Raw macro feature DataFrame.
        z_window:   Rolling normalisation window.
        denoise:    Whether to apply MODWT denoising.

    Returns:
        features: (T × D) normalised feature DataFrame.
        delta_t:  (T,) elapsed-days Series.
    """
    log.info("Preprocessing: denoise=%s, z_window=%d", denoise, z_window)

    if denoise:
        returns_df = denoise_dataframe(returns_df)
        macro_df = denoise_dataframe(macro_df)

    returns_z = rolling_zscore(returns_df, z_window)
    macro_z = rolling_zscore(macro_df, z_window)

    features = pd.concat([returns_z, macro_z], axis=1)
    features = features.ffill().fillna(0.0)

    delta_t = compute_delta_t(features.index)

    log.info("Feature matrix: %s", features.shape)
    return features, delta_t
