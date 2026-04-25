"""wavelet.py — MODWT (stationary wavelet transform) wrapper using PyWavelets."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pywt


def modwt(
    series:  pd.Series | np.ndarray,
    wavelet: str = "db4",
    level:   int = 4,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Apply MODWT decomposition.

    Args:
        series:  Input time series.
        wavelet: Wavelet family (default db4, consistent with suite).
        level:   Decomposition level.

    Returns:
        approx:  Approximation (low-frequency) coefficients.
        details: List of detail coefficient arrays (high-frequency).
    """
    values = np.asarray(series, dtype=float)
    coeffs = pywt.swt(values, wavelet=wavelet, level=level, trim_approx=False)
    approx  = coeffs[-1][0]
    details = [c[1] for c in coeffs]
    return approx, details


def modwt_denoise(
    series:    pd.Series | np.ndarray,
    wavelet:   str   = "db4",
    level:     int   = 4,
    threshold: float | None = None,
) -> np.ndarray:
    """MODWT denoising via universal soft thresholding.

    If threshold is None it is estimated via the median absolute deviation
    of the finest detail coefficients (Donoho & Johnstone 1994).
    """
    values = np.asarray(series, dtype=float)
    coeffs = pywt.swt(values, wavelet=wavelet, level=level, trim_approx=False)

    denoised = []
    for i, (approx, detail) in enumerate(coeffs):
        if threshold is None:
            thr = np.median(np.abs(detail)) / 0.6745 * np.sqrt(2 * np.log(len(detail) + 1))
        else:
            thr = threshold
        detail_d = pywt.threshold(detail, thr, mode="soft")
        denoised.append((approx, detail_d))

    return pywt.iswt(denoised, wavelet=wavelet)
