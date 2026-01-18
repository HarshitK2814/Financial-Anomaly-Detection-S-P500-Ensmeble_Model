import numpy as np
from typing import Tuple

def realized_volatility(series: np.ndarray, window: int = 20) -> np.ndarray:
    """Compute rolling realized volatility (standard deviation) of a 1‑D series.

    Args:
        series: 1‑D array of price or return values.
        window: Rolling window size (default 20 periods).
    Returns:
        np.ndarray of same length as ``series`` where the first ``window-1``
        entries are ``np.nan`` and subsequent entries contain the rolling
        standard deviation.
    """
    if series.ndim != 1:
        raise ValueError("realized_volatility expects a 1‑D array")
    # Use numpy's stride tricks for efficient rolling std
    # Pad with nan for the initial window
    vol = np.full_like(series, fill_value=np.nan, dtype=float)
    if len(series) >= window:
        # Compute rolling std using a sliding window
        for i in range(window - 1, len(series)):
            vol[i] = np.std(series[i - window + 1 : i + 1], ddof=1)
    return vol

def assign_regime(vol_series: np.ndarray) -> np.ndarray:
    """Assign a regime label (0=low, 1=medium, 2=high) based on quantiles.

    The function calculates the 33rd and 66th percentiles of the *non‑nan*
    volatility values and assigns regimes accordingly.
    """
    if vol_series.ndim != 1:
        raise ValueError("assign_regime expects a 1‑D array")
    # Compute quantiles ignoring NaNs
    valid = vol_series[~np.isnan(vol_series)]
    if len(valid) == 0:
        raise ValueError("vol_series contains only NaNs")
    q33, q66 = np.quantile(valid, [0.33, 0.66])
    regimes = np.full_like(vol_series, fill_value=-1, dtype=int)
    regimes[vol_series <= q33] = 0
    regimes[(vol_series > q33) & (vol_series <= q66)] = 1
    regimes[vol_series > q66] = 2
    return regimes
