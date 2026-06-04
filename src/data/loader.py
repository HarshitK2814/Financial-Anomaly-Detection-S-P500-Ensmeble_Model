import os
import json
import numpy as np
import pandas as pd
from typing import Union, Tuple
from sklearn.preprocessing import StandardScaler
from ..utils.volatility import realized_volatility, assign_regime

def load_data(path: str):
    """
    Load dataset from CSV, JSON/JSONL or numpy (.npy / .npz).
    Returns a numpy array or pandas DataFrame depending on file type.
    """
    if not path:
        raise ValueError("No path provided to load_data()")
    if not os.path.exists(path):
        raise ValueError(f"Path does not exist: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".json", ".jsonl"):
        with open(path, "r", encoding="utf-8") as fh:
            if ext == ".jsonl":
                return [json.loads(l) for l in fh if l.strip()]
            return json.load(fh)
    if ext in (".npy", ".npz"):
        loaded = np.load(path, allow_pickle=True)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            keys = list(loaded.files)
            if not keys:
                raise ValueError("Empty .npz archive")
            return loaded[keys[0]]
        return loaded
    raise ValueError(f"Unsupported file format ({ext}). Supported: .csv, .json, .jsonl, .npy, .npz")

def load_and_preprocess_data(
    path: str,
    window_size: int = 128,
    stride: int = 1,
    normalize: bool = True,
    quantile: float = 0.99,
) -> Tuple[np.ndarray, StandardScaler, np.ndarray, dict]:
    """Load, preprocess, and label anomalies in time‑series data.

    This version adds:
    1. Realized volatility calculation and regime assignment.
    2. Conditional upper‑tail quantile thresholds per regime.
    3. Binary anomaly labels (1 if |r| exceeds the regime‑specific quantile).
    4. The anomaly flag is appended as an extra feature/channel.
    """
    # Load raw data
    data = load_data(path)

    # Ensure numpy array
    if isinstance(data, pd.DataFrame):
        data = data.values
    if data.ndim == 1:
        data = data[:, None]

    # ------------------------------------------------------------
    # 2. Realized volatility & regime assignment (use first column)
    # ------------------------------------------------------------
    price_series = data[:, 0].astype(float)
    vol_series = realized_volatility(price_series)
    regimes = assign_regime(vol_series)

    # ------------------------------------------------------------
    # 3 + 4. Anomaly labeling via rolling expanding-window quantile
    #
    # \u26a0\ufe0f  R2 LEAKAGE FIX: replaced regime-conditional full-dataset
    # quantile thresholds with a rolling expanding-window quantile.
    # Old code:  thresholds[r] = np.quantile(|price[regime_mask]|, q)
    # This used ALL rows (including future test data) to set the threshold.
    #
    # Fix: at each time t the threshold is computed from |price[0:t]| only.
    # min_periods=20 means the first 19 rows use the threshold from row 19
    # (earliest available; minor init artifact, unavoidable for expanding windows).
    # ------------------------------------------------------------
    returns_abs    = pd.Series(np.abs(price_series))
    rolling_thresh = (
        returns_abs
        .expanding(min_periods=20)
        .quantile(quantile)
        .bfill()       # fill first <20 NaN rows with earliest computed threshold
        .values
    )
    anomaly_labels = (returns_abs.values > rolling_thresh).astype(int)

    # Thresholds dict: per-regime mean of rolling threshold (API compatibility)
    thresholds: dict = {
        int(r): float(rolling_thresh[regimes == r].mean())
        for r in np.unique(regimes)
    }

    # Append anomaly label as extra column
    data = np.column_stack((data, anomaly_labels))

    # ------------------------------------------------------------
    # 5. Normalisation (exclude the binary label from scaling)
    # ------------------------------------------------------------
    scaler = StandardScaler()
    if normalize:
        data[:, :-1] = scaler.fit_transform(data[:, :-1])
    else:
        scaler.fit(data[:, :-1])

    # ------------------------------------------------------------
    # 6. Sliding windows (if required)
    # ------------------------------------------------------------
    if len(data.shape) == 2:
        windows = []
        for i in range(0, len(data) - window_size + 1, stride):
            windows.append(data[i : i + window_size])
        data = np.array(windows)

    return data, scaler, regimes, thresholds