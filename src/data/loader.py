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
    # 3. Conditional quantile thresholds per regime
    # ------------------------------------------------------------
    thresholds: dict = {}
    for r in np.unique(regimes):
        mask = regimes == r
        if not np.any(mask):
            thresholds[int(r)] = np.nan
            continue
        thresholds[int(r)] = np.quantile(np.abs(price_series[mask]), quantile)

    # ------------------------------------------------------------
    # 4. Anomaly labeling
    # ------------------------------------------------------------
    anomaly_labels = (np.abs(price_series) > np.vectorize(thresholds.get)(regimes)).astype(int)
    # Append as extra column
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