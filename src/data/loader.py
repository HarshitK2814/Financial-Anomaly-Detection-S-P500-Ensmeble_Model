import os
import json
import numpy as np
import pandas as pd
from typing import Union, Tuple
from sklearn.preprocessing import StandardScaler

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

def load_and_preprocess_data(path: str, 
                           window_size: int = 128,
                           stride: int = 1,
                           normalize: bool = True) -> Tuple[np.ndarray, StandardScaler]:
    """
    Load and preprocess time series data with sliding windows and normalization
    """
    data = load_data(path)
    
    # If DataFrame, convert to numpy array
    if isinstance(data, pd.DataFrame):
        data = data.values
        
    # Normalize data
    scaler = StandardScaler()
    if normalize:
        if len(data.shape) == 3:  # Already windowed
            # Reshape to 2D for scaling
            orig_shape = data.shape
            data = data.reshape(-1, data.shape[-1])
            data = scaler.fit_transform(data)
            data = data.reshape(orig_shape)
        else:
            data = scaler.fit_transform(data)
    
    # Create sliding windows if not already windowed
    if len(data.shape) == 2:
        windows = []
        for i in range(0, len(data) - window_size + 1, stride):
            windows.append(data[i:i + window_size])
        data = np.array(windows)
    
    return data, scaler