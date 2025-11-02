import numpy as np
import os

def truncate_features(input_path, output_path, n_features=10):
    # Load full dataset
    X = np.load(input_path)
    print(f"Original shape: {X.shape}")
    
    # Truncate to first n_features
    X_truncated = X[:, :, :n_features]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save truncated data
    np.save(output_path, X_truncated)
    print(f"Saved truncated shape: {X_truncated.shape}")

if __name__ == "__main__":
    truncate_features(
        input_path="artifacts/market_windows.npy",
        output_path="artifacts/market_windows_10f.npy"
    )