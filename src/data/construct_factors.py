"""
🏗️ JOURNAL-GRADE FACTOR CONSTRUCTION
====================================
Transform raw time-series windows into interpretable Economic Factors.

Input:  (N_samples, Window_Size, 10_Features)
Output: (N_samples, K_Factors)

Mappings (based on Forensic Analysis):
- Returns: Feat 0
- Momentum: Feat 1 (Proxy)
- Volatility: Feat 2, 3, 7, 8, 9 (Proxies)
- Latent/Microstructure: Feat 4, 5, 6

Steps:
1. Aggregate windows into scalar scores (Mean/Last).
2. Compress redundant Volatility features into 'Composite_Vol'.
3. Compute explicit Tail Risk (Kurtosis).
4. Add Temporal Lags [1, 5].
5. Save as clean CSV/NPY for modelling.
"""
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
import os

def construct_factors(input_path, output_dir):
    print("🏗️ CONSTRUCTING JOURNAL FACTORS")
    print("==============================")
    
    # Load raw
    X = np.load(input_path) # (977, 128, 10)
    y = np.load('artifacts/market_labels.npy')
    print(f"Input Shape: {X.shape}")
    
    # Placeholder for factor DataFrame
    df = pd.DataFrame()
    
    # 1. PRIMARY FACTORS (Aggregated from Windows)
    # We use 'Mean' to capture the "state" over the window, 
    # and 'Last' to capture the "instantaneous" signal?
    # Strategy: Mean is robust.
    
    # F0: Returns -> Trend/Drift
    df['Return_Trend'] = np.mean(X[:, :, 0], axis=1)
    
    # F1: Momentum Proxy
    df['Momentum_Factor'] = np.mean(X[:, :, 1], axis=1)
    
    # F2,3,7,8,9: Volatility Proxies
    # Compress into one Composite Volatility Score (Mean of all proxies)
    vol_channels = X[:, :, [2, 3, 7, 8, 9]]
    # Mean across channels, then mean across time
    # Or: Mean across time, then mean of factors? same thing.
    df['Composite_Volatility'] = np.mean(np.mean(vol_channels, axis=2), axis=1)
    
    # F4,5,6: Latent Microstructure
    # Keep separate as they are likely orthogonal signals
    df['Latent_State_A'] = np.mean(X[:, :, 4], axis=1)
    df['Latent_State_B'] = np.mean(X[:, :, 5], axis=1)
    df['Latent_State_C'] = np.mean(X[:, :, 6], axis=1)
    
    # 2. EXPLICIT TAIL RISK (Calculated from F0 Returns)
    # Kurtosis of the return distribution within the window
    # High Kurtosis = Fat Tails = Crash Risk
    df['Tail_Kurtosis'] = kurtosis(X[:, :, 0], axis=1)
    df['Tail_Skew'] = skew(X[:, :, 0], axis=1)
    
    print(f"Base Factors: {df.shape[1]}")
    print(df.head())
    
    # 3. TEMPORAL LAGS (Depth)
    # Shift the factors to give the model context of "Previous State"
    # IMPORTANT: The data samples might be overlapping windows or sequential days.
    # Assuming sequentiality based on filename 'market_windows...' usually implies sliding.
    # If stride=1, row i and row i-1 are very close.
    # Lags of *samples* (rows) might be redundant if stride=1.
    # But 'Lag 1' row might mean 'State 1 day ago' if stride=1.
    # Let's add them.
    
    lags = [1, 5]
    cols_to_lag = df.columns.tolist()
    
    for lag in lags:
        for col in cols_to_lag:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
            
    # Drop NaN from lags
    df.dropna(inplace=True)
    # Align y
    y = y[df.index]
    
    print(f"Final Factor Set: {df.shape[1]} features")
    print(f"Final Samples: {len(df)}")
    
    # 4. Save
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV for Inspection (Interpretability)
    df.to_csv(os.path.join(output_dir, 'journal_factors.csv'), index=False)
    
    # Save NPY for Training
    np.save(os.path.join(output_dir, 'X_factors.npy'), df.values)
    np.save(os.path.join(output_dir, 'y_factors.npy'), y)
    
    print("\n💾 Saved to artifacts/journal_factors.csv")
    print("💾 Saved to artifacts/X_factors.npy")

if __name__ == '__main__':
    construct_factors(
        input_path='artifacts/market_windows_10f.npy',
        output_dir='artifacts'
    )
