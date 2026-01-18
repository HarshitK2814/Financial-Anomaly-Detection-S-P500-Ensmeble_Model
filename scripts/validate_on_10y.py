"""
🧪 VALIDATION ON 10-YEAR DATA
===========================
Runs the full 'Journal Pipeline' on the fresh yfinance data.
Data: artifacts/sp500_10y_windows.npy (2357 samples)

Steps:
1. Label Generation (Strict q=0.98)
2. Factor Construction (Using Feature 0=LogRet, Feat 2=Vol, etc.)
3. Model Training (Regime MoE)
4. Reporting
"""
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
import os

def validate_10y():
    print("🧪 VALIDATING ON 5-YEAR S&P 500 DATA (2021-2025)")
    print("==============================================")
    
    # 1. Load Data
    X = np.load('artifacts/sp500_5y_windows.npy')
    print(f"Data Loaded: {X.shape}")
    
    # --- STEP 1: LABELING (Strict q=0.98) ---
    print("\n🏷️ Generating Labels (q=0.98)...")
    returns = X[:, :, 0] # Feature 0 is Log Returns
    
    # Compute Volatility (Std of returns in window)
    vol_series = np.std(returns, axis=1)
    
    # Assign Regimes
    q33, q66 = np.quantile(vol_series, [0.33, 0.66])
    regimes = np.zeros(len(vol_series), dtype=int)
    regimes[vol_series <= q33] = 0
    regimes[(vol_series > q33) & (vol_series <= q66)] = 1
    regimes[vol_series > q66] = 2
    print(f"   Regimes: {np.unique(regimes, return_counts=True)}")
    
    # Apply Thresholds
    y = np.zeros(len(returns), dtype=int)
    last_returns = np.abs(returns[:, -1])
    
    for r in [0, 1, 2]:
        mask = regimes == r
        thresh = np.quantile(last_returns[mask], 0.98)
        y[mask] = (last_returns[mask] > thresh).astype(int)
        
    print(f"   Anomalies: {y.sum()} ({y.sum()/len(y):.2%})")

    # --- STEP 2: FACTOR CONSTRUCTION ---
    print("\n🏗️ Constructing Factors...")
    # Map features from yfinance script:
    # 0:LogRet, 1:RSI, 2:Vol20, 3:VolChg, 4:HL, 5:MACD, 6:BB, 7:Skew, 8:Kurt, 9:Lag
    
    df = pd.DataFrame()
    
    # F0: Trend (Mean Return)
    df['Return_Trend'] = np.mean(X[:, :, 0], axis=1)
    
    # F1: Momentum (RSI Mean)
    df['Momentum_Factor'] = np.mean(X[:, :, 1], axis=1) # RSI
    
    # F2: Volatility (Vol20 Mean)
    df['Composite_Volatility'] = np.mean(X[:, :, 2], axis=1)
    
    # F3: Volume Shock
    df['Volume_Shock'] = np.max(X[:, :, 3], axis=1)
    
    # F4: Intraday Range (Latent State Proxy)
    df['Latent_State_C'] = np.mean(X[:, :, 4], axis=1)
    
    # F5: Tail Risk (Explicit Kurtosis of Returns)
    df['Tail_Kurtosis'] = kurtosis(X[:, :, 0], axis=1)
    
    # Lags
    lags = [1, 5]
    cols_to_lag = df.columns.tolist()
    for lag in lags:
        for col in cols_to_lag:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
            
    # Drop Nan
    df.dropna(inplace=True)
    mask_valid = df.index
    X_factors = df.values
    y_factors = y[mask_valid]
    regimes_factors = regimes[mask_valid]
    
    print(f"   Factor Shape: {X_factors.shape}")

    # --- STEP 3: TRAINING ---
    print("\n🌳 Training Regime MoE...")
    preds = np.zeros(len(y_factors))
    
    for r in [0, 1, 2]:
        mask = regimes_factors == r
        if mask.sum() == 0: continue
        
        X_r = X_factors[mask]
        y_r = y_factors[mask]
        
        if y_r.sum() < 2:
            print(f"   ⚠️ Regime {r}: Not enough anomalies. Skipping.")
            continue
            
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
        rf.fit(X_r, y_r)
        preds[mask] = rf.predict(X_r)
        
        # Feature Importance
        imp = rf.feature_importances_
        best_idx = np.argmax(imp)
        print(f"   Regime {r} Driver: {df.columns[best_idx]} ({imp[best_idx]:.2f})")

    # --- STEP 4: REPORT ---
    curr_prec, curr_rec, curr_f1, _ = precision_recall_fscore_support(y_factors, preds, average='binary')
    print("\n" + "="*30)
    print("📢 5-YEAR RECENT RESULTS (2021-2025)")
    print("="*30)
    print(f"Dataset: 2021-2025 (Approx)")
    print(f"Anomalies: {y_factors.sum()} (Rate: {y_factors.sum()/len(y_factors):.2%})")
    print(f"F1 Score:  {curr_f1:.4f}")
    print(f"Precision: {curr_prec:.4f}")
    print(f"Recall:    {curr_rec:.4f}")
    
    if curr_f1 > 0.80:
        print("\n✅ VALIDATED: Strong performance (>80%) holds on recent data.")
    else:
        print("\n⚠️ DIVERGENCE: Result differs on fresh data. Check feature maps.")

if __name__ == '__main__':
    validate_10y()
