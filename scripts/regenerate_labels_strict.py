"""
🔧 REGENERATE LABELS (STRICT JOURNAL MODE)
==========================================
Problem: Previous labels had ~42% anomaly rate (q=0.58). Too high.
Fix: Regenerate labels using Strict Tail Quantile (q=0.98).

Method:
1. Load `market_windows_10f.npy`.
2. Extract Feature 0 (Returns).
3. Compute Volatility & Regimes (Low/Med/High).
4. Apply q=0.98 threshold per regime.
5. Save new `artifacts/market_labels.npy`.
"""
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from src.utils.volatility import realized_volatility, assign_regime
except ImportError:
    # Fallback if running from proper root
    pass

def realize_vol_window(window_returns):
    # Realized vol of a window = std dev of returns in that window
    return np.std(window_returns, axis=1)

def main():
    print("🔧 REGENERATING LABELS (STRICT q=0.98)")
    print("======================================")
    
    # 1. Load Data
    data_path = 'artifacts/market_windows_10f.npy'
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found.")
        return
        
    X = np.load(data_path)
    print(f"Loaded Data: {X.shape}")
    
    # 2. Extract Returns (Feature 0)
    # Shape (N_samples, Window_Len)
    returns = X[:, :, 0]
    
    # 3. Compute Volatility & Regimes
    # We define 'Sample Volatility' as the Realized Vol of the window
    vol_series = realize_vol_window(returns)
    
    # Assign Regimes (0=Low, 1=Med, 2=High)
    # We use the utility function logic: q33, q66 breakpoints
    q33, q66 = np.quantile(vol_series, [0.33, 0.66])
    regimes = np.zeros_like(vol_series, dtype=int)
    regimes[vol_series <= q33] = 0
    regimes[(vol_series > q33) & (vol_series <= q66)] = 1
    regimes[vol_series > q66] = 2
    
    print(f"Regime Counts: {np.unique(regimes, return_counts=True)}")
    
    # 4. Apply Strict Thresholds (q=0.98)
    QUANTILE = 0.98
    new_labels = np.zeros(len(returns), dtype=int)
    
    thresholds = {}
    
    # To detect anomaly in a window, we check if the LAST return 
    # (or Max return in window?) exceeds the threshold.
    # Usually Anomaly Detection targets the 'current' point (last in window).
    # Let's use the LAST point in the window as the target to classify.
    last_returns = np.abs(returns[:, -1])
    
    for r in [0, 1, 2]:
        mask = regimes == r
        if not np.any(mask):
            continue
            
        # Determine Threshold for this regime
        # strictly based on the distribution of returns WITHIN this regime
        regime_returns = last_returns[mask]
        thresh = np.quantile(regime_returns, QUANTILE)
        thresholds[r] = thresh
        
        # Label
        # Anomaly if |Return| > Threshold
        new_labels[mask] = (last_returns[mask] > thresh).astype(int)
        
    # Stats
    n_anom = new_labels.sum()
    rate = n_anom / len(new_labels)
    print(f"\n📊 New Statistics (q={QUANTILE}):")
    print(f"   Total Samples: {len(new_labels)}")
    print(f"   Anomalies:     {n_anom}")
    print(f"   Rate:          {rate:.2%} (Target: <5%)")
    print(f"   Thresholds:    {thresholds}")
    
    if rate > 0.10:
        print("⚠️ WARNING: Rate still high. Check data distribution.")
    else:
        print("✅ Rate is defensible.")
        
    # 5. Save
    out_path = 'artifacts/market_labels.npy'
    np.save(out_path, new_labels)
    print(f"\n💾 Overwrote {out_path}")

if __name__ == '__main__':
    main()
