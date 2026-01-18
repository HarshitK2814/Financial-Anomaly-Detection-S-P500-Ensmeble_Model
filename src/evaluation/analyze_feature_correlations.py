"""
🕵️‍♂️ FORENSIC FEATURE IDENTIFICATION
=================================
Goal: Identify what the "Unknown Features 1-9" likely represent
      by correlating them with explicitly calculated financial indicators.
      
Methodology:
1. Treat Feature 0 as the 'Anchor' (Return series).
2. Calculate standard indicators:
   - Volatility (Std Dev)
   - Momentum (RSI, ROC)
   - Trend (SMA Diff)
   - Tail (Kurtosis)
3. Correlate these 'Explicit Factors' with 'Unknown Features 1-9'.
4. Rename 'Unknowns' based on highest correlation (>0.5).
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os

def calculate_rsi(data, window=14):
    """Relative Strength Index"""
    # Simple manual implementation for numpy
    # This is rough on windows, but okay for correlation check
    # Vectorized approximation for speed
    delta = np.diff(data, prepend=0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Simple moving average for simplicity in this forensic check
    avg_gain = pd.Series(gain).rolling(window=window).mean().fillna(0).values
    avg_loss = pd.Series(loss).rolling(window=window).mean().fillna(0).values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def main():
    print("🕵️‍♂️ STARTING FORENSIC FEATURE ANALYSIS")
    print("======================================")
    
    # 1. Load Data
    # Shape: (Samples, Window, Features) -> (977, 128, 10)
    X = np.load('artifacts/market_windows_10f.npy')
    print(f"Data Shape: {X.shape}")
    
    # Flatten checks?
    # We should correlate Time-Series behavior.
    # Approach: Take the *last value* of each window to build a Time Series of length 977
    # Or average? Last value is best for 'current state'.
    
    # Extract Time Series (Taking last step of each window to reconstruct the series)
    # Actually, windows overlap. Let's just use the feature vectors of the *last* time step 
    # for each sample. It's a valid proxy for the relationship.
    
    X_last = X[:, -1, :] # (977, 10)
    
    # Feature 0 is Anchor (Returns)
    returns = X_last[:, 0]
    
    # 2. construct Explicit Factors
    print("\n🏗️ Constructing Explicit Economic Factors...")
    factors = {}
    
    # Volatility (Rolling Std of returns)
    # Since we have windows, we can compute realized vol directly from the window!
    # Realized Vol = Std(window_returns)
    factors['Explicit_Vol'] = np.std(X[:, :, 0], axis=1) # (977,)
    
    # Momentum (RSI of the price derived from returns)
    # We can compute RSI *within* the window or globally.
    # Let's compute 'Window Momentum' = Sum(Returns) or Mean(Returns)
    factors['Explicit_Mom_Mean'] = np.mean(X[:, :, 0], axis=1)
    factors['Explicit_Mom_Sum'] = np.sum(X[:, :, 0], axis=1)
    
    # Tail Risk (Kurtosis of window)
    from scipy.stats import kurtosis, skew
    factors['Explicit_Kurt'] = kurtosis(X[:, :, 0], axis=1)
    factors['Explicit_Skew'] = skew(X[:, :, 0], axis=1)
    
    # Trend (Linear slope of window)
    def slope(y):
        x = np.arange(len(y))
        return np.polyfit(x, y, 1)[0]
    factors['Explicit_Trend'] = np.apply_along_axis(slope, 1, X[:, :, 0])
    
    # 3. Correlation Matrix
    print("\n🔍 Correlating Unknowns with Explicit Factors...")
    
    results = []
    
    print(f"{'Unknown Cloud':<15} | {'Best Match':<20} | {'Corr':<6} | {'Interpretation'}")
    print("-" * 65)
    
    feature_map = {0: "Returns (Anchor)"}
    
    for i in range(1, 10):
        unknown_feat = X_last[:, i] # Or mean of window? Let's try Mean first, as some feats might be noise
        # Actually, let's try correlation with both Mean and Last to be sure.
        # But 'Realized Vol' is a scalar summary of the window. So 'unknown' might be a summary too?
        # But the dataset is (N, T, F). 
        # If the features are technical indicators, they are usually time-series themselves.
        # So we compare 'Last Value' of Unknown vs 'Last Value' of Computed (or 'Scalar' computed from window).
        
        # Let's assume Unknowns are also Time Series.
        # Compare "Mean of Unknown Window" vs Factors? 
        # Or "Last of Unknown" vs Factors?
        
        # Taking "Mean of Unknown" as a robust proxy for "Level of Feature i"
        feat_vec = np.mean(X[:, :, i], axis=1)
        
        best_corr = 0
        best_name = "None"
        
        for name, factor_vec in factors.items():
            corr, _ = pearsonr(feat_vec, factor_vec)
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_name = name
        
        # Heuristic Naming
        interpretation = "Latent/Unknown"
        if abs(best_corr) > 0.6:
            interpretation = f"Likely {best_name.replace('Explicit_', '')}"
        elif abs(best_corr) > 0.3:
            interpretation = f"Weak {best_name.replace('Explicit_', '')}"
            
        print(f"Feature {i:<14} | {best_name:<20} | {best_corr:+.3f}  | {interpretation}")
        
        # Assign Name if strong match
        if abs(best_corr) > 0.5:
            feature_map[i] = best_name.replace('Explicit_', '') + "_Proxy"
        else:
            feature_map[i] = f"Latent_State_{i}"

    print("-" * 65)
    print("\n✅ Proposed Feature Map (Journal Ready):")
    for k, v in feature_map.items():
        print(f"   Feature {k} -> {v}")
        
    # Save Map for next steps
    import json
    with open('artifacts/feature_map_proposal.json', 'w') as f:
        json.dump(feature_map, f, indent=4)
    print("\n💾 Saved map to artifacts/feature_map_proposal.json")

if __name__ == '__main__':
    main()
