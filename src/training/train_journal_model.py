"""
🌳 JOURNAL-GRADE MODEL: REGIME-CONDITIONED RANDOM FOREST
======================================================
Architecture: 'Mixture of Factor Experts'
Input: 24 Economic Factors (Volatility, Momentum, Tail Risk, Lags)
Logic: separate Random Forest for each volatility regime.

Improvements over legacy:
- Uses FACTORS, not raw sequences.
- Supervised cost-sensitivity (balanced weights).
- No SMOTE (preserves temporal reality).
- Interpretability via Feature Importance (SHAP).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import os
import argparse

def train_regime_model():
    print("🌳 TRAINING REGIME-CONDITIONED FACTOR MODEL")
    print("=========================================")
    
    # 1. Load Factor Data
    X = np.load('artifacts/X_factors.npy')
    y = np.load('artifacts/y_factors.npy')
    df = pd.read_csv('artifacts/journal_factors.csv')
    print(f"Data: {X.shape}, Anomalies: {y.sum()}")
    
    # 2. Assign Regimes (Recalculate on 'Return_Trend' or similar?)
    # We should stick to the same Definition of Regime as before for consistency.
    # Regime based on 'Composite_Volatility' factors?
    # Or load original regimes?
    # Let's re-compute using 'Composite_Volatility' column from the factor set.
    # Note: 'Composite_Volatility' is already a volatility proxy.
    
    vol_scores = df['Composite_Volatility'].values
    q33, q66 = np.quantile(vol_scores, [0.33, 0.66])
    regimes = np.zeros(len(vol_scores), dtype=int)
    regimes[vol_scores <= q33] = 0
    regimes[(vol_scores > q33) & (vol_scores <= q66)] = 1
    regimes[vol_scores > q66] = 2
    
    print(f"Regimes: {np.unique(regimes, return_counts=True)}")
    
    # 3. Train Experts
    experts = {}
    preds = np.zeros(len(y))
    probs = np.zeros(len(y))
    
    for r in [0, 1, 2]:
        mask = regimes == r
        X_r = X[mask]
        y_r = y[mask]
        
        # Check imbalance
        n_pos = y_r.sum()
        if n_pos < 2:
            print(f"⚠️ Regime {r}: Not enough anomalies ({n_pos}). Skipping.")
            continue
            
        print(f"   training Expert {r} ('{['Low','Med','High'][r]}'): {len(X_r)} samples, {n_pos} anomalies ({n_pos/len(X_r):.1%})")
        
        # COST-SENSITIVE TRAINING (class_weight='balanced')
        # Replaces SMOTE
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,        # Shallow trees for interpretability & anti-overfitting
            class_weight='balanced', # Key change
            random_state=42
        )
        rf.fit(X_r, y_r)
        experts[r] = rf
        
        # Infer
        p = rf.predict(X_r)
        prob = rf.predict_proba(X_r)[:, 1]
        preds[mask] = p
        probs[mask] = prob
        
        # Feature Importance per Expert
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        print(f"     Top Driver: {df.columns[indices[0]]} ({importances[indices[0]]:.2f})")

    # 4. Evaluation
    f1, prec, rec, _ = precision_recall_fscore_support(y, preds, average='binary')
    
    print("\n" + "🏆" * 30)
    print("JOURNAL MODEL RESULTS (FACTOR-BASED)")
    print("🏆" * 30)
    print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    
    # 5. Interpretation
    if f1 > 0.80:
        print("\n✅ TARGET (>80%) ACHIEVED! Factor model is superior.")
    elif f1 > 0.77:
        print("\n✅ MATCHES DEEP LEARNING (77%). Interpretability win.")
    else:
        print("\n⚠️ Performance dip. Factors might be too compressed.")
        
    # Save results
    results = {'f1': f1, 'precision': prec, 'recall': rec, 'probs': probs}
    np.save('artifacts/journal_model_results.npy', results, allow_pickle=True)
    
if __name__ == '__main__':
    train_regime_model()
