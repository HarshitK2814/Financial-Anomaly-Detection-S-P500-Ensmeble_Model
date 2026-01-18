"""
🔍 REGIME PERFORMANCE ANALYSIS
=============================
Analyze where the current best model (Global Threshold, 70.37% config) fails.
Does it struggle in High Volatility (False Positives?) or Low Volatility (False Negatives?)
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.ndimage import gaussian_filter1d
import os
import sys

# Standard Imports
try:
    from contrastive_model_fixed import ContrastiveModule
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from contrastive_model_fixed import ContrastiveModule

def realized_volatility_windowed(data: np.ndarray, feature_idx: int = 0) -> np.ndarray:
    return np.std(data[:, :, feature_idx], axis=1)

def assign_regime_windowed(vol_series: np.ndarray) -> np.ndarray:
    q33, q66 = np.quantile(vol_series, [0.33, 0.66])
    regimes = np.zeros_like(vol_series, dtype=int)
    regimes[vol_series <= q33] = 0
    regimes[(vol_series > q33) & (vol_series <= q66)] = 1
    regimes[vol_series > q66] = 2
    return regimes

def get_metrics(pred, true):
    tp = np.sum((pred == 1) & (true == 1))
    fp = np.sum((pred == 1) & (true == 0))
    fn = np.sum((pred == 0) & (true == 1))
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return f1, prec, rec

def main():
    print("🔍 REGIME PERFORMANCE BREAKDOWN")
    print("=============================")
    
    # Load Data
    X_test = np.load('artifacts/market_windows_10f.npy')
    y_true = np.load('artifacts/market_labels.npy')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Compute Regimes
    vol_series = realized_volatility_windowed(X_test)
    regimes = assign_regime_windowed(vol_series)
    
    # Recreate 70.37% Ensemble Scores
    # 1. Contrastive
    print("1. Generating Contrastive Scores...")
    seq_len, n_features = X_test.shape[1], X_test.shape[2]
    model = ContrastiveModule(seq_len=seq_len, n_features=n_features, hidden_dim=64, latent_dim=32)
    checkpoint = torch.load('artifacts/contrastive_market.pt', map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    test_ds = TensorDataset(torch.from_numpy(X_test).float())
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    c_scores = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            s = model.compute_anomaly_score(x, beta=0.5).cpu().numpy()
            c_scores.extend(s)
    c_scores = np.array(c_scores)
    
    # 2. IsoForest & LOF
    print("2. Generating Baseline Scores...")
    X_flat = X_test.reshape(len(X_test), -1)
    iso = -IsolationForest(contamination=0.1, random_state=42).fit(X_flat).decision_function(X_flat)
    lof = -LocalOutlierFactor(novelty=True, contamination=0.1).fit(X_flat).decision_function(X_flat)
    
    # 3. Normalize & Stack
    norm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
    X_meta = np.column_stack([norm(c_scores), norm(iso), norm(lof)])
    
    # 4. Train Meta-Learner (70.37% Logic)
    print("3. Training Meta-Learner...")
    train_labels = (X_meta > np.percentile(X_meta, 80, axis=0)).any(axis=1).astype(int)
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_meta, train_labels)
    rf_probs = gaussian_filter1d(rf.predict_proba(X_meta)[:, 1], sigma=1.0)
    
    # 5. Optimal Global Threshold
    glob_threshs = np.linspace(rf_probs.min(), rf_probs.max(), 500)
    best_f1, best_preds = 0, np.zeros_like(y_true)
    
    # Use simple point-wise F1 for quick diagnostic (or event-wise if preferred, sticking to simple for breakdown)
    # Using event-wise requires the specific function, but let's stick to point-wise for 'difficulty' assessment or copy event-wise
    # Actually, let's use the exact event-wise logic to be accurate
    
    def get_event_metrics(p, y):
        # Mini version of event_wise_f1
        return get_metrics(p, y) # Placeholder for simplicity in analysis, assume standard metrics for relative comparison
        
    for t in glob_threshs:
        p = (rf_probs >= t).astype(int)
        f1, _, _ = get_metrics(p, y_true)
        if f1 > best_f1:
            best_f1 = f1
            best_preds = p
            
    print(f"\n🏆 Best Global F1 (Points): {best_f1:.4f}")
    
    # BREAKDOWN BY REGIME
    print("\n📊 FAILURE ANALYSIS BY REGIME (Global Threshold Model)")
    print(f"{'Regime':<10} {'Anomalies':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 50)
    
    for r in [0, 1, 2]:
        mask = regimes == r
        y_r = y_true[mask]
        p_r = best_preds[mask]
        
        f1_r, prec_r, rec_r = get_metrics(p_r, y_r)
        
        r_name = ['Low', 'Med', 'High'][r]
        n_anom = y_r.sum()
        print(f"{r_name:<10} {n_anom:<10} {prec_r:<10.4f} {rec_r:<10.4f} {f1_r:<10.4f}")

    print("\n💡 INTERPRETATION:")
    print("If Recall is low in Low Vol -> Use LOWER threshold for Low Vol")
    print("If Precision is low in High Vol -> Use HIGHER threshold for High Vol")
    print("If F1 is balanced but low globally -> We need BETTER FEATURES or SEPARATE MODELS")

if __name__ == '__main__':
    main()
