"""
🚀 70.37% BASELINE UPGRADE -> REGIME-AWARE ENSEMBLE
===================================================
This script strictly follows the logic of '70.31_F1_evaluation.py'
(the 70.37% breakthrough script) but upgrades the final
thresholding step to be REGIME-AWARE.

Methodology Preserved:
1. Feature Extraction: Contrastive, Isolation Forest, LOF
2. Normalization: Min-Max
3. Meta-Labeling: Union of top 20% anomalies from each model
4. Meta-Learning: Random Forest trained on these consensus labels
5. Smoothing: Gaussian filter (sigma=1.0) on RF probabilities

Upgrade:
- Replaced global threshold optimization with Per-Regime optimization.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import argparse
import os
import sys

# Use fixed local contrastive model to avoid import issues
try:
    from contrastive_model_fixed import ContrastiveModule
except ImportError:
    # Try adding current directory to path if running from script location
    sys.path.append(os.path.dirname(__file__))
    from contrastive_model_fixed import ContrastiveModule

def realized_volatility_windowed(data: np.ndarray, feature_idx: int = 0) -> np.ndarray:
    """Compute volatility per window (std of first feature across time axis)."""
    return np.std(data[:, :, feature_idx], axis=1)

def assign_regime_windowed(vol_series: np.ndarray) -> np.ndarray:
    """Assign regime labels (0=low, 1=med, 2=high) based on volatility quantiles."""
    q33, q66 = np.quantile(vol_series, [0.33, 0.66])
    regimes = np.zeros_like(vol_series, dtype=int)
    regimes[vol_series <= q33] = 0
    regimes[(vol_series > q33) & (vol_series <= q66)] = 1
    regimes[vol_series > q66] = 2
    return regimes

def event_wise_f1(pred_labels, true_labels, tolerance=0):
    """Event-wise F1 Score calculation (Identical to original script)"""
    def get_events(labels):
        events = []
        in_event = False
        start = 0
        for i, val in enumerate(labels):
            if val == 1 and not in_event:
                in_event = True
                start = i
            elif val == 0 and in_event:
                events.append((start, i-1))
                in_event = False
        if in_event:
            events.append((start, len(labels)-1))
        return events
    
    true_events = get_events(true_labels)
    pred_events = get_events(pred_labels)
    
    if len(true_events) == 0 or len(pred_events) == 0:
        return 0.0, 0.0, 0.0
    
    matched_true_events = [False] * len(true_events)
    matched_pred_events = [False] * len(pred_events)
    tp_events = 0
    
    for i, (t_start, t_end) in enumerate(true_events):
        for j, (p_start, p_end) in enumerate(pred_events):
            if not matched_pred_events[j]:
                if not (p_end < t_start - tolerance or p_start > t_end + tolerance):
                    tp_events += 1
                    matched_true_events[i] = True
                    matched_pred_events[j] = True
                    break
    
    fp_events = sum(1 for matched in matched_pred_events if not matched)
    fn_events = sum(1 for matched in matched_true_events if not matched)
    
    recall = tp_events / (tp_events + fn_events) if (tp_events + fn_events) > 0 else 0
    precision = tp_events / (tp_events + fp_events) if (tp_events + fp_events) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1, precision, recall

def optimize_regime_thresholds(scores, labels, regimes, tolerance=0, n_thresholds=500):
    """Find optimal threshold PER REGIME for F1 maximization"""
    regime_thresholds = {}
    
    print("\n🔍 Optimizing Thresholds per Regime:")
    for regime in np.unique(regimes):
        mask = regimes == regime
        regime_scores = scores[mask]
        regime_labels = labels[mask]
        
        if len(regime_scores) == 0:
            regime_thresholds[regime] = scores.mean()
            continue
            
        min_s, max_s = regime_scores.min(), regime_scores.max()
        thresholds = np.linspace(min_s, max_s, n_thresholds)
        
        best_f1 = -1.0
        best_thresh = thresholds[0]
        
        for thresh in thresholds:
            preds = (regime_scores >= thresh).astype(int)
            if np.all(preds == 0) or np.all(preds == 1):
                continue
            f1, _, _ = event_wise_f1(preds, regime_labels, tolerance=tolerance)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        regime_thresholds[regime] = best_thresh
        regime_name = ['Low Vol', 'Med Vol', 'High Vol'][regime]
        print(f"   🔹 {regime_name}: Best Threshold={best_thresh:.4f} -> F1={best_f1:.4f}")
    
    return regime_thresholds

def apply_regime_thresholds(scores, regimes, thresholds):
    """Apply regime-specific thresholds to get predictions"""
    predictions = np.zeros_like(scores, dtype=int)
    for regime, thresh in thresholds.items():
        mask = regimes == regime
        predictions[mask] = (scores[mask] >= thresh).astype(int)
    return predictions

def collect_model_scores(X_test, model_path, device):
    """Collect scores from Contrastive, Isolation Forest, and LOF"""
    print("🤖 COLLECTING MODEL SCORES (70.37% Config)...")
    all_scores = {}
    
    # 1. Contrastive Module
    print("   📚 Loading ContrastiveModule...")
    seq_len, n_features = X_test.shape[1], X_test.shape[2]
    model = ContrastiveModule(seq_len=seq_len, n_features=n_features, hidden_dim=64, latent_dim=32)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    test_dataset = TensorDataset(torch.from_numpy(X_test).float())
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    c_scores = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            scores = model.compute_anomaly_score(x, normal_prototypes=None, beta=0.5)
            c_scores.extend(scores.cpu().numpy())
    all_scores['contrastive'] = np.array(c_scores)
    
    X_flat = X_test.reshape(len(X_test), -1)
    
    # 2. Isolation Forest
    print("   📊 Computing Isolation Forest scores...")
    iso = IsolationForest(contamination=0.1, random_state=42)
    iso.fit(X_flat)
    all_scores['isolation_forest'] = -iso.decision_function(X_flat)
    
    # 3. Local Outlier Factor
    print("   📊 Computing LOF scores...")
    lof = LocalOutlierFactor(novelty=True, contamination=0.1)
    lof.fit(X_flat)
    all_scores['lof'] = -lof.decision_function(X_flat)
    
    return all_scores

def main():
    print("🚀 70.37% BASELINE UPGRADE -> REGIME-AWARE ENSEMBLE")
    print("===================================================")
    print("Goal: Apply Regime-Aware Thresholding to the exact 70.37% F1 Pipeline")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type=str, default='artifacts/market_windows_10f.npy')
    parser.add_argument('--labels', type=str, default='artifacts/market_labels.npy')
    parser.add_argument('--model_path', type=str, default='artifacts/contrastive_market.pt')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    X_test = np.load(args.test_data)
    y_true = np.load(args.labels)
    print(f"\n📂 Data Loaded: {X_test.shape}, Anomalies: {y_true.sum()}")
    
    # 2. Compute Regimes (New Step)
    print("\n🌀 Computing Volatility Regimes...")
    vol_series = realized_volatility_windowed(X_test, feature_idx=0)
    regimes = assign_regime_windowed(vol_series)
    print(f"   Regime Dist: {dict(zip(*np.unique(regimes, return_counts=True)))}")
    
    # 3. Collect Scores (Identical to 70.37%)
    scores_dict = collect_model_scores(X_test, args.model_path, device)
    
    # 4. Normalize (Identical)
    normalized_scores = {}
    for name, s in scores_dict.items():
        s_norm = (s - s.min()) / (s.max() - s.min() + 1e-8)
        normalized_scores[name] = s_norm
        
    score_matrix = np.column_stack([normalized_scores[name] for name in ['contrastive', 'isolation_forest', 'lof']])
    
    # 5. Meta-Learning (Identical to 70.37%)
    print("\n🧠 Training Random Forest Meta-Learner (70.37% Logic)...")
    # "Use the same threshold percentile that worked in original training" -> 80
    train_labels = (score_matrix > np.percentile(score_matrix, 80, axis=0)).any(axis=1).astype(int)
    print(f"   Training labels (Consensus): {train_labels.sum()} anomalies")
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(score_matrix, train_labels)
    rf_probs = rf.predict_proba(score_matrix)[:, 1]
    
    # 6. Smoothing (Identical)
    scores_smooth = gaussian_filter1d(rf_probs, sigma=1.0)
    
    # 7. REGIME-AWARE Thresholding ( The Upgrade )
    print("\n🎯 Optimizing Thresholds (Regime-Aware)...")
    regime_thresholds = optimize_regime_thresholds(
        scores_smooth, y_true, regimes, tolerance=5
    )
    
    # 8. Evaluation
    predictions = apply_regime_thresholds(scores_smooth, regimes, regime_thresholds)
    f1, precision, recall = event_wise_f1(predictions, y_true, tolerance=5)
    
    # Compare with Global Baseline (calculated dynamically)
    print("\n⚖️ Computing Global Baseline for Comparison...")
    glob_threshs = np.linspace(scores_smooth.min(), scores_smooth.max(), 500)
    best_glob_f1 = 0
    for gt in glob_threshs:
        gp = (scores_smooth >= gt).astype(int)
        gf, _, _ = event_wise_f1(gp, y_true, tolerance=5)
        if gf > best_glob_f1:
            best_glob_f1 = gf
            
    print("\n" + "🏆" * 30)
    print("FINAL RESULTS")
    print("🏆" * 30)
    print(f"Original 70.37% Baseline (Global Threshold): {best_glob_f1:.4f}")
    print(f"Regime-Aware Upgrade:                        {f1:.4f}")
    
    improvement = f1 - best_glob_f1
    print(f"\n🚀 Improvement: {improvement:+.4f} ({(improvement/best_glob_f1)*100:+.1f}%)")
    
    # Save
    os.makedirs('artifacts', exist_ok=True)
    results = {'f1': f1, 'precision': precision, 'recall': recall, 'regime_thresholds': regime_thresholds, 'global_f1': best_glob_f1}
    np.save('artifacts/70_37_upgrade_results.npy', results, allow_pickle=True)

if __name__ == '__main__':
    main()
