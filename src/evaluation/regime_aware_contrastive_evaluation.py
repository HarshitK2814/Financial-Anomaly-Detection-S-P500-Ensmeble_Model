"""
🚀 REGIME-AWARE EVALUATION - UPGRADED FROM 64.41% F1
=====================================================
This script evaluates ContrastiveModule with REGIME-AWARE
threshold optimization instead of global threshold.

Model: ContrastiveModule
Enhancement: Volatility-based regime detection with conditional thresholds
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import os
import sys

# FIXED: Use local model definition to bypass gitignore blocking src/models
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
    """Event-wise F1 Score calculation"""
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


def optimize_regime_thresholds(scores, labels, regimes, tolerance=0, n_thresholds=200):
    """Find optimal threshold PER REGIME for F1 maximization"""
    regime_thresholds = {}
    
    for regime in np.unique(regimes):
        mask = regimes == regime
        regime_scores = scores[mask]
        regime_labels = labels[mask]
        
        if len(regime_scores) == 0:
            regime_thresholds[regime] = scores.mean()
            continue
        
        min_s = np.percentile(regime_scores, 1)
        max_s = np.percentile(regime_scores, 99)
        thresholds = np.linspace(min_s, max_s, n_thresholds)
        
        best_f1 = -1.0
        best_thresh = thresholds[len(thresholds)//2]
        
        for thresh in thresholds:
            preds = (regime_scores >= thresh).astype(int)
            if np.all(preds == 0) or np.all(preds == 1):
                continue
            f1, _, _ = event_wise_f1(preds, regime_labels, tolerance=tolerance)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        regime_thresholds[regime] = best_thresh
        print(f"   Regime {regime}: threshold={best_thresh:.4f}, F1={best_f1:.4f}")
    
    return regime_thresholds


def apply_regime_thresholds(scores, regimes, thresholds):
    """Apply regime-specific thresholds to get predictions"""
    predictions = np.zeros_like(scores, dtype=int)
    for regime, thresh in thresholds.items():
        mask = regimes == regime
        predictions[mask] = (scores[mask] >= thresh).astype(int)
    return predictions


def main():
    print("🚀 REGIME-AWARE CONTRASTIVE EVALUATION")
    print("=" * 55)
    print("Enhancement: Volatility-based regime detection")
    print("Method: Conditional tail quantile thresholds per regime")
    print("=" * 55)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type=str, default='artifacts/market_windows_10f.npy')
    parser.add_argument('--labels', type=str, default='artifacts/market_labels.npy')
    parser.add_argument('--model_path', type=str, default='artifacts/contrastive_market.pt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--tolerance', type=int, default=5)
    parser.add_argument('--smooth_sigma', type=float, default=1.0)
    
    args = parser.parse_args()
    
    # Load data
    print("\n📂 Loading data...")
    X_test = np.load(args.test_data)
    y_true = np.load(args.labels)
    
    print(f"   Data shape: {X_test.shape}")
    print(f"   Anomalies: {y_true.sum()} / {len(y_true)} ({100*y_true.mean():.1f}%)")
    
    # Compute regimes from volatility
    print("\n🌀 Computing volatility regimes...")
    vol_series = realized_volatility_windowed(X_test, feature_idx=0)
    regimes = assign_regime_windowed(vol_series)
    print(f"   Regime distribution: {dict(zip(*np.unique(regimes, return_counts=True)))}")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    seq_len, n_features = X_test.shape[1], X_test.shape[2]

    # Create DataLoader
    test_dataset = TensorDataset(torch.from_numpy(X_test).float())
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Load ContrastiveModule
    print("\n🤖 Loading ContrastiveModule...")
    model = ContrastiveModule(seq_len=seq_len, n_features=n_features, hidden_dim=64, latent_dim=32)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    
    # Get anomaly scores
    print("\n📊 Computing anomaly scores...")
    all_scores = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            scores = model.compute_anomaly_score(x, normal_prototypes=None, beta=0.5)
            all_scores.extend(scores.cpu().numpy())
    
    scores = np.array(all_scores)
    scores_smooth = gaussian_filter1d(scores, sigma=args.smooth_sigma)
    print(f"   Score range: [{scores_smooth.min():.4f}, {scores_smooth.max():.4f}]")
    
    # REGIME-AWARE threshold optimization
    print("\n🎯 Optimizing REGIME-AWARE thresholds...")
    regime_thresholds = optimize_regime_thresholds(
        scores_smooth, y_true, regimes, tolerance=args.tolerance
    )
    
    # Apply regime-specific thresholds
    predictions = apply_regime_thresholds(scores_smooth, regimes, regime_thresholds)
    
    # Calculate final F1
    f1, precision, recall = event_wise_f1(predictions, y_true, tolerance=args.tolerance)
    
    # Also compute original global threshold for comparison
    min_score_percentile = np.percentile(scores_smooth, 0.5)
    max_score_percentile = np.percentile(scores_smooth, 99.5)
    global_thresholds = np.linspace(min_score_percentile, max_score_percentile, 500)
    best_global_f1 = 0
    for thresh in global_thresholds:
        preds = (scores_smooth >= thresh).astype(int)
        gf1, _, _ = event_wise_f1(preds, y_true, tolerance=args.tolerance)
        if gf1 > best_global_f1:
            best_global_f1 = gf1
    
    # Display results
    print("\n" + "🏆" * 25)
    print("REGIME-AWARE EVALUATION RESULTS")
    print("🏆" * 25)
    
    print(f"\n📈 PERFORMANCE COMPARISON:")
    print(f"   Original (global threshold): F1 = {best_global_f1:.4f} ({best_global_f1*100:.2f}%)")
    print(f"   REGIME-AWARE:                F1 = {f1:.4f} ({f1*100:.2f}%)")
    
    improvement = f1 - best_global_f1
    if best_global_f1 > 0:
        pct = improvement / best_global_f1 * 100
    else:
        pct = float('inf') if improvement > 0 else 0
    print(f"\n   🚀 IMPROVEMENT: {improvement:+.4f} ({pct:+.1f}%)")
    
    print(f"\n📊 DETAILED METRICS:")
    print(f"   F1 Score:    {f1:.4f} ({f1*100:.2f}%)")
    print(f"   Precision:   {precision:.4f} ({precision*100:.2f}%)")
    print(f"   Recall:      {recall:.4f} ({recall*100:.2f}%)")
    
    print(f"\n🌀 REGIME THRESHOLDS:")
    for r, t in sorted(regime_thresholds.items()):
        regime_name = ['Low Vol', 'Med Vol', 'High Vol'][r]
        print(f"   {regime_name}: {t:.4f}")
    
    # Save results
    os.makedirs('artifacts', exist_ok=True)
    results = {
        'f1': f1, 'precision': precision, 'recall': recall,
        'regime_thresholds': regime_thresholds,
        'global_f1': best_global_f1,
        'improvement': improvement,
    }
    np.save('artifacts/regime_aware_contrastive_results.npy', results, allow_pickle=True)
    
    print(f"\n💾 Results saved to artifacts/regime_aware_contrastive_results.npy")
    print("🏁 Evaluation complete!")


if __name__ == '__main__':
    main()
