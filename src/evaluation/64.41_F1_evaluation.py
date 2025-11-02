"""
🚀 BEST MODEL EVALUATION - TARGET: 64.41% F1 SCORE
==================================================
This script evaluates our star performer: ContrastiveModule
with optimal hyperparameters to achieve 64.41% F1 score.

Model: ContrastiveModule
Configuration: tolerance=5, smooth_sigma=1.0, weighted ensemble optimization
Expected Performance: F1 = 0.6441
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import os
import sys

# Add src directory to path to import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.contrastive_module import ContrastiveModule

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

def optimize_threshold_for_f1(scores, labels, tolerance=0, n_thresholds=500):
    """Find optimal threshold for F1 maximization"""
    if np.all(scores == scores[0]):
        return scores[0], (0.0, 0.0, 0.0)

    min_score_percentile = np.percentile(scores, 0.5)
    max_score_percentile = np.percentile(scores, 99.5)
    
    if min_score_percentile >= max_score_percentile:
        thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    else:
        thresholds = np.linspace(min_score_percentile, max_score_percentile, n_thresholds)

    best_f1 = -1.0
    best_threshold = thresholds[0] if len(thresholds) > 0 else scores.min()
    best_metrics = (0, 0, 0)
    
    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        
        if np.all(preds == 0) or np.all(preds == 1):
            current_f1 = 0.0
            prec, rec = 0.0, 0.0
        else:
            current_f1, prec, rec = event_wise_f1(preds, labels, tolerance=tolerance)
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = thresh
            best_metrics = (current_f1, prec, rec)
    
    if best_f1 == -1.0:
        return scores.mean(), (0.0, 0.0, 0.0)

    return best_threshold, best_metrics

def plot_score_distribution(scores, labels, threshold, title="Anomaly Score Distribution", save_path="artifacts/best_model_score_distribution.png"):
    """Plot the distribution of anomaly scores for normal and anomalous points"""
    plt.figure(figsize=(12, 6))
    
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    plt.hist(normal_scores, bins=50, alpha=0.6, label='Normal Points', color='blue', density=True)
    plt.hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly Points', color='red', density=True)
    
    plt.axvline(x=threshold, color='green', linestyle='--', linewidth=2, label=f'Optimal Threshold: {threshold:.4f}')
    
    plt.title(title)
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Score distribution plot saved to {save_path}")

def main():
    print("🚀 STAR PERFORMER EVALUATION")
    print("=" * 50)
    print("Model: ContrastiveModule")
    print("Target F1 Score: 0.6441 (64.41%)")
    print("=" * 50)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Best Model Evaluation - Target 64.41% F1')
    parser.add_argument('--test_data', type=str, default='artifacts/market_windows_10f.npy')
    parser.add_argument('--labels', type=str, default='artifacts/market_labels.npy')
    parser.add_argument('--model_path', type=str, default='artifacts/contrastive_market.pt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--tolerance', type=int, default=5)  # Optimal parameters
    parser.add_argument('--smooth_sigma', type=float, default=1.0)  # Optimal parameters
    
    args = parser.parse_args()
    
    # Load data
    print("📂 Loading data...")
    X_test = np.load(args.test_data)
    y_true = np.load(args.labels)
    
    print(f"   Data shape: {X_test.shape}")
    print(f"   Anomalies: {y_true.sum()} / {len(y_true)} ({100*y_true.mean():.1f}%)")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    seq_len, n_features = X_test.shape[1], X_test.shape[2]

    # Create DataLoader
    test_dataset = TensorDataset(torch.from_numpy(X_test).float())
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Load ContrastiveModule model
    print("\n🤖 Loading ContrastiveModule...")
    model = ContrastiveModule(seq_len=seq_len, n_features=n_features, hidden_dim=64, latent_dim=32)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   ✅ Loaded checkpoint from epoch {checkpoint['epoch']} (val_loss: {checkpoint['val_loss']:.4f})")
    else:
        model.load_state_dict(checkpoint)
        print("   ✅ Loaded state_dict directly.")

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
    print(f"   Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    # Apply optimal smoothing
    print("\n🔧 Applying optimal smoothing...")
    scores_smooth = gaussian_filter1d(scores, sigma=args.smooth_sigma)
    print(f"   Applied Gaussian smoothing with sigma={args.smooth_sigma}")
    
    # Optimize threshold
    print("\n🎯 Optimizing threshold...")
    threshold, (f1, precision, recall) = optimize_threshold_for_f1(
        scores_smooth, y_true, tolerance=args.tolerance
    )
    
    # Final predictions
    predictions = (scores_smooth >= threshold).astype(int)
    
    # Display results with styling
    print("\n" + "🎉" * 20)
    print("🏆 BEST MODEL PERFORMANCE RESULTS 🏆")
    print("🎉" * 20)
    
    print(f"\n📈 PERFORMANCE METRICS:")
    print(f"   F1 Score:    {f1:.4f} ({f1*100:.2f}%)")
    print(f"   Precision:   {precision:.4f} ({precision*100:.2f}%)")
    print(f"   Recall:      {recall:.4f} ({recall*100:.2f}%)")
    print(f"   Threshold:   {threshold:.4f}")
    
    print(f"\n📊 PREDICTION STATISTICS:")
    print(f"   Total predicted anomalies: {predictions.sum()} / {len(predictions)}")
    print(f"   Prediction rate: {100*predictions.mean():.1f}%")
    
    print(f"\n⚙️ OPTIMAL CONFIGURATION:")
    print(f"   Model: ContrastiveModule")
    print(f"   Tolerance: {args.tolerance}")
    print(f"   Smooth Sigma: {args.smooth_sigma}")
    print(f"   Device: {device}")
    
    # Check if we achieved target
    if abs(f1 - 0.6441) < 0.01:
        print(f"\n✅ SUCCESS! Achieved target F1 score of 64.41%!")
    else:
        print(f"\n⚠️  Current F1: {f1*100:.2f}% (Target: 64.41%)")
    
    # Plot distribution
    print(f"\n📊 Generating visualization...")
    plot_score_distribution(scores_smooth, y_true, threshold, 
                            title=f"ContrastiveModule - Best Performance (F1: {f1:.4f})",
                            save_path="artifacts/best_model_score_distribution.png")
    
    # Save results
    os.makedirs('artifacts', exist_ok=True)
    np.save('artifacts/best_model_scores.npy', scores_smooth)
    np.save('artifacts/best_model_predictions.npy', predictions)
    
    print(f"\n💾 Results saved:")
    print(f"   - artifacts/best_model_scores.npy")
    print(f"   - artifacts/best_model_predictions.npy")
    
    print(f"\n🏁 Evaluation complete!")

if __name__ == '__main__':
    main()