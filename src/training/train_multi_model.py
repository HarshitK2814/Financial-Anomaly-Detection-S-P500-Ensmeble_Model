"""
🚀 UNIFIED MULTI-MODEL TRAINING SCRIPT
=========================================
Master training script that trains all model variants (RF, XGBoost,
LightGBM, LSTM, GRU, RA-TAN, Hybrid) across all regimes.

Usage:
    python -m src.training.train_multi_model --models all --output_dir artifacts
"""
import numpy as np
import pandas as pd
import torch
import os
import json
import time
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from typing import Dict

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def load_data(artifacts_dir: str = 'artifacts') -> dict:
    """Load all data needed for training."""
    data = {}

    # Factor data
    factors_path = os.path.join(artifacts_dir, 'X_factors.npy')
    labels_path = os.path.join(artifacts_dir, 'y_factors.npy')
    windows_path = os.path.join(artifacts_dir, 'market_windows_10f.npy')
    csv_path = os.path.join(artifacts_dir, 'journal_factors.csv')

    if os.path.exists(factors_path):
        data['X_factors'] = np.load(factors_path)
        print(f"  Factors: {data['X_factors'].shape}")
    if os.path.exists(labels_path):
        data['y'] = np.load(labels_path)
        print(f"  Labels: {data['y'].shape}, anomalies: {data['y'].sum()}")
    if os.path.exists(windows_path):
        data['X_windows'] = np.load(windows_path)
        print(f"  Windows: {data['X_windows'].shape}")
    if os.path.exists(csv_path):
        data['factor_df'] = pd.read_csv(csv_path)
        data['feature_names'] = data['factor_df'].columns.tolist()

    return data


def compute_regimes(data: dict) -> np.ndarray:
    """Compute regime labels from Composite_Volatility."""
    if 'factor_df' in data and 'Composite_Volatility' in data['factor_df'].columns:
        vol = data['factor_df']['Composite_Volatility'].values
    elif 'X_windows' in data:
        vol = np.std(data['X_windows'][:, :, 0], axis=1)
    else:
        vol = np.std(data['X_factors'], axis=1)

    q33, q66 = np.quantile(vol, [0.33, 0.66])
    regimes = np.zeros(len(vol), dtype=int)
    regimes[vol <= q33] = 0
    regimes[(vol > q33) & (vol <= q66)] = 1
    regimes[vol > q66] = 2
    return regimes


def train_tree_experts(X: np.ndarray, y: np.ndarray,
                        regimes: np.ndarray,
                        feature_names: list,
                        output_dir: str) -> dict:
    """Train RF, XGBoost, and LightGBM per-regime experts."""
    import joblib
    results = {}
    regime_names = {0: 'Low', 1: 'Med', 2: 'High'}

    for model_type in ['RF', 'XGBoost', 'LightGBM']:
        print(f"\n  Training {model_type} experts...")
        model_results = {}

        for r in range(3):
            mask = regimes == r
            X_r, y_r = X[mask], y[mask]
            rname = regime_names[r]

            if len(np.unique(y_r)) < 2:
                print(f"    {rname}: Skipped (single class)")
                continue

            X_train, X_val, y_train, y_val = train_test_split(
                X_r, y_r, test_size=0.2, random_state=42, stratify=y_r
            )

            if model_type == 'RF':
                model = RandomForestClassifier(
                    n_estimators=200, max_depth=5,
                    class_weight='balanced', random_state=42
                )
                model.fit(X_train, y_train)

            elif model_type == 'XGBoost':
                try:
                    from src.models.xgboost_expert import XGBoostExpert
                    expert = XGBoostExpert(regime_id=r)
                    expert.fit(X_train, y_train, eval_set=[(X_val, y_val)])
                    model = expert.model
                except ImportError:
                    print(f"    {rname}: XGBoost not available, skipping")
                    continue

            elif model_type == 'LightGBM':
                try:
                    from src.models.lightgbm_expert import LightGBMExpert
                    expert = LightGBMExpert(regime_id=r)
                    expert.fit(X_train, y_train)
                    model = expert.model
                except ImportError:
                    print(f"    {rname}: LightGBM not available, skipping")
                    continue

            # Evaluate
            from sklearn.metrics import f1_score
            val_preds = model.predict(X_val)
            f1 = f1_score(y_val, val_preds, zero_division=0)

            model_results[rname] = {
                'f1': float(f1),
                'n_train': len(y_train),
                'n_val': len(y_val),
                'anomaly_rate': float(y_r.mean()),
            }

            # Save model
            model_path = os.path.join(output_dir, f'{model_type}_{rname}_expert.joblib')
            joblib.dump(model, model_path)
            print(f"    {rname}: F1={f1:.4f} (train={len(y_train)}, val={len(y_val)})")

        results[model_type] = model_results

    return results


def train_dl_experts(X_windows: np.ndarray, y: np.ndarray,
                      regimes: np.ndarray,
                      output_dir: str,
                      device: str = 'cpu') -> dict:
    """Train deep learning per-regime experts."""
    results = {}
    regime_names = {0: 'Low', 1: 'Med', 2: 'High'}

    # LSTM-Attention
    print("\n  Training LSTM-Attention experts...")
    try:
        from src.models.lstm_attention_expert import LSTMAttentionExpert, LSTMAttentionTrainer

        for r in range(3):
            mask = regimes == r
            X_r, y_r = X_windows[mask], y[mask]
            rname = regime_names[r]

            if len(np.unique(y_r)) < 2:
                continue

            n_val = max(int(0.2 * len(y_r)), 10)
            X_train, X_val = X_r[:-n_val], X_r[-n_val:]
            y_train, y_val = y_r[:-n_val], y_r[-n_val:]

            model = LSTMAttentionExpert(
                n_features=X_windows.shape[2],
                hidden_dim=64,
                n_layers=2,
                n_heads=4,
            )
            trainer = LSTMAttentionTrainer(model, lr=1e-3, device=device)
            history = trainer.fit(
                X_train, y_train, X_val, y_val,
                epochs=30, batch_size=32, patience=10, verbose=False
            )

            # Save
            torch.save(model.state_dict(),
                       os.path.join(output_dir, f'LSTM_{rname}_expert.pt'))
            print(f"    LSTM {rname}: train_loss={history['train_loss'][-1]:.4f}")

        results['LSTM'] = {'status': 'trained'}
    except Exception as e:
        print(f"    LSTM training failed: {e}")
        results['LSTM'] = {'status': f'failed: {e}'}

    # RA-TAN
    print("\n  Training RA-TAN (Regime-Aware Temporal Attention Network)...")
    try:
        from src.models.temporal_attention_network import RegimeAwareTemporalAttention, RATANTrainer

        n_val = max(int(0.2 * len(y)), 50)
        X_train, X_val = X_windows[:-n_val], X_windows[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]
        r_train, r_val = regimes[:-n_val], regimes[-n_val:]

        model = RegimeAwareTemporalAttention(
            n_features=X_windows.shape[2],
            d_model=64,
            n_heads=4,
            n_layers=3,
        )
        trainer = RATANTrainer(model, lr=5e-4, device=device)
        history = trainer.fit(
            X_train, y_train, r_train,
            X_val, y_val, r_val,
            epochs=30, batch_size=32, patience=10, verbose=False
        )

        torch.save(model.state_dict(),
                   os.path.join(output_dir, 'RATAN_expert.pt'))
        print(f"    RA-TAN: train_loss={history['train_loss'][-1]:.4f}")
        results['RATAN'] = {'status': 'trained'}
    except Exception as e:
        print(f"    RA-TAN training failed: {e}")
        results['RATAN'] = {'status': f'failed: {e}'}

    return results


def main(args):
    print("🚀 UNIFIED MULTI-MODEL TRAINING")
    print("=" * 55)

    # Setup
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("\n📁 Loading data...")
    data = load_data(args.artifacts_dir)

    if 'X_factors' not in data or 'y' not in data:
        print("ERROR: Required data not found in artifacts directory.")
        return

    # Compute regimes
    regimes = compute_regimes(data)
    print(f"\n🌀 Regimes: {dict(zip(*np.unique(regimes, return_counts=True)))}")

    all_results = {}

    # Align lengths
    min_len = min(len(data['X_factors']), len(data['y']))
    X_factors = data['X_factors'][:min_len]
    y = data['y'][:min_len]
    regimes = regimes[:min_len]
    feature_names = data.get('feature_names', [f'F{i}' for i in range(X_factors.shape[1])])

    # Train tree-based models
    if args.models in ['all', 'tree']:
        print("\n" + "=" * 55)
        print("🌲 TRAINING TREE-BASED EXPERTS")
        print("=" * 55)
        tree_results = train_tree_experts(
            X_factors, y, regimes, feature_names, output_dir
        )
        all_results.update(tree_results)

    # Train deep learning models
    if args.models in ['all', 'dl'] and 'X_windows' in data:
        print("\n" + "=" * 55)
        print("🧠 TRAINING DEEP LEARNING EXPERTS")
        print("=" * 55)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Device: {device}")

        X_windows = data['X_windows'][:min_len]
        dl_results = train_dl_experts(X_windows, y, regimes, output_dir, device)
        all_results.update(dl_results)

    # Save results
    results_path = os.path.join(output_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n💾 Results saved to {results_path}")

    print("\n✅ Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train all model variants')
    parser.add_argument('--models', choices=['all', 'tree', 'dl'], default='all',
                        help='Which models to train')
    parser.add_argument('--artifacts_dir', default='artifacts',
                        help='Path to artifacts directory')
    parser.add_argument('--output_dir', default='artifacts/models',
                        help='Output directory for trained models')
    args = parser.parse_args()
    main(args)
