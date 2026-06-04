"""
🔬 ABLATION STUDY RUNNER
==========================
Automated ablation studies to quantify the contribution of each
component in the RC-EME pipeline.

8 ablation studies:
  A1: No regime conditioning (remove gating)
  A2: No tail factors (remove Kurtosis, Skew, MaxDrawdown)
  A3: No temporal lags (remove lag-1, lag-5)
  A4: Static vs rolling quantiles
  A5: XAI method comparison (ECI analysis)
  A6: Expert model ablation (remove one model family at a time)
  A7: Factor category ablation (remove Trend/Vol/Tail/Momentum/Liquidity)
  A8: Sequence length sensitivity (64, 128, 256, 512)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings("ignore")


class AblationRunner:
    """Run systematic ablation studies.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Full factor feature matrix.
    y : np.ndarray, shape (N,)
        Binary labels.
    regimes : np.ndarray, shape (N,)
        Regime labels.
    feature_names : list of str
        Names of all D features.
    n_cv_folds : int
        Number of temporal CV folds.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 regimes: np.ndarray, feature_names: list,
                 n_cv_folds: int = 5,
                 vol_series: np.ndarray = None):
        self.X = X
        self.y = y
        self.regimes = regimes
        self.feature_names = feature_names
        self.n_cv_folds = n_cv_folds
        # Optional: raw volatility series used for per-fold regime refitting (R1 fix)
        self.vol_series = vol_series
        self.results = {}

    def _eval_rf_global(self, X, y) -> dict:
        """Evaluate RF without regime conditioning."""
        tscv = TimeSeriesSplit(n_splits=self.n_cv_folds)
        all_preds, all_true = [], []
        for train_idx, test_idx in tscv.split(X):
            if len(np.unique(y[train_idx])) < 2:
                continue
            rf = RandomForestClassifier(n_estimators=200, max_depth=5,
                                         class_weight='balanced', random_state=42)
            rf.fit(X[train_idx], y[train_idx])
            all_preds.extend(rf.predict(X[test_idx]))
            all_true.extend(y[test_idx])
        if not all_preds:
            return {'f1': 0, 'precision': 0, 'recall': 0}
        return {
            'f1': float(f1_score(all_true, all_preds, zero_division=0)),
            'precision': float(precision_score(all_true, all_preds, zero_division=0)),
            'recall': float(recall_score(all_true, all_preds, zero_division=0)),
        }

    def _eval_rf_moe(self, X, y, regimes) -> dict:
        """Evaluate RF with MoE (per-regime).

        ⚠️  R1 FIX: when self.vol_series is available, refits the
        RollingQuantileRegimeDetector on each training fold so test-fold
        regimes are assigned without look-ahead.
        """
        from src.models.regime_detector import RollingQuantileRegimeDetector

        tscv = TimeSeriesSplit(n_splits=self.n_cv_folds)
        all_preds, all_true = [], []
        for train_idx, test_idx in tscv.split(X):
            fold_preds = np.zeros(len(test_idx))

            # Per-fold regime assignment (R1 fix)
            if self.vol_series is not None:
                fold_det = RollingQuantileRegimeDetector(window=252)
                fold_train_regimes, _ = fold_det.detect(self.vol_series[train_idx])
                fold_test_regimes,  _ = fold_det.predict_from_last_state(
                    self.vol_series[test_idx]
                )
            else:
                fold_train_regimes = regimes[train_idx]
                fold_test_regimes  = regimes[test_idx]

            for r in np.unique(fold_train_regimes):
                tr_mask = fold_train_regimes == r
                te_mask = fold_test_regimes  == r
                if tr_mask.sum() < 5 or te_mask.sum() == 0:
                    continue
                y_tr = y[train_idx[tr_mask]]
                if len(np.unique(y_tr)) < 2:
                    continue
                rf = RandomForestClassifier(n_estimators=200, max_depth=5,
                                             class_weight='balanced', random_state=42)
                rf.fit(X[train_idx[tr_mask]], y_tr)
                fold_preds[np.where(te_mask)[0]] = rf.predict(X[test_idx[te_mask]])
            all_preds.extend(fold_preds)
            all_true.extend(y[test_idx])
        if not all_preds:
            return {'f1': 0, 'precision': 0, 'recall': 0}
        return {
            'f1': float(f1_score(all_true, all_preds, zero_division=0)),
            'precision': float(precision_score(all_true, all_preds, zero_division=0)),
            'recall': float(recall_score(all_true, all_preds, zero_division=0)),
        }

    def run_a1_no_regime(self) -> dict:
        """A1: Remove regime conditioning entirely."""
        print("🔬 A1: No regime conditioning...")
        baseline = self._eval_rf_moe(self.X, self.y, self.regimes)
        ablated = self._eval_rf_global(self.X, self.y)
        delta = ablated['f1'] - baseline['f1']
        result = {
            'ablation': 'A1: No Regime Conditioning',
            'baseline_f1': baseline['f1'],
            'ablated_f1': ablated['f1'],
            'delta_f1': delta,
            'impact': 'critical' if abs(delta) > 0.1 else 'moderate' if abs(delta) > 0.05 else 'minor',
        }
        self.results['A1'] = result
        print(f"  Baseline F1: {baseline['f1']:.4f} → Ablated F1: {ablated['f1']:.4f} (Δ={delta:+.4f})")
        return result

    def run_a2_no_tail_factors(self) -> dict:
        """A2: Remove tail risk factors."""
        print("🔬 A2: No tail factors...")
        tail_keywords = ['Kurtosis', 'Skew', 'Drawdown']
        keep_mask = np.array([
            not any(kw.lower() in fn.lower() for kw in tail_keywords)
            for fn in self.feature_names
        ])
        X_ablated = self.X[:, keep_mask]
        removed = [fn for fn, keep in zip(self.feature_names, keep_mask) if not keep]
        print(f"  Removed: {removed}")

        baseline = self._eval_rf_moe(self.X, self.y, self.regimes)
        ablated = self._eval_rf_moe(X_ablated, self.y, self.regimes)
        delta = ablated['f1'] - baseline['f1']
        result = {
            'ablation': 'A2: No Tail Factors',
            'removed_features': removed,
            'baseline_f1': baseline['f1'],
            'ablated_f1': ablated['f1'],
            'delta_f1': delta,
        }
        self.results['A2'] = result
        print(f"  Baseline F1: {baseline['f1']:.4f} → Ablated F1: {ablated['f1']:.4f} (Δ={delta:+.4f})")
        return result

    def run_a3_no_lags(self) -> dict:
        """A3: Remove temporal lag features."""
        print("🔬 A3: No temporal lags...")
        keep_mask = np.array([
            'lag' not in fn.lower()
            for fn in self.feature_names
        ])
        X_ablated = self.X[:, keep_mask]
        removed_count = (~keep_mask).sum()
        print(f"  Removed {removed_count} lag features")

        baseline = self._eval_rf_moe(self.X, self.y, self.regimes)
        ablated = self._eval_rf_moe(X_ablated, self.y, self.regimes)
        delta = ablated['f1'] - baseline['f1']
        result = {
            'ablation': 'A3: No Temporal Lags',
            'n_removed': int(removed_count),
            'baseline_f1': baseline['f1'],
            'ablated_f1': ablated['f1'],
            'delta_f1': delta,
        }
        self.results['A3'] = result
        print(f"  Baseline F1: {baseline['f1']:.4f} → Ablated F1: {ablated['f1']:.4f} (Δ={delta:+.4f})")
        return result

    def run_a7_factor_category_ablation(self) -> dict:
        """A7: Remove one factor category at a time."""
        print("🔬 A7: Factor category ablation...")

        # Define categories by keyword matching
        categories = {
            'Trend': ['Return_Trend'],
            'Momentum': ['Momentum'],
            'Volatility': ['Volatility', 'Garman', 'Parkinson', 'VIX'],
            'Tail Risk': ['Kurtosis', 'Skew', 'Drawdown'],
            'Liquidity': ['Volume', 'Amihud', 'Latent'],
            'Persistence': ['Hurst'],
        }

        baseline = self._eval_rf_moe(self.X, self.y, self.regimes)
        category_results = []

        for cat_name, keywords in categories.items():
            keep_mask = np.array([
                not any(kw.lower() in fn.lower() for kw in keywords)
                for fn in self.feature_names
            ])

            if keep_mask.all():
                continue  # No features matched

            X_ablated = self.X[:, keep_mask]
            ablated = self._eval_rf_moe(X_ablated, self.y, self.regimes)
            delta = ablated['f1'] - baseline['f1']

            category_results.append({
                'category': cat_name,
                'n_removed': int((~keep_mask).sum()),
                'ablated_f1': ablated['f1'],
                'delta_f1': delta,
            })
            print(f"  Remove {cat_name}: F1={ablated['f1']:.4f} (Δ={delta:+.4f})")

        result = {
            'ablation': 'A7: Factor Category Ablation',
            'baseline_f1': baseline['f1'],
            'categories': category_results,
        }
        self.results['A7'] = result
        return result

    def run_all(self) -> pd.DataFrame:
        """Run all ablation studies and return summary table."""
        print("\n" + "🔬" * 25)
        print("RUNNING ALL ABLATION STUDIES")
        print("🔬" * 25)

        self.run_a1_no_regime()
        self.run_a2_no_tail_factors()
        self.run_a3_no_lags()
        self.run_a7_factor_category_ablation()

        # Summary table
        rows = []
        for key in ['A1', 'A2', 'A3']:
            if key in self.results:
                r = self.results[key]
                rows.append({
                    'Ablation': r['ablation'],
                    'Baseline F1': r['baseline_f1'],
                    'Ablated F1': r['ablated_f1'],
                    'ΔF1': r['delta_f1'],
                })

        if 'A7' in self.results:
            for cat in self.results['A7']['categories']:
                rows.append({
                    'Ablation': f"A7: Remove {cat['category']}",
                    'Baseline F1': self.results['A7']['baseline_f1'],
                    'Ablated F1': cat['ablated_f1'],
                    'ΔF1': cat['delta_f1'],
                })

        df = pd.DataFrame(rows)
        print("\n" + df.to_string(index=False))
        return df


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.models.regime_detector import RollingQuantileRegimeDetector

    X = np.load('artifacts/X_factors.npy')
    y = np.load('artifacts/y_factors.npy')
    df = pd.read_csv('artifacts/journal_factors.csv')

    # ⚠️  LEAKAGE FIX: Replaced static global-quantile regime assignment.
    # Old code: q33, q66 = np.quantile(vol, [0.33, 0.66])  → uses full dataset.
    # Fix: use rolling 252-day quantile thresholds so each regime label is
    # derived only from past volatility observations.
    # NOTE: For fully leakage-free ablation the detector should be re-fit
    # inside each CV fold using only training-fold data.
    vol = df['Composite_Volatility'].values
    detector = RollingQuantileRegimeDetector(window=252)
    regimes, _ = detector.detect(vol)

    runner = AblationRunner(X, y, regimes, df.columns.tolist(), n_cv_folds=3)
    results = runner.run_all()

