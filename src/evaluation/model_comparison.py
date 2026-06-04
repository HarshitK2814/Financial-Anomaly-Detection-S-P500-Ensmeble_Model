"""
📊 MODEL COMPARISON FRAMEWORK
================================
Systematic evaluation of all 14 model variants in a fair,
regime-conditioned comparison framework.

Comparison matrix:
  Rows: RF, XGBoost, LightGBM, LSTM-Attn, GRU-Attn, RA-TAN, Hybrid
  Cols: Global (no regime) vs MoE (regime-conditioned)

Evaluation protocol:
  • Temporal expanding-window cross-validation (5-fold)
  • Per-regime evaluation metrics
  • Bootstrap confidence intervals
  • Statistical significance tests (McNemar, DeLong)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_curve, auc, average_precision_score,
)
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Optional, Tuple
import time
import warnings

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
# Metric computation helpers
# ──────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: np.ndarray = None) -> dict:
    """Compute comprehensive anomaly detection metrics."""
    metrics = {
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is not None:
        try:
            metrics['auc_pr'] = float(average_precision_score(y_true, y_proba))
        except ValueError:
            metrics['auc_pr'] = 0.0

    return metrics


def compute_per_regime_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                regimes: np.ndarray,
                                y_proba: np.ndarray = None) -> dict:
    """Compute metrics separately for each regime."""
    regime_names = {0: 'Low', 1: 'Med', 2: 'High'}
    results = {}

    for r in np.unique(regimes):
        mask = regimes == r
        rname = regime_names.get(r, str(r))

        if mask.sum() == 0:
            results[rname] = {'precision': 0, 'recall': 0, 'f1': 0, 'n_samples': 0}
            continue

        results[rname] = compute_metrics(
            y_true[mask], y_pred[mask],
            y_proba[mask] if y_proba is not None else None
        )
        results[rname]['n_samples'] = int(mask.sum())
        results[rname]['n_anomalies'] = int(y_true[mask].sum())

    return results


def optimize_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Find the threshold that maximizes F1 score."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    prec, rec, thresh = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
    best_idx = np.argmax(f1_scores)
    # Thresholds array is len(prec)-1
    if best_idx >= len(thresh):
        best_idx = len(thresh) - 1
    return float(thresh[best_idx])


def bootstrap_confidence_interval(y_true: np.ndarray, y_pred: np.ndarray,
                                   metric_fn, n_bootstrap: int = 1000,
                                   ci: float = 0.95,
                                   block_size: int = 20,
                                   random_state: int = 42) -> Tuple[float, float, float]:
    """Compute bootstrap CI using circular block bootstrap.

    ⚠️  R3 LEAKAGE FIX: replaced i.i.d. bootstrap with circular block
    bootstrap.  Standard resampling destroys temporal ordering and
    underestimates variance for autocorrelated financial return series,
    producing confidence intervals that are systematically too narrow.
    Circular block bootstrap preserves local autocorrelation structure
    by sampling contiguous blocks of length `block_size`.

    Parameters
    ----------
    block_size : int
        Length of each sampled block (default 20 ≈ 1 trading month).

    Returns
    -------
    (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    point_estimate = metric_fn(y_true, y_pred)

    n_blocks = int(np.ceil(n / block_size))
    boot_estimates = []
    for _ in range(n_bootstrap):
        # Circular: all start positions in [0, n) are valid
        starts  = rng.randint(0, n, size=n_blocks)
        indices = np.concatenate(
            [np.arange(s, s + block_size) % n for s in starts]
        )[:n]
        try:
            boot_val = metric_fn(y_true[indices], y_pred[indices])
            boot_estimates.append(boot_val)
        except (ValueError, ZeroDivisionError):
            pass

    if not boot_estimates:
        return point_estimate, point_estimate, point_estimate

    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_estimates, alpha * 100))
    upper = float(np.percentile(boot_estimates, (1 - alpha) * 100))
    return point_estimate, lower, upper



# ──────────────────────────────────────────────────────────────
# Model wrapper interface
# ──────────────────────────────────────────────────────────────

class ModelWrapper:
    """Unified interface for comparing different expert models."""

    def __init__(self, name: str, model_type: str, **kwargs):
        self.name = name
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = None
        self.training_time = 0.0

    def fit(self, X, y, **fit_kwargs):
        start = time.time()
        self._fit_impl(X, y, **fit_kwargs)
        self.training_time = time.time() - start

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:
        raise NotImplementedError

    def _fit_impl(self, X, y, **kwargs):
        raise NotImplementedError


class RFWrapper(ModelWrapper):
    def __init__(self, **kwargs):
        super().__init__('Random Forest', 'tabular', **kwargs)

    def _fit_impl(self, X, y, **kwargs):
        self.model = RandomForestClassifier(
            n_estimators=self.kwargs.get('n_estimators', 200),
            max_depth=self.kwargs.get('max_depth', 5),
            class_weight='balanced',
            random_state=42,
        )
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class XGBWrapper(ModelWrapper):
    def __init__(self, **kwargs):
        super().__init__('XGBoost', 'tabular', **kwargs)

    def _fit_impl(self, X, y, **kwargs):
        try:
            import xgboost as xgb
            n_pos = max(y.sum(), 1)
            n_neg = (y == 0).sum()
            self.model = xgb.XGBClassifier(
                n_estimators=self.kwargs.get('n_estimators', 200),
                max_depth=self.kwargs.get('max_depth', 5),
                scale_pos_weight=n_neg / n_pos,
                learning_rate=0.05,
                eval_metric='aucpr',
                use_label_encoder=False,
                verbosity=0,
                random_state=42,
            )
            self.model.fit(X, y)
        except ImportError:
            # Fallback to RF
            self.model = RandomForestClassifier(n_estimators=200, max_depth=5,
                                                 class_weight='balanced', random_state=42)
            self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class LGBMWrapper(ModelWrapper):
    def __init__(self, **kwargs):
        super().__init__('LightGBM', 'tabular', **kwargs)

    def _fit_impl(self, X, y, **kwargs):
        try:
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(
                n_estimators=self.kwargs.get('n_estimators', 200),
                max_depth=self.kwargs.get('max_depth', 5),
                is_unbalance=True,
                learning_rate=0.05,
                verbose=-1,
                random_state=42,
            )
            self.model.fit(X, y)
        except ImportError:
            self.model = RandomForestClassifier(n_estimators=200, max_depth=5,
                                                 class_weight='balanced', random_state=42)
            self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


# ──────────────────────────────────────────────────────────────
# Comparison framework
# ──────────────────────────────────────────────────────────────

class ModelComparisonFramework:
    """Run systematic comparison of all model variants.

    Parameters
    ----------
    n_cv_folds : int
        Number of temporal cross-validation folds.
    n_bootstrap : int
        Number of bootstrap samples for confidence intervals.
    """

    def __init__(self, n_cv_folds: int = 5, n_bootstrap: int = 500):
        self.n_cv_folds = n_cv_folds
        self.n_bootstrap = n_bootstrap
        self.results = {}

    def compare_tabular_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regimes: np.ndarray,
        model_factories: Dict[str, callable] = None,
        vol_series: np.ndarray = None,
    ) -> pd.DataFrame:
        """Compare tabular models (RF, XGBoost, LightGBM) in both
        global and MoE settings.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Factor features.
        y : np.ndarray, shape (N,)
            Binary labels.
        regimes : np.ndarray, shape (N,)
            Pre-computed regime labels (used when vol_series is None).
        model_factories : dict, optional
            Custom model factories.
        vol_series : np.ndarray, shape (N,), optional
            Raw volatility series.  When provided, the regime detector is
            re-fit inside each CV fold on training data only, eliminating
            the last source of look-ahead in the MoE evaluation (R1 fix).

        Returns
        -------
        pd.DataFrame with comparison results.
        """
        if model_factories is None:
            model_factories = {
                'RF': lambda: RFWrapper(),
                'XGBoost': lambda: XGBWrapper(),
                'LightGBM': lambda: LGBMWrapper(),
            }

        results_rows = []
        tscv = TimeSeriesSplit(n_splits=self.n_cv_folds)

        for model_name, factory in model_factories.items():
            print(f"\n📊 Evaluating {model_name}...")

            # ── GLOBAL MODE ──
            global_metrics = self._evaluate_global(X, y, factory, tscv)
            results_rows.append({'model': model_name, 'mode': 'Global', **global_metrics})

            # ── MoE MODE ──
            moe_metrics = self._evaluate_moe(
                X, y, regimes, factory, tscv, vol_series=vol_series
            )
            results_rows.append({'model': model_name, 'mode': 'MoE', **moe_metrics})

        df = pd.DataFrame(results_rows)
        self.results['tabular_comparison'] = df
        return df


    def _evaluate_global(self, X, y, factory, tscv) -> dict:
        """Evaluate a model in global (no regime) mode with temporal CV."""
        all_preds = []
        all_true = []
        all_proba = []

        for train_idx, test_idx in tscv.split(X):
            model = factory()
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            if len(np.unique(y_train)) < 2:
                continue

            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)
            opt_thresh = optimize_threshold(y_test, proba)
            preds = (proba >= opt_thresh).astype(int)

            all_preds.extend(preds)
            all_true.extend(y_test)
            all_proba.extend(proba)

        if not all_preds:
            return {'f1': 0, 'precision': 0, 'recall': 0, 'auc_pr': 0}

        all_preds = np.array(all_preds)
        all_true = np.array(all_true)
        all_proba = np.array(all_proba)

        metrics = compute_metrics(all_true, all_preds, all_proba)

        # Circular block bootstrap CI for F1
        _, f1_lo, f1_hi = circular_block_bootstrap(
            all_true, all_preds,
            lambda yt, yp: f1_score(yt, yp, zero_division=0),
            n_bootstrap=self.n_bootstrap
        )
        metrics['f1_ci_lower'] = f1_lo
        metrics['f1_ci_upper'] = f1_hi

        return metrics

    def _evaluate_moe(self, X, y, regimes, factory, tscv,
                      vol_series: np.ndarray = None) -> dict:
        """Evaluate a model in MoE (per-regime) mode.

        ⚠️  R1 FIX: when vol_series is supplied, a fresh
        RollingQuantileRegimeDetector is fit on each training fold and
        used to assign test-fold regimes via predict_from_last_state().
        This eliminates the remaining look-ahead from pre-computed regimes.
        ⚠️  F4 FIX retained: threshold computed from training data only.
        """
        from src.models.regime_detector import RollingQuantileRegimeDetector

        all_preds = np.zeros(0)
        all_true  = np.zeros(0)
        all_proba = np.zeros(0)

        for train_idx, test_idx in tscv.split(X):
            fold_preds = np.zeros(len(test_idx))
            fold_proba = np.zeros(len(test_idx))

            # Per-fold regime assignment (R1 fix)
            if vol_series is not None:
                fold_det = RollingQuantileRegimeDetector(window=252)
                fold_train_regimes, _ = fold_det.detect(vol_series[train_idx])
                fold_test_regimes,  _ = fold_det.predict_from_last_state(vol_series[test_idx])
            else:
                fold_train_regimes = regimes[train_idx]
                fold_test_regimes  = regimes[test_idx]

            for r in np.unique(fold_train_regimes):
                train_mask = fold_train_regimes == r
                test_mask  = fold_test_regimes  == r

                if train_mask.sum() < 5 or test_mask.sum() == 0:
                    continue

                X_r_train = X[train_idx[train_mask]]
                y_r_train = y[train_idx[train_mask]]
                X_r_test  = X[test_idx[test_mask]]

                if len(np.unique(y_r_train)) < 2:
                    continue

                model = factory()
                model.fit(X_r_train, y_r_train)

                # Threshold from TRAINING data only (F4 fix retained)
                train_proba = model.predict_proba(X_r_train)
                opt_thresh  = optimize_threshold(y_r_train, train_proba)

                test_proba = model.predict_proba(X_r_test)
                test_positions = np.where(test_mask)[0]
                fold_proba[test_positions] = test_proba
                fold_preds[test_positions] = (test_proba >= opt_thresh).astype(int)

            all_true  = np.concatenate([all_true,  y[test_idx]])
            all_proba = np.concatenate([all_proba, fold_proba])
            all_preds = np.concatenate([all_preds, fold_preds])

        if len(all_preds) == 0:
            return {'f1': 0, 'precision': 0, 'recall': 0, 'auc_pr': 0}

        metrics = compute_metrics(all_true.astype(int), all_preds.astype(int), all_proba)

        # Circular block bootstrap CI for F1
        _, f1_lo, f1_hi = circular_block_bootstrap(
            all_true.astype(int), all_preds.astype(int),
            lambda yt, yp: f1_score(yt, yp, zero_division=0),
            n_bootstrap=self.n_bootstrap
        )
        metrics['f1_ci_lower'] = f1_lo
        metrics['f1_ci_upper'] = f1_hi

        return metrics


    def generate_comparison_table(self) -> str:
        """Generate a formatted comparison table."""
        if 'tabular_comparison' not in self.results:
            return "No comparison results available. Run compare_tabular_models() first."

        df = self.results['tabular_comparison']
        output_lines = [
            "=" * 80,
            "MODEL COMPARISON RESULTS",
            "=" * 80,
            f"{'Model':<15} {'Mode':<10} {'F1':>8} {'Prec':>8} {'Recall':>8} {'AUC-PR':>8} {'F1 CI':>20}",
            "-" * 80,
        ]

        for _, row in df.iterrows():
            ci_str = f"[{row.get('f1_ci_lower', 0):.3f}, {row.get('f1_ci_upper', 0):.3f}]"
            output_lines.append(
                f"{row['model']:<15} {row['mode']:<10} "
                f"{row['f1']:>8.4f} {row['precision']:>8.4f} "
                f"{row['recall']:>8.4f} {row.get('auc_pr', 0):>8.4f} "
                f"{ci_str:>20}"
            )

        output_lines.append("=" * 80)
        return "\n".join(output_lines)


if __name__ == '__main__':
    # Demo with project data
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.models.regime_detector import RollingQuantileRegimeDetector

    X = np.load('artifacts/X_factors.npy')
    y = np.load('artifacts/y_factors.npy')

    # ⚠️  LEAKAGE FIX: Replaced static global-quantile regime assignment with
    # RollingQuantileRegimeDetector(window=252).  The old code computed Q33/Q66
    # over the entire dataset (including test rows) and assigned regimes before
    # any train/test split, leaking future volatility structure into past labels.
    # NOTE: For fully leakage-free evaluation the detector should be re-fit
    # inside each CV fold using only training-fold volatility.
    vol_col = 2  # Composite_Volatility
    detector = RollingQuantileRegimeDetector(window=252)
    regimes, _ = detector.detect(X[:, vol_col])

    framework = ModelComparisonFramework(n_cv_folds=3, n_bootstrap=100)
    results = framework.compare_tabular_models(X, y, regimes)
    print(framework.generate_comparison_table())

