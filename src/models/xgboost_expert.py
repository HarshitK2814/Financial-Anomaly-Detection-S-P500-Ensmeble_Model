"""
🌲 XGBOOST REGIME EXPERT
==========================
Gradient-boosted tree expert for regime-conditioned anomaly detection.

Advantages over Random Forest for financial anomaly detection:
  • Native handling of class imbalance (scale_pos_weight)
  • Regularization (L1/L2) prevents overfitting on small per-regime samples
  • Monotonic constraints can encode economic priors
  • Built-in TreeSHAP support for exact Shapley values
"""
import numpy as np
from typing import Optional


class XGBoostExpert:
    """XGBoost-based expert model for a single volatility regime.

    Parameters
    ----------
    regime_id : int
        Which regime this expert specializes in (0=Low, 1=Med, 2=High).
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Maximum tree depth. Kept shallow for interpretability.
    scale_pos_weight : float or None
        If None, auto-computed as (neg / pos) ratio.
    learning_rate : float
        Step shrinkage. Lower = more regularized.
    subsample : float
        Row subsampling ratio per tree.
    colsample_bytree : float
        Feature subsampling ratio per tree.
    reg_alpha : float
        L1 regularization on leaf weights.
    reg_lambda : float
        L2 regularization on leaf weights.
    random_state : int
        Random seed.
    """

    def __init__(self, regime_id: int = 0,
                 n_estimators: int = 200,
                 max_depth: int = 5,
                 scale_pos_weight: Optional[float] = None,
                 learning_rate: float = 0.05,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 random_state: int = 42):
        self.regime_id = regime_id
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self._scale_pos_weight = scale_pos_weight
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.model = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray,
            eval_set: list = None, verbose: bool = False):
        """Train the XGBoost classifier.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Factor feature matrix for this regime.
        y : np.ndarray, shape (N,)
            Binary anomaly labels.
        eval_set : list of (X, y) tuples, optional
            Validation sets for early stopping.
        verbose : bool
            Whether to print training progress.
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError(
                "xgboost is required. Install with: pip install xgboost"
            )

        # Auto-compute imbalance weight if not specified
        spw = self._scale_pos_weight
        if spw is None:
            n_pos = np.sum(y == 1)
            n_neg = np.sum(y == 0)
            spw = n_neg / max(n_pos, 1)

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            scale_pos_weight=spw,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            eval_metric='aucpr',
            use_label_encoder=False,
            random_state=self.random_state,
            verbosity=0,
        )

        fit_kwargs = {'verbose': verbose}
        if eval_set:
            fit_kwargs['eval_set'] = eval_set

        self.model.fit(X, y, **fit_kwargs)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary anomaly labels."""
        self._check_fitted()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly probabilities."""
        self._check_fitted()
        return self.model.predict_proba(X)[:, 1]

    def feature_importances(self) -> np.ndarray:
        """Return feature importances (gain-based)."""
        self._check_fitted()
        return self.model.feature_importances_

    def explain(self, X: np.ndarray) -> np.ndarray:
        """Compute TreeSHAP values for the input samples.

        Returns
        -------
        np.ndarray, shape (N, D)
            SHAP values for each sample and feature.
        """
        self._check_fitted()
        try:
            import shap
        except ImportError:
            raise ImportError(
                "shap is required for explanations. Install with: pip install shap"
            )

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        return shap_values

    def get_decision_paths(self, X: np.ndarray) -> list:
        """Extract human-readable decision paths (for XAI).

        Returns list of decision path strings for each sample.
        """
        self._check_fitted()
        # XGBoost doesn't expose paths as easily as sklearn trees
        # Use feature importances as a proxy
        importances = self.feature_importances()
        top_features = np.argsort(importances)[::-1][:3]
        return top_features

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

    def __repr__(self):
        regime_names = {0: 'Low', 1: 'Med', 2: 'High'}
        return (f"XGBoostExpert(regime={regime_names.get(self.regime_id, self.regime_id)}, "
                f"n_estimators={self.n_estimators}, max_depth={self.max_depth})")
