"""
🌲 LIGHTGBM REGIME EXPERT
===========================
Light Gradient Boosting Machine expert for regime-conditioned detection.

Advantages over XGBoost for this use case:
  • Leaf-wise (best-first) tree growth → faster convergence
  • Native categorical feature support
  • Lower memory footprint for ensemble evaluation
  • Competitive with XGBoost on tabular financial data
"""
import numpy as np
from typing import Optional


class LightGBMExpert:
    """LightGBM-based expert model for a single volatility regime.

    Parameters
    ----------
    regime_id : int
        Which regime this expert specializes in.
    n_estimators : int
        Number of boosting iterations.
    max_depth : int
        Max tree depth. -1 for unlimited (controlled by num_leaves).
    num_leaves : int
        Max number of leaves per tree.
    is_unbalance : bool
        If True, LightGBM auto-handles class imbalance.
    learning_rate : float
        Shrinkage rate.
    subsample : float
        Bagging fraction.
    colsample_bytree : float
        Feature fraction per tree.
    reg_alpha : float
        L1 regularization.
    reg_lambda : float
        L2 regularization.
    random_state : int
        Random seed.
    """

    def __init__(self, regime_id: int = 0,
                 n_estimators: int = 200,
                 max_depth: int = 5,
                 num_leaves: int = 31,
                 is_unbalance: bool = True,
                 learning_rate: float = 0.05,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 random_state: int = 42):
        self.regime_id = regime_id
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.is_unbalance = is_unbalance
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
        """Train the LightGBM classifier."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "lightgbm is required. Install with: pip install lightgbm"
            )

        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            is_unbalance=self.is_unbalance,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            verbose=-1,
        )

        fit_kwargs = {}
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
        """Compute TreeSHAP values."""
        self._check_fitted()
        try:
            import shap
        except ImportError:
            raise ImportError("shap is required for explanations.")

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        # LightGBM SHAP returns [class0_shap, class1_shap] for binary
        if isinstance(shap_values, list):
            return shap_values[1]  # Anomaly class
        return shap_values

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

    def __repr__(self):
        regime_names = {0: 'Low', 1: 'Med', 2: 'High'}
        return (f"LightGBMExpert(regime={regime_names.get(self.regime_id, self.regime_id)}, "
                f"n_estimators={self.n_estimators}, num_leaves={self.num_leaves})")
