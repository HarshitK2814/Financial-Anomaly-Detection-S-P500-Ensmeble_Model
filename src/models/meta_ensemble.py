"""
🎯 REGIME META-ENSEMBLE
=========================
Stacks predictions from multiple expert models within each regime
into a final anomaly probability.

For each regime:
  1. Collect predictions from RF, XGBoost, LightGBM, DL models
  2. Stack them as features for a meta-learner
  3. Meta-learner produces final calibrated probability

This is a level-2 stacker in the Mixture of Experts framework.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict
from typing import Dict, List, Optional


class RegimeMetaEnsemble:
    """Per-regime meta-ensemble that stacks multiple expert predictions.

    Parameters
    ----------
    n_regimes : int
        Number of volatility regimes.
    calibrate : bool
        Whether to apply Platt scaling for probability calibration.
    meta_model_type : str
        Type of meta-learner: 'logistic', 'ridge_logistic'.
    """

    def __init__(self, n_regimes: int = 3,
                 calibrate: bool = True,
                 meta_model_type: str = 'logistic'):
        self.n_regimes = n_regimes
        self.calibrate = calibrate
        self.meta_model_type = meta_model_type
        self.meta_models = {}  # regime_id -> fitted meta-model
        self.expert_names = []
        self._fitted = False

    def fit(self, expert_predictions: Dict[str, np.ndarray],
            y: np.ndarray, regimes: np.ndarray):
        """Train per-regime meta-learners on stacked expert predictions.

        Parameters
        ----------
        expert_predictions : dict
            Mapping of expert_name -> np.ndarray of shape (N,) with anomaly
            probabilities from each expert model.
        y : np.ndarray, shape (N,)
            True binary labels.
        regimes : np.ndarray, shape (N,)
            Regime labels.
        """
        self.expert_names = sorted(expert_predictions.keys())
        N = len(y)

        # Stack predictions into feature matrix
        X_meta = np.column_stack([
            expert_predictions[name] for name in self.expert_names
        ])

        for r in range(self.n_regimes):
            mask = regimes == r
            X_r = X_meta[mask]
            y_r = y[mask]

            if len(np.unique(y_r)) < 2:
                # Not enough classes in this regime
                self.meta_models[r] = None
                continue

            # Create meta-learner
            if self.meta_model_type == 'logistic':
                base = LogisticRegression(
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=42,
                )
            elif self.meta_model_type == 'ridge_logistic':
                base = LogisticRegression(
                    penalty='l2',
                    C=0.1,
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=42,
                )
            else:
                raise ValueError(f"Unknown meta_model_type: {self.meta_model_type}")

            # Calibration requires enough samples of each class for CV
            min_class_count = min(np.bincount(y_r.astype(int)))
            if self.calibrate and len(y_r) >= 10 and min_class_count >= 4:
                model = CalibratedClassifierCV(base, cv=min(3, min_class_count))
            else:
                model = base

            try:
                model.fit(X_r, y_r)
            except ValueError:
                # Fallback to uncalibrated model
                base.fit(X_r, y_r)
                model = base
            self.meta_models[r] = model

        self._fitted = True

    def predict_proba(self, expert_predictions: Dict[str, np.ndarray],
                      regimes: np.ndarray) -> np.ndarray:
        """Predict anomaly probabilities using per-regime meta-learners.

        Parameters
        ----------
        expert_predictions : dict
            Same structure as in fit().
        regimes : np.ndarray, shape (N,)

        Returns
        -------
        np.ndarray, shape (N,)
            Final fused anomaly probabilities.
        """
        self._check_fitted()

        X_meta = np.column_stack([
            expert_predictions[name] for name in self.expert_names
        ])

        N = len(regimes)
        probs = np.zeros(N)

        for r in range(self.n_regimes):
            mask = regimes == r
            if not mask.any():
                continue

            if self.meta_models.get(r) is None:
                # Fallback: simple average
                probs[mask] = X_meta[mask].mean(axis=1)
            else:
                probs[mask] = self.meta_models[r].predict_proba(X_meta[mask])[:, 1]

        return probs

    def predict(self, expert_predictions: Dict[str, np.ndarray],
                regimes: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels."""
        probs = self.predict_proba(expert_predictions, regimes)
        return (probs >= threshold).astype(int)

    def get_expert_contributions(self, expert_predictions: Dict[str, np.ndarray],
                                  regimes: np.ndarray) -> Dict[str, float]:
        """Compute the relative contribution of each expert model.

        For logistic regression meta-learners, contribution is proportional
        to the absolute coefficient magnitude.

        Returns
        -------
        dict mapping regime_name -> dict of expert_name -> contribution_weight
        """
        self._check_fitted()
        contributions = {}
        regime_names = {0: 'Low', 1: 'Med', 2: 'High'}

        for r in range(self.n_regimes):
            rname = regime_names.get(r, str(r))
            model = self.meta_models.get(r)

            if model is None:
                contributions[rname] = {
                    name: 1.0 / len(self.expert_names) for name in self.expert_names
                }
                continue

            # Extract coefficients
            if hasattr(model, 'calibrated_classifiers_'):
                # CalibratedClassifierCV wraps the base model
                coefs = np.mean([
                    np.abs(cc.estimator.coef_[0])
                    for cc in model.calibrated_classifiers_
                ], axis=0)
            elif hasattr(model, 'coef_'):
                coefs = np.abs(model.coef_[0])
            else:
                coefs = np.ones(len(self.expert_names))

            # Normalize to sum to 1
            total = coefs.sum() + 1e-8
            contributions[rname] = {
                name: float(coefs[i] / total)
                for i, name in enumerate(self.expert_names)
            }

        return contributions

    def explain_prediction(self, expert_predictions: Dict[str, np.ndarray],
                            regimes: np.ndarray,
                            sample_idx: int) -> dict:
        """Generate explanation for a single prediction.

        Returns
        -------
        dict with:
            - 'regime': which regime the sample belongs to
            - 'expert_scores': individual expert predictions
            - 'expert_contributions': how much each expert influenced the result
            - 'final_probability': the fused probability
        """
        self._check_fitted()

        regime = regimes[sample_idx]
        regime_names = {0: 'Low Vol', 1: 'Med Vol', 2: 'High Vol'}

        expert_scores = {
            name: float(expert_predictions[name][sample_idx])
            for name in self.expert_names
        }

        contributions = self.get_expert_contributions(expert_predictions, regimes)
        regime_name = regime_names.get(regime, str(regime))

        # Compute final probability for this sample
        X_meta = np.array([[
            expert_predictions[name][sample_idx]
            for name in self.expert_names
        ]])

        if self.meta_models.get(regime) is not None:
            final_prob = float(self.meta_models[regime].predict_proba(X_meta)[0, 1])
        else:
            final_prob = float(X_meta.mean())

        return {
            'regime': regime_name,
            'regime_id': int(regime),
            'expert_scores': expert_scores,
            'expert_contributions': contributions.get(regime_name, {}),
            'final_probability': final_prob,
            'is_anomaly': final_prob >= 0.5,
        }

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("MetaEnsemble not fitted. Call .fit() first.")

    def __repr__(self):
        return (f"RegimeMetaEnsemble(n_regimes={self.n_regimes}, "
                f"experts={self.expert_names}, "
                f"calibrate={self.calibrate})")


if __name__ == '__main__':
    # Demo with synthetic expert predictions
    N = 500
    regimes = np.random.choice([0, 1, 2], size=N)
    y = np.random.choice([0, 1], size=N, p=[0.98, 0.02])

    expert_preds = {
        'RF': np.random.rand(N),
        'XGBoost': np.random.rand(N),
        'LSTM': np.random.rand(N),
    }

    meta = RegimeMetaEnsemble(n_regimes=3)
    meta.fit(expert_preds, y, regimes)
    probs = meta.predict_proba(expert_preds, regimes)
    print(f"Fused probabilities: mean={probs.mean():.4f}, std={probs.std():.4f}")

    contributions = meta.get_expert_contributions(expert_preds, regimes)
    for regime, contribs in contributions.items():
        print(f"Regime {regime}: {contribs}")

    explanation = meta.explain_prediction(expert_preds, regimes, sample_idx=0)
    print(f"\nSample 0 explanation: {explanation}")
