"""
📊 TREE MODEL EXPLAINER
========================
Unified SHAP-based explainability for all tree ensemble experts
(Random Forest, XGBoost, LightGBM).

Features:
  • TreeSHAP for exact Shapley values (fast, tree-specific)
  • Per-regime SHAP aggregation
  • Factor importance ranking with economic interpretation
  • Decision path extraction for Random Forest
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union


class TreeExplainer:
    """Unified SHAP explainer for tree-based models.

    Parameters
    ----------
    model : fitted sklearn/xgboost/lightgbm model
    feature_names : list of str
        Names of the input features.
    regime_id : int, optional
        Which regime this model was trained on.
    """

    def __init__(self, model, feature_names: List[str],
                 regime_id: Optional[int] = None):
        self.model = model
        self.feature_names = feature_names
        self.regime_id = regime_id
        self._explainer = None

    def _get_explainer(self):
        if self._explainer is None:
            try:
                import shap
                self._explainer = shap.TreeExplainer(self.model)
            except ImportError:
                raise ImportError("shap is required. Install: pip install shap")
        return self._explainer

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """Compute TreeSHAP values.

        Returns
        -------
        np.ndarray, shape (N, D)
            SHAP values for each sample and feature.
        """
        explainer = self._get_explainer()
        values = explainer.shap_values(X)
        # Handle different formats (binary classification)
        if isinstance(values, list):
            return values[1]  # Anomaly class
        return values

    def global_importance(self, X: np.ndarray,
                          top_k: int = 10) -> pd.DataFrame:
        """Compute global SHAP-based feature importance.

        Returns DataFrame sorted by mean |SHAP|.
        """
        sv = self.shap_values(X)
        mean_abs_shap = np.abs(sv).mean(axis=0)

        df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_abs_shap,
            'std_shap': np.std(sv, axis=0),
        }).sort_values('mean_abs_shap', ascending=False)

        if top_k:
            df = df.head(top_k)
        return df.reset_index(drop=True)

    def local_explanation(self, X: np.ndarray,
                          sample_idx: int) -> dict:
        """Generate per-sample SHAP explanation.

        Returns
        -------
        dict with feature attributions and prediction decomposition.
        """
        sv = self.shap_values(X)
        sample_shap = sv[sample_idx]

        # Sort by absolute SHAP
        sorted_idx = np.argsort(np.abs(sample_shap))[::-1]

        attributions = []
        for i in sorted_idx[:5]:  # Top 5
            attributions.append({
                'feature': self.feature_names[i],
                'shap_value': float(sample_shap[i]),
                'direction': 'increases anomaly' if sample_shap[i] > 0 else 'decreases anomaly',
                'feature_value': float(X[sample_idx, i]),
            })

        return {
            'sample_index': sample_idx,
            'regime_id': self.regime_id,
            'top_attributions': attributions,
            'base_value': float(self._get_explainer().expected_value) if hasattr(self._get_explainer(), 'expected_value') else None,
        }

    def regime_explanation_profile(self, X: np.ndarray,
                                     regimes: np.ndarray) -> Dict[str, pd.DataFrame]:
        """Compute SHAP importance profiles per regime.

        Novel contribution: shows how factor importance shifts across
        regimes, providing regime-conditional explanation profiles.
        """
        sv = self.shap_values(X)
        regime_names = {0: 'Low', 1: 'Med', 2: 'High'}
        profiles = {}

        for r in np.unique(regimes):
            mask = regimes == r
            if mask.sum() == 0:
                continue
            regime_sv = sv[mask]
            mean_abs = np.abs(regime_sv).mean(axis=0)
            rname = regime_names.get(r, str(r))

            profiles[rname] = pd.DataFrame({
                'feature': self.feature_names,
                'mean_abs_shap': mean_abs,
                'rank': np.argsort(np.argsort(-mean_abs)) + 1,
            }).sort_values('mean_abs_shap', ascending=False)

        return profiles

    def feature_importance_native(self) -> pd.DataFrame:
        """Return model's native feature importance (Gini/Gain)."""
        if hasattr(self.model, 'feature_importances_'):
            imp = self.model.feature_importances_
        else:
            return pd.DataFrame({'feature': self.feature_names,
                                  'importance': 0.0})

        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': imp,
        }).sort_values('importance', ascending=False).reset_index(drop=True)

    def decision_path_summary(self, X: np.ndarray,
                               sample_idx: int) -> Optional[list]:
        """Extract decision path for a sample (sklearn RF only).

        Returns list of decision rules for the sample.
        """
        if not hasattr(self.model, 'estimators_'):
            return None

        # Use first 3 trees for interpretability
        rules = []
        for tree_idx, tree in enumerate(self.model.estimators_[:3]):
            node_indicator = tree.decision_path(X[sample_idx:sample_idx+1])
            node_indices = node_indicator.indices
            feature_ids = tree.tree_.feature
            thresholds = tree.tree_.threshold

            path_rules = []
            for node_id in node_indices:
                if feature_ids[node_id] >= 0:
                    fname = self.feature_names[feature_ids[node_id]]
                    thresh = thresholds[node_id]
                    val = X[sample_idx, feature_ids[node_id]]
                    direction = '≤' if val <= thresh else '>'
                    path_rules.append(f"{fname} {direction} {thresh:.4f}")

            rules.append({
                'tree_index': tree_idx,
                'path': path_rules,
            })

        return rules
