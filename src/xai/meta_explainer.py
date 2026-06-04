"""
🎯 META EXPLAINER
==================
Explains the meta-ensemble's final decisions by decomposing
contributions from individual expert models.

Also provides counterfactual explanations:
"What minimal change would flip this prediction?"
"""
import numpy as np
from typing import Dict, Optional


class MetaExplainer:
    """Explainability for the meta-ensemble layer.

    Parameters
    ----------
    meta_ensemble : RegimeMetaEnsemble
        Fitted meta-ensemble instance.
    expert_explainers : dict
        Mapping of expert_name -> explainer instance (Tree/Deep).
    """

    def __init__(self, meta_ensemble=None, expert_explainers: dict = None):
        self.meta_ensemble = meta_ensemble
        self.expert_explainers = expert_explainers or {}

    def expert_contribution_breakdown(
        self,
        expert_predictions: Dict[str, np.ndarray],
        regimes: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Show how much each expert model contributes per regime.

        Delegates to RegimeMetaEnsemble.get_expert_contributions().
        """
        if self.meta_ensemble is None:
            return {}
        return self.meta_ensemble.get_expert_contributions(
            expert_predictions, regimes
        )

    def prediction_decomposition(
        self,
        expert_predictions: Dict[str, np.ndarray],
        regimes: np.ndarray,
        sample_idx: int,
    ) -> dict:
        """Decompose a single prediction into expert contributions.

        Shows exactly how each model influenced the final output.
        """
        if self.meta_ensemble is None:
            return {}
        return self.meta_ensemble.explain_prediction(
            expert_predictions, regimes, sample_idx
        )

    def counterfactual_explanation(
        self,
        expert_predictions: Dict[str, np.ndarray],
        regimes: np.ndarray,
        sample_idx: int,
        target_class: int = 0,
        max_changes: int = 3,
    ) -> dict:
        """Find minimal expert prediction changes to flip the decision.

        "What if the LSTM had predicted 0.2 instead of 0.8?
         Would the final prediction change?"

        Returns
        -------
        dict with counterfactual expert predictions and the resulting change.
        """
        if self.meta_ensemble is None:
            return {'error': 'No meta_ensemble fitted'}

        expert_names = sorted(expert_predictions.keys())
        original_scores = {n: float(expert_predictions[n][sample_idx]) for n in expert_names}

        # Get original prediction
        original_pred = self.meta_ensemble.predict_proba(expert_predictions, regimes)[sample_idx]
        target = 0.5 if original_pred >= 0.5 else 0.5  # Flip target

        # Greedy search: change one expert at a time
        best_counterfactual = None
        best_distance = float('inf')

        for n_changes in range(1, min(max_changes + 1, len(expert_names) + 1)):
            from itertools import combinations
            for experts_to_change in combinations(expert_names, n_changes):
                # Try flipping each selected expert's prediction
                cf_preds = {n: expert_predictions[n].copy() for n in expert_names}
                for expert in experts_to_change:
                    # Flip: if high → low, if low → high
                    current = cf_preds[expert][sample_idx]
                    cf_preds[expert][sample_idx] = 1.0 - current

                new_pred = self.meta_ensemble.predict_proba(cf_preds, regimes)[sample_idx]
                changed = (original_pred >= 0.5) != (new_pred >= 0.5)

                if changed:
                    distance = n_changes
                    if distance < best_distance:
                        best_distance = distance
                        best_counterfactual = {
                            'changed_experts': list(experts_to_change),
                            'original_predictions': {e: original_scores[e] for e in experts_to_change},
                            'counterfactual_predictions': {e: 1.0 - original_scores[e] for e in experts_to_change},
                            'original_probability': float(original_pred),
                            'counterfactual_probability': float(new_pred),
                            'n_changes': n_changes,
                        }

        if best_counterfactual is None:
            return {
                'message': 'No counterfactual found within max_changes',
                'original_probability': float(original_pred),
            }

        return best_counterfactual
