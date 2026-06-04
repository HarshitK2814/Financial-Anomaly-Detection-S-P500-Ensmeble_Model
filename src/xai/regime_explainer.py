"""
🌀 REGIME EXPLAINER
=====================
XAI for the regime detection and gating stage.

Explains:
  1. Why a sample was assigned to a particular regime
  2. Regime transition patterns before anomalies
  3. HMM emission distributions per regime
  4. Regime boundary analysis
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional


class RegimeExplainer:
    """Explainability module for regime detection and gating decisions.

    Parameters
    ----------
    gating : AdaptiveGating instance
        The fitted gating mechanism with detector results.
    """

    def __init__(self, gating=None):
        self.gating = gating

    def explain_assignment(self, vol_series: np.ndarray,
                            sample_idx: int) -> dict:
        """Explain why a particular sample was assigned to its regime.

        Returns
        -------
        dict with volatility value, thresholds, detector votes, and confidence.
        """
        vol_value = float(vol_series[sample_idx])

        explanation = {
            'sample_index': sample_idx,
            'volatility': vol_value,
        }

        if self.gating and hasattr(self.gating, '_detector_results'):
            explanation['detector_votes'] = self.gating.explain_routing(sample_idx)

        return explanation

    def transition_analysis(self, regimes: np.ndarray,
                             y: np.ndarray,
                             lookback: int = 10) -> dict:
        """Analyze regime transitions before anomaly events.

        Computes the frequency of regime transitions in the window
        preceding each anomaly. This reveals whether anomalies tend
        to occur after regime transitions (a key structural insight).

        Parameters
        ----------
        regimes : np.ndarray
            Regime labels.
        y : np.ndarray
            Binary anomaly labels.
        lookback : int
            Window before each anomaly to analyze.

        Returns
        -------
        dict with transition statistics.
        """
        anomaly_indices = np.where(y == 1)[0]
        normal_indices = np.where(y == 0)[0]

        def count_transitions(indices):
            transition_counts = []
            for idx in indices:
                start = max(0, idx - lookback)
                window = regimes[start:idx]
                if len(window) > 1:
                    transitions = np.sum(np.diff(window) != 0)
                    transition_counts.append(transitions)
            return np.array(transition_counts) if transition_counts else np.array([0])

        anomaly_transitions = count_transitions(anomaly_indices)
        normal_transitions = count_transitions(
            np.random.choice(normal_indices, min(len(anomaly_indices) * 5, len(normal_indices)), replace=False)
        )

        return {
            'anomaly_mean_transitions': float(np.mean(anomaly_transitions)),
            'normal_mean_transitions': float(np.mean(normal_transitions)),
            'transition_ratio': float(np.mean(anomaly_transitions) / (np.mean(normal_transitions) + 1e-8)),
            'anomalies_preceded_by_transition': float(np.mean(anomaly_transitions > 0)),
            'insight': 'Anomalies are preceded by more regime transitions' if np.mean(anomaly_transitions) > np.mean(normal_transitions) else 'Anomalies occur within stable regimes',
        }

    def regime_distribution_by_outcome(self, regimes: np.ndarray,
                                         y: np.ndarray) -> pd.DataFrame:
        """Show how anomalies distribute across regimes.

        Returns
        -------
        pd.DataFrame with regime-level statistics.
        """
        regime_names = {0: 'Low Vol', 1: 'Med Vol', 2: 'High Vol'}
        rows = []
        for r in sorted(np.unique(regimes)):
            mask = regimes == r
            n_total = mask.sum()
            n_anomaly = (y[mask] == 1).sum()
            rows.append({
                'regime': regime_names.get(r, str(r)),
                'n_total': int(n_total),
                'n_anomalies': int(n_anomaly),
                'anomaly_rate': float(n_anomaly / max(n_total, 1)),
            })
        return pd.DataFrame(rows)

    def boundary_analysis(self, vol_series: np.ndarray,
                           regimes: np.ndarray,
                           y: np.ndarray) -> dict:
        """Analyze samples near regime boundaries.

        Examines how model performance differs for samples near
        vs far from regime boundaries.
        """
        from scipy.stats import percentileofscore

        # Compute how close each sample is to a boundary
        q33 = np.quantile(vol_series, 0.33)
        q66 = np.quantile(vol_series, 0.66)

        dist_to_boundary = np.minimum(
            np.abs(vol_series - q33),
            np.abs(vol_series - q66)
        )

        # Split into "near boundary" (bottom 20% distance) and "far"
        threshold = np.quantile(dist_to_boundary, 0.2)
        near_mask = dist_to_boundary <= threshold
        far_mask = ~near_mask

        return {
            'near_boundary': {
                'n_samples': int(near_mask.sum()),
                'anomaly_rate': float(y[near_mask].mean()) if near_mask.any() else 0,
            },
            'far_from_boundary': {
                'n_samples': int(far_mask.sum()),
                'anomaly_rate': float(y[far_mask].mean()) if far_mask.any() else 0,
            },
            'boundary_threshold': float(threshold),
        }
