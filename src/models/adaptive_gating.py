"""
🚪 ADAPTIVE REGIME GATING MODULE
==================================
Novel contribution: Confidence-aware regime fusion that combines
multiple regime detectors and controls expert routing.

When detectors agree → high confidence → hard routing to one expert
When detectors disagree → low confidence → soft routing (weighted
prediction across multiple experts)

This module replaces the simple quantile-based hard gating in the
original system.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple


class AdaptiveGating:
    """Confidence-aware regime gating mechanism.

    Combines predictions from multiple regime detectors and outputs:
      - A primary regime assignment
      - A confidence score
      - Optional soft routing weights for multi-expert prediction

    Parameters
    ----------
    detectors : dict
        Mapping of detector_name -> detector_instance.
        Each detector must implement .detect(vol_series) -> (regimes, confidence).
    fusion_method : str
        How to combine detector outputs:
        - 'majority': Simple majority vote (hard)
        - 'confidence_weighted': Weight by detector confidence (soft)
        - 'conservative': Use highest-confidence detector (hard)
    soft_threshold : float
        Below this fused confidence, use soft routing across experts.
        Above this, use hard routing to single expert.
    """

    def __init__(self, detectors: dict,
                 fusion_method: str = 'confidence_weighted',
                 soft_threshold: float = 0.6,
                 n_regimes: int = 3):
        self.detectors = detectors
        self.fusion_method = fusion_method
        self.soft_threshold = soft_threshold
        self.n_regimes = n_regimes

        # Store per-detector results for XAI
        self._detector_results = {}

    def detect_all(self, vol_series: np.ndarray) -> Dict[str, Tuple]:
        """Run all detectors and store results.

        Returns
        -------
        dict mapping detector_name -> (regimes, confidence).
        """
        self._detector_results = {}
        for name, detector in self.detectors.items():
            regimes, confidence = detector.detect(vol_series)
            self._detector_results[name] = (regimes, confidence)
        return self._detector_results

    def fuse(self, vol_series: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fuse regime detections into a single assignment with confidence.

        Parameters
        ----------
        vol_series : np.ndarray, shape (N,)
            Realized volatility series.

        Returns
        -------
        fused_regimes : np.ndarray, shape (N,)
            Final regime assignments.
        fused_confidence : np.ndarray, shape (N,)
            Confidence in the fused regime assignment.
        routing_weights : np.ndarray, shape (N, n_regimes)
            Soft routing weights. If hard routing, this is one-hot.
        """
        if not self._detector_results:
            self.detect_all(vol_series)

        N = len(vol_series)
        n_detectors = len(self._detector_results)

        if n_detectors == 0:
            raise ValueError("No detectors configured.")

        if n_detectors == 1:
            # Single detector: use directly
            name = list(self._detector_results.keys())[0]
            regimes, confidence = self._detector_results[name]
            routing = self._make_routing_weights(regimes, confidence)
            return regimes, confidence, routing

        if self.fusion_method == 'majority':
            return self._fuse_majority(N)
        elif self.fusion_method == 'confidence_weighted':
            return self._fuse_confidence_weighted(N)
        elif self.fusion_method == 'conservative':
            return self._fuse_conservative(N)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

    def _fuse_majority(self, N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simple majority vote across detectors."""
        # Collect votes
        votes = np.zeros((N, self.n_regimes))
        for name, (regimes, confidence) in self._detector_results.items():
            for i in range(N):
                votes[i, regimes[i]] += 1

        fused_regimes = np.argmax(votes, axis=1)

        # Confidence = fraction of detectors agreeing
        n_detectors = len(self._detector_results)
        fused_confidence = np.max(votes, axis=1) / n_detectors

        routing = self._make_routing_weights(fused_regimes, fused_confidence)
        return fused_regimes, fused_confidence, routing

    def _fuse_confidence_weighted(self, N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Confidence-weighted vote: each detector's vote is scaled by
        its confidence score.
        """
        weighted_votes = np.zeros((N, self.n_regimes))
        total_confidence = np.zeros(N)

        for name, (regimes, confidence) in self._detector_results.items():
            for i in range(N):
                weighted_votes[i, regimes[i]] += confidence[i]
                total_confidence[i] += confidence[i]

        # Normalize to get soft probabilities
        total_confidence = np.maximum(total_confidence, 1e-8)
        soft_probs = weighted_votes / total_confidence[:, None]

        fused_regimes = np.argmax(soft_probs, axis=1)
        fused_confidence = np.max(soft_probs, axis=1)

        routing = self._make_routing_weights(fused_regimes, fused_confidence)
        return fused_regimes, fused_confidence, routing

    def _fuse_conservative(self, N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Use the detector with highest confidence for each sample."""
        fused_regimes = np.zeros(N, dtype=int)
        fused_confidence = np.zeros(N)

        for i in range(N):
            best_conf = -1.0
            best_regime = 1  # Default: Medium
            for name, (regimes, confidence) in self._detector_results.items():
                if confidence[i] > best_conf:
                    best_conf = confidence[i]
                    best_regime = regimes[i]
            fused_regimes[i] = best_regime
            fused_confidence[i] = best_conf

        routing = self._make_routing_weights(fused_regimes, fused_confidence)
        return fused_regimes, fused_confidence, routing

    def _make_routing_weights(self, regimes: np.ndarray,
                               confidence: np.ndarray) -> np.ndarray:
        """Create routing weight matrix.

        For high-confidence samples: one-hot (hard routing).
        For low-confidence samples: distribute weight across neighbors.
        """
        N = len(regimes)
        routing = np.zeros((N, self.n_regimes))

        for i in range(N):
            if confidence[i] >= self.soft_threshold:
                # Hard routing
                routing[i, regimes[i]] = 1.0
            else:
                # Soft routing: primary expert gets proportional weight,
                # neighboring regimes share the remainder
                primary_weight = confidence[i]
                remainder = 1.0 - primary_weight
                routing[i, regimes[i]] = primary_weight

                # Distribute remainder to adjacent regimes
                neighbors = []
                if regimes[i] > 0:
                    neighbors.append(regimes[i] - 1)
                if regimes[i] < self.n_regimes - 1:
                    neighbors.append(regimes[i] + 1)

                if neighbors:
                    share = remainder / len(neighbors)
                    for n in neighbors:
                        routing[i, n] = share
                else:
                    routing[i, regimes[i]] = 1.0  # Edge case

        return routing

    def get_detector_agreement(self) -> float:
        """Compute pairwise agreement rate between detectors."""
        if len(self._detector_results) < 2:
            return 1.0

        names = list(self._detector_results.keys())
        agreements = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r_i = self._detector_results[names[i]][0]
                r_j = self._detector_results[names[j]][0]
                agreements.append(np.mean(r_i == r_j))

        return np.mean(agreements)

    def get_regime_stability(self) -> dict:
        """Analyze regime transition frequency for each detector."""
        stability = {}
        for name, (regimes, _) in self._detector_results.items():
            transitions = np.sum(np.diff(regimes) != 0)
            stability[name] = {
                'n_transitions': int(transitions),
                'transition_rate': transitions / (len(regimes) - 1),
                'regime_lengths': self._compute_regime_lengths(regimes),
            }
        return stability

    @staticmethod
    def _compute_regime_lengths(regimes: np.ndarray) -> dict:
        """Compute average and median regime duration."""
        lengths = []
        current_regime = regimes[0]
        current_length = 1
        for i in range(1, len(regimes)):
            if regimes[i] == current_regime:
                current_length += 1
            else:
                lengths.append(current_length)
                current_regime = regimes[i]
                current_length = 1
        lengths.append(current_length)

        return {
            'mean': float(np.mean(lengths)),
            'median': float(np.median(lengths)),
            'min': int(np.min(lengths)),
            'max': int(np.max(lengths)),
        }

    def explain_routing(self, idx: int) -> dict:
        """Generate XAI explanation for routing decision at sample idx.

        Returns a dict with detector-level explanations and routing rationale.
        """
        explanation = {
            'sample_index': idx,
            'detector_votes': {},
            'fusion_method': self.fusion_method,
        }

        for name, (regimes, confidence) in self._detector_results.items():
            regime_names = {0: 'Low Vol', 1: 'Med Vol', 2: 'High Vol'}
            explanation['detector_votes'][name] = {
                'regime': regime_names.get(regimes[idx], f'Unknown({regimes[idx]})'),
                'regime_id': int(regimes[idx]),
                'confidence': float(confidence[idx]),
            }

        return explanation


def create_default_gating(use_hmm: bool = True,
                          rolling_window: int = 252) -> AdaptiveGating:
    """Factory function to create a standard adaptive gating setup.

    Parameters
    ----------
    use_hmm : bool
        Whether to include HMM detector (requires hmmlearn).
    rolling_window : int
        Window size for rolling quantile detector.

    Returns
    -------
    AdaptiveGating instance with pre-configured detectors.
    """
    from .regime_detector import (
        RollingQuantileRegimeDetector,
        StaticQuantileRegimeDetector,
    )

    detectors = {
        'rolling_quantile': RollingQuantileRegimeDetector(window=rolling_window),
        'static_quantile': StaticQuantileRegimeDetector(),
    }

    if use_hmm:
        try:
            from .regime_detector import HMMRegimeDetector
            detectors['hmm'] = HMMRegimeDetector()
        except ImportError:
            pass

    return AdaptiveGating(
        detectors=detectors,
        fusion_method='confidence_weighted',
        soft_threshold=0.6,
    )


if __name__ == '__main__':
    from regime_detector import (
        RollingQuantileRegimeDetector,
        StaticQuantileRegimeDetector,
        realized_volatility_windowed,
    )

    X = np.load('artifacts/market_windows_10f.npy')
    vol = realized_volatility_windowed(X)

    print("🚪 ADAPTIVE GATING DEMO")
    print("=" * 50)

    detectors = {
        'rolling': RollingQuantileRegimeDetector(window=252),
        'static': StaticQuantileRegimeDetector(),
    }

    gating = AdaptiveGating(
        detectors=detectors,
        fusion_method='confidence_weighted',
        soft_threshold=0.6,
    )

    regimes, confidence, routing = gating.fuse(vol)

    print(f"Fused regimes: {dict(zip(*np.unique(regimes, return_counts=True)))}")
    print(f"Mean confidence: {confidence.mean():.3f}")
    print(f"Hard-routed samples: {(confidence >= 0.6).sum()} / {len(confidence)}")
    print(f"Soft-routed samples: {(confidence < 0.6).sum()} / {len(confidence)}")
    print(f"Detector agreement: {gating.get_detector_agreement():.1%}")

    # Sample explanation
    ex = gating.explain_routing(0)
    print(f"\nSample 0 routing explanation: {ex}")
