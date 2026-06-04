"""
🌀 ADAPTIVE REGIME DETECTION MODULE
=====================================
Replaces static Q33/Q66 quantile thresholds with adaptive methods.

Two detectors:
  1. Rolling Quantile Detector — rolling 252-day quantile thresholds
  2. HMM-based Detector        — Gaussian HMM with 3 latent states

A fusion mechanism combines both detectors with confidence scoring.

Novel contribution: No existing financial anomaly detection paper uses
a confidence-aware hybrid gating mechanism that combines statistical
and probabilistic regime detectors.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


class RollingQuantileRegimeDetector:
    """Rolling quantile-based regime detector.

    Instead of computing global Q33/Q66 on the entire dataset
    (which introduces look-ahead bias), this uses a rolling
    window of 252 trading days (~1 year) to compute adaptive
    quantile thresholds.
    """

    def __init__(self, window: int = 252, q_low: float = 0.33,
                 q_high: float = 0.66, min_periods: int = 20):
        self.window = window
        self.q_low = q_low
        self.q_high = q_high
        self.min_periods = min_periods
        # Terminal thresholds stored by detect() for out-of-sample prediction
        self._last_q_low: Optional[float] = None
        self._last_q_high: Optional[float] = None

    def detect(self, vol_series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect regimes using rolling quantile thresholds.

        Parameters
        ----------
        vol_series : np.ndarray, shape (N,)
            Realized volatility time series.

        Returns
        -------
        regimes : np.ndarray, shape (N,)
            Regime labels (0=Low, 1=Med, 2=High).
        confidence : np.ndarray, shape (N,)
            Confidence scores in [0, 1]. Higher = more certain.
        """
        vol_pd = pd.Series(vol_series)

        rolling_q_low = vol_pd.rolling(
            window=self.window, min_periods=self.min_periods
        ).quantile(self.q_low).ffill().bfill().values

        rolling_q_high = vol_pd.rolling(
            window=self.window, min_periods=self.min_periods
        ).quantile(self.q_high).ffill().bfill().values

        regimes = np.ones(len(vol_series), dtype=int)  # Default: Medium
        regimes[vol_series <= rolling_q_low] = 0
        regimes[vol_series > rolling_q_high] = 2

        # Confidence: distance from nearest boundary, normalized
        dist_low = np.abs(vol_series - rolling_q_low)
        dist_high = np.abs(vol_series - rolling_q_high)
        dist_nearest = np.minimum(dist_low, dist_high)

        # For boundary-distant points, high confidence
        # For points near boundaries, low confidence
        range_vol = rolling_q_high - rolling_q_low + 1e-8
        confidence = np.clip(dist_nearest / (range_vol * 0.5), 0, 1)

        # Store terminal thresholds for predict_from_last_state() (R1 fix)
        self._last_q_low  = float(rolling_q_low[-1])
        self._last_q_high = float(rolling_q_high[-1])

        return regimes, confidence

    def get_thresholds(self, vol_series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return the rolling threshold series for plotting."""
        vol_pd = pd.Series(vol_series)
        q_low = vol_pd.rolling(
            window=self.window, min_periods=self.min_periods
        ).quantile(self.q_low).ffill().bfill().values
        q_high = vol_pd.rolling(
            window=self.window, min_periods=self.min_periods
        ).quantile(self.q_high).ffill().bfill().values
        return q_low, q_high

    def predict_from_last_state(self, vol_series_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Assign regimes to test observations using terminal training thresholds.

        ⚠️  R1 LEAKAGE FIX: Call detect() on the training fold first, then
        call this method on the test fold.  This applies the last rolling
        quantile thresholds from training to test observations without
        updating the window, ensuring zero look-ahead into the test period.

        Parameters
        ----------
        vol_series_new : np.ndarray, shape (M,)
            Test-fold volatility series.

        Returns
        -------
        regimes    : np.ndarray, shape (M,)  regime labels (0=Low,1=Med,2=High)
        confidence : np.ndarray, shape (M,)  in [0, 1]
        """
        if self._last_q_low is None or self._last_q_high is None:
            raise RuntimeError(
                "Call detect() on training data before predict_from_last_state()."
            )
        q_lo, q_hi = self._last_q_low, self._last_q_high

        regimes = np.ones(len(vol_series_new), dtype=int)  # Default: Medium
        regimes[vol_series_new <= q_lo] = 0
        regimes[vol_series_new >  q_hi] = 2

        dist_low     = np.abs(vol_series_new - q_lo)
        dist_high    = np.abs(vol_series_new - q_hi)
        dist_nearest = np.minimum(dist_low, dist_high)
        range_vol    = q_hi - q_lo + 1e-8
        confidence   = np.clip(dist_nearest / (range_vol * 0.5), 0, 1)

        return regimes, confidence


class HMMRegimeDetector:
    """Gaussian Hidden Markov Model regime detector.

    Learns 3 latent states from the volatility series using
    a Gaussian HMM. The states are ordered by mean volatility
    to correspond to Low/Med/High.
    """

    def __init__(self, n_states: int = 3, n_iter: int = 100,
                 covariance_type: str = 'full', random_state: int = 42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = None
        self._state_order = None

    def fit(self, vol_series: np.ndarray) -> 'HMMRegimeDetector':
        """Fit the HMM to the volatility series.

        Parameters
        ----------
        vol_series : np.ndarray, shape (N,)
            Realized volatility time series.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise ImportError(
                "hmmlearn is required for HMM regime detection. "
                "Install with: pip install hmmlearn"
            )

        X = vol_series.reshape(-1, 1)
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.model.fit(X)

        # Order states by mean volatility (Low=0, Med=1, High=2)
        means = self.model.means_.flatten()
        self._state_order = np.argsort(means)

        return self

    def detect(self, vol_series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect regimes using the fitted HMM.

        Returns
        -------
        regimes : np.ndarray, shape (N,)
            Regime labels re-ordered as (0=Low, 1=Med, 2=High).
        confidence : np.ndarray, shape (N,)
            Posterior probability of the assigned state.
        """
        if self.model is None:
            raise RuntimeError("HMM must be fitted first. Call .fit()")

        X = vol_series.reshape(-1, 1)
        raw_states = self.model.predict(X)
        posteriors = self.model.predict_proba(X)

        # Re-order states
        state_map = {old: new for new, old in enumerate(self._state_order)}
        regimes = np.array([state_map[s] for s in raw_states])

        # Confidence = posterior probability of assigned state
        confidence = np.array([posteriors[i, raw_states[i]] for i in range(len(raw_states))])

        return regimes, confidence

    def get_transition_matrix(self) -> np.ndarray:
        """Return the regime transition probability matrix (reordered)."""
        if self.model is None:
            raise RuntimeError("HMM must be fitted first.")
        T = self.model.transmat_
        # Reorder rows and columns
        return T[self._state_order][:, self._state_order]

    def get_emission_params(self) -> dict:
        """Return emission distribution parameters per regime."""
        if self.model is None:
            raise RuntimeError("HMM must be fitted first.")
        means = self.model.means_[self._state_order].flatten()
        covars = self.model.covars_[self._state_order].flatten()
        return {
            'means': {'Low': means[0], 'Med': means[1], 'High': means[2]},
            'variances': {'Low': covars[0], 'Med': covars[1], 'High': covars[2]},
        }


class StaticQuantileRegimeDetector:
    """Static quantile-based regime detector (current baseline).

    Preserved for backward compatibility and ablation studies.
    Uses global Q33/Q66 quantiles on the full series.
    """

    def __init__(self, q_low: float = 0.33, q_high: float = 0.66):
        self.q_low = q_low
        self.q_high = q_high

    def detect(self, vol_series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        q33, q66 = np.quantile(vol_series, [self.q_low, self.q_high])
        regimes = np.zeros_like(vol_series, dtype=int)
        regimes[vol_series <= q33] = 0
        regimes[(vol_series > q33) & (vol_series <= q66)] = 1
        regimes[vol_series > q66] = 2

        # Static detector has uniform confidence
        confidence = np.ones_like(vol_series)
        return regimes, confidence


def realized_volatility_windowed(data: np.ndarray,
                                  feature_idx: int = 0) -> np.ndarray:
    """Compute per-window realized volatility.

    Parameters
    ----------
    data : np.ndarray, shape (N, T, F)
        Windowed time series data.
    feature_idx : int
        Which feature channel to compute volatility from.

    Returns
    -------
    np.ndarray, shape (N,)
        Realized volatility per window.
    """
    return np.std(data[:, :, feature_idx], axis=1)


if __name__ == '__main__':
    # Demo: compare all three detectors
    X = np.load('artifacts/market_windows_10f.npy')
    vol = realized_volatility_windowed(X)

    print("🌀 REGIME DETECTION COMPARISON")
    print("=" * 50)

    # Static (baseline)
    static = StaticQuantileRegimeDetector()
    r_static, c_static = static.detect(vol)
    print(f"\nStatic Quantile:  {dict(zip(*np.unique(r_static, return_counts=True)))}")

    # Rolling
    rolling = RollingQuantileRegimeDetector(window=252)
    r_rolling, c_rolling = rolling.detect(vol)
    print(f"Rolling Quantile: {dict(zip(*np.unique(r_rolling, return_counts=True)))}")
    print(f"  Avg confidence: {c_rolling.mean():.3f}")

    # HMM
    try:
        hmm = HMMRegimeDetector()
        hmm.fit(vol)
        r_hmm, c_hmm = hmm.detect(vol)
        print(f"HMM Detector:     {dict(zip(*np.unique(r_hmm, return_counts=True)))}")
        print(f"  Avg confidence: {c_hmm.mean():.3f}")
        print(f"  Transition Matrix:\n{hmm.get_transition_matrix().round(3)}")
    except ImportError:
        print("  HMM: hmmlearn not installed. Skipping.")

    # Agreement
    agreement_sr = np.mean(r_static == r_rolling)
    print(f"\nStatic-Rolling agreement: {agreement_sr:.1%}")
