"""
📏 EXPLAINABILITY CONSISTENCY INDEX (ECI)
==========================================
NOVEL METRIC CONTRIBUTION

Measures whether multiple XAI methods agree on the same explanation
for a given prediction. High ECI = trustworthy explanation.

ECI(x_i) = 1 - (1/C(K,2)) * Σ_{j<k} D(a^(j)(x_i), a^(k)(x_i))

Where:
  • K = number of XAI methods
  • a^(j)(x_i) = normalized attribution vector from method j
  • D(·, ·) = Kendall's tau rank distance

No existing paper quantifies inter-method explanation consistency
in financial anomaly detection. This metric can become a standard
for trustworthy XAI in regulated domains.
"""
import numpy as np
from scipy.stats import kendalltau, spearmanr
from itertools import combinations
from typing import Dict, List, Tuple


def normalize_attributions(attributions: np.ndarray) -> np.ndarray:
    """Normalize attribution vector to unit L1 norm.

    Parameters
    ----------
    attributions : np.ndarray, shape (D,)
        Raw attribution values (SHAP, IG, saliency, etc.).

    Returns
    -------
    np.ndarray, shape (D,)
        Normalized attribution vector.
    """
    abs_sum = np.abs(attributions).sum()
    if abs_sum < 1e-10:
        return np.ones_like(attributions) / len(attributions)
    return attributions / abs_sum


def rank_distance_kendall(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall's tau rank distance between two attribution vectors.

    Returns value in [0, 1] where 0 = perfect agreement, 1 = total disagreement.
    """
    if len(a) < 2:
        return 0.0
    
    # Use absolute attributions for ranking
    tau, _ = kendalltau(np.abs(a), np.abs(b))
    # Convert correlation [-1, 1] to distance [0, 1]
    return (1 - tau) / 2


def rank_distance_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank distance between two attribution vectors."""
    if len(a) < 2:
        return 0.0
    rho, _ = spearmanr(np.abs(a), np.abs(b))
    return (1 - rho) / 2


def compute_eci_single(attributions: Dict[str, np.ndarray],
                        distance_fn: str = 'kendall') -> float:
    """Compute ECI for a single sample across multiple XAI methods.

    Parameters
    ----------
    attributions : dict
        Mapping of method_name -> attribution vector (np.ndarray).
    distance_fn : str
        'kendall' or 'spearman'.

    Returns
    -------
    float
        ECI score in [0, 1]. Higher = more consistent.
    """
    methods = list(attributions.keys())
    K = len(methods)
    if K < 2:
        return 1.0  # Only one method; trivially consistent

    # Select distance function
    if distance_fn == 'kendall':
        dist_func = rank_distance_kendall
    elif distance_fn == 'spearman':
        dist_func = rank_distance_spearman
    else:
        raise ValueError(f"Unknown distance function: {distance_fn}")

    # Normalize all attributions
    normalized = {
        m: normalize_attributions(attributions[m]) for m in methods
    }

    # Compute pairwise distances
    n_pairs = 0
    total_distance = 0.0
    for m1, m2 in combinations(methods, 2):
        d = dist_func(normalized[m1], normalized[m2])
        total_distance += d
        n_pairs += 1

    # ECI = 1 - average pairwise distance
    avg_distance = total_distance / max(n_pairs, 1)
    return 1.0 - avg_distance


def compute_eci_batch(attributions_batch: Dict[str, np.ndarray],
                       distance_fn: str = 'kendall') -> np.ndarray:
    """Compute ECI for a batch of samples.

    Parameters
    ----------
    attributions_batch : dict
        Mapping of method_name -> np.ndarray of shape (N, D).

    Returns
    -------
    np.ndarray, shape (N,)
        ECI scores per sample.
    """
    methods = list(attributions_batch.keys())
    N = attributions_batch[methods[0]].shape[0]
    eci_scores = np.zeros(N)

    for i in range(N):
        sample_attrs = {m: attributions_batch[m][i] for m in methods}
        eci_scores[i] = compute_eci_single(sample_attrs, distance_fn)

    return eci_scores


def eci_summary(eci_scores: np.ndarray,
                regimes: np.ndarray = None) -> dict:
    """Generate summary statistics for ECI scores.

    Returns
    -------
    dict with global and per-regime ECI statistics.
    """
    summary = {
        'global': {
            'mean': float(np.mean(eci_scores)),
            'median': float(np.median(eci_scores)),
            'std': float(np.std(eci_scores)),
            'min': float(np.min(eci_scores)),
            'max': float(np.max(eci_scores)),
            'pct_high_consistency': float(np.mean(eci_scores > 0.7)),
            'pct_low_consistency': float(np.mean(eci_scores < 0.3)),
        }
    }

    if regimes is not None:
        regime_names = {0: 'Low', 1: 'Med', 2: 'High'}
        per_regime = {}
        for r in np.unique(regimes):
            mask = regimes == r
            rname = regime_names.get(r, str(r))
            per_regime[rname] = {
                'mean': float(np.mean(eci_scores[mask])),
                'std': float(np.std(eci_scores[mask])),
                'n_samples': int(mask.sum()),
            }
        summary['per_regime'] = per_regime

    return summary


def top_k_agreement(attributions: Dict[str, np.ndarray],
                     k: int = 3) -> float:
    """Compute top-k agreement across XAI methods.

    Measures the fraction of pairwise overlap in the top-k
    most important features across methods.

    Parameters
    ----------
    attributions : dict
        Method name -> attribution vector.
    k : int
        Number of top features to compare.

    Returns
    -------
    float
        Agreement score in [0, 1].
    """
    methods = list(attributions.keys())
    if len(methods) < 2:
        return 1.0

    top_k_sets = {}
    for m in methods:
        attr = np.abs(attributions[m])
        top_k_sets[m] = set(np.argsort(attr)[::-1][:k])

    agreements = []
    for m1, m2 in combinations(methods, 2):
        overlap = len(top_k_sets[m1] & top_k_sets[m2])
        agreements.append(overlap / k)

    return float(np.mean(agreements))


if __name__ == '__main__':
    # Demo
    D = 10  # features
    np.random.seed(42)

    # Simulate attributions from 3 XAI methods
    # Methods that mostly agree
    base_importance = np.array([0.3, 0.25, 0.15, 0.1, 0.05, 0.04, 0.03, 0.03, 0.03, 0.02])
    attrs = {
        'SHAP':      base_importance + np.random.normal(0, 0.02, D),
        'IntGrad':   base_importance + np.random.normal(0, 0.03, D),
        'Saliency':  base_importance + np.random.normal(0, 0.05, D),
    }

    eci = compute_eci_single(attrs)
    print(f"ECI (agreeing methods):    {eci:.4f}  (should be high)")

    # Methods that disagree
    attrs_disagree = {
        'SHAP':      np.array([0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01]),
        'IntGrad':   np.array([0.01, 0.02, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.12, 0.4]),
        'Saliency':  np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    }

    eci_d = compute_eci_single(attrs_disagree)
    print(f"ECI (disagreeing methods): {eci_d:.4f}  (should be low)")

    # Top-k agreement
    tk = top_k_agreement(attrs, k=3)
    print(f"Top-3 agreement (agreeing):    {tk:.4f}")
    tk_d = top_k_agreement(attrs_disagree, k=3)
    print(f"Top-3 agreement (disagreeing): {tk_d:.4f}")
