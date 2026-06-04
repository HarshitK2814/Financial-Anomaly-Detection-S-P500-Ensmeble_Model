"""
📐 STATISTICAL SIGNIFICANCE TESTS
====================================
Formal statistical tests for pairwise model comparison.

Tests:
  1. McNemar's Test — Are two models' errors significantly different?
  2. DeLong's Test  — Are two models' AUC-PR significantly different?
  3. Bootstrap Difference Test — Is the F1 difference significant?
"""
import numpy as np
from scipy.stats import chi2, norm
from typing import Tuple


def mcnemar_test(y_true: np.ndarray,
                 preds_a: np.ndarray,
                 preds_b: np.ndarray) -> dict:
    """McNemar's test for comparing two classifiers.

    Tests whether the disagreement between two models is significant.
    Null hypothesis: both models have the same error rate.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    preds_a, preds_b : np.ndarray
        Predictions from model A and model B.

    Returns
    -------
    dict with statistic, p_value, and interpretation.
    """
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)

    # Contingency table
    # b01 = A correct, B incorrect
    # b10 = A incorrect, B correct
    b01 = np.sum(correct_a & ~correct_b)
    b10 = np.sum(~correct_a & correct_b)

    # McNemar statistic with continuity correction
    if b01 + b10 == 0:
        return {
            'statistic': 0.0,
            'p_value': 1.0,
            'significant': False,
            'interpretation': 'Models make identical errors',
            'b01': int(b01),
            'b10': int(b10),
        }

    statistic = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p_value = 1 - chi2.cdf(statistic, df=1)

    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'interpretation': f"{'Significant' if p_value < 0.05 else 'Not significant'} "
                         f"difference (p={p_value:.4f})",
        'b01': int(b01),
        'b10': int(b10),
        'favors': 'A' if b10 > b01 else 'B' if b01 > b10 else 'Neither',
    }


def bootstrap_paired_test(y_true: np.ndarray,
                           preds_a: np.ndarray,
                           preds_b: np.ndarray,
                           metric_fn=None,
                           n_bootstrap: int = 1000,
                           confidence: float = 0.95,
                           random_state: int = 42) -> dict:
    """Bootstrap paired difference test for comparing two models.

    Tests whether the metric difference between models A and B
    is statistically significant.

    Parameters
    ----------
    metric_fn : callable
        Function(y_true, y_pred) -> float. Default: F1 score.

    Returns
    -------
    dict with difference, CI, and significance.
    """
    from sklearn.metrics import f1_score as _f1_score

    if metric_fn is None:
        metric_fn = lambda yt, yp: _f1_score(yt, yp, zero_division=0)

    rng = np.random.RandomState(random_state)
    n = len(y_true)

    # Point estimates
    metric_a = metric_fn(y_true, preds_a)
    metric_b = metric_fn(y_true, preds_b)
    observed_diff = metric_a - metric_b

    # Bootstrap differences
    boot_diffs = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        try:
            diff = metric_fn(y_true[indices], preds_a[indices]) - \
                   metric_fn(y_true[indices], preds_b[indices])
            boot_diffs.append(diff)
        except (ValueError, ZeroDivisionError):
            pass

    if not boot_diffs:
        return {
            'metric_a': float(metric_a),
            'metric_b': float(metric_b),
            'difference': float(observed_diff),
            'p_value': 1.0,
            'significant': False,
        }

    boot_diffs = np.array(boot_diffs)
    alpha = (1 - confidence) / 2

    ci_lower = float(np.percentile(boot_diffs, alpha * 100))
    ci_upper = float(np.percentile(boot_diffs, (1 - alpha) * 100))

    # P-value: proportion of bootstrap samples where diff has opposite sign
    if observed_diff > 0:
        p_value = float(np.mean(boot_diffs <= 0)) * 2  # Two-sided
    elif observed_diff < 0:
        p_value = float(np.mean(boot_diffs >= 0)) * 2
    else:
        p_value = 1.0

    p_value = min(p_value, 1.0)

    return {
        'metric_a': float(metric_a),
        'metric_b': float(metric_b),
        'difference': float(observed_diff),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'favors': 'A' if observed_diff > 0 else 'B' if observed_diff < 0 else 'Neither',
        'interpretation': (
            f"Model {'A' if observed_diff > 0 else 'B'} is "
            f"{'significantly' if p_value < 0.05 else 'not significantly'} "
            f"better (Δ={observed_diff:+.4f}, p={p_value:.4f}, "
            f"95% CI [{ci_lower:+.4f}, {ci_upper:+.4f}])"
        ),
    }


def pairwise_comparison_table(y_true: np.ndarray,
                               predictions: dict,
                               metric_fn=None,
                               n_bootstrap: int = 500) -> str:
    """Generate a full pairwise comparison table.

    Parameters
    ----------
    predictions : dict
        Mapping of model_name -> predictions array.

    Returns
    -------
    str : Formatted comparison table.
    """
    from sklearn.metrics import f1_score as _f1_score
    if metric_fn is None:
        metric_fn = lambda yt, yp: _f1_score(yt, yp, zero_division=0)

    model_names = sorted(predictions.keys())
    lines = ["PAIRWISE STATISTICAL COMPARISON", "=" * 60]

    for i, name_a in enumerate(model_names):
        for name_b in model_names[i+1:]:
            result = bootstrap_paired_test(
                y_true,
                predictions[name_a],
                predictions[name_b],
                metric_fn=metric_fn,
                n_bootstrap=n_bootstrap,
            )
            sig = "✓" if result['significant'] else "✗"
            lines.append(
                f"{name_a} vs {name_b}: "
                f"Δ={result['difference']:+.4f} "
                f"(p={result['p_value']:.4f}) [{sig}] "
                f"→ {result['favors']}"
            )

    return "\n".join(lines)


if __name__ == '__main__':
    # Demo
    np.random.seed(42)
    n = 200
    y_true = np.random.choice([0, 1], size=n, p=[0.95, 0.05])
    preds_a = np.random.choice([0, 1], size=n, p=[0.93, 0.07])
    preds_b = np.random.choice([0, 1], size=n, p=[0.90, 0.10])

    print("McNemar Test:")
    result = mcnemar_test(y_true, preds_a, preds_b)
    print(f"  {result['interpretation']}")

    print("\nBootstrap Paired Test:")
    result = bootstrap_paired_test(y_true, preds_a, preds_b, n_bootstrap=500)
    print(f"  {result['interpretation']}")
