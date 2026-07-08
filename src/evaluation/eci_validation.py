"""
Priority 5 — ECI Validation (H-01)

Resolves three open issues:
1. Cross-model vs cross-method reconciliation:
   computes BOTH types of ECI and reports them side-by-side.
2. Null model baseline: expected ECI under random feature rankings.
3. Bootstrap 95% CIs on per-regime ECI + Wilcoxon test across regimes.

Saves: artifacts/eci_validation_results.json

Run from repo root:
  .venv/Scripts/python.exe -m src.evaluation.eci_validation
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import joblib
from scipy.stats import kendalltau, wilcoxon
from itertools import combinations

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

REGIME_MAP = {0: "Low", 1: "Med", 2: "High"}
MODELS = ["RF", "XGBoost", "LightGBM"]
MODEL_FILES = {
    "RF":       "RF_{}_expert.joblib",
    "XGBoost":  "XGBoost_{}_expert.joblib",
    "LightGBM": "LightGBM_{}_expert.joblib",
}
ARTIFACTS = "artifacts"
B_BOOT = 2000  # bootstrap replicates


# ----------------------------------------------------------------------------─
# helpers
# ----------------------------------------------------------------------------─
def _kendall_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized Kendall tau distance in [0, 1]; 0=perfect, 1=reverse."""
    if len(a) < 2:
        return 0.0
    tau, _ = kendalltau(np.abs(a), np.abs(b))
    return (1.0 - tau) / 2.0


def _eci_from_vectors(vecs: list[np.ndarray]) -> float:
    """ECI = 1 − mean pairwise Kendall-tau distance."""
    K = len(vecs)
    if K < 2:
        return 1.0
    dists = [_kendall_dist(a, b) for a, b in combinations(vecs, 2)]
    return 1.0 - float(np.mean(dists))


def _null_eci(dim: int, K: int = 3, n: int = 2000, rng=None) -> np.ndarray:
    """ECI distribution under H0: each model's importance is a random permutation."""
    if rng is None:
        rng = np.random.default_rng(0)
    ecis = np.empty(n)
    base = np.arange(dim, dtype=float)
    for i in range(n):
        vecs = [rng.permutation(base) for _ in range(K)]
        ecis[i] = _eci_from_vectors(vecs)
    return ecis


def _boot_ci_eci(vecs: list[np.ndarray], B: int = B_BOOT, alpha: float = 0.05,
                 rng=None) -> tuple:
    """Bootstrap CI for ECI by resampling feature indices with replacement."""
    if rng is None:
        rng = np.random.default_rng(42)
    D = len(vecs[0])
    boot = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, D, size=D)
        boot[b] = _eci_from_vectors([v[idx] for v in vecs])
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return lo, hi, boot


# ----------------------------------------------------------------------------─
# load helpers
# ----------------------------------------------------------------------------─
def _load_model(name: str, regime_str: str):
    fname = MODEL_FILES[name].format(regime_str)
    path = os.path.join(ARTIFACTS, "models", fname)
    return joblib.load(path)


def _get_regime_data(X, y, regime_arr, k):
    mask = regime_arr == k
    return X[mask], y[mask]


def _compute_regime_arr(X):
    """Assign regime per sample using Composite_Volatility (feature index 2)."""
    cv = X[:, 2]
    q33, q66 = np.quantile(cv, [0.33, 0.66])
    return np.where(cv <= q33, 0, np.where(cv <= q66, 1, 2))


# ----------------------------------------------------------------------------─
# SHAP-based importance (cross-method reconciliation)
# ----------------------------------------------------------------------------─
def _shap_importance(model, X, model_name: str) -> np.ndarray:
    """Mean |SHAP| per feature for the anomaly class."""
    import shap
    try:
        expl = shap.TreeExplainer(model)
        sv = expl.shap_values(X)
        if isinstance(sv, list):          # binary outputs → pick class 1
            sv = sv[1]
        elif sv.ndim == 3:                # (n, features, classes)
            sv = sv[:, :, 1]
        return np.abs(sv).mean(axis=0)
    except Exception as e:
        print(f"    SHAP failed for {model_name}: {e} — using tree FI fallback")
        return model.feature_importances_


# ----------------------------------------------------------------------------─
# main
# ----------------------------------------------------------------------------─
def run():
    rng = np.random.default_rng(42)

    X = np.load(os.path.join(ARTIFACTS, "X_factors.npy"))
    y = np.load(os.path.join(ARTIFACTS, "y_factors.npy"))
    regime_arr = _compute_regime_arr(X)

    D = X.shape[1]
    print(f"Data: {X.shape}, anomalies={y.sum()}, features={D}")
    print(f"Regimes: {dict(zip(*np.unique(regime_arr, return_counts=True)))}")

    results = {}

    # -- Per-regime analysis --------------------------------------------------─
    regime_boot_ecis = {}   # to store bootstrap arrays for Wilcoxon
    for k, nm in REGIME_MAP.items():
        Xk, yk = _get_regime_data(X, y, regime_arr, k)
        if len(Xk) < 10:
            print(f"  {nm}-Vol: too few samples ({len(Xk)}), skipping")
            continue
        print(f"\n{nm}-Vol regime  (n={len(Xk)}, anomalies={yk.sum()})")

        # Load models
        mods = {nm_m: _load_model(nm_m, nm) for nm_m in MODELS}

        # -- (A) Cross-model ECI: tree feature_importances_ ------------------─
        fi_vecs = [mods[m].feature_importances_ for m in MODELS]
        eci_cm = _eci_from_vectors(fi_vecs)
        ci_lo, ci_hi, boot_arr = _boot_ci_eci(fi_vecs, B=B_BOOT, rng=rng)
        regime_boot_ecis[nm] = boot_arr
        print(f"  Cross-model ECI (tree FI):  {eci_cm:.4f}  "
              f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")

        # -- (B) Cross-method ECI per model: tree FI vs SHAP importance ------─
        shap_vecs = {m: _shap_importance(mods[m], Xk, m) for m in MODELS}
        cm_method_ecis = {}
        for m in MODELS:
            fi = mods[m].feature_importances_
            sh = shap_vecs[m]
            # 2-method ECI (one pair → ECI = 1 − dist)
            d = _kendall_dist(fi, sh)
            cm_method_ecis[m] = round(1.0 - d, 4)
        avg_cross_method = float(np.mean(list(cm_method_ecis.values())))
        print(f"  Cross-method ECI per model (FI vs SHAP): {cm_method_ecis}")
        print(f"    avg cross-method ECI: {avg_cross_method:.4f}")

        # -- (C) Null model baseline ------------------------------------------─
        null_dist = _null_eci(D, K=3, n=2000, rng=rng)
        null_mean = float(null_dist.mean())
        null_ci_lo = float(np.percentile(null_dist, 2.5))
        null_ci_hi = float(np.percentile(null_dist, 97.5))
        z_score = (eci_cm - null_mean) / null_dist.std()
        p_null = float(np.mean(null_dist >= eci_cm))
        print(f"  Null model ECI: {null_mean:.4f} [{null_ci_lo:.4f}, {null_ci_hi:.4f}]")
        print(f"    z = {z_score:.2f}, p(H0: ECI >= observed) = {p_null:.4f}")

        results[nm] = {
            "n_samples": int(len(Xk)),
            "n_anomalies": int(yk.sum()),
            "cross_model_eci": round(eci_cm, 4),
            "ci_lo": round(ci_lo, 4),
            "ci_hi": round(ci_hi, 4),
            "cross_method_eci_per_model": cm_method_ecis,
            "avg_cross_method_eci": round(avg_cross_method, 4),
            "null_mean": round(null_mean, 4),
            "null_ci_lo": round(null_ci_lo, 4),
            "null_ci_hi": round(null_ci_hi, 4),
            "z_vs_null": round(z_score, 3),
            "p_vs_null": round(p_null, 4),
            "above_null": bool(eci_cm > null_mean),
        }

    # -- Global ECI (pool all regime importances into single per-model avg) ----
    print("\n-- Global ECI --")
    global_fi = {}
    for m in MODELS:
        fi_all = []
        for k, nm in REGIME_MAP.items():
            Xk, _ = _get_regime_data(X, y, regime_arr, k)
            if len(Xk) < 10:
                continue
            mod = _load_model(m, nm)
            fi_all.append(mod.feature_importances_)
        global_fi[m] = np.stack(fi_all).mean(axis=0)

    global_fi_vecs = [global_fi[m] for m in MODELS]
    eci_global = _eci_from_vectors(global_fi_vecs)
    ci_lo_g, ci_hi_g, _ = _boot_ci_eci(global_fi_vecs, B=B_BOOT, rng=rng)
    print(f"Global ECI: {eci_global:.4f}  95% CI [{ci_lo_g:.4f}, {ci_hi_g:.4f}]")

    results["Global"] = {
        "cross_model_eci": round(eci_global, 4),
        "ci_lo": round(ci_lo_g, 4),
        "ci_hi": round(ci_hi_g, 4),
    }

    # -- Wilcoxon test: pairwise regime comparisons ----------------------------
    print("\n-- Wilcoxon pairwise tests (bootstrap ECI distributions) --")
    wilcox = {}
    regime_keys = [nm for nm in REGIME_MAP.values() if nm in regime_boot_ecis]
    for r1, r2 in combinations(regime_keys, 2):
        try:
            stat, p = wilcoxon(regime_boot_ecis[r1], regime_boot_ecis[r2])
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
            print(f"  {r1} vs {r2}: W={stat:.0f}, p={p:.4f} {sig}")
            wilcox[f"{r1}_vs_{r2}"] = {"W": round(float(stat), 1), "p": round(p, 4)}
        except Exception as e:
            print(f"  {r1} vs {r2}: failed ({e})")
    results["wilcoxon_pairwise"] = wilcox

    # -- Summary for paper update ----------------------------------------------
    print("\n-- PAPER SUMMARY --")
    print("Cross-model ECI values to report in Table / text:")
    for nm in ["Low", "Med", "High", "Global"]:
        if nm in results:
            eci = results[nm]["cross_model_eci"]
            if "ci_lo" in results[nm]:
                ci_l = results[nm]["ci_lo"]
                ci_h = results[nm]["ci_hi"]
                print(f"  {nm}: ECI={eci:.3f}  [{ci_l:.3f}, {ci_h:.3f}]")
            else:
                print(f"  {nm}: ECI={eci:.3f}")

    print("\nPreviously reported: Low=0.605, Med=0.735, High=0.650, Global=0.659")
    print("Null model baseline (expected): see null_mean per regime above")

    # Save
    out_path = os.path.join(ARTIFACTS, "eci_validation_results.json")
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nSaved -> {out_path}")
    return results


if __name__ == "__main__":
    run()
