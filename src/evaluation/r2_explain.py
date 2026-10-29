"""
Round-2 revision: ECI and SHAP on the unified labels (S&P 500, 2006-2025).

Per-regime RF / XGBoost / LightGBM experts are fitted on all labelled days of
the regime.  ECI = mean pairwise normalised Kendall agreement of the three
feature-importance rankings (24 features).  Uncertainty comes from B circular
block-bootstrap refits (20-day blocks) within each regime; the null is random
rankings.  SHAP: LightGBM TreeSHAP (pred_contrib), mean |SHAP| aggregated over
each factor and its two lags; we also report how often each factor ranks first
across the bootstrap refits.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from src.evaluation import r2_common as C
from src.evaluation.r2_models import make_rf, make_xgb, make_lgb

B = 200
BLOCK = 20


def eci_from(models):
    imps = [m.feature_importances_ for m in models]
    vals = []
    for i in range(3):
        for j in range(i + 1, 3):
            tau = kendalltau(imps[i], imps[j]).correlation
            vals.append((1 + (0 if np.isnan(tau) else tau)) / 2)
    return float(np.mean(vals))


def fit3(X, y):
    pw = max((1 - y.mean()) / max(y.mean(), 1e-3), 1.0)
    return [make_rf().fit(X, y), make_xgb(pw).fit(X, y), make_lgb().fit(X, y)]


def shap_factor(lgbm, X):
    contrib = lgbm.predict_proba(X, pred_contrib=True) if False else lgbm.booster_.predict(X, pred_contrib=True)
    a = np.abs(contrib[:, :-1]).mean(axis=0)          # 24 features
    return a[0:8] + a[8:16] + a[16:24]


def cbb(n, rng):
    st = rng.integers(0, n, size=int(np.ceil(n / BLOCK)))
    return ((st[:, None] + np.arange(BLOCK)[None, :]).ravel()[:n]) % n


def main():
    d = pd.read_pickle(os.path.join(C.ART, "dataset_SP500.pkl"))
    fc = C.feature_cols(d)
    rng = np.random.default_rng(0)
    out = {}
    null = []
    for _ in range(2000):
        rk = [rng.permutation(24) for _ in range(3)]
        null.append(np.mean([(1 + kendalltau(rk[i], rk[j]).correlation) / 2 for i in range(3) for j in range(i + 1, 3)]))
    null = np.array(null)
    out["null"] = {"mean": float(null.mean()), "ci": [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))], "sd": float(null.std())}
    scopes = {"Global": np.ones(len(d), bool), 0: d.regime.values == 0, 1: d.regime.values == 1, 2: d.regime.values == 2}
    for nm, s in scopes.items():
        X = d.loc[s, fc].values
        y = d.loc[s, "y"].values.astype(int)
        ms = fit3(X, y)
        e0 = eci_from(ms)
        sh0 = shap_factor(ms[2], X)
        eb, shb, top = [], [], []
        for _ in range(B):
            i = cbb(len(y), rng)
            if y[i].sum() < 5:
                continue
            mb = fit3(X[i], y[i])
            eb.append(eci_from(mb))
            sb = shap_factor(mb[2], X[i])
            shb.append(sb)
            top.append(int(np.argmax(sb)))
        eb, shb = np.array(eb), np.array(shb)
        out[str(nm)] = {"n": int(len(y)), "pos": int(y.sum()), "eci": e0,
                        "ci": [float(np.percentile(eb, 2.5)), float(np.percentile(eb, 97.5))],
                        "z": float((e0 - null.mean()) / null.std()),
                        "shap": dict(zip(C.FACTOR_NAMES, sh0.tolist())),
                        "shap_ci": {f: [float(np.percentile(shb[:, j], 2.5)), float(np.percentile(shb[:, j], 97.5))] for j, f in enumerate(C.FACTOR_NAMES)},
                        "top_share": {f: float(np.mean(np.array(top) == j)) for j, f in enumerate(C.FACTOR_NAMES)}}
        o = out[str(nm)]
        print(nm, o["n"], o["pos"], f"ECI {e0:.3f} [{o['ci'][0]:.3f},{o['ci'][1]:.3f}] z={o['z']:.2f}",
              "top", max(o["top_share"], key=o["top_share"].get), round(max(o["top_share"].values()), 2),
              {k: round(v, 3) for k, v in o["shap"].items()}, flush=True)
    json.dump(out, open(os.path.join(C.ART, "explain.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
