"""
Out-of-sample L x K sensitivity with the FULL tree ensemble (Reviewer 1, Query 4).

Same data, labels, folds, 128-day purge and per-fold regime thresholds as
run_sensitivity.py; the only change is the expert: RF + XGBoost + LightGBM
probability average per regime (instead of RF alone), with the fixed
decision threshold tau = 0.50 of Algorithm 1 (no threshold tuning, so no
test information enters the decision rule).

Also reports BIC/AIC of 2-, 3- and 4-state Gaussian HMMs fitted to the
log window volatility (theoretical support for K = 3).

Run from model root:  python -u src/evaluation/sensitivity_full_ensemble.py
All estimators use n_jobs=1 (WMI hang on this machine).
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_sensitivity import build_dataset, assign_regimes_from_thresholds

warnings.filterwarnings("ignore")
TAU = 0.50


def ensemble(X, y):
    pos = max(int(y.sum()), 1)
    spw = (len(y) - pos) / pos
    models = [
        RandomForestClassifier(n_estimators=200, max_depth=5, class_weight="balanced",
                               random_state=42, n_jobs=1),
        xgb.XGBClassifier(n_estimators=200, max_depth=5, reg_alpha=0.1, reg_lambda=1.0,
                          scale_pos_weight=spw, eval_metric="logloss", random_state=42,
                          n_jobs=1, verbosity=0),
        lgb.LGBMClassifier(n_estimators=200, num_leaves=31, is_unbalance=True,
                           random_state=42, n_jobs=1, verbose=-1),
    ]
    for m in models:
        m.fit(X, y)
    return lambda Z: np.mean([m.predict_proba(Z)[:, 1] for m in models], axis=0)


def oos_cv_ensemble(X, y, K, n_folds=5, purge=128):
    n = len(y)
    fold = n // n_folds
    pred = np.full(n, -1, dtype=int)
    for f in range(n_folds - 1):
        tr_end = (f + 1) * fold
        te_start, te_end = tr_end + purge, min(tr_end + purge + fold, n)
        if te_start >= n:
            break
        X_tr, y_tr, X_te = X[:tr_end], y[:tr_end], X[te_start:te_end]
        thr = np.quantile(X_tr[:, 2], np.linspace(0, 1, K + 1)[1:-1])
        r_tr = assign_regimes_from_thresholds(X_tr[:, 2], thr)
        r_te = assign_regimes_from_thresholds(X_te[:, 2], thr)
        sc = StandardScaler().fit(X_tr)
        Xtr, Xte = sc.transform(X_tr), sc.transform(X_te)

        g = ensemble(Xtr, y_tr)
        out = (g(Xte) >= TAU).astype(int)
        for k in range(K):
            m_tr, m_te = r_tr == k, r_te == k
            if m_tr.sum() < 10 or y_tr[m_tr].sum() < 3 or m_te.sum() == 0:
                continue  # sparse regime: keep global prediction
            e = ensemble(Xtr[m_tr], y_tr[m_tr])
            out[m_te] = (e(Xte[m_te]) >= TAU).astype(int)
        pred[te_start:te_end] = out
    ev = pred >= 0
    return (f1_score(y[ev], pred[ev], zero_division=0),
            precision_score(y[ev], pred[ev], zero_division=0),
            recall_score(y[ev], pred[ev], zero_division=0))


def hmm_bic(vol):
    from hmmlearn.hmm import GaussianHMM
    x = np.log(vol).reshape(-1, 1)
    rows = []
    for K in (2, 3, 4):
        best = None
        for seed in range(5):
            m = GaussianHMM(n_components=K, covariance_type="full", n_iter=500, random_state=seed)
            m.fit(x)
            ll = m.score(x)
            if best is None or ll > best:
                best = ll
        p = K * (K - 1) + (K - 1) + 2 * K      # transitions + initial + means + variances
        rows.append(dict(K=K, logL=best, n_params=p,
                         AIC=-2 * best + 2 * p, BIC=-2 * best + p * np.log(len(x))))
    return rows


def main():
    df = pd.read_csv("artifacts/ohlcv_GSPC.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    res = []
    for L in (64, 96, 128, 192):
        X, y = build_dataset(df, L)
        for K in (2, 3, 4):
            f1, pr, rc = oos_cv_ensemble(X, y, K)
            res.append(dict(L=L, K=K, OOS_F1=round(100 * f1, 1), OOS_Prec=round(100 * pr, 1),
                            OOS_Rec=round(100 * rc, 1)))
            print(res[-1], flush=True)
        if L == 128:
            bic = hmm_bic(X[:, 2])
            for r in bic:
                print("HMM", r, flush=True)
    out = dict(ensemble_oos=res, hmm_L128=bic)
    Path("artifacts/sensitivity_full_ensemble.json").write_text(json.dumps(out, indent=2))
    print("saved artifacts/sensitivity_full_ensemble.json")


if __name__ == "__main__":
    main()
