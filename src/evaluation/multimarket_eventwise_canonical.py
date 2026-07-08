"""
CANONICAL multi-market event-wise detection evaluation (RC-MoE paper).

This is the single source of truth for the multi-market DETECTION table.
It reuses the exact feature-engineering + model stack from
`multimarket_full_pipeline.py` but adds three things the paper needs:

  1. McNemar test (per market + pooled) on the paired Global/MoE predictions.
  2. A locked, principled anomaly threshold: q = 0.95  (the same 95th-percentile
     tail definition used by the EVT/VaR_95 analysis elsewhere in the paper),
     so the result is fully reproducible from committed code.
  3. A DATED out-of-sample prediction CSV for S&P 500
     (date, y_true, proba_global, pred_global, proba_moe, pred_moe) so the
     publication figures use real, temporally-aligned model output instead of
     last-N-rows guesses.

Primary metric: event-wise F1 (tolerance = 5 windows), identical to the main
S&P 500 model. Point-wise F1 is also reported for transparency.

Run:  .venv/Scripts/python.exe -m src.evaluation.multimarket_eventwise_canonical
Outputs:
  artifacts/multimarket_eventwise_canonical.json
  artifacts/sp500_oos_dated_predictions.csv
"""
import os, json, warnings
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

from src.evaluation.multimarket_full_pipeline import (
    download_and_build_features, build_windows_and_labels, extract_24_factors,
    _ensemble_proba, event_wise_f1, _opt_threshold_event,
    WINDOW, N_FOLDS, GAUSS_SIGMA,
)

warnings.filterwarnings("ignore")

# All three model makers force n_jobs=1. This is REQUIRED on this Windows box:
# with the default n_jobs, sklearn/XGBoost/LightGBM call joblib/loky's physical-
# core counter, which shells out to `wmic` -> WMI is broken here and the call
# hangs forever. Setting n_jobs=1 skips that probe entirely.


def _make_rf():
    return RandomForestClassifier(
        n_estimators=300, max_depth=7, class_weight="balanced",
        min_samples_leaf=2, random_state=42, n_jobs=1,
    )


def _make_xgb(pos_weight: float = 19.0):
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=6, scale_pos_weight=pos_weight,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        eval_metric="aucpr", verbosity=0, random_state=42, n_jobs=1,
    )


def _make_lgb():
    return lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, is_unbalance=True,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        verbose=-1, random_state=42, n_jobs=1,
    )

ARTIFACTS  = "artifacts"
PURGE      = WINDOW
ANOMALY_Q  = 0.95          # locked: 95th-pct tail (== VaR_95 definition in paper)

MARKETS = [
    ("^GSPC",   "SP500",       "2016-01-01"),
    ("^IXIC",   "NASDAQ",      "2016-01-01"),
    ("^RUT",    "Russell2000", "2016-01-01"),
    ("BTC-USD", "Bitcoin",     "2016-01-01"),
]


def mcnemar(true, pred_a, pred_b):
    """McNemar test with continuity correction on paired classifiers.
    b01 = A wrong, B right; b10 = A right, B wrong (vs ground truth)."""
    true = np.asarray(true); a = np.asarray(pred_a); b = np.asarray(pred_b)
    a_correct = (a == true); b_correct = (b == true)
    b01 = int(np.sum(~a_correct & b_correct))   # global wrong, moe right
    b10 = int(np.sum(a_correct & ~b_correct))   # global right, moe wrong
    n = b01 + b10
    if n == 0:
        return {"chi2": 0.0, "p": 1.0, "b01": b01, "b10": b10}
    chi2 = (abs(b01 - b10) - 1) ** 2 / n
    from scipy.stats import chi2 as chi2_dist
    p = float(chi2_dist.sf(chi2, df=1))
    return {"chi2": round(float(chi2), 3), "p": p, "b01": b01, "b10": b10}


def cv_eval_dated(X, y, reg, w_dates, pos_weight=19.0):
    """5-fold walk-forward CV with purge gap. Returns pooled paired predictions
    + per-window dates so callers can build dated CSVs and McNemar tests."""
    n = len(y)
    fold_size = n // N_FOLDS
    pred_g, pred_m, proba_g, proba_m, true_all, dates_all = [], [], [], [], [], []
    fold_records = []

    for fold in range(1, N_FOLDS):
        train_end  = fold * fold_size
        test_start = train_end + PURGE
        test_end   = min(test_start + fold_size, n)
        if test_start >= n or test_end <= test_start:
            continue

        X_tr, y_tr = X[:train_end], y[:train_end]
        X_te, y_te = X[test_start:test_end], y[test_start:test_end]
        r_tr, r_te = reg[:train_end], reg[test_start:test_end]
        d_te       = w_dates[test_start:test_end]

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        if y_tr.sum() < 2:
            continue

        rf = _make_rf(); rf.fit(X_tr_s, y_tr)
        xb = _make_xgb(pos_weight); xb.fit(X_tr_s, y_tr)
        lb = _make_lgb(); lb.fit(X_tr_s, y_tr)
        models = [rf, xb, lb]

        g_proba_tr = _ensemble_proba(models, X_tr_s)
        g_proba_te = _ensemble_proba(models, X_te_s)
        g_thr   = _opt_threshold_event(g_proba_tr, y_tr)
        g_smooth= gaussian_filter1d(g_proba_te, GAUSS_SIGMA)
        g_pred  = (g_smooth >= g_thr).astype(int)

        moe_pred  = np.zeros(len(y_te), dtype=int)
        moe_proba = np.zeros(len(y_te), dtype=float)
        for k in range(3):
            mask_tr = r_tr == k; mask_te = r_te == k
            if mask_te.sum() == 0:
                continue
            if mask_tr.sum() < 10 or y_tr[mask_tr].sum() < 3:
                moe_pred[mask_te]  = g_pred[mask_te]
                moe_proba[mask_te] = g_smooth[mask_te]
                continue
            e_rf = _make_rf(); e_rf.fit(X_tr_s[mask_tr], y_tr[mask_tr])
            e_xb = _make_xgb(pos_weight); e_xb.fit(X_tr_s[mask_tr], y_tr[mask_tr])
            e_lb = _make_lgb(); e_lb.fit(X_tr_s[mask_tr], y_tr[mask_tr])
            ep_tr = _ensemble_proba([e_rf, e_xb, e_lb], X_tr_s[mask_tr])
            ep_te = _ensemble_proba([e_rf, e_xb, e_lb], X_te_s[mask_te])
            e_thr = _opt_threshold_event(ep_tr, y_tr[mask_tr])
            e_smooth = gaussian_filter1d(ep_te, GAUSS_SIGMA)
            moe_proba[mask_te] = e_smooth
            moe_pred[mask_te]  = (e_smooth >= e_thr).astype(int)

        pred_g.extend(g_pred.tolist());   pred_m.extend(moe_pred.tolist())
        proba_g.extend(g_smooth.tolist()); proba_m.extend(moe_proba.tolist())
        true_all.extend(y_te.tolist());    dates_all.extend(list(d_te))

        gf, gp, gr = event_wise_f1(g_pred, y_te)
        mf, mp, mr = event_wise_f1(moe_pred, y_te)
        fold_records.append({"fold": fold, "n_test": len(y_te),
                             "n_pos_test": int(y_te.sum()),
                             "f1_global": round(gf, 4), "f1_moe": round(mf, 4)})
        print(f"    Fold {fold}: Global F1={gf:.3f} | MoE F1={mf:.3f} "
              f"[{int(y_te.sum())} pos / {len(y_te)} win]")

    pg = np.array(pred_g); pm = np.array(pred_m); pt = np.array(true_all)
    gf1, gpr, grc = event_wise_f1(pg, pt)
    mf1, mpr, mrc = event_wise_f1(pm, pt)
    return {
        "n_total": int(len(pt)), "n_anomalies": int(pt.sum()),
        "f1_global_ew": round(gf1, 4), "prec_global_ew": round(gpr, 4),
        "rec_global_ew": round(grc, 4),
        "f1_moe_ew": round(mf1, 4), "prec_moe_ew": round(mpr, 4),
        "rec_moe_ew": round(mrc, 4),
        "f1_global_pw": round(float(f1_score(pt, pg, zero_division=0)), 4),
        "f1_moe_pw": round(float(f1_score(pt, pm, zero_division=0)), 4),
        "mcnemar": mcnemar(pt, pg, pm),
        "folds": fold_records,
        "_pred_g": pg, "_pred_m": pm, "_true": pt,
        "_proba_g": np.array(proba_g), "_proba_m": np.array(proba_m),
        "_dates": np.array(dates_all),
    }


def run():
    os.makedirs(ARTIFACTS, exist_ok=True)
    # patch the module-level ANOMALY_Q used by build_windows_and_labels
    import src.evaluation.multimarket_full_pipeline as mfp
    mfp.ANOMALY_Q = ANOMALY_Q

    results = {}
    pool_g, pool_m, pool_t = [], [], []

    for ticker, mkt, start in MARKETS:
        print(f"\n{'='*60}\n  {mkt} ({ticker})\n{'='*60}")
        feats, log_ret, dates = download_and_build_features(ticker, start)
        X_win, y, reg, w_dates = build_windows_and_labels(
            feats, log_ret, dates, q=ANOMALY_Q)
        print(f"  Windows: {len(y)} | Anomalies: {y.sum()} ({100*y.mean():.1f}%)")
        Xf = extract_24_factors(X_win)
        anom_rate = max(y.mean(), 0.01)
        pos_weight = (1.0 - anom_rate) / anom_rate

        cv = cv_eval_dated(Xf, y, reg, w_dates, pos_weight=pos_weight)
        print(f"  Global EW-F1={cv['f1_global_ew']:.4f}  MoE EW-F1={cv['f1_moe_ew']:.4f}  "
              f"(d={cv['f1_moe_ew']-cv['f1_global_ew']:+.4f})  "
              f"McNemar chi2={cv['mcnemar']['chi2']}")

        results[mkt] = {k: v for k, v in cv.items() if not k.startswith("_")}
        results[mkt].update({"ticker": ticker, "anomaly_rate": round(float(y.mean()), 4),
                             "n_features": int(Xf.shape[1])})

        pool_g.append(cv["_pred_g"]); pool_m.append(cv["_pred_m"]); pool_t.append(cv["_true"])

        # Save dated S&P predictions for figures
        if mkt == "SP500":
            df_dated = pd.DataFrame({
                "date": pd.to_datetime(cv["_dates"]),
                "y_true": cv["_true"],
                "proba_global": cv["_proba_g"], "pred_global": cv["_pred_g"],
                "proba_moe": cv["_proba_m"], "pred_moe": cv["_pred_m"],
            }).sort_values("date")
            csv_p = os.path.join(ARTIFACTS, "sp500_oos_dated_predictions.csv")
            df_dated.to_csv(csv_p, index=False)
            print(f"  [saved dated predictions] {csv_p}  ({len(df_dated)} rows, "
                  f"{df_dated['date'].min().date()}..{df_dated['date'].max().date()})")

    pg = np.concatenate(pool_g); pm = np.concatenate(pool_m); pt = np.concatenate(pool_t)
    gf1, gpr, grc = event_wise_f1(pg, pt)
    mf1, mpr, mrc = event_wise_f1(pm, pt)
    results["pooled"] = {
        "n_total": int(len(pt)), "n_anomalies": int(pt.sum()),
        "f1_global_ew": round(gf1, 4), "prec_global_ew": round(gpr, 4),
        "rec_global_ew": round(grc, 4),
        "f1_moe_ew": round(mf1, 4), "prec_moe_ew": round(mpr, 4),
        "rec_moe_ew": round(mrc, 4),
        "f1_global_pw": round(float(f1_score(pt, pg, zero_division=0)), 4),
        "f1_moe_pw": round(float(f1_score(pt, pm, zero_division=0)), 4),
        "mcnemar": mcnemar(pt, pg, pm),
    }
    print(f"\n{'='*60}\n  POOLED: Global EW-F1={gf1:.4f}  MoE EW-F1={mf1:.4f}  "
          f"McNemar chi2={results['pooled']['mcnemar']['chi2']}\n{'='*60}")

    out = os.path.join(ARTIFACTS, "multimarket_eventwise_canonical.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    run()
