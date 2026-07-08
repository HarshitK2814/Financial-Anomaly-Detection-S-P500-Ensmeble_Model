"""
Multi-market RC-MoE detection retrain (Task 4).

For each of 4 markets (S&P 500, NASDAQ, Russell 2000, Bitcoin):
  1. Construct 128-day sliding windows from daily returns.
  2. Extract 8 returns-based features + lag-1/lag-5 (24 total).
  3. Label anomalies via rolling expanding-window q=0.98 quantile.
  4. Assign regimes via rolling 252-day volatility terciles.
  5. Run leakage-free 5-fold temporal CV (RF + XGB + LGBM MoE).
  6. Report per-market OOS F1 / precision / recall.
  7. Pool predictions across 4 markets and run McNemar (Global vs MoE LightGBM).

Saves: artifacts/multimarket_detection_results.json

Run from repo root:
  .venv/Scripts/python.exe -m src.evaluation.multimarket_detection
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

WINDOW = 128
REGIME_WIN = 252
N_FOLDS = 5
ARTIFACTS = "artifacts"


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction from returns (returns-computable subset, 8 base features)
# ─────────────────────────────────────────────────────────────────────────────
def _hurst(x):
    """Rough Hurst exponent via R/S analysis."""
    n = len(x)
    if n < 20:
        return 0.5
    rs_vals = []
    for sub_n in [max(4, n // 4), max(4, n // 2), n]:
        m = int(sub_n)
        s = x[-m:]
        mean_s = np.mean(s)
        z = np.cumsum(s - mean_s)
        R = z.max() - z.min()
        S = np.std(s, ddof=1) + 1e-12
        rs_vals.append((m, R / S))
    if len(rs_vals) < 2:
        return 0.5
    ns = np.log([r[0] for r in rs_vals])
    rs = np.log([r[1] for r in rs_vals])
    try:
        return float(np.polyfit(ns, rs, 1)[0])
    except Exception:
        return 0.5


def extract_features_from_window(w: np.ndarray) -> np.ndarray:
    """Extract 8 features from a (WINDOW,) returns vector."""
    r = w.astype(float)
    trend = float(np.mean(r))
    # Momentum: RSI-like: sum(pos_r) / sum(|r|)
    pos_sum = np.sum(r[r > 0])
    abs_sum = np.sum(np.abs(r)) + 1e-12
    momentum = pos_sum / abs_sum
    vol = float(np.std(r, ddof=1))
    kurt = float(scipy_kurtosis(r))
    sk = float(scipy_skew(r))
    mdd_path = np.cumsum(r)
    mdd = float((mdd_path.cummax() - mdd_path).max()) if hasattr(mdd_path, "cummax") else 0.0
    # For numpy:
    cm = np.maximum.accumulate(mdd_path)
    mdd = float(np.max(cm - mdd_path))
    hurst = _hurst(r)
    # 5-day rolling mean as momentum proxy
    roll5 = float(np.mean(r[-5:])) if len(r) >= 5 else trend
    return np.array([trend, momentum, vol, kurt, sk, mdd, hurst, roll5])


def build_dataset(returns: np.ndarray, window: int = WINDOW, regime_win: int = REGIME_WIN):
    """
    Build (X, y, regimes) from a 1D returns array.
    X: (N_windows, 24) — 8 features × 3 time contexts (base, lag1, lag5)
    y: (N_windows,) — binary anomaly labels
    regimes: (N_windows,) — 0/1/2
    """
    n = len(returns)
    # Rolling expanding-window anomaly threshold (q=0.98) — no look-ahead
    abs_r = np.abs(returns)
    thresholds = np.full(n, np.nan)
    for t in range(20, n):
        thresholds[t] = np.quantile(abs_r[:t], 0.98)
    anomaly_daily = (abs_r > thresholds).astype(int)

    # Regime from rolling vol terciles
    vol = pd.Series(returns).rolling(regime_win).std().bfill().values
    q33, q66 = np.quantile(vol, [0.33, 0.66])
    regimes_daily = np.where(vol <= q33, 0, np.where(vol <= q66, 1, 2))

    # Build windows
    base_feats, labels, reg_labels = [], [], []
    for t in range(window, n):
        w = returns[t - window:t]
        feats = extract_features_from_window(w)
        # window-level label: 1 if the last day is anomaly
        lbl = int(anomaly_daily[t - 1])
        reg = int(regimes_daily[t - 1])
        base_feats.append(feats)
        labels.append(lbl)
        reg_labels.append(reg)

    B = np.array(base_feats)  # (N, 8)
    y = np.array(labels)
    reg_arr = np.array(reg_labels)

    # Add lag-1 and lag-5 (shift rows)
    B1 = np.vstack([B[:1], B[:-1]])      # lag-1
    B5 = np.vstack([B[:5], B[:-5]])      # lag-5
    X = np.hstack([B, B1, B5])           # (N, 24)
    return X, y, reg_arr


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────
def make_model(mt: str):
    if mt == "RF":
        return RandomForestClassifier(
            n_estimators=200, max_depth=5, class_weight="balanced", random_state=42
        )
    elif mt == "XGBoost":
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators=200, max_depth=5, scale_pos_weight=10,
            learning_rate=0.05, eval_metric="aucpr",
            verbosity=0, random_state=42,
        )
    elif mt == "LightGBM":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, is_unbalance=True,
            learning_rate=0.05, verbose=-1, random_state=42,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Leakage-free 5-fold CV: Global vs MoE per model type
# ─────────────────────────────────────────────────────────────────────────────
def cv_eval(X: np.ndarray, y: np.ndarray, reg_arr: np.ndarray,
            mt: str = "LightGBM") -> dict:
    """5-fold temporal CV: return per-fold predictions for Global + MoE."""
    n = len(y)
    fold_size = n // N_FOLDS
    all_preds_global, all_preds_moe, all_true = [], [], []

    for fold in range(1, N_FOLDS):
        train_end = fold * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)

        X_tr, y_tr = X[:train_end], y[:train_end]
        X_te, y_te = X[test_start:test_end], y[test_start:test_end]
        reg_tr = reg_arr[:train_end]
        reg_te = reg_arr[test_start:test_end]

        # Per-fold normalization
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # ── Global model ──────────────────────────────────────────────
        gm = make_model(mt)
        if y_tr.sum() == 0:
            all_preds_global.extend([0] * len(y_te))
            all_preds_moe.extend([0] * len(y_te))
            all_true.extend(y_te.tolist())
            continue
        gm.fit(X_tr_s, y_tr)
        g_proba = gm.predict_proba(X_te_s)[:, 1]

        # F1-optimal threshold on training fold
        tr_proba = gm.predict_proba(X_tr_s)[:, 1]
        best_t, best_f1 = 0.5, 0.0
        for thr in np.linspace(0.01, 0.99, 200):
            f1 = f1_score(y_tr, (tr_proba >= thr).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, thr
        g_pred = (g_proba >= best_t).astype(int)

        # ── MoE model ─────────────────────────────────────────────────
        moe_pred = np.zeros(len(y_te), dtype=int)
        for k in range(3):
            mask_tr = reg_tr == k
            mask_te = reg_te == k
            if mask_tr.sum() < 5 or y_tr[mask_tr].sum() == 0:
                continue
            em = make_model(mt)
            em.fit(X_tr_s[mask_tr], y_tr[mask_tr])
            if mask_te.sum() == 0:
                continue
            ep = em.predict_proba(X_te_s[mask_te])[:, 1]
            # Threshold from training regime
            tr_ep = em.predict_proba(X_tr_s[mask_tr])[:, 1]
            best_et, best_ef1 = 0.5, 0.0
            for thr in np.linspace(0.01, 0.99, 200):
                ef1 = f1_score(y_tr[mask_tr], (tr_ep >= thr).astype(int),
                               zero_division=0)
                if ef1 > best_ef1:
                    best_ef1, best_et = ef1, thr
            moe_pred[mask_te] = (ep >= best_et).astype(int)

        all_preds_global.extend(g_pred.tolist())
        all_preds_moe.extend(moe_pred.tolist())
        all_true.extend(y_te.tolist())

    pg = np.array(all_preds_global)
    pm = np.array(all_preds_moe)
    pt = np.array(all_true)
    return {
        "true": pt, "pred_global": pg, "pred_moe": pm,
        "f1_global":   round(float(f1_score(pt, pg, zero_division=0)), 4),
        "f1_moe":      round(float(f1_score(pt, pm, zero_division=0)), 4),
        "prec_global": round(float(precision_score(pt, pg, zero_division=0)), 4),
        "prec_moe":    round(float(precision_score(pt, pm, zero_division=0)), 4),
        "rec_global":  round(float(recall_score(pt, pg, zero_division=0)), 4),
        "rec_moe":     round(float(recall_score(pt, pm, zero_division=0)), 4),
        "n_anomalies": int(pt.sum()),
        "n_total":     int(len(pt)),
    }


def mcnemar_test(pred_a: np.ndarray, pred_b: np.ndarray,
                 true: np.ndarray) -> dict:
    """McNemar chi2 test: is b10 (B correct, A wrong) > b01 (A correct, B wrong)?"""
    from scipy.stats import chi2
    b01 = int(((pred_a == true) & (pred_b != true)).sum())  # A correct, B wrong
    b10 = int(((pred_a != true) & (pred_b == true)).sum())  # B correct, A wrong
    n = b01 + b10
    if n < 1:
        return {"chi2": 0.0, "p": 1.0, "b01": b01, "b10": b10}
    chi2_stat = (abs(b10 - b01) - 1) ** 2 / n
    p = float(1 - chi2.cdf(chi2_stat, df=1))
    return {"chi2": round(chi2_stat, 3), "p": round(p, 4), "b01": b01, "b10": b10}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
MARKETS = {
    "SP500":       ("sp500_full_daily_returns.csv", "^GSPC",    "2016-01-01"),
    "NASDAQ":      ("returns_NASDAQ.csv",           "^IXIC",    "2016-01-01"),
    "Russell2000": ("returns_Russell2000.csv",      "^RUT",     "2016-01-01"),
    "Bitcoin":     ("returns_Bitcoin.csv",          "BTC-USD",  "2016-01-01"),
}


def run():
    results = {}
    all_true_pool, all_global_pool, all_moe_pool = [], [], []

    for mkt, (fname, col, start) in MARKETS.items():
        print(f"\n=== {mkt} ===")
        path = os.path.join(ARTIFACTS, fname)
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df[df["Date"] >= start].sort_values("Date").reset_index(drop=True)
        returns = df[col].values.astype(float)
        # Remove NaN/Inf
        returns = np.where(np.isfinite(returns), returns, 0.0)

        X, y, reg_arr = build_dataset(returns)
        print(f"  Windows: {len(y)}, Anomalies: {y.sum()} ({100*y.mean():.1f}%)")
        print(f"  Regimes: Low={int((reg_arr==0).sum())}, "
              f"Med={int((reg_arr==1).sum())}, High={int((reg_arr==2).sum())}")

        cv = cv_eval(X, y, reg_arr, mt="LightGBM")
        print(f"  LightGBM Global: F1={cv['f1_global']:.3f}, "
              f"Prec={cv['prec_global']:.3f}, Rec={cv['rec_global']:.3f}")
        print(f"  LightGBM MoE:    F1={cv['f1_moe']:.3f}, "
              f"Prec={cv['prec_moe']:.3f}, Rec={cv['rec_moe']:.3f}")

        mc = mcnemar_test(cv["pred_global"], cv["pred_moe"], cv["true"])
        print(f"  McNemar: chi2={mc['chi2']:.2f}, p={mc['p']:.4f}, "
              f"b01={mc['b01']}, b10={mc['b10']}")

        results[mkt] = {
            "n_windows": int(len(y)),
            "n_anomalies": cv["n_anomalies"],
            "f1_global":   cv["f1_global"],
            "prec_global": cv["prec_global"],
            "rec_global":  cv["rec_global"],
            "f1_moe":      cv["f1_moe"],
            "prec_moe":    cv["prec_moe"],
            "rec_moe":     cv["rec_moe"],
            "mcnemar":     mc,
        }
        all_true_pool.extend(cv["true"].tolist())
        all_global_pool.extend(cv["pred_global"].tolist())
        all_moe_pool.extend(cv["pred_moe"].tolist())

    # Pooled McNemar across all 4 markets
    pt = np.array(all_true_pool)
    pg = np.array(all_global_pool)
    pm = np.array(all_moe_pool)
    pooled_mc = mcnemar_test(pg, pm, pt)
    print(f"\n=== POOLED (4 markets) ===")
    print(f"  n={len(pt)}, anomalies={pt.sum()}")
    print(f"  Pooled McNemar: chi2={pooled_mc['chi2']:.2f}, p={pooled_mc['p']:.4f}, "
          f"b01={pooled_mc['b01']}, b10={pooled_mc['b10']}")
    print(f"  Pooled F1 Global: {f1_score(pt, pg, zero_division=0):.3f}")
    print(f"  Pooled F1 MoE:    {f1_score(pt, pm, zero_division=0):.3f}")
    results["pooled"] = {
        "n_total": int(len(pt)),
        "n_anomalies": int(pt.sum()),
        "f1_global": round(float(f1_score(pt, pg, zero_division=0)), 4),
        "f1_moe":    round(float(f1_score(pt, pm, zero_division=0)), 4),
        "mcnemar": pooled_mc,
    }

    out = os.path.join(ARTIFACTS, "multimarket_detection_results.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved -> {out}")
    return results


if __name__ == "__main__":
    run()
