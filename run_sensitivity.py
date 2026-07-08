"""
Sensitivity analysis: window size L x regime count K
Reports in-sample F1 (structural separability) and 5-fold OOS F1
for RF-based RC-MoE. Reference point: L=128, K=3.
All models use n_jobs=1 to avoid WMI hang on Windows.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew
import json, warnings
warnings.filterwarnings('ignore')


# ── Factor computation ──────────────────────────────────────────────────────

def compute_atr_series(highs, lows, closes, window=20):
    """Rolling ATR at the daily level."""
    n = len(highs)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i]  - closes[i - 1]))
    atr = np.full(n, np.nan)
    for i in range(window - 1, n):
        atr[i] = np.mean(tr[i - window + 1:i + 1])
    return atr


def window_factors(opens, highs, lows, closes, atrs):
    """8 core factors for a single window slice."""
    eps = 1e-10
    rets = np.diff(np.log(np.maximum(closes, eps)))

    trend    = float(np.mean(rets))
    momentum = float(np.sum(np.maximum(rets, 0)) / (np.sum(np.abs(rets)) + eps))
    vol      = float(np.std(rets)) if len(rets) > 1 else 0.0
    kurt     = float(sp_kurtosis(rets, fisher=True)) if len(rets) > 3 else 0.0
    sk       = float(sp_skew(rets))                  if len(rets) > 2 else 0.0

    gaps  = (opens[1:] - closes[:-1]) / (np.maximum(closes[:-1], eps))
    ls_a  = float(np.mean(gaps))

    intra = np.maximum(highs - lows, eps)
    ls_b  = float(np.mean((highs - opens) / intra))

    valid_atr = atrs[~np.isnan(atrs)]
    mean_atr  = float(np.mean(valid_atr)) + eps if len(valid_atr) > 0 else eps
    ls_c      = float(np.mean((highs - lows) / mean_atr))

    return np.array([trend, momentum, vol, kurt, sk, ls_a, ls_b, ls_c],
                    dtype=np.float64)


def build_dataset(df, L):
    """Build (X_24dim, y) from OHLCV frame with window size L."""
    opens  = df['Open'].values.astype(np.float64)
    highs  = df['High'].values.astype(np.float64)
    lows   = df['Low'].values.astype(np.float64)
    closes = df['Close'].values.astype(np.float64)
    atrs   = compute_atr_series(highs, lows, closes, window=20)

    eps         = 1e-10
    log_closes  = np.log(np.maximum(closes, eps))
    daily_rets  = np.diff(log_closes)            # length N-1

    N = len(closes)

    # -- compute one 8-factor vector per window ending at day t ---------------
    raw = []
    end_idx = []
    for t in range(L, N):
        f = window_factors(opens[t-L:t], highs[t-L:t], lows[t-L:t],
                           closes[t-L:t], atrs[t-L:t])
        raw.append(f)
        end_idx.append(t)

    raw     = np.array(raw)      # (n_windows, 8)
    end_idx = np.array(end_idx)

    # -- lag augmentation [f_t, f_{t-1}, f_{t-5}] ----------------------------
    X_list, valid_t = [], []
    for i in range(5, len(raw)):
        X_list.append(np.concatenate([raw[i], raw[i-1], raw[i-5]]))
        valid_t.append(end_idx[i])

    X = np.array(X_list, dtype=np.float64)   # (n, 24)
    valid_t = np.array(valid_t)

    # -- rolling expanding-window anomaly labels (98th percentile) ------------
    y = np.zeros(len(valid_t), dtype=int)
    abs_rets = np.abs(daily_rets)
    for j, t in enumerate(valid_t):
        past = abs_rets[:t-1]
        if len(past) < 20:
            continue
        thr = np.quantile(past, 0.98)
        ret_idx = t - 1          # return for day t is daily_rets[t-1]
        if ret_idx < len(daily_rets) and abs_rets[ret_idx] > thr:
            y[j] = 1

    return X, y


# ── Regime assignment ───────────────────────────────────────────────────────

def assign_regimes_from_thresholds(vol, thresholds):
    """Map volatility values to regime indices 0..K-1 given sorted thresholds."""
    r = np.zeros(len(vol), dtype=int)
    for thr in thresholds:
        r[vol > thr] += 1
    return r


# ── In-sample structural separability ──────────────────────────────────────

def insample_f1(X, y, K):
    vol    = X[:, 2]
    thr    = np.quantile(vol, np.linspace(0, 1, K+1)[1:-1])
    r      = assign_regimes_from_thresholds(vol, thr)
    sc     = StandardScaler().fit(X)
    Xs     = sc.transform(X)

    global_rf = RandomForestClassifier(n_estimators=200, max_depth=5,
                                       class_weight='balanced', random_state=42,
                                       n_jobs=1)
    global_rf.fit(Xs, y)
    global_p = global_rf.predict(Xs)

    preds = np.zeros(len(y), dtype=int)
    for k in range(K):
        m = (r == k)
        if m.sum() < 10 or y[m].sum() < 3:
            preds[m] = global_p[m]
        else:
            rf = RandomForestClassifier(n_estimators=200, max_depth=5,
                                        class_weight='balanced', random_state=42,
                                        n_jobs=1)
            rf.fit(Xs[m], y[m])
            preds[m] = rf.predict(Xs[m])

    return float(f1_score(y, preds, zero_division=0))


# ── 5-fold temporal OOS ─────────────────────────────────────────────────────

def oos_cv(X, y, K, n_folds=5, purge=128):
    n         = len(y)
    fold_size = n // n_folds
    all_pred  = np.full(n, -1, dtype=int)

    for fold in range(n_folds - 1):
        tr_end   = (fold + 1) * fold_size
        te_start = tr_end + purge
        te_end   = min(te_start + fold_size, n)
        if te_start >= n:
            break

        X_tr, y_tr = X[:tr_end], y[:tr_end]
        X_te       = X[te_start:te_end]
        if len(X_te) == 0:
            continue

        # Per-fold regime thresholds from training data only
        vol_tr = X_tr[:, 2]
        thr    = np.quantile(vol_tr, np.linspace(0, 1, K+1)[1:-1])
        r_tr   = assign_regimes_from_thresholds(vol_tr, thr)
        r_te   = assign_regimes_from_thresholds(X_te[:, 2], thr)

        sc   = StandardScaler().fit(X_tr)
        Xtr  = sc.transform(X_tr)
        Xte  = sc.transform(X_te)

        global_rf = RandomForestClassifier(n_estimators=200, max_depth=5,
                                           class_weight='balanced', random_state=42,
                                           n_jobs=1)
        global_rf.fit(Xtr, y_tr)
        global_p = global_rf.predict(Xte)

        preds_te = np.zeros(len(X_te), dtype=int)
        for k in range(K):
            m_tr = (r_tr == k)
            m_te = (r_te == k)
            if m_tr.sum() < 10 or y_tr[m_tr].sum() < 3 or m_te.sum() == 0:
                preds_te[m_te] = global_p[m_te]
            else:
                rf = RandomForestClassifier(n_estimators=200, max_depth=5,
                                            class_weight='balanced', random_state=42,
                                            n_jobs=1)
                rf.fit(Xtr[m_tr], y_tr[m_tr])
                preds_te[m_te] = rf.predict(Xte[m_te])

        all_pred[te_start:te_end] = preds_te

    ev = all_pred >= 0
    if ev.sum() == 0:
        return 0.0, 0.0, 0.0

    return (float(f1_score(y[ev], all_pred[ev], zero_division=0)),
            float(precision_score(y[ev], all_pred[ev], zero_division=0)),
            float(recall_score(y[ev], all_pred[ev], zero_division=0)))


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv('artifacts/ohlcv_GSPC.csv', parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    print(f"Data: {len(df)} trading days ({df['Date'].iloc[0].date()} - {df['Date'].iloc[-1].date()})\n")

    window_sizes  = [64, 96, 128, 192]
    regime_counts = [2, 3, 4]
    results       = []

    for L in window_sizes:
        print(f"=== L={L} ===")
        X, y = build_dataset(df, L)
        print(f"  windows={len(y)}, anomalies={y.sum()} ({y.mean()*100:.1f}%)")

        for K in regime_counts:
            print(f"  K={K}  in-sample ...", end='', flush=True)
            is_f1 = insample_f1(X, y, K)
            print(f" {is_f1*100:.1f}%   OOS ...", end='', flush=True)
            oos_f1, oos_pr, oos_rc = oos_cv(X, y, K)
            print(f" F1={oos_f1*100:.1f}%  Prec={oos_pr*100:.1f}%  Rec={oos_rc*100:.1f}%")

            results.append(dict(
                L=L, K=K,
                anom_pct=round(y.mean() * 100, 1),
                IS_F1=round(is_f1 * 100, 1),
                OOS_F1=round(oos_f1 * 100, 1),
                OOS_Prec=round(oos_pr * 100, 1),
                OOS_Rec=round(oos_rc * 100, 1),
            ))

    with open('artifacts/sensitivity_results.json', 'w') as fh:
        json.dump(results, fh, indent=2)

    print("\n\nFinal table:")
    print(f"{'L':>5}  {'K':>3}  {'IS-F1':>7}  {'OOS-F1':>8}  {'OOS-Prec':>10}  {'OOS-Rec':>9}")
    print("-" * 55)
    for r in results:
        star = " *" if r['L'] == 128 and r['K'] == 3 else ""
        print(f"{r['L']:>5}  {r['K']:>3}  {r['IS_F1']:>6.1f}%  {r['OOS_F1']:>7.1f}%  "
              f"{r['OOS_Prec']:>9.1f}%  {r['OOS_Rec']:>8.1f}%{star}")
    print("\n(* = configuration used in the paper)")


if __name__ == '__main__':
    main()
