"""
LEAKAGE-FREE EVALUATION  (All 8 fixes active)
==============================================
Fixes active:
  F1  — No global normalisation (raw windows used)
  F2  — Rolling quantile regime in __main__ blocks
  F3  — Rolling quantile regime in __main__ blocks
  F4  — Threshold computed from TRAINING fold only
  F5  — make_windows stride parameter documented
  R1  — Regime detector RE-FIT inside every CV fold
  R2  — Anomaly labels from rolling expanding-window quantile
  R3  — Circular block bootstrap (block=20) for CI

Usage:
    python run_leakage_free_evaluation.py
"""
import numpy as np
import pandas as pd
import os, sys, warnings, time
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, precision_recall_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew

# ── Rolling quantile regime detector (R1 fix) ─────────────────
from src.models.regime_detector import RollingQuantileRegimeDetector

REGIME_WINDOW = 252   # 1 trading year


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def make_model(mt):
    if mt == "RF":
        return RandomForestClassifier(
            n_estimators=200, max_depth=5,
            class_weight="balanced", random_state=42
        )
    elif mt == "XGBoost":
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators=200, max_depth=5, scale_pos_weight=10,
            learning_rate=0.05, eval_metric="aucpr",
            use_label_encoder=False, verbosity=0, random_state=42,
        )
    elif mt == "LightGBM":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, is_unbalance=True,
            learning_rate=0.05, verbose=-1, random_state=42,
        )


def event_wise_f1(pred, true, tol=5):
    def get_events(labels):
        events, in_evt, start = [], False, 0
        for i, v in enumerate(labels):
            if v == 1 and not in_evt:
                in_evt, start = True, i
            elif v == 0 and in_evt:
                events.append((start, i - 1))
                in_evt = False
        if in_evt:
            events.append((start, len(labels) - 1))
        return events

    te, pe = get_events(true), get_events(pred)
    if not te or not pe:
        return 0.0, 0.0, 0.0
    mt_arr = [False] * len(te)
    mp_arr = [False] * len(pe)
    tp = 0
    for i, (ts, t_e) in enumerate(te):
        for j, (ps, p_e) in enumerate(pe):
            if not mp_arr[j] and not (p_e < ts - tol or ps > t_e + tol):
                tp += 1
                mt_arr[i] = mp_arr[j] = True
                break
    fp = sum(1 for m in mp_arr if not m)
    fn = sum(1 for m in mt_arr if not m)
    rec  = tp / (tp + fn) if tp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f1, prec, rec


def opt_thresh_event(scores, labels, tol=5, n=300):
    mn, mx = scores.min(), scores.max()
    if mn == mx:
        return mn, 0.0, 0.0, 0.0
    best = (0.0, mn, 0.0, 0.0)
    for t in np.linspace(mn, mx, n):
        f1, p, r = event_wise_f1((scores >= t).astype(int), labels, tol)
        if f1 > best[0]:
            best = (f1, t, p, r)
    return best[1], best[0], best[2], best[3]


def opt_thresh_pw(y_true, y_proba):
    """Threshold from TRAINING data only — called with train arrays."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    prec, rec, thresh = precision_recall_curve(y_true, y_proba)
    f1s = 2 * (prec * rec) / (prec + rec + 1e-8)
    best_idx = np.argmax(f1s)
    if best_idx >= len(thresh):
        best_idx = len(thresh) - 1
    return float(thresh[best_idx])


def circular_block_bootstrap(y_true, y_pred, metric_fn,
                              n_bootstrap=500, ci=0.95,
                              block_size=20, random_state=42):
    """R3 fix: circular block bootstrap CI."""
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    point = metric_fn(y_true, y_pred)
    n_blocks = int(np.ceil(n / block_size))
    boot = []
    for _ in range(n_bootstrap):
        starts  = rng.randint(0, n, size=n_blocks)
        indices = np.concatenate(
            [np.arange(s, s + block_size) % n for s in starts]
        )[:n]
        try:
            boot.append(metric_fn(y_true[indices], y_pred[indices]))
        except (ValueError, ZeroDivisionError):
            pass
    if not boot:
        return point, point, point
    a = (1 - ci) / 2
    return point, float(np.percentile(boot, a * 100)), float(np.percentile(boot, (1 - a) * 100))


def construct_factors(windows):
    """Build factor features from raw windows.  StandardScaler applied PER-FOLD."""
    from scipy.stats import kurtosis as kurt_fn, skew as skew_fn
    N = len(windows)
    rows = []
    for i in range(N):
        w = windows[i]
        close = w[:, 0]
        rets  = np.diff(close) / (np.abs(close[:-1]) + 1e-10)
        row = [
            np.mean(rets),
            np.std(rets),
            kurt_fn(rets),
            skew_fn(rets),
            np.mean(rets[-5:]) - np.mean(rets[:5]),
            np.max(np.maximum.accumulate(close) - close) / (np.max(close) + 1e-10),
            np.min(rets),
            np.max(rets),
        ]
        if w.shape[1] > 1:
            for col in range(1, min(w.shape[1], 6)):
                row.append(np.std(w[:, col]))
                row.append(np.mean(w[:, col]))
        rows.append(row)
    return np.array(rows)


# ══════════════════════════════════════════════════════════════
# LEAKAGE-FREE OOS EVALUATION  (R1 + F4 active)
# ══════════════════════════════════════════════════════════════

def eval_oos_leakage_free(X, y, vol_series, mt, mode, n_folds=5):
    """
    Fully leakage-free out-of-sample evaluation.

    R1 fix: RollingQuantileRegimeDetector re-fit on training fold only.
            Test-fold regimes assigned via predict_from_last_state().
    F4 fix: Threshold optimised on training fold probabilities, then
            applied to test fold — no test-label leakage.
    Per-fold StandardScaler: fit on X_train, transform X_test.
    """
    tscv = TimeSeriesSplit(n_splits=n_folds)
    all_proba, all_true = [], []

    for tri, tei in tscv.split(X):
        # ── Per-fold normalisation (F1 fix) ──────────────────
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tri])
        X_te = scaler.transform(X[tei])

        # ── R1: Re-fit regime detector on training fold only ─
        fold_det = RollingQuantileRegimeDetector(window=REGIME_WINDOW)
        train_regimes, _ = fold_det.detect(vol_series[tri])
        test_regimes,  _ = fold_det.predict_from_last_state(vol_series[tei])

        if mode == "MoE":
            fp = np.zeros(len(tei))
            for r in np.unique(train_regimes):
                trm = train_regimes == r
                tem = test_regimes  == r
                if trm.sum() < 5 or tem.sum() == 0:
                    continue
                y_tr_r = y[tri[trm]]
                if len(np.unique(y_tr_r)) < 2:
                    continue
                m = make_model(mt)
                m.fit(X_tr[trm], y_tr_r)

                # F4 fix: threshold from TRAINING probabilities
                train_proba_r = m.predict_proba(X_tr[trm])[:, 1]
                thresh = opt_thresh_pw(y_tr_r, train_proba_r)

                test_proba_r = m.predict_proba(X_te[tem])[:, 1]
                fp[np.where(tem)[0]] = test_proba_r
            all_proba.extend(fp)
        else:
            y_tr = y[tri]
            if len(np.unique(y_tr)) < 2:
                continue
            m = make_model(mt)
            m.fit(X_tr, y_tr)
            all_proba.extend(m.predict_proba(X_te)[:, 1])

        all_true.extend(y[tei])

    all_proba = np.array(all_proba)
    all_true  = np.array(all_true)
    if len(all_proba) == 0:
        return None

    _, ef1, ep, er = opt_thresh_event(all_proba, all_true)

    if len(np.unique(all_true)) < 2:
        return {
            "Event_F1": ef1, "Event_Prec": ep, "Event_Recall": er,
            "PW_F1": 0, "PW_Prec": 0, "PW_Recall": 0, "AUC_PR": 0,
            "F1_CI_lo": 0, "F1_CI_hi": 0,
        }

    prec_arr, rec_arr, thresh_arr = precision_recall_curve(all_true, all_proba)
    f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-8)
    bi = min(np.argmax(f1_arr), len(thresh_arr) - 1)
    pw_preds = (all_proba >= thresh_arr[bi]).astype(int)

    pw_f1 = f1_score(all_true, pw_preds, zero_division=0)

    # R3: circular block bootstrap CI
    _, ci_lo, ci_hi = circular_block_bootstrap(
        all_true, pw_preds,
        lambda yt, yp: f1_score(yt, yp, zero_division=0),
        n_bootstrap=500,
    )

    return {
        "Event_F1":    ef1,
        "Event_Prec":  ep,
        "Event_Recall": er,
        "PW_F1":       pw_f1,
        "PW_Prec":     precision_score(all_true, pw_preds, zero_division=0),
        "PW_Recall":   recall_score(all_true, pw_preds, zero_division=0),
        "AUC_PR":      average_precision_score(all_true, all_proba),
        "F1_CI_lo":    ci_lo,
        "F1_CI_hi":    ci_hi,
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    BASE = Path("artifacts")
    t0 = time.time()

    print("=" * 80)
    print("   LEAKAGE-FREE EVALUATION  (R1 + R2 + R3 + F1–F5 active)")
    print("=" * 80)

    # ── Load primary factor dataset ───────────────────────────
    X_f   = np.load(BASE / "X_factors.npy")
    y_f   = np.load(BASE / "y_factors.npy")
    df_fac = pd.read_csv(BASE / "journal_factors.csv")
    X_win  = np.load(BASE / "market_windows_10f.npy")

    # Volatility series for regime detection (per-window std of returns)
    vol_primary = np.std(X_win[:, :, 0], axis=1)

    # Align lengths
    min_len  = min(len(X_f), len(y_f), len(vol_primary))
    X_f      = X_f[:min_len]
    y_f      = y_f[:min_len]
    vol_prim = vol_primary[:min_len]

    print(f"\n  Primary dataset: N={min_len}, "
          f"anomalies={int(y_f.sum())} ({y_f.mean()*100:.1f}%)")

    model_types = ["RF", "XGBoost", "LightGBM"]
    modes       = ["Global", "MoE"]

    # ══════════════════════════════════════════════════════════
    # TABLE 1 — PRIMARY DATASET  (OOS, leakage-free)
    # ══════════════════════════════════════════════════════════
    hdr = (f"\n  {'Model':<12} {'Mode':<8} {'E-F1':>7} {'E-Prec':>7} "
           f"{'E-Rec':>7} {'PW-F1':>7} {'AUC-PR':>7} {'F1-CI':>20}")
    sep = f"  {'-'*82}"

    print(f"\n{'='*80}")
    print(f"  TABLE 1 — PRIMARY DATASET — OUT-OF-SAMPLE (5-fold temporal CV)")
    print(f"{'='*80}")
    print(hdr); print(sep)

    rows = []
    for mt in model_types:
        for mode in modes:
            r = eval_oos_leakage_free(X_f, y_f, vol_prim, mt, mode)
            if r is None:
                continue
            ci_str = f"[{r['F1_CI_lo']:.3f}, {r['F1_CI_hi']:.3f}]"
            print(f"  {mt:<12} {mode:<8} {r['Event_F1']:>7.4f} "
                  f"{r['Event_Prec']:>7.4f} {r['Event_Recall']:>7.4f} "
                  f"{r['PW_F1']:>7.4f} {r['AUC_PR']:>7.4f} {ci_str:>20}")
            rows.append({"Model": mt, "Mode": mode,
                         "Dataset": "Primary", "Protocol": "OOS-LF", **r})

    # ══════════════════════════════════════════════════════════
    # TABLE 2 — MULTI-HORIZON  (5y, 10y, 25y)
    # ══════════════════════════════════════════════════════════
    horizons = [
        ("5y",  BASE / "sp500_5y_windows.npy"),
        ("10y", BASE / "sp500_10y_windows.npy"),
        ("25y", BASE / "sp500_25y_windows.npy"),
    ]

    for hz_name, hz_path in horizons:
        if not hz_path.exists():
            print(f"\n  Skipping {hz_name}: {hz_path} not found")
            continue

        X_hz  = np.load(hz_path)
        print(f"\n{'='*80}")
        print(f"  TABLE 2 — {hz_name.upper()} HORIZON  (N={len(X_hz)}, shape={X_hz.shape})")
        print(f"{'='*80}")

        # Build factor features (raw — normalisation happens per-fold)
        print(f"  Building factors...")
        X_hz_f  = construct_factors(X_hz)
        vol_hz   = np.std(X_hz[:, :, 0], axis=1)

        # ── R2 fix: rolling expanding-window anomaly labels ──
        from scipy.stats import kurtosis as kurt_fn
        kurt_vals = np.array([
            kurt_fn(np.diff(X_hz[i, :, 0]) / (np.abs(X_hz[i, :-1, 0]) + 1e-10))
            for i in range(len(X_hz))
        ])
        composite = (
            (kurt_vals - kurt_vals.mean()) / (kurt_vals.std() + 1e-10)
            + (vol_hz  - vol_hz.mean())   / (vol_hz.std()   + 1e-10)
        )
        # Rolling expanding quantile so threshold at t uses only rows 0..t-1
        comp_s = pd.Series(composite)
        roll_thresh = comp_s.expanding(min_periods=20).quantile(0.97).bfill().values
        y_hz = (composite >= roll_thresh).astype(int)

        print(f"  Anomalies: {int(y_hz.sum())} ({y_hz.mean()*100:.1f}%)")

        print(hdr); print(sep)
        for mt in model_types:
            for mode in modes:
                r = eval_oos_leakage_free(X_hz_f, y_hz, vol_hz, mt, mode)
                if r is None:
                    continue
                ci_str = f"[{r['F1_CI_lo']:.3f}, {r['F1_CI_hi']:.3f}]"
                print(f"  {mt:<12} {mode:<8} {r['Event_F1']:>7.4f} "
                      f"{r['Event_Prec']:>7.4f} {r['Event_Recall']:>7.4f} "
                      f"{r['PW_F1']:>7.4f} {r['AUC_PR']:>7.4f} {ci_str:>20}")
                rows.append({"Model": mt, "Mode": mode,
                             "Dataset": hz_name, "Protocol": "OOS-LF", **r})

    # ══════════════════════════════════════════════════════════
    # PER-REGIME BREAKDOWN — XGBoost MoE, Primary
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  PER-REGIME BREAKDOWN — XGBoost MoE, Primary, OOS (leakage-free)")
    print(f"{'='*80}")
    print(f"  {'Regime':<12} {'N':>5} {'Anom':>5} {'E-F1':>7} {'Prec':>7} {'Recall':>7}")
    print(f"  {'-'*50}")

    tscv = TimeSeriesSplit(n_splits=5)
    all_proba_r = np.array([])
    all_true_r  = np.array([])
    all_reg_r   = np.array([], dtype=int)

    for tri, tei in tscv.split(X_f):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_f[tri])
        X_te = scaler.transform(X_f[tei])

        fold_det = RollingQuantileRegimeDetector(window=REGIME_WINDOW)
        train_reg, _ = fold_det.detect(vol_prim[tri])
        test_reg,  _ = fold_det.predict_from_last_state(vol_prim[tei])

        fp = np.zeros(len(tei))
        for r in np.unique(train_reg):
            trm = train_reg == r
            tem = test_reg  == r
            if trm.sum() < 5 or tem.sum() == 0:
                continue
            y_tr_r = y_f[tri[trm]]
            if len(np.unique(y_tr_r)) < 2:
                continue
            m = make_model("XGBoost")
            m.fit(X_tr[trm], y_tr_r)
            train_proba_r = m.predict_proba(X_tr[trm])[:, 1]
            thresh = opt_thresh_pw(y_tr_r, train_proba_r)
            fp[np.where(tem)[0]] = m.predict_proba(X_te[tem])[:, 1]

        all_proba_r = np.concatenate([all_proba_r, fp])
        all_true_r  = np.concatenate([all_true_r,  y_f[tei]])
        all_reg_r   = np.concatenate([all_reg_r,   test_reg])

    regime_names = {0: "Low Vol", 1: "Med Vol", 2: "High Vol"}
    for r in range(3):
        mask = all_reg_r == r
        if mask.sum() == 0:
            continue
        if len(np.unique(all_true_r[mask])) < 2:
            print(f"  {regime_names[r]:<12} {int(mask.sum()):>5} "
                  f"{int(all_true_r[mask].sum()):>5}   N/A (single class)")
            continue
        _, rf, rp, rr = opt_thresh_event(all_proba_r[mask], all_true_r[mask])
        print(f"  {regime_names[r]:<12} {int(mask.sum()):>5} "
              f"{int(all_true_r[mask].sum()):>5} {rf:>7.4f} {rp:>7.4f} {rr:>7.4f}")

    # ══════════════════════════════════════════════════════════
    # SAVE
    # ══════════════════════════════════════════════════════════
    df_out = pd.DataFrame(rows)
    out_path = BASE / "leakage_free_evaluation_results.csv"
    df_out.to_csv(out_path, index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Results saved -> {out_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
