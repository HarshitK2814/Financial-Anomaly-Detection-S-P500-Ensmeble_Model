"""
Path A: Full RC-MoE Multi-Market Pipeline

Implements the SAME feature engineering as the main S&P 500 paper model:
  calculate_technical_indicators  ->  (N, 128, 10) windows
  ->  8 base factors + lag-1 + lag-5  =  24 features  (matches X_factors.npy)

Performance boosters ON TOP of the base pipeline:
  * RF + XGBoost + LightGBM probability averaging  (3-model ensemble)
  * Gaussian smoothing (sigma=1.0) on ensemble scores
  * Event-wise F1 with tolerance=5  (same metric the main model uses)
  * q=0.95 anomaly threshold  (5 percent base rate -> richer training signal)
  * 128-day purge gap in walk-forward CV  (eliminates overlapping-window leakage)
  * F1-optimal threshold tuned on TRAINING fold only  (honest OOS)
  * Regime fallback to Global when expert has < 3 positives

Markets: S&P 500 (^GSPC), NASDAQ (^IXIC), Russell 2000 (^RUT), Bitcoin (BTC-USD)
Period : 2016-01-01 to 2025-12-31

Outputs:
  artifacts/multimarket_path_a_results.json
  correct figures/fig_multimarket_path_a.png

Run from repo root:
  .venv/Scripts/python.exe -m src.evaluation.multimarket_full_pipeline
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew
from scipy.ndimage import gaussian_filter1d
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import yfinance as yf

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

WINDOW       = 128
REGIME_WIN   = 252
N_FOLDS      = 5
PURGE        = WINDOW          # purge gap = window size to eliminate lookahead
ANOMALY_Q    = 0.90            # top-10% extreme returns labelled anomalous (more training signal)
GAUSS_SIGMA  = 1.0
EVENT_TOL    = 5               # event-wise F1 tolerance (windows)
FBETA        = 2.0             # F-beta for threshold opt (beta>1 rewards recall over precision)
ARTIFACTS    = "artifacts"
FIGURES      = "correct figures"

MARKETS = [
    ("^GSPC",   "SP500",       "2016-01-01"),
    ("^IXIC",   "NASDAQ",      "2016-01-01"),
    ("^RUT",    "Russell2000", "2016-01-01"),
    ("BTC-USD", "Bitcoin",     "2016-01-01"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — data download + 10-feature engineering (same as fetch_yfinance_data)
# ─────────────────────────────────────────────────────────────────────────────

def _calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirror of fetch_yfinance_data.calculate_technical_indicators.
    Returns DataFrame with 10 features (same channel order as market_windows_10f.npy).

    Ch0  Log_Ret   — log daily return
    Ch1  RSI       — 14-day RSI normalized 0-1
    Ch2  Vol_20    — 20-day realized vol
    Ch3  Vol_Change— log volume change
    Ch4  HL_Range  — (H-L)/C intraday range
    Ch5  MACD      — EMA12 - EMA26
    Ch6  BB_Pos    — Bollinger band position
    Ch7  Skew_20   — 20-day rolling skewness
    Ch8  Kurt_20   — 20-day rolling kurtosis
    Ch9  Lag_Ret   — t-1 log return
    """
    close = df["Close"]
    vol   = df.get("Volume", pd.Series(np.ones(len(df)), index=df.index))

    out = pd.DataFrame(index=df.index)

    # Ch0: log return
    out["Log_Ret"] = np.log(close / close.shift(1))

    # Ch1: RSI (14)
    delta = close.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs    = gain / (loss + 1e-8)
    out["RSI"] = (100 - (100 / (1 + rs))) / 100.0

    # Ch2: 20-day realized vol
    out["Vol_20"] = out["Log_Ret"].rolling(20).std()

    # Ch3: log volume change
    out["Vol_Change"] = np.log(vol / vol.shift(1).clip(lower=1))

    # Ch4: (H-L)/C intraday range
    out["HL_Range"] = (df["High"] - df["Low"]) / close.clip(lower=1e-8)

    # Ch5: MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26

    # Ch6: Bollinger band position
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std() + 1e-8
    out["BB_Pos"] = (close - (sma20 - 2*std20)) / (4*std20 + 1e-8)

    # Ch7/Ch8: 20-day rolling skew / kurtosis on log returns
    out["Skew_20"] = out["Log_Ret"].rolling(20).skew()
    out["Kurt_20"] = out["Log_Ret"].rolling(20).kurt()

    # Ch9: lag-1 log return
    out["Lag_Ret"] = out["Log_Ret"].shift(1)

    return out[["Log_Ret","RSI","Vol_20","Vol_Change","HL_Range",
                "MACD","BB_Pos","Skew_20","Kurt_20","Lag_Ret"]]


def download_and_build_features(ticker: str, start: str,
                                 cache_dir: str = ARTIFACTS):
    """
    Download OHLCV, compute 10-feature technical indicators, return
    (feats_array, log_returns, dates) after warm-up trim.
    """
    safe      = ticker.replace("^","").replace("-","")
    ohlcv_csv = os.path.join(cache_dir, f"ohlcv_{safe}.csv")

    if os.path.exists(ohlcv_csv):
        df = pd.read_csv(ohlcv_csv, parse_dates=["Date"], index_col="Date")
        print(f"  [cache] {ticker}: {len(df)} rows")
    else:
        print(f"  [download] {ticker}...")
        df = yf.download(ticker, start=start, end="2025-12-31",
                         interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open","High","Low","Close","Volume"]].dropna(subset=["Close"])
        df.index.name = "Date"
        df.to_csv(ohlcv_csv)
        print(f"  [saved]  {ticker}: {len(df)} rows")

    # Ensure numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["Close"], inplace=True)

    feats = _calculate_technical_indicators(df)
    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    feats.dropna(inplace=True)          # drops warm-up rows
    feats.fillna(0, inplace=True)

    # Clip extreme outliers per channel (outside 4-sigma)
    for col in feats.columns:
        mu, sg = feats[col].mean(), feats[col].std() + 1e-8
        feats[col] = feats[col].clip(mu - 4*sg, mu + 4*sg)

    log_returns = feats["Log_Ret"].values.astype(float)
    dates       = feats.index.values
    arr         = feats.values.astype(float)          # (N_days, 10)
    return arr, log_returns, dates


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — sliding windows, anomaly labels, regime labels
# ─────────────────────────────────────────────────────────────────────────────

def build_windows_and_labels(feats: np.ndarray, log_ret: np.ndarray,
                              dates: np.ndarray,
                              window: int = WINDOW,
                              regime_win: int = REGIME_WIN,
                              q: float = ANOMALY_Q):
    """
    Returns:
      X      : (N, window, 10)  — raw feature windows (un-scaled)
      y      : (N,)             — binary anomaly (q-threshold on |return|)
      regimes: (N,)             — 0=Low / 1=Med / 2=High vol
      w_dates: (N,)             — date of the last day in each window
    """
    n = len(feats)

    # Rolling expanding-window quantile — no look-ahead
    abs_r  = np.abs(log_ret)
    thresh = np.full(n, np.nan)
    for t in range(20, n):
        thresh[t] = np.quantile(abs_r[:t], q)
    thresh = np.nan_to_num(thresh, nan=np.inf)
    anom_daily = (abs_r > thresh).astype(int)

    # Regime: 252-day rolling std terciles
    vol    = pd.Series(log_ret).rolling(regime_win).std().bfill().values
    q33, q66 = np.quantile(vol, [0.33, 0.66])
    reg_daily = np.where(vol <= q33, 0, np.where(vol <= q66, 1, 2))

    X_list, y_list, r_list, d_list = [], [], [], []
    for t in range(window, n):
        W = feats[t - window : t]      # (128, 10)
        if not np.all(np.isfinite(W)):
            continue
        X_list.append(W)
        y_list.append(int(anom_daily[t - 1]))
        r_list.append(int(reg_daily[t - 1]))
        d_list.append(dates[t - 1])

    X       = np.array(X_list, dtype=np.float32)   # (N, 128, 10)
    y       = np.array(y_list,  dtype=int)
    regimes = np.array(r_list,  dtype=int)
    w_dates = np.array(d_list)
    return X, y, regimes, w_dates


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — 24-factor extraction (same as main model's X_factors.npy)
# ─────────────────────────────────────────────────────────────────────────────

def extract_24_factors(X: np.ndarray) -> np.ndarray:
    """
    Extract 8 base factors from (N, T, 10) windows, then add lag-1 and lag-5
    to produce (N, 24) feature matrix — identical to the main paper's
    construct_factors.py pipeline.

    Factor map:
      0  Return_Trend          mean log return across window
      1  Momentum_Factor       mean RSI (Ch1) across window
      2  Composite_Volatility  mean of vol-related channels (2,3,7,8,9)
      3  Latent_State_A        mean HL_Range (Ch4)
      4  Latent_State_B        mean MACD (Ch5)
      5  Latent_State_C        mean BB_Pos (Ch6)
      6  Tail_Kurtosis         excess kurtosis of log returns in window
      7  Tail_Skew             skewness of log returns in window
     8-15  lag-1 of factors 0-7
    16-23  lag-5 of factors 0-7
    """
    f0 = np.mean(X[:, :, 0], axis=1)
    f1 = np.mean(X[:, :, 1], axis=1)
    vc = X[:, :, [2, 3, 7, 8, 9]]
    f2 = np.mean(np.mean(vc, axis=2), axis=1)
    f3 = np.mean(X[:, :, 4], axis=1)
    f4 = np.mean(X[:, :, 5], axis=1)
    f5 = np.mean(X[:, :, 6], axis=1)
    f6 = sp_kurtosis(X[:, :, 0], axis=1)
    f7 = sp_skew(X[:, :, 0], axis=1)

    B  = np.column_stack([f0, f1, f2, f3, f4, f5, f6, f7])  # (N, 8)

    B1 = np.vstack([B[:1], B[:-1]])   # lag-1
    B5 = np.vstack([B[:5], B[:-5]])   # lag-5

    out = np.hstack([B, B1, B5])      # (N, 24)
    # Replace any remaining NaN/Inf
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — models and ensemble
# ─────────────────────────────────────────────────────────────────────────────

def _make_rf():
    return RandomForestClassifier(
        n_estimators=300, max_depth=7, class_weight="balanced",
        min_samples_leaf=2, random_state=42, n_jobs=-1,
    )

def _make_xgb(pos_weight: float = 19.0):
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=6, scale_pos_weight=pos_weight,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        eval_metric="aucpr", verbosity=0, random_state=42,
    )

def _make_lgb():
    return lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, is_unbalance=True,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        verbose=-1, random_state=42,
    )

def _ensemble_proba(models: list, X: np.ndarray) -> np.ndarray:
    """Average predicted probabilities from a list of fitted classifiers."""
    probas = np.stack([m.predict_proba(X)[:, 1] for m in models], axis=0)
    return probas.mean(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — event-wise F1 (same implementation as main paper)
# ─────────────────────────────────────────────────────────────────────────────

def event_wise_f1(pred: np.ndarray, true: np.ndarray,
                  tolerance: int = EVENT_TOL):
    """
    Event-based F1 with tolerance.
    A predicted event within +/-tolerance windows of a true event = TP.
    """
    def get_events(labels):
        events, in_ev, start = [], False, 0
        for i, v in enumerate(labels):
            if v == 1 and not in_ev:
                in_ev, start = True, i
            elif v == 0 and in_ev:
                events.append((start, i - 1))
                in_ev = False
        if in_ev:
            events.append((start, len(labels) - 1))
        return events

    true_ev = get_events(true)
    pred_ev = get_events(pred)
    if len(true_ev) == 0 or len(pred_ev) == 0:
        return 0.0, 0.0, 0.0

    matched_t = [False] * len(true_ev)
    matched_p = [False] * len(pred_ev)
    tp = 0
    for i, (ts, te) in enumerate(true_ev):
        for j, (ps, pe) in enumerate(pred_ev):
            if not matched_p[j]:
                if not (pe < ts - tolerance or ps > te + tolerance):
                    tp += 1
                    matched_t[i] = matched_p[j] = True
                    break

    fp  = sum(1 for m in matched_p if not m)
    fn  = sum(1 for m in matched_t if not m)
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1  = 2*pre*rec / (pre + rec) if (pre + rec) > 0 else 0.0
    return f1, pre, rec


def _opt_threshold_event(proba: np.ndarray, y: np.ndarray,
                          sigma: float = GAUSS_SIGMA,
                          beta: float = FBETA) -> float:
    """
    Find the threshold that maximises event-wise F-beta on (proba, y).
    beta > 1 weights recall more than precision (beta=2 gives recall
    twice the weight), which reduces missed detections at the cost of
    more false alarms.  Applied to TRAINING fold only (honest OOS).
    """
    smoothed = gaussian_filter1d(proba, sigma=sigma)
    hi, lo   = smoothed.max(), smoothed.min()
    if hi <= lo:
        return lo + 1e-6        # predict nothing — conservative fallback
    best_t, best_fb = lo + 1e-6, 0.0
    b2 = beta ** 2
    for thr in np.linspace(lo, hi, 300):
        pred = (smoothed >= thr).astype(int)
        f1, pre, rec = event_wise_f1(pred, y)
        fb = (1 + b2) * pre * rec / (b2 * pre + rec + 1e-12)
        if fb > best_fb:
            best_fb, best_t = fb, thr
    return best_t


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — walk-forward cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def cv_eval(X: np.ndarray, y: np.ndarray, reg: np.ndarray,
            pos_weight: float = 19.0) -> dict:
    """
    5-fold walk-forward CV with 128-day purge gap.
    Returns per-fold metrics + pooled event-wise F1.
    """
    n         = len(y)
    fold_size = n // N_FOLDS

    all_pred_g, all_pred_m, all_true = [], [], []
    fold_records = []

    for fold in range(1, N_FOLDS):
        train_end  = fold * fold_size
        test_start = train_end + PURGE          # purge gap
        test_end   = min(test_start + fold_size, n)

        if test_start >= n or test_end <= test_start:
            continue

        X_tr, y_tr = X[:train_end],          y[:train_end]
        X_te, y_te = X[test_start:test_end], y[test_start:test_end]
        r_tr        = reg[:train_end]
        r_te        = reg[test_start:test_end]

        # Per-fold feature standardisation (fit on train, apply to test)
        sc      = StandardScaler()
        X_tr_s  = sc.fit_transform(X_tr)
        X_te_s  = sc.transform(X_te)

        # ── Global ensemble ───────────────────────────────────────
        if y_tr.sum() < 2:
            all_pred_g.extend([0]*len(y_te))
            all_pred_m.extend([0]*len(y_te))
            all_true.extend(y_te.tolist())
            continue

        rf  = _make_rf();   rf.fit(X_tr_s, y_tr)
        xb  = _make_xgb(pos_weight); xb.fit(X_tr_s, y_tr)
        lb  = _make_lgb();  lb.fit(X_tr_s, y_tr)
        models = [rf, xb, lb]

        g_proba_tr = _ensemble_proba(models, X_tr_s)
        g_proba_te = _ensemble_proba(models, X_te_s)

        g_thr   = _opt_threshold_event(g_proba_tr, y_tr)
        g_smooth= gaussian_filter1d(g_proba_te, GAUSS_SIGMA)
        g_pred  = (g_smooth >= g_thr).astype(int)

        # ── MoE ensemble (regime-specific, fallback to Global) ────
        moe_pred   = np.zeros(len(y_te), dtype=int)
        moe_proba  = np.zeros(len(y_te), dtype=float)

        for k in range(3):
            mask_tr = r_tr == k
            mask_te = r_te == k
            if mask_te.sum() == 0:
                continue
            # Fallback to global when regime is too sparse for expert training
            if mask_tr.sum() < 10 or y_tr[mask_tr].sum() < 3:
                moe_pred[mask_te]  = g_pred[mask_te]   # inherit global decision
                moe_proba[mask_te] = g_smooth[mask_te]
                continue

            e_rf = _make_rf();   e_rf.fit(X_tr_s[mask_tr], y_tr[mask_tr])
            e_xb = _make_xgb(pos_weight)
            e_xb.fit(X_tr_s[mask_tr], y_tr[mask_tr])
            e_lb = _make_lgb();  e_lb.fit(X_tr_s[mask_tr], y_tr[mask_tr])
            e_models = [e_rf, e_xb, e_lb]

            ep_tr = _ensemble_proba(e_models, X_tr_s[mask_tr])
            ep_te = _ensemble_proba(e_models, X_te_s[mask_te])
            e_thr = _opt_threshold_event(ep_tr, y_tr[mask_tr])

            e_smooth           = gaussian_filter1d(ep_te, GAUSS_SIGMA)
            moe_proba[mask_te] = e_smooth
            moe_pred[mask_te]  = (e_smooth >= e_thr).astype(int)

        all_pred_g.extend(g_pred.tolist())
        all_pred_m.extend(moe_pred.tolist())
        all_true.extend(y_te.tolist())

        g_f1, g_pr, g_rc  = event_wise_f1(g_pred, y_te)
        m_f1, m_pr, m_rc  = event_wise_f1(moe_pred, y_te)
        fold_records.append({
            "fold": fold,
            "n_test": len(y_te), "n_pos_test": int(y_te.sum()),
            "f1_global": round(g_f1, 4), "prec_global": round(g_pr, 4),
            "rec_global": round(g_rc, 4),
            "f1_moe":    round(m_f1, 4), "prec_moe": round(m_pr, 4),
            "rec_moe":   round(m_rc, 4),
        })
        print(f"    Fold {fold}: Global F1={g_f1:.3f} | MoE F1={m_f1:.3f} "
              f"[{int(y_te.sum())} positives in {len(y_te)} test windows]")

    pg = np.array(all_pred_g)
    pm = np.array(all_pred_m)
    pt = np.array(all_true)

    gf1, gpr, grc = event_wise_f1(pg, pt)
    mf1, mpr, mrc = event_wise_f1(pm, pt)

    # Point-wise F1 for reference / comparison with prior scripts
    gf1_pt = float(f1_score(pt, pg, zero_division=0))
    mf1_pt = float(f1_score(pt, pm, zero_division=0))

    return {
        "folds":          fold_records,
        "n_total":        int(len(pt)),
        "n_anomalies":    int(pt.sum()),
        # Event-wise (primary metric — same as main paper)
        "f1_global_ew":   round(gf1, 4),
        "prec_global_ew": round(gpr, 4),
        "rec_global_ew":  round(grc, 4),
        "f1_moe_ew":      round(mf1, 4),
        "prec_moe_ew":    round(mpr, 4),
        "rec_moe_ew":     round(mrc, 4),
        # Point-wise (for comparison)
        "f1_global_pw":   round(gf1_pt, 4),
        "f1_moe_pw":      round(mf1_pt, 4),
        # Raw arrays for pooled analysis
        "_pred_g": pg.tolist(),
        "_pred_m": pm.tolist(),
        "_true":   pt.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — main
# ─────────────────────────────────────────────────────────────────────────────

def run():
    os.makedirs(ARTIFACTS, exist_ok=True)
    os.makedirs(FIGURES,   exist_ok=True)

    results = {}
    all_g_pool, all_m_pool, all_t_pool = [], [], []

    for ticker, mkt, start in MARKETS:
        print(f"\n{'='*62}")
        print(f"  {mkt}  ({ticker})")
        print(f"{'='*62}")

        feats, log_ret, dates = download_and_build_features(ticker, start)
        print(f"  Daily rows  : {len(feats)}")

        X_win, y, reg, w_dates = build_windows_and_labels(feats, log_ret, dates)
        print(f"  Windows     : {len(y)}  |  "
              f"Anomalies: {y.sum()} ({100*y.mean():.1f}%)")
        print(f"  Regimes     : Low={int((reg==0).sum())}  "
              f"Med={int((reg==1).sum())}  High={int((reg==2).sum())}")

        # Extract 24-factor feature matrix
        print("  Extracting 24 factors...")
        Xf = extract_24_factors(X_win)
        print(f"  Feature matrix: {Xf.shape}")

        # pos_weight for XGBoost: (1-base_rate)/base_rate
        anom_rate   = max(y.mean(), 0.01)
        pos_weight  = (1.0 - anom_rate) / anom_rate

        print("  Running 5-fold walk-forward CV (RF + XGB + LGBM ensemble)...")
        cv = cv_eval(Xf, y, reg, pos_weight=pos_weight)

        print(f"  Global  — Event-wise F1 : {cv['f1_global_ew']:.4f}  "
              f"Prec: {cv['prec_global_ew']:.4f}  Rec: {cv['rec_global_ew']:.4f}")
        print(f"  RC-MoE  — Event-wise F1 : {cv['f1_moe_ew']:.4f}  "
              f"Prec: {cv['prec_moe_ew']:.4f}  Rec: {cv['rec_moe_ew']:.4f}")
        print(f"  (Point-wise F1 for ref  : Global={cv['f1_global_pw']:.3f}  "
              f"MoE={cv['f1_moe_pw']:.3f})")

        results[mkt] = {
            "ticker":       ticker,
            "n_windows":    cv["n_total"],
            "n_anomalies":  cv["n_anomalies"],
            "anomaly_rate": round(float(y.mean()), 4),
            "n_features":   Xf.shape[1],
            # Primary metric
            "f1_global_ew":   cv["f1_global_ew"],
            "prec_global_ew": cv["prec_global_ew"],
            "rec_global_ew":  cv["rec_global_ew"],
            "f1_moe_ew":      cv["f1_moe_ew"],
            "prec_moe_ew":    cv["prec_moe_ew"],
            "rec_moe_ew":     cv["rec_moe_ew"],
            # Point-wise for comparison
            "f1_global_pw": cv["f1_global_pw"],
            "f1_moe_pw":    cv["f1_moe_pw"],
            "folds":        cv["folds"],
        }

        all_g_pool.extend(cv["_pred_g"])
        all_m_pool.extend(cv["_pred_m"])
        all_t_pool.extend(cv["_true"])

    # ── Pooled cross-market event-wise F1 ────────────────────────
    pg = np.array(all_g_pool)
    pm = np.array(all_m_pool)
    pt = np.array(all_t_pool)

    p_gf1, p_gpr, p_grc = event_wise_f1(pg, pt)
    p_mf1, p_mpr, p_mrc = event_wise_f1(pm, pt)

    print(f"\n{'='*62}")
    print(f"  POOLED  (all 4 markets)")
    print(f"{'='*62}")
    print(f"  n={len(pt)}  anomalies={int(pt.sum())}")
    print(f"  Global  Event-wise F1 : {p_gf1:.4f}")
    print(f"  RC-MoE  Event-wise F1 : {p_mf1:.4f}")

    results["pooled"] = {
        "n_total":       int(len(pt)),
        "n_anomalies":   int(pt.sum()),
        "f1_global_ew":  round(p_gf1, 4),
        "prec_global_ew": round(p_gpr, 4),
        "rec_global_ew": round(p_grc, 4),
        "f1_moe_ew":     round(p_mf1, 4),
        "prec_moe_ew":   round(p_mpr, 4),
        "rec_moe_ew":    round(p_mrc, 4),
        "f1_global_pw":  round(float(f1_score(pt, pg, zero_division=0)), 4),
        "f1_moe_pw":     round(float(f1_score(pt, pm, zero_division=0)), 4),
    }

    out = os.path.join(ARTIFACTS, "multimarket_path_a_v2_results.json")
    # Drop raw arrays before saving
    save_results = {k: {sk: sv for sk, sv in v.items()
                        if not sk.startswith("_")}
                    for k, v in results.items()}
    json.dump(save_results, open(out, "w"), indent=2)
    print(f"\nSaved -> {out}")

    _make_figure(save_results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — publication figure
# ─────────────────────────────────────────────────────────────────────────────

def _make_figure(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load old returns-only baseline for comparison
    # Use v1 results as baseline comparison if available
    baseline_path = os.path.join(ARTIFACTS, "multimarket_path_a_results.json")
    baseline: dict = {}
    if os.path.exists(baseline_path):
        baseline = json.load(open(baseline_path))

    mkts    = ["SP500", "NASDAQ", "Russell2000", "Bitcoin"]
    display = ["S&P 500", "NASDAQ", "Russell 2000", "Bitcoin"]
    x = np.arange(len(mkts))
    w = 0.25

    CB_BLUE   = "#0072B2"
    CB_ORANGE = "#D55E00"
    CB_GREEN  = "#009E73"

    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "figure.dpi": 300})
    fig, ax = plt.subplots(figsize=(13, 5.5))

    # Returns-only baseline MoE (point-wise F1)
    f1_base = [baseline.get(m, {}).get("f1_global", 0) for m in mkts]
    # Full pipeline Global (event-wise)
    f1_g_ew  = [results.get(m, {}).get("f1_global_ew", 0) for m in mkts]
    # Full pipeline MoE (event-wise) — PRIMARY result
    f1_m_ew  = [results.get(m, {}).get("f1_moe_ew", 0) for m in mkts]

    b1 = ax.bar(x - w, f1_base, w, label="Returns-only baseline (point-wise F1)",
                color=CB_BLUE, alpha=0.65, edgecolor="white")
    b2 = ax.bar(x,     f1_g_ew, w, label="Full pipeline — Global (event-wise F1, tol=5)",
                color=CB_ORANGE, alpha=0.75, edgecolor="white")
    b3 = ax.bar(x + w, f1_m_ew, w, label="Full pipeline — RC-MoE (event-wise F1, tol=5)",
                color=CB_GREEN, edgecolor="white")

    # Main S&P 500 in-sample reference line
    ax.axhline(0.738, color="gray", ls="--", lw=1.4,
               label="Main model in-sample (S&P 500, 0.738)")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
                        f"{h:.3f}", ha="center", va="bottom",
                        fontsize=7.5, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(display, fontsize=11)
    ax.set_ylabel("F1 Score")
    ax.set_title(
        "Multi-Market OOS Anomaly Detection — RC-MoE Full Pipeline\n"
        "(24 economic factors, RF+XGB+LGBM ensemble, 5-fold walk-forward CV, 2016-2025)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper right")
    ymax = max(max(f1_base + f1_g_ew + f1_m_ew), 0.738) + 0.05
    ax.set_ylim(0, min(ymax * 1.20, 1.05))

    fig.tight_layout()
    p = os.path.join(FIGURES, "fig_multimarket_path_a_v2.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved -> {p}")


if __name__ == "__main__":
    run()
