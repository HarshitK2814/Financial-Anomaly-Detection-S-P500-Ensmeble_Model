"""
Full OHLCV OOS Retrain — S&P 500 / NASDAQ / Russell 2000 / Bitcoin.

Uses the same 16-factor OHLCV feature engineering pipeline as the main paper
model (Garman-Klass vol, Parkinson vol, Amihud illiquidity, VIX proxy, Hurst
exponent, etc.) with lag-1 and lag-5 → 48 features total.

Compares against the simplified 8-feature returns-only multimarket results
(multimarket_detection_results.json).

Saves:
  artifacts/multimarket_ohlcv_results.json
  correct figures/fig_multimarket_ohlcv_comparison.png

Run from repo root:
  .venv/Scripts/python.exe -m src.evaluation.multimarket_ohlcv_retrain
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew
import lightgbm as lgb
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import yfinance as yf

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

WINDOW     = 128
REGIME_WIN = 252
N_FOLDS    = 5
ARTIFACTS  = "artifacts"
FIGURES    = "correct figures"

# Market definitions: (yfinance ticker, display name, start date)
MARKETS = [
    ("^GSPC",   "SP500",       "2016-01-01"),
    ("^IXIC",   "NASDAQ",      "2016-01-01"),
    ("^RUT",    "Russell2000", "2016-01-01"),
    ("BTC-USD", "Bitcoin",     "2016-01-01"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Data download & OHLCV feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def download_ohlcv(ticker: str, start: str,
                   cache_dir: str = ARTIFACTS) -> pd.DataFrame:
    """Download OHLCV via yfinance and cache as CSV. Returns daily OHLCV."""
    safe = ticker.replace("^", "").replace("-", "")
    cache_path = os.path.join(cache_dir, f"ohlcv_{safe}.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
        print(f"  [cache] {ticker}: {len(df)} rows")
        return df

    print(f"  [download] {ticker} from {start}...")
    df = yf.download(ticker, start=start, end="2025-12-31",
                     interval="1d", progress=False, auto_adjust=True)

    # Flatten MultiIndex columns (yfinance >= 0.2)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df[df["Close"].notna()].copy()
    df.index.name = "Date"
    df.to_csv(cache_path)
    print(f"  [saved]  {ticker}: {len(df)} rows -> {cache_path}")
    return df


def build_ohlcv_channels(df: pd.DataFrame) -> np.ndarray:
    """
    Convert raw OHLCV DataFrame to a (N-1, 5) daily feature matrix.

    Channels (all daily log ratios / normalized measures):
      0: log(Close_t / Close_{t-1})              — close-to-close return
      1: log(High_t  / Close_{t-1})              — upside intraday extent
      2: log(Low_t   / Close_{t-1})              — downside intraday extent
      3: log(Volume_t / Volume_{t-1})            — volume change
      4: (High_t - Low_t) / Close_t              — normalized HL range
    """
    close = df["Close"].values.astype(float)
    high  = df["High"].values.astype(float)
    low   = df["Low"].values.astype(float)
    vol   = df["Volume"].values.astype(float) + 1.0  # avoid log(0)

    n   = len(close)
    ret   = np.zeros(n, dtype=float)
    h_ret = np.zeros(n, dtype=float)
    l_ret = np.zeros(n, dtype=float)
    v_chg = np.zeros(n, dtype=float)
    hl_r  = np.zeros(n, dtype=float)

    for t in range(1, n):
        if close[t-1] > 0:
            ret[t]   = np.log(close[t]  / close[t-1])
            h_ret[t] = np.log(max(high[t],  close[t-1]) / close[t-1])
            l_ret[t] = np.log(max(low[t],   close[t-1]) / close[t-1])
        if vol[t-1] > 0:
            v_chg[t] = np.log(vol[t] / vol[t-1])
        if close[t] > 0:
            hl_r[t] = (high[t] - low[t]) / close[t]

    data = np.column_stack([ret, h_ret, l_ret, v_chg, hl_r])
    data = data[1:]  # drop row 0 (all zeros from initialization)

    # Clip extreme outliers per channel (winsorise at 1st/99th ± 3×IQR)
    for ch in range(data.shape[1]):
        p1, p99 = np.percentile(data[:, ch], [1, 99])
        iqr = p99 - p1
        data[:, ch] = np.clip(data[:, ch], p1 - 3*iqr, p99 + 3*iqr)

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Factor extraction from a single OHLCV window
# ─────────────────────────────────────────────────────────────────────────────

def _hurst(x: np.ndarray) -> float:
    """Hurst exponent via R/S analysis on 3 sub-lags."""
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
    ns = np.log([r[0] for r in rs_vals])
    rs = np.log([r[1] for r in rs_vals])
    try:
        return float(np.clip(np.polyfit(ns, rs, 1)[0], 0.0, 1.0))
    except Exception:
        return 0.5


def extract_ohlcv_factors(W: np.ndarray) -> np.ndarray:
    """
    Extract 16 OHLCV-based factors from a (WINDOW, 5) window.

    Factor map (matches Table I factor taxonomy in paper):
      0  Return_Trend          — mean close return
      1  Momentum_Factor       — RSI-like: Σ(pos_ret) / Σ|ret|
      2  Composite_Volatility  — mean std across all 5 channels
      3  Latent_State_A        — mean log(H/C_prev)  (upside pressure)
      4  Latent_State_B        — mean log(L/C_prev)  (downside pressure)
      5  Latent_State_C        — mean HL range/Close (breadth)
      6  Tail_Kurtosis         — excess kurtosis of close returns
      7  Tail_Skew             — skewness of close returns
      8  Volume_Shock          — (last_vol - 20d mean) / 20d mean
      9  Garman_Klass_Vol      — GK estimator via log(H/L)^2 − (2ln2−1)log(C/O)^2
     10  Parkinson_Vol         — sqrt(mean(log(H/L)^2) / 4ln2)
     11  Amihud_Illiquidity    — mean(|ret| / |vol_change|)
     12  Realized_Skew         — skewness of close returns (separate from 7)
     13  Max_Drawdown          — max drawdown over cumulative returns
     14  VIX_Proxy             — std(ret[-20:])×sqrt(252) / |mean(ret)|
     15  Hurst_Exponent        — R/S Hurst exponent
    """
    ret  = W[:, 0]   # close returns
    h    = W[:, 1]   # log(H/C_prev)
    l    = W[:, 2]   # log(L/C_prev)
    vol  = W[:, 3]   # volume change
    hl_r = W[:, 4]   # (H-L)/C

    # 0. Return Trend
    f_trend = float(np.mean(ret))

    # 1. Momentum (RSI-like)
    pos_sum  = float(np.sum(ret[ret > 0]))
    abs_sum  = float(np.sum(np.abs(ret))) + 1e-12
    f_mom    = pos_sum / abs_sum

    # 2. Composite Volatility
    f_cvol   = float(np.mean([np.std(ret, ddof=1), np.std(h, ddof=1),
                               np.std(l, ddof=1),   np.std(vol, ddof=1),
                               np.std(hl_r, ddof=1)]))

    # 3–5. Latent States
    f_ls_a   = float(np.mean(h))
    f_ls_b   = float(np.mean(l))
    f_ls_c   = float(np.mean(hl_r))

    # 6–7. Tail Risk
    f_kurt   = float(scipy_kurtosis(ret))
    f_skew   = float(scipy_skew(ret))

    # 8. Volume Shock
    lookback = min(20, len(vol))
    last_vol  = vol[-1]
    mean_vol  = float(np.mean(vol[-lookback:]))
    f_vshock  = (last_vol - mean_vol) / (abs(mean_vol) + 1e-8)

    # 9. Garman-Klass volatility
    #    h ≈ log(H/C_prev), l ≈ log(L/C_prev)  →  h-l ≈ log(H/L)
    hl_log  = h - l                              # ≈ log(H/L)
    co_log  = ret                                # ≈ log(C/O) approximation
    gk      = 0.5 * np.mean(hl_log ** 2) - (2*np.log(2) - 1) * np.mean(co_log ** 2)
    f_gk    = float(np.sqrt(max(gk, 0.0)))

    # 10. Parkinson volatility
    f_pk    = float(np.sqrt(np.mean(hl_log ** 2) / (4.0 * np.log(2) + 1e-12)))

    # 11. Amihud illiquidity
    f_amihud = float(np.mean(np.abs(ret) / (np.abs(vol) + 1e-8)))

    # 12. Realized Skewness (within-window crash asymmetry)
    f_rskew  = float(scipy_skew(ret))

    # 13. Max Drawdown
    cum         = np.cumsum(ret)
    running_max = np.maximum.accumulate(cum)
    f_mdd       = float(np.max(running_max - cum))

    # 14. VIX Proxy
    atr     = np.std(ret[-20:]) * np.sqrt(252)
    f_vix   = atr / (abs(np.mean(ret)) + 1e-8)

    # 15. Hurst Exponent
    f_hurst = _hurst(ret)

    return np.array([f_trend, f_mom,   f_cvol,   f_ls_a, f_ls_b,  f_ls_c,
                     f_kurt,  f_skew,  f_vshock, f_gk,   f_pk,    f_amihud,
                     f_rskew, f_mdd,   f_vix,    f_hurst], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def build_factor_dataset(ohlcv_data: np.ndarray,
                          window:     int = WINDOW,
                          regime_win: int = REGIME_WIN):
    """
    Build leakage-free (X, y, regimes) from daily (N, 5) OHLCV matrix.

    X       : (N_windows, 48)  — 16 factors × {base, lag-1, lag-5}
    y       : (N_windows,)     — binary anomaly (1 = extreme |return|)
    regimes : (N_windows,)     — 0=Low / 1=Med / 2=High vol regime
    """
    n       = len(ohlcv_data)
    returns = ohlcv_data[:, 0]

    # Rolling expanding-window anomaly threshold (q=0.98) — no look-ahead
    abs_r      = np.abs(returns)
    thresholds = np.full(n, np.nan)
    for t in range(20, n):
        thresholds[t] = np.quantile(abs_r[:t], 0.98)
    thresh_filled   = np.nan_to_num(thresholds, nan=np.inf)
    anomaly_daily   = (abs_r > thresh_filled).astype(int)

    # Regime labels from rolling 252-day vol terciles
    vol_series  = pd.Series(returns).rolling(regime_win).std().bfill().values
    q33, q66    = np.quantile(vol_series, [0.33, 0.66])
    reg_daily   = np.where(vol_series <= q33, 0,
                           np.where(vol_series <= q66, 1, 2))

    base_feats, labels, reg_labels = [], [], []
    for t in range(window, n):
        W = ohlcv_data[t - window : t]   # (128, 5)
        if not np.all(np.isfinite(W)):
            continue
        feats = extract_ohlcv_factors(W)
        if not np.all(np.isfinite(feats)):
            continue
        base_feats.append(feats)
        labels.append(int(anomaly_daily[t - 1]))
        reg_labels.append(int(reg_daily[t - 1]))

    B       = np.array(base_feats)    # (N, 16)
    y       = np.array(labels)
    reg_arr = np.array(reg_labels)

    # Add lag-1 and lag-5 (temporal context)
    B1 = np.vstack([B[:1], B[:-1]])
    B5 = np.vstack([B[:5], B[:-5]])
    X  = np.hstack([B, B1, B5])       # (N, 48)
    return X, y, reg_arr


# ─────────────────────────────────────────────────────────────────────────────
# RC-MoE temporal cross-validation (LightGBM)
# ─────────────────────────────────────────────────────────────────────────────

def _make_lgbm() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=200, max_depth=5, is_unbalance=True,
        learning_rate=0.05, verbose=-1, random_state=42,
    )


def _opt_threshold(model, X_tr: np.ndarray, y_tr: np.ndarray) -> float:
    """F1-optimal classification threshold on the training fold."""
    proba    = model.predict_proba(X_tr)[:, 1]
    best_t   = 0.5
    best_f1  = 0.0
    for thr in np.linspace(0.01, 0.99, 200):
        f1 = f1_score(y_tr, (proba >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, thr
    return best_t


def cv_eval_lgbm(X: np.ndarray, y: np.ndarray,
                  reg_arr: np.ndarray) -> dict:
    """5-fold walk-forward CV: Global LightGBM vs RC-MoE LightGBM."""
    n         = len(y)
    fold_size = n // N_FOLDS
    all_g, all_m, all_t = [], [], []

    for fold in range(1, N_FOLDS):
        train_end  = fold * fold_size
        test_start = train_end
        test_end   = min(test_start + fold_size, n)

        X_tr, y_tr = X[:train_end],       y[:train_end]
        X_te, y_te = X[test_start:test_end], y[test_start:test_end]
        reg_tr     = reg_arr[:train_end]
        reg_te     = reg_arr[test_start:test_end]

        scaler  = StandardScaler()
        X_tr_s  = scaler.fit_transform(X_tr)
        X_te_s  = scaler.transform(X_te)

        if y_tr.sum() == 0:
            all_g.extend([0] * len(y_te))
            all_m.extend([0] * len(y_te))
            all_t.extend(y_te.tolist())
            continue

        # ── Global model ──────────────────────────────────────────
        gm      = _make_lgbm()
        gm.fit(X_tr_s, y_tr)
        g_t     = _opt_threshold(gm, X_tr_s, y_tr)
        g_pred  = (gm.predict_proba(X_te_s)[:, 1] >= g_t).astype(int)

        # ── MoE model ─────────────────────────────────────────────
        moe_pred = np.zeros(len(y_te), dtype=int)
        for k in range(3):
            mask_tr = reg_tr == k
            mask_te = reg_te == k
            if mask_tr.sum() < 5 or y_tr[mask_tr].sum() == 0:
                continue
            em = _make_lgbm()
            em.fit(X_tr_s[mask_tr], y_tr[mask_tr])
            if mask_te.sum() == 0:
                continue
            et              = _opt_threshold(em, X_tr_s[mask_tr], y_tr[mask_tr])
            ep              = em.predict_proba(X_te_s[mask_te])[:, 1]
            moe_pred[mask_te] = (ep >= et).astype(int)

        all_g.extend(g_pred.tolist())
        all_m.extend(moe_pred.tolist())
        all_t.extend(y_te.tolist())

    pg = np.array(all_g)
    pm = np.array(all_m)
    pt = np.array(all_t)
    return {
        "true":       pt,
        "pred_global": pg,
        "pred_moe":    pm,
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
    """McNemar chi-squared: does MoE significantly outperform Global?"""
    from scipy.stats import chi2
    b01      = int(((pred_a == true) & (pred_b != true)).sum())
    b10      = int(((pred_a != true) & (pred_b == true)).sum())
    n        = b01 + b10
    if n < 1:
        return {"chi2": 0.0, "p": 1.0, "b01": b01, "b10": b10}
    chi2_stat = (abs(b10 - b01) - 1) ** 2 / n
    p         = float(1 - chi2.cdf(chi2_stat, df=1))
    return {"chi2": round(chi2_stat, 3), "p": round(p, 4),
            "b01": b01, "b10": b10}


# ─────────────────────────────────────────────────────────────────────────────
# Comparison figure
# ─────────────────────────────────────────────────────────────────────────────

def make_comparison_figure(results_ohlcv: dict) -> None:
    """Bar chart: 8-feature returns-only vs 16-factor OHLCV RC-MoE."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    simple_path = os.path.join(ARTIFACTS, "multimarket_detection_results.json")
    simple: dict = {}
    if os.path.exists(simple_path):
        simple = json.load(open(simple_path))

    mkts    = ["SP500", "NASDAQ", "Russell2000", "Bitcoin"]
    display = ["S&P 500", "NASDAQ", "Russell 2000", "Bitcoin"]
    x = np.arange(len(mkts))
    w = 0.20

    CB_BLUE   = "#0072B2"
    CB_ORANGE = "#D55E00"

    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "figure.dpi": 300})
    fig, ax = plt.subplots(figsize=(13, 5))

    f1_sg = [simple.get(m, {}).get("f1_global", 0)       for m in mkts]
    f1_sm = [simple.get(m, {}).get("f1_moe",    0)       for m in mkts]
    f1_og = [results_ohlcv.get(m, {}).get("f1_global", 0) for m in mkts]
    f1_om = [results_ohlcv.get(m, {}).get("f1_moe",    0) for m in mkts]

    b1 = ax.bar(x - 1.5*w, f1_sg, w, label="8-feat Returns  · Global",
                color=CB_BLUE, alpha=0.65, edgecolor="white")
    b2 = ax.bar(x - 0.5*w, f1_sm, w, label="8-feat Returns  · RC-MoE",
                color=CB_BLUE, edgecolor="white")
    b3 = ax.bar(x + 0.5*w, f1_og, w, label="16-factor OHLCV · Global",
                color=CB_ORANGE, alpha=0.65, edgecolor="white")
    b4 = ax.bar(x + 1.5*w, f1_om, w, label="16-factor OHLCV · RC-MoE",
                color=CB_ORANGE, edgecolor="white")

    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.005:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.004,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7.5,
                        rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(display, fontsize=11)
    ax.set_ylabel("OOS F1 Score")
    ax.set_title(
        "Multi-Market OOS Detection: 8-Feature Returns-Only vs Full 16-Factor OHLCV RC-MoE\n"
        "(128-day sliding windows · 5-fold walk-forward CV · LightGBM · 2016–2025)",
        fontsize=11,
    )
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    ymax = max(max(f1_sg + f1_sm + f1_og + f1_om), 0.1)
    ax.set_ylim(0, ymax * 1.30)

    fig.tight_layout()
    p = os.path.join(FIGURES, "fig_multimarket_ohlcv_comparison.png")
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved -> {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run():
    os.makedirs(ARTIFACTS, exist_ok=True)

    results: dict = {}
    all_true_pool, all_global_pool, all_moe_pool = [], [], []

    for ticker, mkt_name, start in MARKETS:
        print(f"\n{'='*60}")
        print(f"  {mkt_name}  ({ticker})")
        print(f"{'='*60}")

        df = download_ohlcv(ticker, start)
        ohlcv = build_ohlcv_channels(df)
        print(f"  Daily rows (after channel engineering): {len(ohlcv)}")

        X, y, reg_arr = build_factor_dataset(ohlcv)
        print(f"  Windows:  {len(y)} | Anomalies: {y.sum()} ({100*y.mean():.1f}%)")
        print(f"  Features: {X.shape[1]}  (16 factors × 3 time contexts)")
        print(f"  Regimes:  Low={int((reg_arr==0).sum())}  "
              f"Med={int((reg_arr==1).sum())}  "
              f"High={int((reg_arr==2).sum())}")

        cv = cv_eval_lgbm(X, y, reg_arr)
        print(f"  LightGBM Global : F1={cv['f1_global']:.4f}  "
              f"Prec={cv['prec_global']:.4f}  Rec={cv['rec_global']:.4f}")
        print(f"  LightGBM RC-MoE : F1={cv['f1_moe']:.4f}  "
              f"Prec={cv['prec_moe']:.4f}  Rec={cv['rec_moe']:.4f}")

        mc = mcnemar_test(cv["pred_global"], cv["pred_moe"], cv["true"])
        print(f"  McNemar (MoE vs Global): chi2={mc['chi2']:.3f}  "
              f"p={mc['p']:.4f}  b01={mc['b01']}  b10={mc['b10']}")

        results[mkt_name] = {
            "ticker":      ticker,
            "n_windows":   int(len(y)),
            "n_anomalies": cv["n_anomalies"],
            "n_features":  X.shape[1],
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

    # ── Pooled cross-market statistics ───────────────────────────
    pt = np.array(all_true_pool)
    pg = np.array(all_global_pool)
    pm = np.array(all_moe_pool)
    pooled_mc = mcnemar_test(pg, pm, pt)

    print(f"\n{'='*60}")
    print(f"  POOLED  (all 4 markets)")
    print(f"{'='*60}")
    print(f"  n={len(pt)}  anomalies={pt.sum()}")
    print(f"  Pooled F1 Global : {f1_score(pt, pg, zero_division=0):.4f}")
    print(f"  Pooled F1 RC-MoE : {f1_score(pt, pm, zero_division=0):.4f}")
    print(f"  Pooled McNemar   : chi2={pooled_mc['chi2']:.3f}  "
          f"p={pooled_mc['p']:.4f}")

    results["pooled"] = {
        "n_total":     int(len(pt)),
        "n_anomalies": int(pt.sum()),
        "f1_global":   round(float(f1_score(pt, pg, zero_division=0)), 4),
        "f1_moe":      round(float(f1_score(pt, pm, zero_division=0)), 4),
        "mcnemar":     pooled_mc,
    }

    out = os.path.join(ARTIFACTS, "multimarket_ohlcv_results.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved -> {out}")

    make_comparison_figure(results)
    return results


if __name__ == "__main__":
    run()
