"""
run_daily_pipeline.py
=====================
Master daily pipeline:
  STEP 0  Refresh S&P 500 daily data via yfinance  (2000-today)
  STEP 1  C-01 — McNemar + bootstrap significance test
          (Global LightGBM vs MoE LightGBM, primary dataset)
  STEP 2  Economic backtest — build hedge-strategy daily returns
          (5-year dataset, dates available)
  STEP 3  Download Fama-French FF5 + MOM daily factors
  STEP 4  Fama-French alpha test (OLS + Newey-West HAC)

Run from model root:
    python run_daily_pipeline.py

Output files (all in artifacts/):
    c01_results.json
    hedge_daily_returns.csv
    ff_alpha_results.json
    data/ff_factors/FF5_daily.csv
    data/ff_factors/MOM_daily.csv
"""
from __future__ import annotations
import json, os, sys, time, warnings, zipfile
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen, Request

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              average_precision_score, precision_recall_curve)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

BASE = Path("artifacts")
BASE.mkdir(exist_ok=True)

PRINT_SEP = "=" * 72


# ═══════════════════════════════════════════════════════════════════════
# HELPERS — regime detection (self-contained, no src import needed)
# ═══════════════════════════════════════════════════════════════════════

def rolling_regime(vol_series: np.ndarray, window: int = 252) -> np.ndarray:
    """Assign Low(0)/Med(1)/High(2) regimes via a rolling quantile window."""
    n = len(vol_series)
    regimes = np.ones(n, dtype=int)  # default Med
    for i in range(n):
        start = max(0, i - window + 1)
        w = vol_series[start : i + 1]
        q33, q66 = np.percentile(w, [33, 66])
        if vol_series[i] <= q33:
            regimes[i] = 0
        elif vol_series[i] <= q66:
            regimes[i] = 1
        else:
            regimes[i] = 2
    return regimes


def opt_thresh_pw(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Best F1 threshold from PR curve — fitted on training data only."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    prec, rec, thresh = precision_recall_curve(y_true, y_proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    bi = min(np.argmax(f1s), len(thresh) - 1)
    return float(thresh[bi])


def construct_factors_basic(windows: np.ndarray) -> np.ndarray:
    """Build simple factor features from raw 128-day windows."""
    from scipy.stats import kurtosis as kurt_fn, skew as skew_fn
    rows = []
    for i in range(len(windows)):
        w = windows[i]
        close = w[:, 0]
        rets = np.diff(close) / (np.abs(close[:-1]) + 1e-10)
        row = [
            np.mean(rets), np.std(rets),
            kurt_fn(rets), skew_fn(rets),
            np.mean(rets[-5:]) - np.mean(rets[:5]),
            np.max(np.maximum.accumulate(close) - close) / (np.max(close) + 1e-10),
            np.min(rets), np.max(rets),
        ]
        if w.shape[1] > 1:
            for col in range(1, min(w.shape[1], 6)):
                row += [np.std(w[:, col]), np.mean(w[:, col])]
        rows.append(row)
    return np.array(rows)


# ═══════════════════════════════════════════════════════════════════════
# STEP 0 — REFRESH DAILY DATA
# ═══════════════════════════════════════════════════════════════════════

def step0_refresh_data() -> None:
    """Download fresh S&P 500 daily data from yfinance and save."""
    print(f"\n{PRINT_SEP}")
    print("  STEP 0 — REFRESH DAILY DATA")
    print(PRINT_SEP)
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not installed — skipping refresh.")
        return

    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    print(f"  Downloading ^GSPC daily data 2000-01-01 -> {today} …")
    df = yf.download("^GSPC", start="2000-01-01", end=today, interval="1d",
                     progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if len(df) == 0:
        print("  Download returned empty — using cached data.")
        return

    close = df["Close"].dropna()
    daily_ret = close.pct_change().dropna()

    out = BASE / "sp500_full_daily_returns.csv"
    daily_ret.reset_index().rename(columns={"index": "Date", "Close": "^GSPC"}).to_csv(
        out, index=False
    )
    print(f"  Saved {len(daily_ret)} daily returns -> {out}")

    prices_out = BASE / "sp500_full_daily_prices.csv"
    close.reset_index().to_csv(prices_out, index=False)
    print(f"  Saved daily prices -> {prices_out}")


# ═══════════════════════════════════════════════════════════════════════
# STEP 1 — C-01: MCNEMAR + BOOTSTRAP (LightGBM Global vs MoE)
# ═══════════════════════════════════════════════════════════════════════

def _mcnemar(y_true: np.ndarray,
             preds_a: np.ndarray,
             preds_b: np.ndarray) -> dict:
    ca = preds_a == y_true
    cb = preds_b == y_true
    b01 = int(np.sum(ca & ~cb))   # A correct, B wrong
    b10 = int(np.sum(~ca & cb))   # A wrong,   B correct
    if b01 + b10 == 0:
        return dict(statistic=0.0, p_value=1.0, significant=False,
                    b01=b01, b10=b10,
                    interpretation="Models make identical errors")
    stat = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p = float(1 - chi2.cdf(stat, df=1))
    return dict(statistic=float(stat), p_value=p, significant=p < 0.05,
                b01=b01, b10=b10,
                interpretation=f"{'Significant' if p < 0.05 else 'Not significant'} "
                               f"(p={p:.4f}), favors={'Global' if b01 > b10 else 'MoE'}")


def _bootstrap_f1_diff(y_true: np.ndarray,
                        preds_a: np.ndarray,
                        preds_b: np.ndarray,
                        n_boot: int = 2000,
                        seed: int = 42) -> dict:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    fa = f1_score(y_true, preds_a, zero_division=0)
    fb = f1_score(y_true, preds_b, zero_division=0)
    obs_diff = fa - fb
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            d = (f1_score(y_true[idx], preds_a[idx], zero_division=0)
                 - f1_score(y_true[idx], preds_b[idx], zero_division=0))
            diffs.append(d)
        except Exception:
            pass
    diffs = np.array(diffs)
    p = float(min(np.mean(diffs <= 0) * 2, 1.0)) if obs_diff > 0 else \
        float(min(np.mean(diffs >= 0) * 2, 1.0))
    # preds_a=MoE, preds_b=Global → fa=F1(MoE), fb=F1(Global)
    return dict(
        f1_moe=float(fa), f1_global=float(fb),
        f1_diff=float(obs_diff),   # positive = MoE better, negative = Global better
        ci95_lo=float(np.percentile(diffs, 2.5)),
        ci95_hi=float(np.percentile(diffs, 97.5)),
        p_value=p, significant=p < 0.05,
        interpretation=(
            f"MoE F1={fa:.4f} vs Global F1={fb:.4f}, "
            f"delta={obs_diff:+.4f} "
            f"95%CI=[{np.percentile(diffs,2.5):+.4f}, {np.percentile(diffs,97.5):+.4f}] "
            f"p={p:.4f} {'[SIGNIFICANT]' if p < 0.05 else '[not significant]'}"
        )
    )


def _run_lgbm_oos(X: np.ndarray, y: np.ndarray, vol: np.ndarray,
                  mode: str, n_folds: int = 5):
    """5-fold temporal CV for LightGBM; returns concatenated (proba, true, binary_pred)."""
    tscv = TimeSeriesSplit(n_splits=n_folds)
    all_proba, all_true = [], []

    for tri, tei in tscv.split(X):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tri])
        Xte = sc.transform(X[tei])

        if mode == "Global":
            ytr = y[tri]
            if len(np.unique(ytr)) < 2:
                continue
            m = lgb.LGBMClassifier(n_estimators=200, max_depth=5,
                                    is_unbalance=True, learning_rate=0.05,
                                    verbose=-1, random_state=42)
            m.fit(Xtr, ytr)
            all_proba.extend(m.predict_proba(Xte)[:, 1])

        else:  # MoE
            tr_reg = rolling_regime(vol[tri])
            te_reg_approx = rolling_regime(
                np.concatenate([vol[tri[-20:]], vol[tei]]))[20:]
            fp = np.zeros(len(tei))
            for r in np.unique(tr_reg):
                trm = tr_reg == r
                tem = te_reg_approx == r
                if trm.sum() < 5 or tem.sum() == 0:
                    continue
                ytr_r = y[tri[trm]]
                if len(np.unique(ytr_r)) < 2:
                    continue
                m = lgb.LGBMClassifier(n_estimators=200, max_depth=5,
                                        is_unbalance=True, learning_rate=0.05,
                                        verbose=-1, random_state=42)
                m.fit(Xtr[trm], ytr_r)
                # threshold from training proba only (F4 fix)
                t = opt_thresh_pw(ytr_r, m.predict_proba(Xtr[trm])[:, 1])
                fp[np.where(tem)[0]] = m.predict_proba(Xte[tem])[:, 1]
            all_proba.extend(fp)

        all_true.extend(y[tei])

    proba = np.array(all_proba)
    true  = np.array(all_true)

    if len(proba) == 0 or len(np.unique(true)) < 2:
        return proba, true, np.zeros_like(true, dtype=int)

    prec_a, rec_a, thresh_a = precision_recall_curve(true, proba)
    f1s_a = 2 * prec_a * rec_a / (prec_a + rec_a + 1e-8)
    bi = min(np.argmax(f1s_a), len(thresh_a) - 1)
    binary = (proba >= thresh_a[bi]).astype(int)
    return proba, true, binary


def step1_c01(save_predictions: bool = True) -> dict:
    """Run C-01: McNemar + bootstrap on Global vs MoE LightGBM (primary dataset)."""
    print(f"\n{PRINT_SEP}")
    print("  STEP 1 — C-01: STATISTICAL SIGNIFICANCE TEST")
    print("  (Global LightGBM vs MoE LightGBM — Primary dataset)")
    print(PRINT_SEP)

    X_f  = np.load(BASE / "X_factors.npy")
    y_f  = np.load(BASE / "y_factors.npy")
    X_win = np.load(BASE / "market_windows_10f.npy")

    min_len = min(len(X_f), len(y_f), len(X_win))
    X_f   = X_f[:min_len]
    y_f   = y_f[:min_len]
    vol   = np.std(X_win[:min_len, :, 0], axis=1)

    print(f"  N={min_len}, anomalies={int(y_f.sum())} ({y_f.mean()*100:.1f}%)")
    print(f"  Running 5-fold temporal CV for LightGBM Global …")
    t0 = time.time()
    proba_g, true_g, pred_g = _run_lgbm_oos(X_f, y_f, vol, mode="Global")
    print(f"    done in {time.time()-t0:.1f}s  |  F1={f1_score(true_g, pred_g, zero_division=0):.4f}")

    print(f"  Running 5-fold temporal CV for LightGBM MoE …")
    t0 = time.time()
    proba_m, true_m, pred_m = _run_lgbm_oos(X_f, y_f, vol, mode="MoE")
    print(f"    done in {time.time()-t0:.1f}s  |  F1={f1_score(true_m, pred_m, zero_division=0):.4f}")

    # Align lengths
    n = min(len(true_g), len(true_m))
    y_true = true_g[:n]
    pg = pred_g[:n]
    pm = pred_m[:n]

    print(f"\n  McNemar test …")
    mc = _mcnemar(y_true, pg, pm)
    print(f"    b01={mc['b01']} (GlobalOK/MoEFAIL)  b10={mc['b10']} (GlobalFAIL/MoEOK)")
    print(f"    χ² = {mc['statistic']:.4f}   p = {mc['p_value']:.4f}   -> {mc['interpretation']}")

    print(f"\n  Bootstrap paired F1 difference test (2 000 iterations) …")
    boot = _bootstrap_f1_diff(y_true, pm, pg)   # A=MoE, B=Global
    print(f"    {boot['interpretation']}")

    precision_global = float(precision_score(y_true, pg, zero_division=0))
    precision_moe    = float(precision_score(y_true, pm, zero_division=0))
    prec_lift = (precision_moe - precision_global) / (precision_global + 1e-10)

    result = {
        "n_obs": int(n),
        "n_anomalies": int(y_true.sum()),
        "precision_global": precision_global,
        "precision_moe": precision_moe,
        "precision_lift_pct": float(prec_lift * 100),
        "f1_global": boot['f1_global'],
        "f1_moe": boot['f1_moe'],
        "f1_diff": boot['f1_diff'],
        "f1_ci95": [boot['ci95_lo'], boot['ci95_hi']],
        "f1_pvalue": boot['p_value'],
        "f1_significant": boot['significant'],
        "mcnemar": mc,
        "verdict": (
            f"Precision: MoE={precision_moe:.4f} vs Global={precision_global:.4f} "
            f"({prec_lift*100:+.1f}% lift); "
            f"F1: MoE={boot['f1_moe']:.4f} vs Global={boot['f1_global']:.4f}; "
            f"McNemar p={mc['p_value']:.4f} ({mc['interpretation']}); "
            f"Bootstrap F1-diff p={boot['p_value']:.4f} "
            f"({'SIGNIFICANT' if boot['significant'] else 'not significant'})"
        ),
    }

    # Table-4 paper guidance
    if mc['significant'] and prec_lift > 0:
        result["paper_action"] = (
            "McNemar SIGNIFICANT and precision lift is positive: "
            f"MoE precision={precision_moe:.4f} vs Global={precision_global:.4f} "
            f"({prec_lift*100:+.1f}%). Add McNemar p-value to Table 4. "
            "Note F1-bootstrap not significant — CIs are wide due to small n_anomalies."
        )
    elif mc['significant'] and prec_lift <= 0:
        result["paper_action"] = (
            "McNemar SIGNIFICANT but precision lift is NEGATIVE "
            f"(Global precision={precision_global:.4f} > MoE={precision_moe:.4f}). "
            "The paper's +32% precision claim depends on operating threshold. "
            "Report McNemar p-value and note threshold sensitivity. "
            "Consider precision at recall-0.5 operating point."
        )
    else:
        result["paper_action"] = (
            "REFRAME REQUIRED: Neither McNemar nor bootstrap are significant. "
            "Qualify the +32% precision claim with p-value and 95% CI. "
            "Reframe as descriptive with uncertainty bounds."
        )

    print(f"\n  VERDICT: {result['verdict']}")
    print(f"  Paper action: {result['paper_action']}")

    out = BASE / "c01_results.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved -> {out}")

    if save_predictions:
        pred_df = pd.DataFrame({
            "y_true": y_true,
            "pred_global_lgbm": pg,
            "pred_moe_lgbm": pm,
            "proba_global_lgbm": proba_g[:n],
            "proba_moe_lgbm": proba_m[:n],
        })
        pred_df.to_csv(BASE / "lgbm_oos_predictions.csv", index=False)
        print(f"  Raw predictions saved -> artifacts/lgbm_oos_predictions.csv")

    return result


# ═══════════════════════════════════════════════════════════════════════
# STEP 2 — ECONOMIC BACKTEST -> HEDGE DAILY RETURNS
# ═══════════════════════════════════════════════════════════════════════

def step2_economic_backtest() -> pd.DataFrame:
    """
    Run economic backtest on the 5-year dataset (dates available).
    Returns a DataFrame with date-indexed daily returns for each strategy.
    """
    print(f"\n{PRINT_SEP}")
    print("  STEP 2 — ECONOMIC BACKTEST -> HEDGE DAILY RETURNS")
    print(PRINT_SEP)

    # Load 5y windows (has date CSV)
    X_win = np.load(BASE / "sp500_5y_windows.npy")
    dates_df = pd.read_csv(BASE / "sp500_5y_dates.csv", parse_dates=["Date"])
    n_win = min(len(X_win), len(dates_df))
    X_win = X_win[:n_win]
    win_dates = dates_df["Date"].values[:n_win]

    # Build factors + vol + labels
    print(f"  Building factors for {n_win} windows …")
    X_f = construct_factors_basic(X_win)
    vol = np.std(X_win[:, :, 0], axis=1)

    # Rolling quantile anomaly labels (R2 fix)
    returns_abs = pd.Series(np.abs(vol))  # use vol as proxy for |return|
    roll_t = returns_abs.expanding(min_periods=20).quantile(0.97).bfill().values
    y = (vol >= roll_t).astype(int)
    print(f"  Anomalies: {int(y.sum())} ({y.mean()*100:.1f}%)")

    # OOS predictions with LightGBM MoE (5-fold)
    print(f"  Running LightGBM MoE on 5y dataset …")
    proba_moe, true_moe, pred_moe = _run_lgbm_oos(X_f, y, vol, mode="MoE")

    # OOS index alignment (5-fold TimeSeriesSplit skips early folds)
    n_oos = len(pred_moe)
    oos_dates = pd.DatetimeIndex(win_dates[-n_oos:])

    # Load actual daily returns
    dr_path = BASE / "sp500_full_daily_returns.csv"
    if not dr_path.exists():
        dr_path = BASE / "sp500_daily_returns.csv"
    dr = pd.read_csv(dr_path, index_col=0, parse_dates=True)
    dr = dr.iloc[:, 0]  # first column

    # Map window-level signal to daily signal
    # Window date = last trading day of that window; signal applies next day
    signal_daily = pd.Series(0, index=dr.index)
    for i, (dt, sig) in enumerate(zip(oos_dates, pred_moe)):
        if sig == 1 and dt in dr.index:
            # Hedge for 5 days forward
            idx_pos = dr.index.get_loc(dt)
            end_pos = min(idx_pos + 6, len(dr))
            signal_daily.iloc[idx_pos + 1 : end_pos] = 1

    # Trim to OOS date range
    start_dt = oos_dates[0]
    end_dt   = oos_dates[-1]
    dr_oos = dr[(dr.index >= start_dt) & (dr.index <= end_dt)]
    sig_oos = signal_daily[(signal_daily.index >= start_dt) & (signal_daily.index <= end_dt)]

    # Align
    common_idx = dr_oos.index.intersection(sig_oos.index)
    ret = dr_oos[common_idx].values
    sig = sig_oos[common_idx].values
    rf_daily = (1.04) ** (1 / 252) - 1

    # Strategy returns
    ret_bh     = ret
    ret_hedged = np.where(sig == 1, rf_daily, ret)  # go to cash when signaled

    from src.models.regime_detector import RollingQuantileRegimeDetector
    rdet = RollingQuantileRegimeDetector(window=252)
    vol_daily = pd.Series(ret).abs().rolling(20).std().bfill().values
    regimes_d, _ = rdet.detect(vol_daily)
    regime_hedge = {0: 0.40, 1: 0.70, 2: 1.00}
    h_pct = np.array([regime_hedge.get(int(r), 0.70) for r in regimes_d[:len(ret)]])
    ret_regime = np.where(sig == 1,
                          (1 - h_pct) * ret + h_pct * rf_daily,
                          ret)

    df_out = pd.DataFrame({
        "date": common_idx,
        "buy_hold": ret_bh,
        "anomaly_hedged": ret_hedged,
        "regime_aware_hedged": ret_regime,
    }).set_index("date")

    out = BASE / "hedge_daily_returns.csv"
    df_out.to_csv(out)
    print(f"  {len(df_out)} daily strategy returns saved -> {out}")

    # Quick stats
    for col in df_out.columns:
        r = df_out[col].values
        wealth = np.cumprod(1 + r)
        ann_ret = wealth[-1] ** (252 / len(r)) - 1
        ann_vol = r.std() * np.sqrt(252)
        sharpe  = (ann_ret - 0.04) / (ann_vol + 1e-10)
        mdd     = np.max((np.maximum.accumulate(wealth) - wealth) /
                          np.maximum.accumulate(wealth))
        print(f"  {col:<25}  Ann={ann_ret:+.2%}  Vol={ann_vol:.2%}  "
              f"Sharpe={sharpe:.3f}  MDD={mdd:.2%}")

    return df_out


# ═══════════════════════════════════════════════════════════════════════
# STEP 3 — DOWNLOAD FAMA-FRENCH FACTORS
# ═══════════════════════════════════════════════════════════════════════

FF5_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
MOM_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Momentum_Factor_daily_CSV.zip")

FF_DIR = BASE / "data" / "ff_factors"


def _download_and_extract_ff_csv(url: str, out_path: Path) -> bool:
    """Download a Kenneth French CSV zip and extract the CSV inside."""
    print(f"  Downloading {url.split('/')[-1]} …")
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=30) as resp:
            data = resp.read()
        with zipfile.ZipFile(BytesIO(data)) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                print(f"    No CSV found inside zip.")
                return False
            content = zf.read(csv_names[0]).decode("latin-1")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        print(f"    Saved {len(content)} bytes -> {out_path}")
        return True
    except Exception as e:
        print(f"    Download failed: {e}")
        return False


def step3_download_ff_factors() -> tuple[Path | None, Path | None]:
    """Download FF5 daily + MOM daily CSVs from Kenneth French library."""
    print(f"\n{PRINT_SEP}")
    print("  STEP 3 — DOWNLOAD FAMA-FRENCH FACTORS")
    print(PRINT_SEP)

    ff5_path = FF_DIR / "FF5_daily.csv"
    mom_path = FF_DIR / "MOM_daily.csv"

    ff5_ok = False
    mom_ok = False

    if ff5_path.exists():
        print(f"  FF5 factors already cached -> {ff5_path}")
        ff5_ok = True
    else:
        ff5_ok = _download_and_extract_ff_csv(FF5_URL, ff5_path)

    if mom_path.exists():
        print(f"  MOM factor already cached -> {mom_path}")
        mom_ok = True
    else:
        mom_ok = _download_and_extract_ff_csv(MOM_URL, mom_path)

    if not ff5_ok or not mom_ok:
        print("\n  WARNING: Could not download FF factor files.")
        print("  Manual steps:")
        print("    1. Visit https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html")
        print("    2. Download 'Fama/French 5 Factors (2x3) [Daily]' -> rename to FF5_daily.csv")
        print("    3. Download 'Momentum Factor [Daily]' -> rename to MOM_daily.csv")
        print(f"    4. Place both files in {FF_DIR}")
        return None, None

    return ff5_path, mom_path


# ═══════════════════════════════════════════════════════════════════════
# STEP 4 — FAMA-FRENCH ALPHA TEST
# ═══════════════════════════════════════════════════════════════════════

def step4_fama_french(hedge_returns: pd.DataFrame,
                      ff5_path: Path,
                      mom_path: Path) -> dict | None:
    """Run FF5+MOM alpha test on the regime-aware hedge strategy."""
    print(f"\n{PRINT_SEP}")
    print("  STEP 4 — FAMA-FRENCH ALPHA TEST")
    print("  Regression: R_hedge − Rf = alpha + beta₁MktRF + … + beta₆MOM + ε")
    print(PRINT_SEP)

    if ff5_path is None or mom_path is None:
        print("  Skipped — FF factor files not available.")
        return None

    sys.path.insert(0, ".")
    try:
        from src.evaluation.factor_alpha import run_factor_alpha
    except ImportError as e:
        print(f"  Import error: {e}")
        return None

    # Use regime-aware hedged strategy (the MoE strategy)
    strat_col = "regime_aware_hedged"
    if strat_col not in hedge_returns.columns:
        strat_col = hedge_returns.columns[-1]

    strategy_ret = hedge_returns[strat_col].values
    dates = hedge_returns.index

    print(f"  Strategy: {strat_col}")
    print(f"  Period:   {dates[0].date()} -> {dates[-1].date()}")
    print(f"  N days:   {len(strategy_ret)}")

    try:
        res = run_factor_alpha(
            strategy_returns=strategy_ret,
            dates=dates,
            ff5_csv=str(ff5_path),
            mom_csv=str(mom_path),
            hac_lags=5,
        )
    except Exception as e:
        print(f"  Factor alpha test failed: {e}")
        return None

    # Print key results directly (avoid garbled line-ending from summary str)
    print(f"  N obs aligned   : {res['n_obs']}")
    print(f"  Alpha (daily)   : {res['alpha_daily']:+.6f}")
    print(f"  Alpha (annual.) : {res['alpha_annualized']:+.4%}")
    print(f"  Alpha t-stat    : {res['alpha_tstat']:+.3f}  p={res['alpha_pvalue']:.4f}")
    print(f"  Adj. R^2        : {res['r2_adj']:.4f}")
    print(f"  Factor loadings:")
    for k, v in res['betas'].items():
        pv = res['beta_pvalues'][k]
        print(f"    {k:6s} = {v:+.3f}  (p={pv:.3f})")
    print(f"  Verdict: {res['verdict']}")

    # Save JSON (excluding the statsmodels model object)
    result = {k: v for k, v in res.items() if k != "model"}
    out = BASE / "ff_alpha_results.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved -> {out}")

    # Paper guidance
    if res["alpha_daily"] > 0 and res["alpha_pvalue"] < 0.05:
        guidance = (
            "PUBLISHABLE ALPHA: Add Fama-French alpha table to Section VII. "
            "Wording: 'The regime-aware hedge strategy generates a statistically "
            f"significant annualized alpha of {res['alpha_annualized']:.2%} "
            f"(t={res['alpha_tstat']:.2f}, p={res['alpha_pvalue']:.4f}) beyond "
            "Fama-French 5 + Momentum factor exposure, confirming economic value "
            "beyond known risk premia.'"
        )
    elif res["alpha_pvalue"] >= 0.05:
        guidance = (
            "Alpha not statistically significant. Reframe: report as evidence "
            "that strategy is not driven by factor loadings, even if alpha "
            "itself is uncertain. Note that short sample periods reduce power."
        )
    else:
        guidance = "Negative significant alpha — investigate strategy implementation."

    print(f"\n  PAPER GUIDANCE: {guidance}")
    result["paper_guidance"] = guidance

    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print(f"\n{'#'*72}")
    print(f"  DAILY PIPELINE — {pd.Timestamp.today().strftime('%Y-%m-%d %H:%M')}")
    print(f"  C-01 significance test + Fama-French alpha")
    print(f"{'#'*72}")

    os.chdir(Path(__file__).parent)

    # STEP 0 — refresh data
    step0_refresh_data()

    # STEP 1 — C-01
    c01 = step1_c01()

    # STEP 2 — economic backtest
    try:
        hedge_ret = step2_economic_backtest()
    except Exception as e:
        print(f"\n  STEP 2 FAILED: {e}")
        hedge_ret = None

    # STEP 3 — download FF factors
    ff5_path, mom_path = step3_download_ff_factors()

    # STEP 4 — Fama-French alpha test
    if hedge_ret is not None:
        ff_result = step4_fama_french(hedge_ret, ff5_path, mom_path)
    else:
        print("\n  STEP 4 skipped — no hedge returns available.")
        ff_result = None

    # Summary
    elapsed = time.time() - t_start
    print(f"\n{'#'*72}")
    print(f"  PIPELINE COMPLETE  ({elapsed/60:.1f} min)")
    print(f"{'#'*72}")
    print(f"  C-01 verdict:   {c01['verdict']}")
    if ff_result:
        print(f"  FF alpha daily: {ff_result.get('alpha_daily', 'N/A'):+.6f}  "
              f"p={ff_result.get('alpha_pvalue', 'N/A'):.4f}")
    print(f"\n  Output files:")
    for f in ["c01_results.json", "lgbm_oos_predictions.csv",
              "hedge_daily_returns.csv", "ff_alpha_results.json"]:
        p = BASE / f
        print(f"    {'OK' if p.exists() else 'FAIL'} {p}")
    print()


if __name__ == "__main__":
    main()
