"""
Kronos-style publication figures for RC-MoE Financial Anomaly Detection paper.
Generates 8 figures into kronos_style_figures/ directory.

REAL-DATA-ONLY policy (no proxy/fabricated series):
  * fig1 radar  : 8 axes, every value sourced from the paper's result tables.
                  ECI removed from the radar (it has its own figure); replaced
                  by Sortino ratio, which is defined for all five strategies.
  * fig2 returns: real hedge series from hedge_daily_returns.csv + a REAL
                  GARCH(1,1) 2 sigma overlay fitted with the `arch` package.
  * fig4/fig8   : real, DATED out-of-sample predictions from
                  artifacts/sp500_oos_dated_predictions.csv (produced by
                  src.evaluation.multimarket_eventwise_canonical). No
                  last-N-rows alignment, no fabricated model panels.

Run: .venv/Scripts/python.exe -m src.utils.make_kronos_style_figs
"""

import sys, os, json, warnings
# Remove this script's directory from sys.path so stdlib modules
# (logging, etc.) are not shadowed by local files in src/utils/
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir in sys.path:
    sys.path.remove(_this_dir)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
warnings.filterwarnings("ignore")

OUT = "kronos_style_figures"
os.makedirs(OUT, exist_ok=True)

# Wong colorblind-safe palette
CB_BLUE, CB_ORANGE, CB_GREEN = "#0072B2", "#E69F00", "#009E73"
CB_RED, CB_PURPLE, CB_CYAN   = "#D55E00", "#CC79A7", "#56B4E9"
CB_YELLOW, CB_GREY, CB_BLACK = "#F0E442", "#999999", "#000000"

REGIME_COLORS = {0: CB_CYAN, 1: CB_ORANGE, 2: CB_RED}
REGIME_LABELS = {0: "Low-Vol", 1: "Med-Vol", 2: "High-Vol"}
REGIME_ALPHA  = {0: 0.15, 1: 0.12, 2: 0.18}

plt.rcParams.update({
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
    "figure.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif",
})

BH_MDD = 0.3392   # Buy-and-Hold max drawdown (Table: backtest)


# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────

def compute_regime_labels(returns_series, window=63, q_lo=0.33, q_hi=0.67):
    """Rolling realized-vol terciles -> 3 regimes (0=Low,1=Med,2=High).
    Same tercile rule used by the model pipeline; derived from returns only."""
    vol = returns_series.rolling(window).std()
    lo, hi = vol.quantile(q_lo), vol.quantile(q_hi)
    labels = np.where(vol < lo, 0, np.where(vol > hi, 2, 1))
    return labels, vol


def sigma_rule_signal(returns, k=2.0, window=21):
    """Rolling mean +/- k*sigma downside breach detector (honest label)."""
    mu  = returns.rolling(window).mean()
    sig = returns.rolling(window).std()
    return (returns < mu - k * sig).astype(int)


def garch_signal(returns_decimal, k=2.0):
    """REAL GARCH(1,1) anomaly detector: |r| > k * conditional_vol.
    Returns a 0/1 numpy array aligned to returns_decimal (a pandas Series)."""
    from arch import arch_model
    r_pct = returns_decimal.dropna() * 100.0
    am = arch_model(r_pct, vol="Garch", p=1, q=1, mean="Constant", dist="normal")
    res = am.fit(disp="off")
    cond_vol = res.conditional_volatility            # percent units
    sig = (np.abs(r_pct.values) > k * cond_vol).astype(int)
    out = pd.Series(0, index=returns_decimal.index)
    out.loc[r_pct.index] = sig
    return out.values


def hedge_from_signal(ret, signal, hold_days=5, rf_annual=0.04):
    """Convert a 0/1 detection signal into a hedged daily-return series:
    on a signal, sit in the risk-free asset for the next `hold_days`."""
    ret = np.asarray(ret, dtype=float)
    signal = np.asarray(signal)
    hedged = ret.copy()
    rf_daily = rf_annual / 252
    mask = np.zeros(len(ret), dtype=bool)
    for i in range(len(signal)):
        if signal[i]:
            mask[i:i + hold_days] = True
    hedged[mask] = rf_daily
    return hedged


def cumret(r):
    return (1 + np.asarray(r, dtype=float)).cumprod() - 1


def normalize_cols(mat):
    mat = np.asarray(mat, dtype=float)
    vmin, vmax = mat.min(axis=0), mat.max(axis=0)
    rng = np.where(vmax - vmin == 0, 1.0, vmax - vmin)
    return (mat - vmin) / rng


# ─────────────────────────────────────────────────────────────
# Figure 1 - Radar overview (real numbers, 8 axes)
# ─────────────────────────────────────────────────────────────

def make_fig1_radar():
    print("Fig 1: radar overview (real numbers) ...")
    metrics = ["Detection\nPrecision", "Detection\nRecall", "Event-F1",
               "Sharpe\nRatio", "Annualized\nReturn", "Max-DD\nProtection",
               "Hedging\nEfficiency", "Sortino\nRatio"]
    # Every value below is from the paper's result tables / OOS evaluation text.
    # [prec, recall, eventF1, sharpe, annret, dd_protection, hedge_eff, sortino]
    models = {
        "LightGBM RC-MoE (Ours)": dict(color=CB_BLUE, lw=2.6, ls="-",
            vals=[0.543, 0.328, 0.540, 0.772, 0.1527, (BH_MDD-0.2071)/BH_MDD, 0.315, 0.964]),
        "LightGBM Global": dict(color=CB_CYAN, lw=1.8, ls="--",
            vals=[0.381, 0.533, 0.444, 0.717, 0.1407, (BH_MDD-0.1800)/BH_MDD, 0.279, 0.874]),
        "XGBoost MoE": dict(color=CB_GREEN, lw=1.8, ls="--",
            vals=[0.435, 0.667, 0.526, 0.748, 0.1457, (BH_MDD-0.2162)/BH_MDD, 0.237, 0.929]),
        "GARCH(1,1) 2.0$\\sigma$": dict(color=CB_RED, lw=1.8, ls="-.",
            vals=[0.394, 0.707, 0.506, 0.777, 0.1424, (BH_MDD-0.1569)/BH_MDD, 0.170, 0.940]),
        "Buy-and-Hold": dict(color=CB_GREY, lw=1.4, ls=":",
            vals=[0.000, 0.000, 0.000, 0.556, 0.1315, 0.000, 0.000, 0.605]),
    }
    raw = np.array([m["vals"] for m in models.values()])
    norm = normalize_cols(raw)
    for i, m in enumerate(models.values()):
        m["norm"] = norm[i]

    N = len(metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8.4, 7.6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=9.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7.5, color="grey")
    ax.tick_params(axis="x", pad=12)

    for name, m in models.items():
        v = m["norm"].tolist(); v += v[:1]
        ax.plot(angles, v, color=m["color"], lw=m["lw"], ls=m["ls"], label=name)
        ax.fill(angles, v, color=m["color"], alpha=0.05)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=3,
              fontsize=8.5, framealpha=0.9, columnspacing=1.2)
    ax.set_title("RC-MoE performance overview across eight metrics\n"
                 "(each axis min-max normalised; farther from centre is better)",
                 pad=28, fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.85, bottom=0.13, left=0.09, right=0.91)
    p = os.path.join(OUT, "fig1_radar_overview.png")
    fig.savefig(p, bbox_inches="tight", dpi=300); plt.close(fig)
    print(f"  saved {p}")


# ─────────────────────────────────────────────────────────────
# Figure 2 - Cumulative returns (real hedge series + real GARCH)
# ─────────────────────────────────────────────────────────────

def make_fig2_cumulative_returns():
    print("Fig 2: cumulative returns (real GARCH) ...")
    h = pd.read_csv("artifacts/hedge_daily_returns.csv", parse_dates=["date"]).set_index("date")
    sp = h["buy_hold"]

    # REAL GARCH(1,1) 2-sigma strategy over the same window
    try:
        g_sig = garch_signal(sp, k=2.0)
        garch_ret = hedge_from_signal(sp.values, g_sig)
        garch_ok = True
    except Exception as e:
        print("  [warn] GARCH fit failed, omitting GARCH curve:", e)
        garch_ok = False

    strats = [
        ("Buy-and-Hold",                sp.values,                  CB_BLACK, 2.0, "-"),
        ("Anomaly-Hedged (LightGBM)",   h["anomaly_hedged"].values, CB_CYAN,  1.7, "--"),
        ("RC-MoE Regime-Aware (Ours)",  h["regime_aware_hedged"].values, CB_BLUE, 2.4, "-"),
    ]
    if garch_ok:
        strats.insert(1, ("GARCH(1,1) 2.0$\\sigma$ (real fit)", garch_ret, CB_RED, 1.6, "-."))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    idx = h.index
    for name, r, col, lw, ls in strats:
        ax1.plot(idx, cumret(r)*100, color=col, lw=lw, ls=ls, label=name)
        if name != "Buy-and-Hold":
            ax2.plot(idx, (cumret(r)-cumret(sp.values))*100, color=col, lw=lw, ls=ls, label=name)

    ax1.axhline(0, color="black", lw=0.6, alpha=0.4)
    ax1.set_ylabel("Cumulative Return (%)")
    ax1.set_title("(a) Cumulative Strategy Returns (2022–2025)", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax1.tick_params(axis="x", rotation=30, labelsize=8)

    ax2.axhline(0, color="black", lw=0.8, alpha=0.5, ls="--")
    ax2.set_ylabel("Cumulative Excess Return\nvs Buy-and-Hold (%)")
    ax2.set_title("(b) Cumulative Excess Return Relative to Buy-and-Hold", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="lower left", framealpha=0.9)
    ax2.tick_params(axis="x", rotation=30, labelsize=8)
    ax2.set_xlabel("Date")

    fig.tight_layout()
    p = os.path.join(OUT, "fig2_cumulative_returns.png")
    fig.savefig(p, bbox_inches="tight", dpi=300); plt.close(fig)
    print(f"  saved {p}")


# ─────────────────────────────────────────────────────────────
# Figure 3 - Regime t-SNE + KDE (real factor data)
# ─────────────────────────────────────────────────────────────

def make_fig3_regime_tsne_kde():
    print("Fig 3: regime t-SNE + KDE ...")
    X = np.load("artifacts/X_factors.npy")
    y = np.load("artifacts/y_factors.npy")
    df = pd.read_csv("artifacts/journal_factors.csv")

    vol_col = df["Composite_Volatility"].values
    q1, q2 = np.percentile(vol_col, [33, 67])
    regimes = np.where(vol_col < q1, 0, np.where(vol_col > q2, 2, 1))

    Xs = StandardScaler().fit_transform(X)
    Xp = PCA(n_components=min(20, X.shape[1])).fit_transform(Xs)
    print("    t-SNE ...")
    Z = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000,
             learning_rate="auto", n_jobs=1).fit_transform(Xp)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    names = ["RC-MoE\n(Low-Vol)", "RC-MoE\n(Med-Vol)", "RC-MoE\n(High-Vol)", "Global\n(No Regime)"]
    rids  = [0, 1, 2, None]
    rt = df["Return_Trend"].values
    x_grid = np.linspace(rt.min(), rt.max(), 200)

    for col, (name, rid) in enumerate(zip(names, rids)):
        at, ak = axes[0, col], axes[1, col]
        if rid is not None:
            mr = regimes == rid
            at.scatter(Z[~mr, 0], Z[~mr, 1], c="lightgrey", s=4, alpha=0.3)
            at.scatter(Z[mr, 0], Z[mr, 1], c=REGIME_COLORS[rid], s=8, alpha=0.7, label=REGIME_LABELS[rid])
            ma = mr & (y == 1)
            at.scatter(Z[ma, 0], Z[ma, 1], c=CB_BLACK, s=22, marker="*", alpha=0.9, label="Anomaly")
            ak.plot(x_grid, gaussian_kde(rt, bw_method=0.3)(x_grid), color=CB_BLACK, lw=1.5, label="Global (all)")
            ak.plot(x_grid, gaussian_kde(rt[mr], bw_method=0.3)(x_grid), color=REGIME_COLORS[rid], lw=2, ls="--", label=REGIME_LABELS[rid])
        else:
            for r in [0, 1, 2]:
                m = regimes == r
                at.scatter(Z[m, 0], Z[m, 1], c=REGIME_COLORS[r], s=5, alpha=0.5, label=REGIME_LABELS[r])
                ak.plot(x_grid, gaussian_kde(rt[m], bw_method=0.3)(x_grid), color=REGIME_COLORS[r], lw=1.8, label=REGIME_LABELS[r])
            ma = y == 1
            at.scatter(Z[ma, 0], Z[ma, 1], c=CB_BLACK, s=22, marker="*", alpha=0.9, label="Anomaly")
        at.set_title(name, fontsize=10, fontweight="bold")
        at.set_xlabel("t-SNE 1", fontsize=8); at.set_ylabel("t-SNE 2", fontsize=8)
        at.legend(fontsize=7, markerscale=1.4, loc="upper right"); at.tick_params(labelsize=7); at.grid(alpha=0.2)
        ak.set_xlabel("Return Trend Factor", fontsize=8); ak.set_ylabel("Density", fontsize=8)
        ak.legend(fontsize=7); ak.tick_params(labelsize=7); ak.grid(alpha=0.2)

    fig.suptitle("Regime Feature-Space Distributions: t-SNE Embeddings (top) and "
                 "Return-Trend KDE (bottom). Regimes from rolling-volatility terciles.",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    p = os.path.join(OUT, "fig3_regime_tsne_kde.png")
    fig.savefig(p, bbox_inches="tight", dpi=300); plt.close(fig)
    print(f"  saved {p}")


# ─────────────────────────────────────────────────────────────
# helper: load real dated S&P OOS predictions
# ─────────────────────────────────────────────────────────────

def _load_dated_preds():
    p = "artifacts/sp500_oos_dated_predictions.csv"
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} not found. Run: .venv/Scripts/python.exe -m "
            "src.evaluation.multimarket_eventwise_canonical")
    d = pd.read_csv(p, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return d


# ─────────────────────────────────────────────────────────────
# Figure 4 - Anomaly detection full series + COVID zoom (real dated preds)
# ─────────────────────────────────────────────────────────────

def make_fig4_anomaly_detection_zoom():
    print("Fig 4: anomaly detection + zoom (real dated preds) ...")
    preds = _load_dated_preds()
    prices = pd.read_csv("artifacts/sp500_full_daily_prices.csv", parse_dates=["Date"])
    prices = prices[prices["Date"] >= "2016-01-01"].reset_index(drop=True)
    prices["Return"] = prices["Close"].pct_change()
    rets = prices["Return"].fillna(0)
    regime_arr, _ = compute_regime_labels(rets)
    garch_sig = pd.Series(garch_signal(rets, k=2.0), index=prices.index)

    # merge dated predictions onto price dates
    pmap = preds.set_index("date")
    prices = prices.set_index("Date")
    prices["y_true"]      = pmap["y_true"].reindex(prices.index).fillna(-1)
    prices["pred_moe"]    = pmap["pred_moe"].reindex(prices.index).fillna(0)
    prices["proba_moe"]   = pmap["proba_moe"].reindex(prices.index)
    prices = prices.reset_index().rename(columns={"index": "Date"})

    covid_start, covid_end = pd.Timestamp("2020-02-01"), pd.Timestamp("2020-07-01")

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[2.5, 1, 1],
                           height_ratios=[1, 1], hspace=0.35, wspace=0.32)
    ax_main = fig.add_subplot(gs[:, 0])
    dates = prices["Date"].values
    close = prices["Close"].values
    ax_main.plot(dates, close, color=CB_BLACK, lw=0.8, label="S&P 500 Close")

    for r in [0, 1, 2]:
        mask = regime_arr == r
        in_r, s = False, 0
        for i, m in enumerate(mask):
            if m and not in_r: s, in_r = i, True
            elif not m and in_r:
                ax_main.axvspan(dates[s], dates[i], color=REGIME_COLORS[r], alpha=REGIME_ALPHA[r]); in_r = False
        if in_r: ax_main.axvspan(dates[s], dates[-1], color=REGIME_COLORS[r], alpha=REGIME_ALPHA[r])

    ta = prices[prices["y_true"] == 1]
    ax_main.scatter(ta["Date"], ta["Close"], color=CB_BLACK, s=32, marker="^", zorder=5, label="True Anomaly (OOS)", alpha=0.85)
    md = prices[prices["pred_moe"] == 1]
    ax_main.scatter(md["Date"], md["Close"], color=CB_BLUE, s=26, marker="o", zorder=4, label="RC-MoE Detection (OOS)", alpha=0.7)
    gd = prices[garch_sig.values == 1]
    ax_main.scatter(gd["Date"], gd["Close"], color=CB_RED, s=16, marker="D", zorder=3, label="GARCH(2σ) Detection", alpha=0.4)

    sub = prices[(prices["Date"] >= covid_start) & (prices["Date"] <= covid_end)]
    y_lo, y_hi = sub["Close"].min()*0.95, sub["Close"].max()*1.05
    ax_main.add_patch(mpatches.Rectangle(
        (matplotlib.dates.date2num(covid_start), y_lo),
        matplotlib.dates.date2num(covid_end)-matplotlib.dates.date2num(covid_start),
        y_hi-y_lo, lw=1.5, edgecolor=CB_ORANGE, facecolor=CB_YELLOW, alpha=0.15, zorder=6))
    ax_main.annotate("COVID-19\nCrash Focus", xy=(covid_start, y_hi),
                     xytext=(pd.Timestamp("2017-06-01"), y_hi*0.8),
                     arrowprops=dict(arrowstyle="->", color=CB_ORANGE, lw=1.5),
                     fontsize=9, color=CB_ORANGE, fontweight="bold")
    reg_patches = [mpatches.Patch(color=REGIME_COLORS[r], alpha=0.5, label=REGIME_LABELS[r]) for r in [0,1,2]]
    ax_main.legend(handles=reg_patches + [
        plt.Line2D([],[],color=CB_BLACK,lw=1,label="S&P 500 Close"),
        plt.scatter([],[],color=CB_BLACK,marker="^",s=40,label="True Anomaly (OOS)"),
        plt.scatter([],[],color=CB_BLUE,marker="o",s=40,label="RC-MoE (OOS)"),
        plt.scatter([],[],color=CB_RED,marker="D",s=30,label="GARCH(2σ)")],
        fontsize=7, loc="upper left", framealpha=0.9)
    ax_main.set_ylabel("S&P 500 Close Price (USD)")
    ax_main.set_title("(a) S&P 500 Anomaly Detection (2016–2025)\n"
                      "Regime shading: Low (blue), Med (orange), High (red); markers at true OOS dates",
                      fontsize=10, fontweight="bold")
    ax_main.tick_params(axis="x", rotation=30, labelsize=8)

    # right zoom panels
    sub = prices[(prices["Date"] >= covid_start) & (prices["Date"] <= covid_end)]
    axp = fig.add_subplot(gs[0, 1])
    axp.plot(sub["Date"], sub["Close"], color=CB_BLACK, lw=1.5)
    axp.fill_between(sub["Date"], sub["Close"], sub["Close"].min()*0.95, alpha=0.1, color=CB_BLUE)
    axp.set_title("(b) COVID Crash - Close Price", fontsize=9, fontweight="bold")
    axp.set_ylabel("Close", fontsize=8); axp.tick_params(axis="x", rotation=40, labelsize=7)

    axg = fig.add_subplot(gs[0, 2])
    gc = garch_sig.reindex(sub.index).fillna(0).values
    axg.bar(sub["Date"], gc, color=CB_RED, width=1, alpha=0.7)
    axg.plot(sub["Date"], sub["Return"].abs(), color=CB_ORANGE, lw=1.2, label="|Return|")
    axg.set_title("(c) GARCH(2σ) Signal", fontsize=9, fontweight="bold")
    axg.set_ylabel("Signal + |Return|", fontsize=8); axg.legend(fontsize=7); axg.tick_params(axis="x", rotation=40, labelsize=7)

    axmp = fig.add_subplot(gs[1, 1])
    has_oos = sub["proba_moe"].notna().any()
    if has_oos:
        axmp.bar(sub["Date"], sub["proba_moe"].fillna(0), color=CB_BLUE, width=1, alpha=0.8)
        axmp.axhline(0.5, color="red", lw=1, ls="--", label="Decision boundary")
        axmp.legend(fontsize=7)
    else:
        axmp.text(0.5, 0.5, "COVID window in\ntraining fold\n(no OOS preds)", ha="center", va="center", transform=axmp.transAxes, fontsize=8, color="grey")
    axmp.set_title("(d) RC-MoE Anomaly Probability", fontsize=9, fontweight="bold")
    axmp.set_ylabel("P(Anomaly)", fontsize=8); axmp.set_ylim(0, 1); axmp.tick_params(axis="x", rotation=40, labelsize=7)

    axt = fig.add_subplot(gs[1, 2])
    ct = sub["y_true"].replace(-1, np.nan)
    axt.bar(sub["Date"], ct.fillna(0).clip(lower=0), color=CB_BLACK, width=1, alpha=0.6, label="True Anomaly")
    axt.step(sub["Date"], (sub["proba_moe"].fillna(0) > 0.5).astype(int), color=CB_BLUE, lw=1.5, where="post", label="RC-MoE Pred")
    axt.set_title("(e) True vs RC-MoE (COVID Zoom)", fontsize=9, fontweight="bold")
    axt.set_ylabel("Anomaly Label", fontsize=8); axt.legend(fontsize=7); axt.tick_params(axis="x", rotation=40, labelsize=7)

    p = os.path.join(OUT, "fig4_anomaly_detection_zoom.png")
    fig.savefig(p, bbox_inches="tight", dpi=300); plt.close(fig)
    print(f"  saved {p}")


# ─────────────────────────────────────────────────────────────
# Figure 5 - Crisis focus OHLC (real OHLCV)
# ─────────────────────────────────────────────────────────────

def make_fig5_crisis_focus():
    print("Fig 5: crisis focus OHLC ...")
    ohlcv = pd.read_csv("artifacts/ohlcv_GSPC.csv", parse_dates=["Date"]).set_index("Date")
    ohlcv["Return"] = ohlcv["Close"].pct_change()
    focus = ohlcv.loc["2020-02-01":"2020-06-30"].copy()
    zoom  = ohlcv.loc["2020-02-20":"2020-04-15"].copy()
    g_sig_f = sigma_rule_signal(focus["Return"], k=2.0)
    regime_full, _ = compute_regime_labels(ohlcv["Return"].fillna(0))
    regime_f = regime_full[ohlcv.index.get_indexer(focus.index)]

    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(4, 3, figure=fig, width_ratios=[3, 1, 1],
                           height_ratios=[3, 2, 2, 2], hspace=0.75, wspace=0.45)
    ax_full = fig.add_subplot(gs[:2, 0])
    ax_zc   = fig.add_subplot(gs[0, 1:])
    ax_gt   = fig.add_subplot(gs[1, 1:])
    ax_moe  = fig.add_subplot(gs[2, 1:])
    ax_reg  = fig.add_subplot(gs[3, 1:])

    up = focus[focus["Close"] >= focus["Open"]]; dn = focus[focus["Close"] < focus["Open"]]
    ax_full.bar(up.index, up["Close"]-up["Open"], bottom=up["Open"], color=CB_GREEN, width=0.6, alpha=0.85)
    ax_full.bar(dn.index, dn["Close"]-dn["Open"], bottom=dn["Open"], color=CB_RED, width=0.6, alpha=0.85)
    ax_full.vlines(focus.index, focus["Low"], focus["High"], color="black", lw=0.5, alpha=0.6)
    for r in [0, 1, 2]:
        m = regime_f == r; dd = focus.index.values; in_r, s = False, 0
        for i, flag in enumerate(m):
            if flag and not in_r: s, in_r = i, True
            elif not flag and in_r: ax_full.axvspan(dd[s], dd[i], color=REGIME_COLORS[r], alpha=0.15); in_r=False
        if in_r: ax_full.axvspan(dd[s], dd[-1], color=REGIME_COLORS[r], alpha=0.15)
    gdt = focus.index[g_sig_f == 1]
    ax_full.scatter(gdt, focus.loc[gdt, "High"]*1.01, color=CB_RED, marker="v", s=40, zorder=5, label="2σ-rule signal")
    z_lo, z_hi = zoom["Low"].min()*0.96, zoom["High"].max()*1.04
    ax_full.add_patch(mpatches.Rectangle(
        (matplotlib.dates.date2num(zoom.index[0]), z_lo),
        matplotlib.dates.date2num(zoom.index[-1])-matplotlib.dates.date2num(zoom.index[0]),
        z_hi-z_lo, lw=2, edgecolor=CB_ORANGE, facecolor=CB_YELLOW, alpha=0.12, zorder=3))
    ax_full.set_title("(a) S&P 500 - COVID-19 Crash (Feb–Jun 2020, OHLC bars)", fontsize=10, fontweight="bold")
    ax_full.set_ylabel("Price (USD)", fontsize=9); ax_full.tick_params(axis="x", rotation=45, labelsize=7)
    ax_full.legend(fontsize=8, loc="upper right")
    axv = ax_full.twinx()
    axv.bar(focus.index, focus["Volume"]/1e9, color=CB_GREY, alpha=0.18, width=0.6)
    axv.set_ylabel("Volume (bn)", fontsize=7, color=CB_GREY); axv.tick_params(labelsize=7, colors=CB_GREY); axv.grid(False)

    ax_zc.plot(zoom.index, zoom["Close"], color=CB_BLUE, lw=2)
    ax_zc.fill_between(zoom.index, zoom["Close"], zoom["Close"].min()*0.97, alpha=0.15, color=CB_BLUE)
    ax_zc.set_title("(b) Focus: Close Price (Feb 20 – Apr 15)", fontsize=9, fontweight="bold")
    ax_zc.set_ylabel("Close", fontsize=8); ax_zc.tick_params(axis="x", rotation=40, labelsize=7)

    uz = zoom[zoom["Close"] >= zoom["Open"]]; dz = zoom[zoom["Close"] < zoom["Open"]]
    ax_gt.bar(uz.index, uz["Close"]-uz["Open"], bottom=uz["Open"], color=CB_GREEN, alpha=0.85, width=0.6)
    ax_gt.bar(dz.index, dz["Close"]-dz["Open"], bottom=dz["Open"], color=CB_RED, alpha=0.85, width=0.6)
    ax_gt.vlines(zoom.index, zoom["Low"], zoom["High"], color="black", lw=0.6, alpha=0.5)
    ax_gt.set_title("(c) OHLC Candles - Zoom", fontsize=9, fontweight="bold")
    ax_gt.set_ylabel("Price", fontsize=8); ax_gt.tick_params(axis="x", rotation=40, labelsize=7)

    gz = sigma_rule_signal(zoom["Return"], k=2.0)
    ax_moe.bar(zoom.index, zoom["Return"].abs()*100, color=CB_ORANGE, alpha=0.5, width=0.6, label="|Return| %")
    ax_moe.bar(zoom.index, gz*zoom["Return"].abs()*100, color=CB_RED, alpha=0.9, width=0.6, label="2σ-rule signal")
    ax_moe.set_title("(d) 2σ-Rule Detection - Zoom", fontsize=9, fontweight="bold")
    ax_moe.set_ylabel("|Return| × Signal (%)", fontsize=8); ax_moe.legend(fontsize=7); ax_moe.tick_params(axis="x", rotation=40, labelsize=7)

    reg_z = regime_full[ohlcv.index.get_indexer(zoom.index)]
    ax_reg.bar(zoom.index, np.ones(len(zoom)), color=[REGIME_COLORS[r] for r in reg_z], alpha=0.7, width=0.6)
    ax_reg.set_yticks([]); ax_reg.set_title("(e) Volatility Regime - Zoom", fontsize=9, fontweight="bold")
    ax_reg.legend(handles=[mpatches.Patch(color=REGIME_COLORS[r], label=REGIME_LABELS[r]) for r in [0,1,2]], fontsize=7, loc="upper right")
    ax_reg.tick_params(axis="x", rotation=40, labelsize=7)

    p = os.path.join(OUT, "fig5_crisis_focus_ohlc.png")
    fig.savefig(p, bbox_inches="tight", dpi=300); plt.close(fig)
    print(f"  saved {p}")


# ─────────────────────────────────────────────────────────────
# Figure 6 - Representative return windows per regime (real windows)
# ─────────────────────────────────────────────────────────────

def make_fig6_regime_patterns():
    print("Fig 6: regime characteristic windows ...")
    windows = np.load("artifacts/sp500_5y_windows.npy")  # (1096,128,10)
    dates = pd.read_csv("artifacts/sp500_5y_dates.csv", parse_dates=["Date"])["Date"]
    ohlcv = pd.read_csv("artifacts/ohlcv_GSPC.csv", parse_dates=["Date"]).set_index("Date")
    rets = ohlcv["Close"].pct_change().fillna(0)
    regime_full, _ = compute_regime_labels(rets)
    wreg = []
    for d in dates:
        wreg.append(int(regime_full[ohlcv.index.get_loc(d)]) if d in ohlcv.index else 1)
    wreg = np.array(wreg)

    n_ex = 3
    fig, axes = plt.subplots(3, n_ex*2, figsize=(18, 10))
    close_col = 3  # Close channel in OHLCV windows
    names = ["Low-Vol Regime", "Med-Vol Regime", "High-Vol Regime"]
    for row, rid in enumerate([0, 1, 2]):
        idxs = np.where(wreg == rid)[0]
        if len(idxs) < n_ex:
            idxs = np.pad(idxs, (0, n_ex-len(idxs)), mode="edge")
        vols = np.array([windows[i, :, close_col].std() for i in idxs])
        chosen = [idxs[np.argmin(np.abs(vols - np.percentile(vols, p)))] for p in (10, 50, 90)]
        for cp, wi in enumerate(chosen):
            cw = windows[wi, :, close_col]
            ret_w = np.diff(cw) / (np.abs(cw[:-1]) + 1e-9)
            ap, av = axes[row, cp*2], axes[row, cp*2+1]
            x = np.arange(len(cw))
            ap.plot(x, cw, color=REGIME_COLORS[rid], lw=1.5)
            ap.fill_between(x, cw, cw.min(), alpha=0.15, color=REGIME_COLORS[rid])
            anom_x = np.where(np.abs(ret_w) > ret_w.mean() + 2*ret_w.std())[0] + 1
            ap.scatter(anom_x, cw[anom_x], color=CB_RED, s=15, zorder=5)
            volw = windows[wi, :, 4] if windows.shape[2] > 4 else np.zeros(len(cw))
            ap.set_title(["Low","Mid","High"][cp]+"-variance", fontsize=8)
            ap.tick_params(labelsize=6); ap.set_xlabel("Day in window", fontsize=7)
            av.bar(x, volw, color=[CB_GREEN if v>=0 else CB_RED for v in np.diff(cw, prepend=cw[0])], alpha=0.7, width=0.8)
            av.tick_params(labelsize=6); av.set_xlabel("Day", fontsize=7); av.set_ylabel("Volume", fontsize=7)
        fig.text(0.005, 0.82 - row*0.33, names[row], va="center", rotation=90,
                 fontsize=10, fontweight="bold", color=REGIME_COLORS[rid])
    fig.suptitle("Representative 128-Day Windows per Regime (Left=Price, Right=Volume; red=within-window anomalies)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0, 1, 0.97])
    p = os.path.join(OUT, "fig6_regime_patterns.png")
    fig.savefig(p, bbox_inches="tight", dpi=300); plt.close(fig)
    print(f"  saved {p}")


# ─────────────────────────────────────────────────────────────
# Figure 7 - Multi-market t-SNE + KDE (real returns)
# ─────────────────────────────────────────────────────────────

def make_fig7_multimarket_distributions():
    print("Fig 7: multi-market distributions ...")
    markets = {
        "S&P 500":      ("artifacts/ohlcv_GSPC.csv", "Close", CB_BLUE),
        "NASDAQ":       ("artifacts/returns_NASDAQ.csv", "^IXIC", CB_RED),
        "Russell 2000": ("artifacts/returns_Russell2000.csv", "^RUT", CB_GREEN),
        "Bitcoin":      ("artifacts/returns_Bitcoin.csv", "BTC-USD", CB_ORANGE),
    }
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for col, (mkt, (fp, cn, mc)) in enumerate(markets.items()):
        df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date").sort_index()
        ret = (df["Close"].pct_change() if cn == "Close" else df[cn]).dropna()
        ret = ret[(ret.index >= "2016-01-01") & (ret.index <= "2025-12-31")]
        ra = ret.values
        reg, _ = compute_regime_labels(ret, window=63)

        win = 20
        Xm = np.lib.stride_tricks.sliding_window_view(ra, win)
        rm = reg[win-1:]
        valid = ~np.any(np.isnan(Xm), axis=1)
        Xm, rm = Xm[valid], rm[valid]
        Xp = PCA(n_components=min(10, Xm.shape[1])).fit_transform(StandardScaler().fit_transform(Xm))
        Z = TSNE(n_components=2, random_state=42, perplexity=25, max_iter=500, learning_rate="auto", n_jobs=1).fit_transform(Xp)

        at, ak = axes[0, col], axes[1, col]
        for r in [0, 1, 2]:
            m = rm == r
            at.scatter(Z[m, 0], Z[m, 1], c=REGIME_COLORS[r], s=4, alpha=0.5, label=REGIME_LABELS[r])
        at.set_title(mkt, fontsize=11, fontweight="bold", color=mc)
        at.set_xlabel("t-SNE 1", fontsize=8); at.set_ylabel("t-SNE 2", fontsize=8); at.tick_params(labelsize=7); at.grid(alpha=0.2)
        if col == 0: at.legend(fontsize=7, markerscale=2)

        xg = np.linspace(np.percentile(ra, 0.5), np.percentile(ra, 99.5), 300)
        for r in [0, 1, 2]:
            vr = ra[reg == r]
            if len(vr) > 10:
                ak.plot(xg, gaussian_kde(vr, bw_method=0.2)(xg), color=REGIME_COLORS[r], lw=2, label=REGIME_LABELS[r])
        ak.plot(xg, gaussian_kde(ra[~np.isnan(ra)], bw_method=0.2)(xg), color=CB_BLACK, lw=1.5, ls="--", label="Global")
        ak.set_xlabel("Daily Return", fontsize=8); ak.set_ylabel("Density", fontsize=8); ak.tick_params(labelsize=7); ak.grid(alpha=0.2)
        if col == 0: ak.legend(fontsize=7)

    fig.suptitle("Multi-Market Regime Distributions: t-SNE Embeddings (top) and Return KDE by Regime (bottom)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0.0, 0, 1, 0.96])
    p = os.path.join(OUT, "fig7_multimarket_distributions.png")
    fig.savefig(p, bbox_inches="tight", dpi=300); plt.close(fig)
    print(f"  saved {p}")


# ─────────────────────────────────────────────────────────────
# Figure 8 - Multi-detector comparison (real detectors only)
# ─────────────────────────────────────────────────────────────

def make_fig8_detection_panels():
    print("Fig 8: multi-detector panels (real only) ...")
    preds = _load_dated_preds()
    ohlcv = pd.read_csv("artifacts/ohlcv_GSPC.csv", parse_dates=["Date"]).set_index("Date")
    rets = ohlcv["Close"].pct_change().fillna(0)

    # rule-based detectors over full series (real)
    std3 = sigma_rule_signal(rets, k=3.0)
    g2   = pd.Series(garch_signal(rets, k=2.0), index=ohlcv.index)

    # restrict to OOS date span for fair comparison
    oos_dates = pd.to_datetime(preds["date"])
    d0, d1 = oos_dates.min(), oos_dates.max()
    span = (ohlcv.index >= d0) & (ohlcv.index <= d1)
    dates = ohlcv.index[span]
    close = ohlcv["Close"].values[span]

    pmap = preds.set_index("date")
    yt   = pmap["y_true"].reindex(dates).fillna(-1).values
    moe  = pmap["pred_moe"].reindex(dates).fillna(0).values
    moe_p= pmap["proba_moe"].reindex(dates).values
    glob = pmap["pred_global"].reindex(dates).fillna(0).values
    glob_p=pmap["proba_global"].reindex(dates).values

    detectors = [
        ("Std-Dev Rule (3σ)",          std3.reindex(dates).fillna(0).values, None,   CB_GREY,  "^"),
        ("GARCH(1,1) 2.0σ (real fit)", g2.reindex(dates).fillna(0).values,   None,   CB_RED,   "D"),
        ("LightGBM Global",            glob,  glob_p, CB_CYAN, "o"),
        ("LightGBM RC-MoE (Ours)",     moe,   moe_p,  CB_BLUE, "*"),
    ]
    fig, axes = plt.subplots(len(detectors), 2, figsize=(16, len(detectors)*3.0))
    true_dates = dates[yt == 1]
    true_close = close[yt == 1]

    for i, (name, sig, proba, col, mk) in enumerate(detectors):
        ap, av = axes[i, 0], axes[i, 1]
        ap.plot(dates, close, color=CB_BLACK, lw=0.8, alpha=0.7)
        ap.scatter(true_dates, true_close, color=CB_BLACK, s=26, marker="^", zorder=5, label="True Anomaly", alpha=0.85)
        det = sig == 1
        ap.scatter(dates[det], close[det], color=col, s=36, marker=mk, zorder=4, label="Detected", alpha=0.7)
        ap.set_title(f"({chr(97+i)}) {name}", fontsize=9, fontweight="bold")
        ap.set_ylabel("Close", fontsize=8); ap.tick_params(axis="x", rotation=40, labelsize=7)
        ap.legend(fontsize=7, loc="upper left"); ap.grid(alpha=0.2)
        if proba is not None:
            av.bar(dates, np.nan_to_num(proba), color=col, alpha=0.6, width=2)
            av.axhline(0.5, color="red", lw=1, ls="--", alpha=0.6); av.set_ylabel("P(Anomaly)", fontsize=8); av.set_ylim(0, 1)
        else:
            av.bar(dates, sig, color=col, alpha=0.7, width=2)
            av.bar(true_dates, np.ones(len(true_dates)), color=CB_BLACK, alpha=0.25, width=2)
            av.set_ylabel("Signal (0/1)", fontsize=8)
        av.set_title("Detection signal", fontsize=8); av.tick_params(axis="x", rotation=40, labelsize=7); av.grid(alpha=0.2)

    fig.suptitle("Multi-Detector Anomaly Comparison - S&P 500 OOS Window\n"
                 "(Left: price + detections; Right: signal/probability; ▲ = true anomaly. All detectors real.)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(OUT, "fig8_detection_panels.png")
    fig.savefig(p, bbox_inches="tight", dpi=300); plt.close(fig)
    print(f"  saved {p}")


if __name__ == "__main__":
    print(f"Output: {os.path.abspath(OUT)}\n")
    make_fig1_radar()
    make_fig2_cumulative_returns()
    make_fig3_regime_tsne_kde()
    make_fig4_anomaly_detection_zoom()
    make_fig5_crisis_focus()
    make_fig6_regime_patterns()
    make_fig7_multimarket_distributions()
    make_fig8_detection_panels()
    print(f"\nAll 8 figures saved to {os.path.abspath(OUT)}/")
