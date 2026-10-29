"""
Round-2 revision figures.  Every figure is drawn from the artifacts written by
the r2_* evaluation scripts (artifacts/r2).  Output: final_figures/*_r2.(png|pdf)

Run:  python -m src.utils.make_r2_figs [name ...]
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

from src.evaluation import r2_common as C

OUT = os.environ.get("R2_FIG_DIR", "final_figures")
ART = C.ART
BLUE, ORANGE, GREEN, VERM, SKY, PURPLE, YELLOW, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#F0E442", "#999999"
REG_COL = {0: GREEN, 1: ORANGE, 2: VERM}
REG_NAME = {0: "Low-Vol", 1: "Med-Vol", 2: "High-Vol"}
MK_NAME = {"SP500": "S&P 500", "NASDAQ": "NASDAQ", "Russell2000": "Russell 2000", "Bitcoin": "Bitcoin"}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "-", "savefig.dpi": 300,
                     "figure.dpi": 100, "legend.frameon": False})


def save(fig, name, pdf=False):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"{name}_r2.{'pdf' if pdf else 'png'}")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def J(name):
    return json.load(open(os.path.join(ART, name)))


def scores(m="SP500"):
    d = pd.read_csv(os.path.join(ART, f"oos_scores_{m}.csv"), parse_dates=["date"], index_col="date")
    f = os.path.join(ART, f"oos_deep_{m}.csv")
    if os.path.exists(f):
        d = d.join(pd.read_csv(f, parse_dates=["date"], index_col="date"))
    return d


def budget_alerts(df, col, b=0.05):
    a = pd.Series(0, index=df.index)
    for _, g in df.groupby("fold"):
        k = int(np.ceil(b * len(g)))
        a.loc[g[col].sort_values(ascending=False).index[:k]] = 1
    return a


# ------------------------------------------------------------------ Fig 1
def phase1():
    fig, ax = plt.subplots(figsize=(13, 8.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    navy = "#33415C"
    ax.add_patch(FancyBboxPatch((0.1, 7.45), 12.3, 0.6, boxstyle="round,pad=0.02,rounding_size=0.1", fc=navy, ec=navy))
    ax.text(6.25, 7.75, "PHASE 1     Factor Engineering Pipeline", color="white", ha="center", va="center", fontsize=15, weight="bold", family="serif")

    def box(x, y, w, h, fc="white", ec="#4A5568", lw=1.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=lw))
    box(1.05, 1.15, 2.15, 5.0)
    ax.text(2.12, 5.6, "Raw OHLCV", ha="center", fontsize=14, weight="bold", family="serif")
    ax.text(2.12, 4.95, r"$W_{t-1}\in\mathbb{R}^{128\times 5}$", ha="center", fontsize=14, color="#2B4CB3")
    rng = np.random.default_rng(3)
    px = 1.35
    lvl = 3.4
    for i in range(12):
        o = lvl
        c = lvl + rng.normal(0, 0.12) - (0.05 if i > 5 else -0.03)
        col = "#16A34A" if c >= o else "#DC2626"
        ax.plot([px + i * 0.13] * 2, [min(o, c) - 0.12, max(o, c) + 0.12], color=col, lw=1)
        ax.add_patch(plt.Rectangle((px + i * 0.13 - 0.04, min(o, c)), 0.08, abs(c - o) + 0.02, color=col))
        lvl = c
    ax.text(2.12, 1.5, "S&P 500 | 2005-2025", ha="center", fontsize=10, color="#6B7280", family="serif")
    box(3.75, 0.85, 4.9, 6.25, fc="#EBEDFB", ec="#4A5568")
    ax.text(6.2, 6.72, "8-Factor Extraction", ha="center", fontsize=14, weight="bold", family="serif")
    groups = [("TREND & MOM", "2 factors", "Return Trend\nMomentum", 4.0, 4.3),
              ("VOLATILITY", "1 factor", "Composite Volatility\n(window std. dev.)", 6.3, 4.3),
              ("MICROSTRUCTURE", "3 factors", "Latent A (gap)\nLatent B (pressure)\nLatent C (range/ATR)", 4.0, 1.15),
              ("TAIL RISK", "2 factors", "Kurtosis\nSkewness", 6.3, 1.15)]
    for t, n, items, x, y in groups:
        box(x, y, 2.1, 2.4, ec="#2563EB")
        ax.text(x + 1.05, y + 2.0, t, ha="center", fontsize=11.5, weight="bold", color="#1E40AF")
        ax.text(x + 1.05, y + 1.72, n, ha="center", fontsize=8.5, color="#4B5563")
        ax.text(x + 1.05, y + 1.0, items, ha="center", va="center", fontsize=9.5, linespacing=1.5)
    box(9.15, 1.15, 1.55, 5.0)
    ax.text(9.92, 5.45, "Lag\nAugmentation", ha="center", fontsize=13, weight="bold", family="serif", linespacing=1.2)
    for i, (lab, c) in enumerate([(r"$f(t-1)$", "#2563EB"), (r"$f(t-2)$", "#60A5FA"), (r"$f(t-6)$", "#93C5FD")]):
        ax.add_patch(plt.Rectangle((9.4, 4.0 - i * 0.75), 1.05, 0.42, fc=c, ec="#4A5568", lw=0.6))
        ax.text(9.92, 4.21 - i * 0.75, lab, ha="center", va="center", fontsize=11, color="white" if i == 0 else "black")
    ax.text(9.92, 1.95, r"$\oplus$ Concat", ha="center", fontsize=12, weight="bold", family="serif")
    ax.text(9.92, 1.5, r"$8\times 3=24$", ha="center", fontsize=12)
    box(11.2, 1.15, 1.45, 5.0, fc="#ECECEC")
    ax.text(11.92, 5.65, r"$F_t\in\mathbb{R}^{24}$", ha="center", fontsize=12)
    ax.text(11.92, 4.75, "Factor\nState\nVector", ha="center", fontsize=12, weight="bold", family="serif", linespacing=1.25)
    cols = ["#60A5FA", "#33415C", "#2563EB", "black"]
    for i in range(6):
        for j in range(4):
            ax.add_patch(plt.Rectangle((11.6 + j * 0.16, 3.4 - i * 0.18), 0.16, 0.18, fc=cols[rng.integers(0, 4)], ec="#9CA3AF", lw=0.4))
    for x0, x1 in [(3.2, 3.75), (8.65, 9.15), (10.7, 11.2)]:
        ax.annotate("", xy=(x1, 3.65), xytext=(x0, 3.65), arrowprops=dict(arrowstyle="->", color="#4A5568", lw=1.3))
    ax.text(3.47, 3.8, r"$\phi(\cdot)$", ha="center", fontsize=12)
    ax.annotate("", xy=(12.65, 0.35), xytext=(1.05, 0.35), arrowprops=dict(arrowstyle="<->", color="#4A5568", lw=1))
    ax.text(6.85, 0.35, " 128-day window ending at $t-1$ $\\rightarrow$ 24-dim factor vector for day $t$ ", ha="center", va="center",
            fontsize=10, color="#6B7280", family="serif", bbox=dict(fc="white", ec="none"))
    save(fig, "phase-1", pdf=True)


# ------------------------------------------------------------------ Fig 4
def topological():
    from sklearn.ensemble import RandomForestClassifier
    d = pd.read_pickle(os.path.join(ART, "dataset_SP500.pkl"))
    fx, fy = "f_Return_Trend", "f_Momentum"
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    for k, ax in enumerate(axs):
        s = d[d.regime == k]
        X = s[[fx, fy]].values
        y = s.y.values.astype(int)
        rf = RandomForestClassifier(n_estimators=200, max_depth=5, class_weight="balanced", random_state=0, n_jobs=4).fit(X, y)
        gx = np.linspace(np.percentile(d[fx], 0.5), np.percentile(d[fx], 99.5), 150)
        gy = np.linspace(np.percentile(d[fy], 0.5), np.percentile(d[fy], 99.5), 150)
        GX, GY = np.meshgrid(gx, gy)
        P = rf.predict_proba(np.c_[GX.ravel(), GY.ravel()])[:, 1].reshape(GX.shape)
        cs = ax.contourf(GX * 1e3, GY, P, levels=np.linspace(0, 1, 11), cmap="Blues", alpha=0.8)
        ax.scatter(s.loc[s.y == 0, fx] * 1e3, s.loc[s.y == 0, fy], s=3, c=GREY, alpha=0.35, lw=0)
        ax.scatter(s.loc[s.y == 1, fx] * 1e3, s.loc[s.y == 1, fy], s=45, marker="*", c=REG_COL[k], ec="k", lw=0.4)
        ax.set_title(f"{REG_NAME[k]} ({int(y.sum())} anomalies / {len(y)} days)", fontsize=10)
        ax.set_xlabel(r"Return trend ($\times10^{-3}$)")
        ax.grid(False)
    axs[0].set_ylabel("Momentum")
    cb = fig.colorbar(cs, ax=axs, shrink=0.9, pad=0.01)
    cb.set_label("Expert anomaly probability")
    save(fig, "fig_m3_topological_boundaries")


# ------------------------------------------------------------------ Fig 5 / 23
def _tsne(X, seed=0):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(X)
    return TSNE(n_components=2, perplexity=40, random_state=seed, init="pca", n_jobs=1).fit_transform(Z)


def tsne_sp():
    from scipy.stats import gaussian_kde
    d = pd.read_pickle(os.path.join(ART, "dataset_SP500.pkl")).loc[C.SAMPLE[0]:]
    E = _tsne(d[C.feature_cols(d)].values)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    for k in range(3):
        m = d.regime.values == k
        a.scatter(E[m, 0], E[m, 1], s=4, c=REG_COL[k], alpha=0.45, lw=0, label=REG_NAME[k])
    an = d.y.values == 1
    a.scatter(E[an, 0], E[an, 1], s=55, marker="*", c="k", label="Anomaly", lw=0)
    a.set_title("(a) t-SNE of ex-ante factor vectors")
    a.legend(markerscale=2, loc="best", fontsize=8)
    a.set_xticks([]); a.set_yticks([])
    xs = np.linspace(-0.08, 0.08, 400)
    for k in range(3):
        r = d.loc[d.regime == k, "r"].values
        b.plot(xs * 100, gaussian_kde(r)(xs) / 100, color=REG_COL[k], lw=2, label=f"{REG_NAME[k]} (sd {100*r.std():.2f}%)")
    b.set_yscale("log"); b.set_ylim(1e-4, 1)
    b.set_xlabel("Daily return $r_t$ (%)"); b.set_ylabel("Density (log)")
    b.set_title("(b) Return density by regime"); b.legend(fontsize=8)
    save(fig, "fig3_regime_tsne_kde")


def mm_distributions():
    from scipy.stats import gaussian_kde
    fig, axs = plt.subplots(2, 4, figsize=(16, 7))
    for j, m in enumerate(C.MARKETS):
        d = pd.read_pickle(os.path.join(ART, f"dataset_{m}.pkl")).loc[C.SAMPLE[0]:]
        E = _tsne(d[C.feature_cols(d)].values)
        for k in range(3):
            s = d.regime.values == k
            axs[0, j].scatter(E[s, 0], E[s, 1], s=3, c=REG_COL[k], alpha=0.45, lw=0, label=REG_NAME[k])
        axs[0, j].set_title(MK_NAME[m]); axs[0, j].set_xticks([]); axs[0, j].set_yticks([])
        lim = 0.15 if m == "Bitcoin" else 0.08
        xs = np.linspace(-lim, lim, 400)
        for k in range(3):
            r = d.loc[d.regime == k, "r"].values
            axs[1, j].plot(xs * 100, gaussian_kde(r)(xs) / 100, color=REG_COL[k], lw=1.8, label=REG_NAME[k])
        axs[1, j].set_yscale("log"); axs[1, j].set_ylim(1e-4, 1); axs[1, j].set_xlabel("Daily return (%)")
    axs[0, 0].legend(markerscale=3, fontsize=8); axs[1, 0].set_ylabel("Density (log)")
    save(fig, "fig7_multimarket_distributions")


# ------------------------------------------------------------------ Fig 6
def regime_patterns():
    d = pd.read_pickle(os.path.join(ART, "dataset_SP500.pkl")).loc[C.SAMPLE[0]:]
    px = C.load_ohlcv("GSPC")
    fig, axs = plt.subplots(2, 3, figsize=(14, 5.2), gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
    for k in range(3):
        s = d[d.regime == k]
        cand = s[s.y == 1]
        t = cand.sigma.sub(s.sigma.median()).abs().idxmin() if len(cand) else s.sigma.sub(s.sigma.median()).abs().idxmin()
        pos = px.index.get_loc(t)
        w = px.iloc[pos - C.WINDOW:pos + 1]
        ax = axs[0, k]
        ax.plot(w.index, w.Close, color=REG_COL[k], lw=1.5)
        an = d.loc[w.index.intersection(d.index)]
        an = an[an.y == 1]
        ax.scatter(an.index, px.loc[an.index, "Close"], marker="v", c="k", s=40, zorder=3, label="Anomaly")
        ax.set_title(f"{REG_NAME[k]} window ending {t.date()}", fontsize=10)
        axs[1, k].bar(w.index, w.Volume / 1e9, color=GREY, width=1)
        axs[1, k].tick_params(axis="x", rotation=30)
    axs[0, 0].set_ylabel("S&P 500 close"); axs[1, 0].set_ylabel("Volume (bn)")
    axs[0, 0].legend(fontsize=8)
    save(fig, "fig6_regime_patterns")


# ------------------------------------------------------------------ Fig 7 (ablation)
def ablation():
    M = J("detection_metrics.json")["SP500"]
    order = ["Vol rule", "Z-score rule", "ConvVAE", "Anomaly Transformer", "Global RF", "RC-MoE"]
    lab = ["Std-dev rule", "Z-score rule", "ConvVAE", "Anomaly Transformer", "Global RF", "RC-MoE"]
    v = [M["det"][o]["auprc"] for o in order]
    lo = [M["det"][o]["auprc"] - M["det"][o]["ci"][0] for o in order]
    hi = [M["det"][o]["ci"][1] - M["det"][o]["auprc"] for o in order]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    cols = [GREY, GREY, SKY, SKY, ORANGE, BLUE]
    ax.barh(lab, v, xerr=[lo, hi], color=cols, ec="k", lw=0.5, capsize=3)
    for i, x in enumerate(v):
        ax.text(x + hi[i] + 0.006, i + 0.1, f"{x:.3f}", fontsize=9)
    ax.axvline(M["prev"], color="k", ls=":", lw=1, label=f"Base rate ({M['prev']:.3f})")
    ax.axvline(M["det"]["GARCH-FHS"]["auprc"], color=VERM, ls="--", lw=1.2, label=f"GARCH-FHS ({M['det']['GARCH-FHS']['auprc']:.3f})")
    ax.set_xlabel("Out-of-sample AUPRC (block-bootstrap 95% CI)")
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()
    save(fig, "ablation_chart_paper")


# ------------------------------------------------------------------ Fig 8
def sensitivity():
    S = J("sensitivity.json")
    Ls, Ks = [64, 96, 128, 192], [2, 3, 4]
    A = np.array([[S[f"L{L}_K{K}"]["auprc_moe"] for K in Ks] for L in Ls])
    G = np.array([S[f"L{L}_K3"]["auprc_global"] for L in Ls])
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.1]})
    im = a.imshow(A, cmap="Blues", vmin=0, vmax=max(A.max(), G.max()) * 1.1)
    for i in range(4):
        for j in range(3):
            a.text(j, i, f"{A[i, j]:.3f}", ha="center", va="center", fontsize=10, weight="bold" if (Ls[i], Ks[j]) == (128, 3) else None)
    a.add_patch(plt.Rectangle((0.5, 1.5), 1, 1, fill=False, ec=VERM, lw=2))
    a.set_xticks(range(3), [f"K={k}" for k in Ks]); a.set_yticks(range(4), [f"L={L}" for L in Ls]); a.grid(False)
    a.set_title("(a) RC-MoE out-of-sample AUPRC")
    fig.colorbar(im, ax=a, shrink=0.8)
    for j, K in enumerate(Ks):
        b.plot(Ls, A[:, j], marker="o", label=f"RC-MoE, K={K}", color=[SKY, BLUE, PURPLE][j])
    b.plot(Ls, G, marker="s", color="k", ls="--", label="Global ensemble")
    b.axhline(S["L128_K3"]["prev"], color=GREY, ls=":", label="Base rate")
    b.set_xticks(Ls); b.set_xlabel("Factor window L (days)"); b.set_ylabel("AUPRC"); b.legend(fontsize=8)
    b.set_title("(b) Comparison with the global ensemble")
    save(fig, "fig_sensitivity_oos")


# ------------------------------------------------------------------ Fig 9 (radar)
def radar():
    M = J("detection_metrics.json")["SP500"]["det"]
    Bt = J("backtest.json")["strategies"]
    names = ["RC-MoE", "Global ensemble", "GBM + regime", "GARCH-FHS"]
    axes = ["AUPRC", "Precision@5%", "Recall@5%", "Sharpe", "Sortino", "MDD reduction", "ES reduction"]
    bh = Bt["Buy-and-hold"]
    raw = {n: [M[n]["auprc"], M[n]["b5"]["prec"], M[n]["b5"]["rec"], Bt[n]["sharpe"], Bt[n]["sortino"],
               bh["mdd"] - Bt[n]["mdd"], bh["es975"] - Bt[n]["es975"]] for n in names}
    R = np.array(list(raw.values()))
    lo, hi = R.min(0), R.max(0)
    N = (R - lo) / np.where(hi > lo, hi - lo, 1) * 0.8 + 0.2
    ang = np.linspace(0, 2 * np.pi, len(axes), endpoint=False).tolist()
    ang += ang[:1]
    fig = plt.figure(figsize=(6.2, 5.6))
    ax = plt.subplot(111, polar=True)
    for i, n in enumerate(names):
        vals = N[i].tolist() + N[i][:1].tolist()
        ax.plot(ang, vals, color=[BLUE, ORANGE, GREEN, VERM][i], lw=2, label=n)
        ax.fill(ang, vals, color=[BLUE, ORANGE, GREEN, VERM][i], alpha=0.08)
    ax.set_xticks(ang[:-1], axes, fontsize=9)
    ax.set_yticks([0.2, 0.6, 1.0], ["min", "", "max"], fontsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=8)
    save(fig, "fig1_radar_overview")


# ------------------------------------------------------------------ Fig 10
def oos_chart():
    M = J("detection_metrics.json")["SP500"]
    order = ["Vol rule", "Z-score rule", "GARCH-FHS", "GARCH-EVT", "Global RF", "Global GBM", "GBM + regime", "Global ensemble", "RC-MoE"]
    lab = ["Std-dev\nrule", "Z-score\nrule", "GARCH\nFHS", "GARCH\nEVT", "Global\nRF", "Global\nGBM", "GBM +\nregime", "Global\nensemble", "RC-MoE"]
    x = np.arange(len(order))
    w = 0.27
    a = [M["det"][o]["auprc"] for o in order]
    err = [[M["det"][o]["auprc"] - M["det"][o]["ci"][0] for o in order], [M["det"][o]["ci"][1] - M["det"][o]["auprc"] for o in order]]
    p = [M["det"][o]["b5"]["prec"] for o in order]
    r = [M["det"][o]["b5"]["rec"] for o in order]
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.bar(x - w, a, w, yerr=err, color=BLUE, ec="k", lw=0.4, capsize=2, label="AUPRC (95% CI)")
    ax.bar(x, p, w, color=SKY, ec="k", lw=0.4, label="Precision at 5% budget")
    ax.bar(x + w, r, w, color=GREY, ec="k", lw=0.4, label="Recall at 5% budget")
    ax.axhline(M["prev"], color="k", ls=":", lw=1, label=f"Base rate ({M['prev']:.3f})")
    ax.set_xticks(x, lab, fontsize=9)
    ax.set_ylabel("Score")
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    save(fig, "oos_performance_chart")


# ------------------------------------------------------------------ Fig 11 / 12 / 13
def timeline():
    d = scores()
    px = C.load_ohlcv("GSPC").loc[d.index]
    a_m = budget_alerts(d, "RC-MoE")
    a_g = budget_alerts(d, "GARCH-FHS")
    fig, axs = plt.subplots(4, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [2, 1, 2, 1.2]})
    ax = axs[0]
    ax.plot(px.index, px.Close, color="k", lw=0.8)
    for k in range(3):
        m = (d.regime == k).values
        ax.fill_between(px.index, px.Close.min(), px.Close.max(), where=m, color=REG_COL[k], alpha=0.15, lw=0, label=REG_NAME[k])
    ax.set_title("(a) S&P 500 close and ex-ante volatility regime"); ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax = axs[1]
    ax.eventplot([d.index[d.y == 1], d.index[a_m == 1], d.index[a_g == 1]], colors=["k", BLUE, VERM], lineoffsets=[2, 1, 0], linelengths=0.7)
    ax.set_yticks([2, 1, 0], ["Anomalies", "RC-MoE alerts", "GARCH-FHS alerts"]); ax.grid(False)
    ax.set_title("(b) Anomalies and alerts at a 5% alert budget")
    ax.set_xlim(axs[0].get_xlim())
    z = d.loc["2020-01-15":"2020-05-15"]
    ax = axs[2]
    ax.bar(z.index, np.abs(z.r) * 100, color=GREY, width=1, label="|r_t|")
    ax.plot(z.index, z.tau * 100, color="k", lw=1.2, ls="--", label=r"Threshold $\tau_t$")
    zm, zg = a_m.loc[z.index], a_g.loc[z.index]
    ax.scatter(z.index[zm == 1], np.abs(z.r[zm == 1]) * 100 + 0.6, marker="v", c=BLUE, s=30, label="RC-MoE alert")
    ax.scatter(z.index[zg == 1], np.abs(z.r[zg == 1]) * 100 + 1.4, marker="v", c=VERM, s=30, label="GARCH-FHS alert")
    ax.set_ylabel("%"); ax.set_title("(c) COVID-19 crash, Feb to May 2020"); ax.legend(fontsize=8, ncol=4)
    ax = axs[3]
    rk = lambda s: s.rank(pct=True)
    ax.plot(z.index, rk(d["RC-MoE"]).loc[z.index], color=BLUE, label="RC-MoE score (percentile)")
    ax.plot(z.index, rk(d["GARCH-FHS"]).loc[z.index], color=VERM, label="GARCH-FHS score (percentile)")
    ax.axhline(0.95, color="k", ls=":", lw=1)
    ax.set_ylim(0, 1.02); ax.legend(fontsize=8, loc="lower left", ncol=2); ax.set_title("(d) Out-of-sample scores (percentile rank)")
    fig.tight_layout()
    save(fig, "fig4_anomaly_detection_zoom")


def crisis():
    d = scores()
    px = C.load_ohlcv("GSPC")
    a_m = budget_alerts(d, "RC-MoE")
    a_g = budget_alerts(d, "GARCH-FHS")
    z = d.loc["2020-02-03":"2020-06-30"]
    p = px.loc[z.index]
    fig, axs = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2, 0.8]})
    ax = axs[0]
    for t, row in p.iterrows():
        col = "#16A34A" if row.Close >= row.Open else "#DC2626"
        ax.plot([t, t], [row.Low, row.High], color=col, lw=0.8)
        ax.add_patch(plt.Rectangle((t - pd.Timedelta(hours=8), min(row.Open, row.Close)), pd.Timedelta(hours=16), abs(row.Close - row.Open) + 0.5, color=col))
    ax.scatter(z.index[a_m.loc[z.index] == 1], p.High[a_m.loc[z.index] == 1] * 1.02, marker="v", c=BLUE, s=35, label="RC-MoE alert")
    ax.scatter(z.index[a_g.loc[z.index] == 1], p.High[a_g.loc[z.index] == 1] * 1.05, marker="v", c=VERM, s=35, label="GARCH-FHS alert")
    ax.set_ylabel("S&P 500"); ax.legend(fontsize=8); ax.set_title("(a) Daily OHLC with alerts (5% budget)")
    ax = axs[1]
    ax.bar(z.index, np.abs(z.r) * 100, color=np.where(z.y == 1, "k", GREY), width=0.8)
    ax.plot(z.index, z.tau * 100, color=VERM, ls="--", lw=1.2, label=r"Regime threshold $\tau_t$")
    ax.set_ylabel("|r| (%)"); ax.legend(fontsize=8); ax.set_title("(b) Absolute returns (anomalies in black)")
    ax = axs[2]
    ax.step(z.index, z.regime, color="k", where="post")
    ax.set_yticks([0, 1, 2], ["Low", "Med", "High"]); ax.set_title("(c) Ex-ante regime")
    fig.tight_layout()
    save(fig, "fig5_crisis_focus_ohlc")


def panels():
    d = scores()
    dets = ["Z-score rule", "GARCH-FHS", "Global GBM", "RC-MoE"]
    z = d.loc["2022-01-01":"2022-12-31"]
    fig, axs = plt.subplots(4, 2, figsize=(13, 8.5), sharex=True)
    for i, c in enumerate(dets):
        a = budget_alerts(d, c).loc[z.index]
        ax = axs[i, 0]
        ax.bar(z.index, z.r * 100, color=GREY, width=1)
        ax.scatter(z.index[z.y == 1], z.r[z.y == 1] * 100, marker="o", fc="none", ec="k", s=40, label="Anomaly")
        hit = (a == 1) & (z.y == 1)
        ax.scatter(z.index[a == 1], z.r[a == 1] * 100, marker="x", c=BLUE, s=25, label="Alert")
        ax.set_ylabel("r (%)"); ax.set_title(f"{c}: {int(a.sum())} alerts, {int(hit.sum())} hits", fontsize=9)
        ax2 = axs[i, 1]
        ax2.plot(z.index, d[c].rank(pct=True).loc[z.index], color=BLUE, lw=1)
        ax2.axhline(0.95, color=VERM, ls="--", lw=1)
        ax2.set_ylim(0, 1.02); ax2.set_ylabel("Score pct.")
    axs[0, 0].legend(fontsize=8, ncol=2)
    for ax in axs[-1]:
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    save(fig, "fig8_detection_panels")


# ------------------------------------------------------------------ Fig 14
def heatmap():
    M = J("detection_metrics.json")["SP500"]["det"]
    dets = ["RC-MoE", "Global ensemble", "GBM + regime", "GARCH-FHS"]
    A = np.array([[M[d]["regime"][str(k)][m] for k in range(3) for m in ["prec", "rec"]] for d in dets])
    fig, ax = plt.subplots(figsize=(8, 3.2))
    im = ax.imshow(A, cmap="Blues", vmin=0, vmax=max(0.5, A.max()))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f"{100*A[i, j]:.1f}", ha="center", va="center", fontsize=9, color="white" if A[i, j] > 0.35 else "k")
    pos = [M["RC-MoE"]["regime"][str(k)]["pos"] for k in range(3)]
    ax.set_xticks(range(6), [f"{REG_NAME[k]}\n{m}" for k in range(3) for m in ["Precision", "Recall"]], fontsize=8)
    ax.set_yticks(range(len(dets)), dets); ax.grid(False)
    ax.set_title(f"Per-regime precision / recall (%), 5% budget; anomalies {pos[0]}/{pos[1]}/{pos[2]}", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, "per_regime_heatmap")


# ------------------------------------------------------------------ Fig 15 / 17 / 18
def execution():
    D = pd.read_csv(os.path.join(ART, "backtest_daily.csv"), parse_dates=[0], index_col=0)
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for c, col in [("Buy-and-hold", "k"), ("RC-MoE", BLUE), ("GARCH-FHS", VERM), ("Global ensemble", ORANGE)]:
        w = (1 + D[c]).cumprod()
        a.plot(D.index, w, color=col, lw=1.4, label=c)
        b.plot(D.index, -(1 - w / w.cummax()) * 100, color=col, lw=1)
    m = D["mask_RC-MoE"].values == 1
    a.fill_between(D.index, 0, 1, where=m, transform=a.get_xaxis_transform(), color=BLUE, alpha=0.08, lw=0)
    a.set_ylabel("Growth of 1"); a.legend(fontsize=8); a.set_title("(a) Cumulative total return (5 bp per switch; shaded: RC-MoE in cash)")
    b.set_ylabel("Drawdown (%)"); b.set_title("(b) Drawdown")
    save(fig, "fig_execution_lag")


def random_timing():
    from src.evaluation.r2_backtest import load_returns, random_timing as rt, causal_alerts
    Bt = J("backtest.json")["strategies"]
    D = pd.read_csv(os.path.join(ART, "backtest_daily.csv"), parse_dates=[0], index_col=0)
    eq = D["Buy-and-hold"].values
    rf = D["rf"].values
    n_al = Bt["RC-MoE"]["alerts"]
    R = rt(eq, rf, n_al, seed=1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8))
    a.hist(R[:, 0], bins=40, color=GREY, alpha=0.8)
    b.hist(R[:, 1] * 100, bins=40, color=GREY, alpha=0.8)
    for c, col in [("RC-MoE", BLUE), ("GARCH-FHS", VERM), ("Global ensemble", ORANGE), ("Buy-and-hold", "k")]:
        a.axvline(Bt[c]["sharpe"], color=col, lw=2, label=c)
        b.axvline(Bt[c]["mdd"] * 100, color=col, lw=2, label=c)
    a.set_xlabel("Sharpe ratio"); a.set_ylabel("Random-timing draws"); a.legend(fontsize=8)
    a.set_title(f"(a) Sharpe: {n_al} random 5-day hedges")
    b.set_xlabel("Maximum drawdown (%)"); b.set_title("(b) Maximum drawdown")
    save(fig, "hedging_efficiency_chart")


def factor_beta():
    import statsmodels.api as sm
    from src.evaluation.factor_alpha import load_ff_factors
    F = J("backtest.json")["ff"]
    D = pd.read_csv(os.path.join(ART, "backtest_daily.csv"), parse_dates=[0], index_col=0)
    ff = load_ff_factors("artifacts/data/ff_factors/FF5_daily.csv", "artifacts/data/ff_factors/MOM_daily.csv")
    fac = ["MktRF", "SMB", "HML", "RMW", "CMA", "MOM"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1, 1.4]})
    x = np.arange(6)
    for i, (c, col) in enumerate([("RC-MoE", BLUE), ("GARCH-FHS", VERM)]):
        a.bar(x + (i - 0.5) * 0.38, [F[c][f]["est"] for f in fac], 0.38, color=col, label=c, ec="k", lw=0.4)
        for j, f in enumerate(fac):
            if F[c][f]["p"] < 0.05:
                a.text(x[j] + (i - 0.5) * 0.38, F[c][f]["est"] + 0.02, "*", ha="center")
    a.set_xticks(x, fac); a.axhline(0, color="k", lw=0.6); a.legend(fontsize=8); a.set_title("(a) FF5+MOM loadings (* p<0.05)")
    df = D.join(ff, how="inner")
    for c, col in [("RC-MoE", BLUE), ("GARCH-FHS", VERM)]:
        ex = df[c] - df["rf"]
        beta = ex.rolling(252).cov(df["MktRF"]) / df["MktRF"].rolling(252).var()
        b.plot(beta.index, beta, color=col, lw=1.3, label=c)
    b.axhline(1, color="k", ls=":", lw=1); b.set_ylabel("Rolling 252-day market beta"); b.legend(fontsize=8)
    b.set_title("(b) Rolling market beta")
    save(fig, "fig_factor_beta")


# ------------------------------------------------------------------ risk figures
def _risk(m):
    return pd.read_csv(os.path.join(ART, f"risk_forecasts_{m}.csv"), parse_dates=[0], index_col=0)


def evt_tail():
    from scipy.stats import genpareto
    d = _risk("SP500")
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for k in range(3):
        L = d.loc[d.regime == k, "loss"].values
        u = np.quantile(L, 0.9)
        exc = np.sort(L[L > u])
        xi, _, be = genpareto.fit(exc - u, floc=0)
        emp = 1 - np.arange(len(exc)) / len(exc)
        ax.scatter(exc * 100, emp * 0.1, s=12, color=REG_COL[k], alpha=0.7)
        xs = np.linspace(u, exc.max() * 1.1, 200)
        ax.plot(xs * 100, 0.1 * genpareto.sf(xs - u, xi, 0, be), color=REG_COL[k], lw=2,
                label=fr"{REG_NAME[k]}: $\hat\xi$={xi:+.3f}, n$_u$={len(exc)}")
    ax.set_yscale("log"); ax.set_xlabel("Daily loss (%)"); ax.set_ylabel("P(loss > x)")
    ax.axhline(0.01, color="k", ls=":", lw=1)
    ax.legend(fontsize=8)
    save(fig, "fig_evt_tail")


def var_coverage():
    R = J("risk.json")
    models = ["HS", "RC-EVT", "GARCH-t", "GARCH-FHS", "GARCH-EVT"]
    cols = [GREY, BLUE, PURPLE, SKY, GREEN]
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.2))
    for j, m in enumerate(C.MARKETS):
        ax = axs[j]
        for i, mod in enumerate(models):
            for k in range(3):
                rr = R[m]["models"][mod]["regime"][str(k)]
                x = k + (i - 2) * 0.16
                ax.bar(x, rr["V"], 0.16, color=cols[i], ec="k", lw=0.3, label=mod if k == 0 else None)
                if rr["p_uc"] < 0.05:
                    ax.scatter(x, rr["V"] + 1.2, marker="x", c=VERM, s=25)
        for k in range(3):
            E = R[m]["models"]["HS"]["regime"][str(k)]["E"]
            ax.plot([k - 0.45, k + 0.45], [E, E], "k--", lw=1.2)
        ax.set_xticks(range(3), ["Low-Vol", "Med-Vol", "High-Vol"]); ax.set_title(MK_NAME[m], weight="bold")
    axs[0].set_ylabel(r"VaR$_{99}$ violations")
    h, l = axs[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color="k", ls="--"), plt.Line2D([], [], marker="x", color=VERM, ls="")]
    l += ["Expected (1%)", r"Kupiec rejects ($p<0.05$)"]
    fig.legend(h, l, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.06), fontsize=9)
    save(fig, "fig_var_coverage")


def capital():
    R = J("risk.json")["SP500"]["models"]
    models = ["HS", "RC-EVT", "GARCH-t", "GARCH-FHS", "GARCH-EVT"]
    cols = [GREY, BLUE, PURPLE, SKY, GREEN]
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, mod in enumerate(models):
        v = [100 * R[mod]["regime"][str(k)]["cap_mean"] for k in range(3)]
        x = np.arange(3) + (i - 2) * 0.16
        ax.bar(x, v, 0.16, color=cols[i], ec="k", lw=0.3, label=mod)
        for xx, vv, k in zip(x, v, range(3)):
            ax.text(xx, vv + 0.3, str(R[mod]["regime"][str(k)]["cap_breach"]), ha="center", fontsize=7)
    ax.set_xticks(range(3), ["Low-Vol", "Med-Vol", "High-Vol"])
    ax.set_ylabel("Mean FRTB ES capital (% of position)")
    ax.set_title("Numbers above bars: days on which the next 10-day loss exceeded capital", fontsize=9)
    ax.legend(fontsize=8, ncol=5, loc="upper left")
    save(fig, "fig_basel_capital")


def multimarket():
    R = J("risk.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(3)
    mk = list(C.MARKETS)
    for j, m in enumerate(mk):
        v = [100 * R[m]["models"]["RC-EVT"]["regime"][str(k)]["v99_mean"] for k in range(3)]
        a.plot(x, v, marker="o", lw=2, label=MK_NAME[m], color=[BLUE, ORANGE, GREEN, VERM][j])
        g = [100 * (R[m]["models"]["HS"]["regime"][str(k)]["cap_mean"] / R[m]["models"]["RC-EVT"]["regime"][str(k)]["cap_mean"] - 1) for k in range(3)]
        b.bar(x + (j - 1.5) * 0.2, g, 0.2, color=[BLUE, ORANGE, GREEN, VERM][j], label=MK_NAME[m], ec="k", lw=0.3)
    a.set_xticks(x, ["Low-Vol", "Med-Vol", "High-Vol"]); a.set_ylabel(r"Mean RC-EVT VaR$_{99}$ (%)")
    a.set_title(r"(a) Regime-conditional VaR$_{99}$"); a.legend(fontsize=8)
    b.axhline(0, color="k", lw=0.8); b.set_xticks(x, ["Low-Vol", "Med-Vol", "High-Vol"])
    b.set_ylabel("Capital gap, HS vs RC-EVT (%)"); b.set_title("(b) Unconditional capital relative to RC-EVT"); b.legend(fontsize=8)
    save(fig, "fig_multimarket")


def var_boundaries():
    d = _risk("SP500")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(d.index, d.r * 100, color=GREY, width=1.5)
    for c, col in [("HS", "k"), ("RC-EVT", BLUE), ("GARCH-EVT", GREEN)]:
        ax.plot(d.index, -d[f"{c}|v99"] * 100, color=col, lw=1, label=fr"{c} VaR$_{{99}}$")
        v = d.loss > d[f"{c}|v99"]
        ax.scatter(d.index[v], d.r[v] * 100, s=14, color=col, zorder=3)
    ax.set_ylabel("Daily return (%)"); ax.legend(fontsize=8, ncol=3, loc="lower left"); ax.set_ylim(-13, 10)
    save(fig, "var_boundaries_chart")


# ------------------------------------------------------------------ ECI / SHAP
def eci():
    E = J("explain.json")
    sc = ["Global", "0", "1", "2"]
    lab = ["Global", "Low-Vol", "Med-Vol", "High-Vol"]
    v = [E[s]["eci"] for s in sc]
    err = [[E[s]["eci"] - E[s]["ci"][0] for s in sc], [E[s]["ci"][1] - E[s]["eci"] for s in sc]]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(lab, v, yerr=err, color=[GREY, GREEN, ORANGE, VERM], ec="k", lw=0.5, capsize=4)
    ax.axhspan(E["null"]["ci"][0], E["null"]["ci"][1], color="k", alpha=0.08, label="Random-ranking null (95%)")
    ax.axhline(E["null"]["mean"], color="k", ls=":", lw=1)
    for i, s in enumerate(sc):
        ax.text(i, E[s]["ci"][1] + 0.01, f"{v[i]:.3f}\n({E[s]['pos']} anom.)", ha="center", fontsize=8)
    ax.set_ylim(0.3, 1.0); ax.set_ylabel("ECI"); ax.legend(fontsize=8, loc="lower right")
    save(fig, "eci_chart")


def shap_fig():
    E = J("explain.json")
    fac = C.FACTOR_NAMES
    lab = ["Return\ntrend", "Momentum", "Composite\nvol.", "Latent A\n(gap)", "Latent B\n(pressure)", "Latent C\n(range)", "Kurtosis", "Skewness"]
    fig, ax = plt.subplots(figsize=(11, 3.9))
    x = np.arange(len(fac))
    for i, k in enumerate(["0", "1", "2"]):
        v = np.array([E[k]["shap"][f] for f in fac])
        lo = v - np.array([E[k]["shap_ci"][f][0] for f in fac])
        hi = np.array([E[k]["shap_ci"][f][1] for f in fac]) - v
        ax.bar(x + (i - 1) * 0.27, v, 0.27, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)], color=REG_COL[int(k)], ec="k", lw=0.3,
               capsize=2, label=f"{REG_NAME[int(k)]} ({E[k]['pos']} anomalies)")
    ax.set_xticks(x, lab, fontsize=9); ax.set_ylabel("Mean |SHAP| (log-odds)"); ax.legend(fontsize=8)
    save(fig, "explainability_attribution")


ALL = {"phase1": phase1, "topological": topological, "tsne": tsne_sp, "patterns": regime_patterns, "ablation": ablation,
       "sensitivity": sensitivity, "radar": radar, "oos": oos_chart, "timeline": timeline, "crisis": crisis, "panels": panels,
       "heatmap": heatmap, "execution": execution, "random": random_timing, "beta": factor_beta, "evt": evt_tail,
       "coverage": var_coverage, "capital": capital, "multimarket": multimarket, "varb": var_boundaries, "eci": eci,
       "shap": shap_fig, "mmdist": mm_distributions}

if __name__ == "__main__":
    for nm in (sys.argv[1:] or list(ALL)):
        ALL[nm]()
