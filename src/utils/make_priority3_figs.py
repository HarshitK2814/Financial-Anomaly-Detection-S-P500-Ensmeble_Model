"""
Priority 3 figure generation for RC-MoE paper.
Produces / updates:
  correct figures/fig_m3_topological_boundaries.png  - 300 DPI PCA decision boundaries
  correct figures/var_boundaries_chart.png           - GARCH VaR + EVT-VaR overlay
  correct figures/fig_phase3_moe.pdf                 - Tree-only architecture (replaces phase-3.pdf)
  correct figures/fig_basel_capital.png              - FIXED: more y-headroom for -24% label
  correct figures/fig_multimarket.png                - FIXED: more y-headroom for +93% label

Run from repo root:
  .venv/Scripts/python.exe -m src.utils.make_priority3_figs
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

OUT = "correct figures"
os.makedirs(OUT, exist_ok=True)

# Colorblind-safe (Wong palette)
CB = {"Low": "#0072B2", "Med": "#E69F00", "High": "#D55E00"}
CB_LIST = ["#0072B2", "#E69F00", "#D55E00"]
GREY = "#999999"
BLACK = "#222222"

plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 300, "font.family": "DejaVu Sans",
})


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def _get_regime_from_returns(returns, q=60):
    """Rolling 60-day vol → regime (0=Low, 1=Med, 2=High) by terciles."""
    vol = pd.Series(returns).rolling(q).std().bfill().values
    q33, q66 = np.quantile(vol, [0.33, 0.66])
    return np.where(vol <= q33, 0, np.where(vol <= q66, 1, 2)), vol


# ─────────────────────────────────────────────────────────────────────────────
# A. Topological Boundaries (Fig 4): PCA decision-boundary scatter per regime
# ─────────────────────────────────────────────────────────────────────────────
def fig_topological_boundaries():
    X = np.load("artifacts/X_factors.npy")       # (972, 24)
    y = np.load("artifacts/y_factors.npy")        # (972,)

    # Compute rolling vol from full returns to assign regimes
    df = pd.read_csv("artifacts/sp500_full_daily_returns.csv", parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    r_full = df["^GSPC"].values
    _, vol = _get_regime_from_returns(r_full)

    # Map 972 windows to the 2016-2025 evaluation window
    # sp500_full has data from ~1999; evaluation starts ~2016
    # Use last 972+128 trading days for the 10y evaluation window
    r_eval = r_full[-len(X)-128:]    # slightly over-index, use eval portion
    regime_arr, _ = _get_regime_from_returns(r_eval)
    # Window end-dates align to regime_arr[128:128+len(X)]
    if len(regime_arr) >= 128 + len(X):
        w_regimes = regime_arr[128:128 + len(X)]
    else:
        # fallback: use global terciles on composite volatility feature (col 2)
        cv = X[:, 2]
        q33, q66 = np.quantile(cv, [0.33, 0.66])
        w_regimes = np.where(cv <= q33, 0, np.where(cv <= q66, 1, 2))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    regime_names = ["Low-Volatility", "Med-Volatility", "High-Volatility"]
    colors_pt = ["#0072B2", "#E69F00", "#D55E00"]

    for k, (ax, name, col) in enumerate(zip(axes, regime_names, colors_pt)):
        mask = w_regimes == k
        Xk = X[mask]
        yk = y[mask]
        if len(Xk) < 10:
            ax.set_title(f"{name}\n(n<10, skip)")
            continue

        # PCA projection
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xk)
        pca = PCA(n_components=2, random_state=42)
        Xp = pca.fit_transform(Xs)
        ev = pca.explained_variance_ratio_

        # Separate normal vs anomaly
        norm_mask = yk == 0
        anom_mask = yk == 1

        # Decision region: KDE-based contour showing anomaly density
        if anom_mask.sum() >= 4:
            try:
                kde = gaussian_kde(Xp[anom_mask].T, bw_method=0.6)
                xmin, xmax = Xp[:, 0].min()-0.5, Xp[:, 0].max()+0.5
                ymin, ymax = Xp[:, 1].min()-0.5, Xp[:, 1].max()+0.5
                xx, yy = np.meshgrid(np.linspace(xmin, xmax, 80),
                                     np.linspace(ymin, ymax, 80))
                Z = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                ax.contourf(xx, yy, Z, levels=6, cmap="Reds", alpha=0.25)
                ax.contour(xx, yy, Z, levels=3, colors=col, linewidths=0.8, alpha=0.5)
            except Exception:
                pass

        ax.scatter(Xp[norm_mask, 0], Xp[norm_mask, 1], s=12, c=GREY,
                   alpha=0.4, zorder=2, label="Normal")
        if anom_mask.sum() > 0:
            ax.scatter(Xp[anom_mask, 0], Xp[anom_mask, 1], s=55, c=col,
                       edgecolors=BLACK, linewidths=0.6, alpha=0.9, zorder=3,
                       label=f"Anomaly (n={anom_mask.sum()})", marker="*")

        ax.set_title(f"{name} regime\n(PC1={ev[0]:.0%}, PC2={ev[1]:.0%})",
                     fontsize=10, color=col)
        ax.set_xlabel("PC1", fontsize=9)
        ax.set_ylabel("PC2", fontsize=9)
        ax.legend(fontsize=8, markerscale=0.9, loc="upper right")
        ax.tick_params(labelsize=8)

    fig.suptitle(
        "Regime-Dependent Anomaly Geometry in Factor Space (PCA 2D projection)\n"
        "RC-MoE: S&P 500 2016–2025, colorblind-safe Wong palette",
        fontsize=11, y=1.02,
    )
    p = os.path.join(OUT, "fig_m3_topological_boundaries.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# ─────────────────────────────────────────────────────────────────────────────
# B. VaR Boundaries + EVT overlay (Fig 9/13)
# ─────────────────────────────────────────────────────────────────────────────
def fig_var_boundaries_evt():
    from src.evaluation.evt_risk import fit_gpd_pot, evt_var_es

    df = pd.read_csv("artifacts/sp500_full_daily_returns.csv", parse_dates=["Date"])
    df = df[(df.Date >= "2016-01-01") & (df.Date <= "2025-12-31")].reset_index(drop=True)
    r = df["^GSPC"].values
    dates = df["Date"].values

    # Regime assignment (60-day rolling vol terciles)
    regime, vol60 = _get_regime_from_returns(r)

    # Rolling GARCH-style VaR99: approximate with rolling 1% quantile
    window = 252
    rolling_var99 = []
    for i in range(len(r)):
        start = max(0, i - window)
        hist = r[start:i] if i > 0 else r[:1]
        rolling_var99.append(np.quantile(hist, 0.01) if len(hist) > 5 else np.nan)
    rolling_var99 = np.array(rolling_var99)

    # EVT-VaR99 per regime from actual GPD fits
    evt_var_by_regime = {}
    regime_names = {0: "Low", 1: "Med", 2: "High"}
    for k, nm in regime_names.items():
        losses = -r[regime == k]
        try:
            g = fit_gpd_pot(losses, 0.90)
            evt_var_by_regime[k] = evt_var_es(g, 0.99)["VaR"]
        except Exception:
            evt_var_by_regime[k] = np.quantile(losses, 0.99)

    # Plot
    fig, ax = plt.subplots(figsize=(13, 4.8))
    # Returns bar-style (thin grey)
    ax.bar(dates, r * 100, color=GREY, alpha=0.3, width=1, zorder=1, label="Daily return")
    # Rolling VaR99 (negative = downside)
    ax.plot(dates, rolling_var99 * 100, color=BLACK, lw=1.3, zorder=3,
            label="Rolling VaR$_{99}$ (1y window)")
    # VaR violations
    violations = r < rolling_var99
    ax.scatter(dates[violations], r[violations] * 100, color="#CC3311", s=20,
               zorder=5, label="VaR violation")

    # EVT-VaR horizontal lines per regime (regime-specific, show at unconditional level)
    regime_colors = [CB["Low"], CB["Med"], CB["High"]]
    for k, nm in regime_names.items():
        v = -evt_var_by_regime[k] * 100  # negative (downside)
        ax.axhline(v, color=regime_colors[k], lw=1.6, ls="--", alpha=0.85,
                   label=f"EVT-VaR$_{{99}}$ {nm}-Vol = {abs(v):.2f}%")

    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Return / VaR (%)", fontsize=10)
    ax.set_title(
        "Rolling VaR$_{99}$ with EVT Regime-Conditional Benchmarks (S&P 500, 2016–2025)\n"
        "Dashed: regime-conditional EVT-VaR$_{99}$ — unconditional model ignores 3×  regime spread",
        fontsize=10,
    )
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    p = os.path.join(OUT, "var_boundaries_chart.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# ─────────────────────────────────────────────────────────────────────────────
# C. Phase-3 architecture diagram — tree experts only (PDF + PNG)
# ─────────────────────────────────────────────────────────────────────────────
def fig_phase3_architecture():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    def box(ax, x, y, w, h, text, fc="#DBEAFE", ec="#1E40AF", fontsize=9,
            bold=False, color="black", radius=0.18):
        bp = FancyBboxPatch((x - w/2, y - h/2), w, h,
                            boxstyle=f"round,pad=0.05,rounding_size={radius}",
                            fc=fc, ec=ec, lw=1.4, zorder=3)
        ax.add_patch(bp)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                fontweight="bold" if bold else "normal", color=color, zorder=4,
                wrap=False)

    def arr(ax, x0, y0, x1, y1, color="#555555", lw=1.2):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))

    # ── Input block ──────────────────────────────────────────────
    box(ax, 1.2, 3.5, 1.8, 0.7, "Factor Vector\n$F_t\\in\\mathbb{R}^{24}$",
        fc="#F0F9FF", ec="#0072B2", fontsize=9, bold=True)

    # ── Gating ───────────────────────────────────────────────────
    box(ax, 3.3, 3.5, 1.8, 0.9,
        "Adaptive Gating\n(Regime $\\hat{R}_t$)\nRoll. Quantile + HMM\n+ Static Q.",
        fc="#FEF9C3", ec="#B45309", fontsize=8)
    arr(ax, 2.1, 3.5, 2.4, 3.5)

    # ── Three regime expert pools ─────────────────────────────────
    regime_cfg = [
        (5.8,  6.0, "Low-Volatility\nExpert Pool",  CB["Low"]),
        (5.8,  3.5, "Med-Volatility\nExpert Pool",  CB["Med"]),
        (5.8,  1.0, "High-Volatility\nExpert Pool", CB["High"]),
    ]
    meta_cfg = [
        (8.4,  6.0, CB["Low"]),
        (8.4,  3.5, CB["Med"]),
        (8.4,  1.0, CB["High"]),
    ]

    expert_models = ["RF", "XGBoost", "LightGBM"]
    for (rx, ry, rlabel, rcol), (mx, my, _) in zip(regime_cfg, meta_cfg):
        # Regime pool box
        box(ax, rx, ry, 2.1, 0.75, rlabel, fc=rcol + "33", ec=rcol, fontsize=8,
            bold=True, color=rcol)
        arr(ax, 4.2, 3.5, rx - 1.05, ry, color=rcol)

        # Three expert model badges inside pool
        for ei, em in enumerate(expert_models):
            ex = rx - 0.72 + ei * 0.72
            ey = ry - 0.15
            box(ax, ex, ey, 0.62, 0.35, em, fc="white", ec=rcol, fontsize=7,
                radius=0.08)

        # Meta-learner
        box(ax, mx, my, 1.4, 0.65, f"Meta-Learner\n$\\hat P_k(A_t=1)$",
            fc="#F5F5F5", ec="#555555", fontsize=8)
        arr(ax, rx + 1.05, ry, mx - 0.7, my)

        # Arrow to final aggregation
        arr(ax, mx + 0.7, my, 10.5, 3.5, color="#555555")

    # ── Final aggregation ─────────────────────────────────────────
    box(ax, 11.2, 3.5, 1.7, 0.9, "Weighted\nAggregation\n$\\sum_k w_t^{(k)}\\hat P_k$",
        fc="#F0FDF4", ec="#15803D", fontsize=8, bold=True)
    arr(ax, 12.05, 3.5, 12.7, 3.5)
    box(ax, 13.3, 3.5, 1.1, 0.7, "$\\hat P(A_t=1)$\n$D_t\\in\\{0,1\\}$",
        fc="#FFF7ED", ec="#C2410C", fontsize=8, bold=True, color="#C2410C")

    # ── Note on excluded deep experts ────────────────────────────
    note = ("Deep experts (RA-TAN, GRU-Attention, Hybrid) were implemented and trained "
            "but excluded from all results (Sec. III-D):\nunder sparse per-regime "
            "samples ($n\\!\\approx\\!64$--67 per fold) they did not exceed tree-based experts.")
    ax.text(7.0, 0.28, note, ha="center", va="bottom", fontsize=7.5,
            color="#666666", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF8F0", ec="#E5C07B", lw=0.8))

    ax.set_title(
        "Phase 3: Regime-Conditioned Meta-Ensemble Architecture (RC-MoE)\n"
        "Evaluated experts: RF, XGBoost, LightGBM (three tree-based families per regime)",
        fontsize=12, fontweight="bold", pad=10,
    )

    # Save both PDF and PNG
    for ext, fn in [("pdf", "phase-3.pdf"), ("png", "fig_phase3_moe.png")]:
        p = os.path.join(OUT, fn)
        fig.savefig(p, dpi=300, bbox_inches="tight", format=ext)
        print("wrote", p)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# D. Fix fig_basel_capital — more y-headroom so -24% label doesn't clip
# ─────────────────────────────────────────────────────────────────────────────
def fig_basel_capital_fixed():
    from src.evaluation.evt_risk import compare_capital_regimes

    df = pd.read_csv("artifacts/sp500_full_daily_returns.csv", parse_dates=["Date"])
    df = df[(df.Date >= "2016-01-01") & (df.Date <= "2025-12-31")].reset_index(drop=True)
    r = df["^GSPC"].values
    regime, _ = _get_regime_from_returns(r)

    cap = compare_capital_regimes(r, regime, 0.99)
    uncond = cap["unconditional"]["capital_10d"]
    regimes = ["Low", "Med", "High"]
    cond = [cap["per_regime"][n]["capital_cond_10d"] for n in regimes]
    gaps = [cap["per_regime"][n]["reservation_gap"] for n in regimes]

    fig, ax = plt.subplots(figsize=(7, 4.8))  # taller than before
    x = np.arange(len(regimes))
    w = 0.38
    ax.bar(x - w/2, [uncond]*3, w, label="Unconditional VaR capital", color=GREY)
    ax.bar(x + w/2, cond, w, label="Regime-conditional capital",
           color=[CB[n] for n in regimes])

    # headroom: extend y-axis 30% above max bar
    ymax = max(max(cond), uncond)
    ax.set_ylim(0, ymax * 1.35)

    for i, (c, gp) in enumerate(zip(cond, gaps)):
        # Place annotation above the taller of the two bars
        ypos = max(c, uncond) * 1.06
        sign_word = "over" if gp > 0 else "under"
        tag = f"{gp:+.0%}\n({sign_word})"
        ax.annotate(tag, (i + w/2, ypos),
                    ha="center", fontsize=9,
                    color="#D55E00" if gp < 0 else "#0072B2",
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}-Vol" for n in regimes])
    ax.set_ylabel("Basel III 10-day capital charge ($k$=3)")
    ax.set_title("Capital Adequacy: Unconditional VaR Over/Under-Reserves\n"
                 "by Regime (S&P 500, 2016–2025)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_basel_capital.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# ─────────────────────────────────────────────────────────────────────────────
# E. Fix fig_multimarket — more y-headroom so +93% label doesn't clip
# ─────────────────────────────────────────────────────────────────────────────
def fig_multimarket_fixed():
    d = json.load(open("artifacts/multimarket_evt_basel.json"))
    mk = list(d.keys())
    x = np.arange(len(mk))
    w = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))  # taller

    # (a) EVT-VaR99 low vs high
    lo = [d[m]["EVT_VaR99_Low"] * 100 for m in mk]
    hi = [d[m]["EVT_VaR99_High"] * 100 for m in mk]
    ax1.bar(x - w/2, lo, w, label="Low-Vol regime", color=CB["Low"])
    ax1.bar(x + w/2, hi, w, label="High-Vol regime", color=CB["High"])
    ax1.set_ylim(0, max(hi) * 1.20)  # 20% headroom
    for i, (a, b) in enumerate(zip(lo, hi)):
        ax1.annotate(f"{a:.1f}%", (i - w/2, a + 0.1), ha="center", va="bottom",
                     fontsize=8)
        ax1.annotate(f"{b:.1f}%", (i + w/2, b + 0.1), ha="center", va="bottom",
                     fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(mk, rotation=15)
    ax1.set_ylabel("EVT-VaR$_{99}$ (%)")
    ax1.set_title("(a) Tail Risk Rises Low$\\rightarrow$High in Every Market")
    ax1.legend(fontsize=9)

    # (b) Basel reservation gaps — positive (over-reserve) and negative (under-reserve)
    gl = [d[m]["Basel_gap_Low"] * 100 for m in mk]
    gh = [d[m]["Basel_gap_High"] * 100 for m in mk]
    ax2.bar(x - w/2, gl, w, label="Low-Vol (over-reserve)", color=CB["Low"])
    ax2.bar(x + w/2, gh, w, label="High-Vol (under-reserve)", color=CB["High"])
    ax2.axhline(0, color="black", lw=0.8)

    # extra headroom: top = max(gl)*1.25, bottom = min(gh)*1.25
    ax2.set_ylim(min(gh) * 1.30, max(gl) * 1.30)

    for i, (a, b) in enumerate(zip(gl, gh)):
        ax2.annotate(f"{a:+.0f}%", (i - w/2, a + 1.5), ha="center", va="bottom",
                     fontsize=8, fontweight="bold")
        ax2.annotate(f"{b:+.0f}%", (i + w/2, b - 1.5), ha="center", va="top",
                     fontsize=8, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(mk, rotation=15)
    ax2.set_ylabel("Basel capital reservation gap (%)")
    ax2.set_title("(b) Unconditional VaR Mis-Reserves in Every Market")
    ax2.legend(fontsize=8, loc="upper right")

    fig.suptitle("Multi-Market Replication of the EVT / Basel III Finding "
                 "(2016–2025)", y=1.02, fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_multimarket.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Priority 3 Figures ===\n")
    print("A. Topological boundaries...")
    fig_topological_boundaries()
    print("B. VaR boundaries + EVT overlay...")
    fig_var_boundaries_evt()
    print("C. Phase-3 architecture (tree-only)...")
    fig_phase3_architecture()
    print("D. Basel capital (y-headroom fix)...")
    fig_basel_capital_fixed()
    print("E. Multi-market (y-headroom fix)...")
    fig_multimarket_fixed()
    print("\nAll done.")
