"""
Figures for the analyses added in the revision (UAAI 262098302), in the style of the
existing paper figures (Wong colour-blind palette, Blues heatmaps, sans-serif, 300 dpi).

  fig_sensitivity_oos   out-of-sample F1 over L x K, RF expert vs full expert  (R1-Q4)
  fig_var_coverage      VaR99 violations by market and regime, uncond vs cond   (R1-Q5)
  fig_execution_lag     lagged overlay vs buy-and-hold, wealth and drawdown     (R1-Q6)

Each figure is written as PDF (for Overleaf), EPS and 300-dpi TIFF (journal upload).
No transparency is used, so the EPS files render the same as the PDFs.

Run from model root:  python src/evaluation/make_revision_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ART = Path("artifacts")
OUT = Path("../Revision_Files/new_figures")
for sub in ("png", "pdf", "eps", "tiff"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)

# Wong colour-blind-safe palette (same as make_kronos_style_figs.py)
CB_BLUE, CB_ORANGE, CB_GREEN = "#0072B2", "#E69F00", "#009E73"
CB_RED, CB_PURPLE, CB_CYAN = "#D55E00", "#CC79A7", "#56B4E9"
CB_GREY, CB_BLACK = "#999999", "#000000"
GRID = "#DDDDDD"
DPI = 300

plt.rcParams.update({
    "font.size": 10, "font.family": "sans-serif", "figure.dpi": DPI,
    "axes.grid": True, "grid.color": GRID, "grid.alpha": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "ps.fonttype": 42, "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(OUT / "png" / f"{name}.png", bbox_inches="tight", dpi=DPI)
    fig.savefig(OUT / "pdf" / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / "eps" / f"{name}.eps", bbox_inches="tight")
    fig.savefig(OUT / "tiff" / f"{name}.tif", bbox_inches="tight", dpi=DPI,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("saved", name)


# ------------------------------------------------------------------ R1-Q4
def fig_sensitivity():
    rf = pd.DataFrame(json.loads((ART / "sensitivity_results.json").read_text()))
    ens = pd.DataFrame(json.loads((ART / "sensitivity_full_ensemble.json").read_text())["ensemble_oos"])
    Ls, Ks = [64, 96, 128, 192], [2, 3, 4]
    panels = [("(a) RF expert", rf), ("(b) Full expert (RF + XGBoost + LightGBM)", ens)]
    vmax = max(rf.OOS_F1.max(), ens.OOS_F1.max())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (title, d) in zip(axes, panels):
        M = d.pivot(index="L", columns="K", values="OOS_F1").loc[Ls, Ks].values
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        ax.grid(False)
        for i in range(len(Ls)):
            best = M[i].max()
            for j in range(len(Ks)):
                v = M[i, j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=11,
                        color="white" if v > 0.55 * vmax else "black",
                        fontweight="bold" if v == best and best > 0 else "normal")
        ax.set_xticks(range(len(Ks)), [f"K = {k}" for k in Ks])
        ax.set_yticks(range(len(Ls)), [f"L = {l}" for l in Ls])
        # outline the paper configuration (L = 128, K = 3)
        ax.add_patch(mpatches.Rectangle((0.5, 1.5), 1, 1, fill=False, ec=CB_RED, lw=2.2))
        ax.set_title(title, fontsize=11, fontweight="bold")
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("Out-of-sample F1 (%)")
    save(fig, "fig_sensitivity_oos")


# ------------------------------------------------------------------ R1-Q5
def fig_coverage():
    cov = pd.read_csv(ART / "coverage_table.csv")
    markets = ["S&P 500", "NASDAQ", "Russell 2000", "Bitcoin"]
    regs = ["Low", "Med", "High"]
    fig, axes = plt.subplots(1, len(markets), figsize=(12, 3.8), sharey=False)
    w = 0.36
    for ax, mk in zip(axes, markets):
        d = cov[cov.market == mk]
        x = np.arange(len(regs))
        for off, model, col, lab in [(-w / 2, "Uncond", CB_GREY, "Unconditional VaR$_{99}$"),
                                     (w / 2, "Cond", CB_BLUE, "Regime-conditional EVT-VaR$_{99}$")]:
            dd = d[d.model == model].set_index("regime").loc[regs]
            bars = ax.bar(x + off, dd.violations, w, color=col, label=lab, zorder=3)
            for b, p in zip(bars, dd.p_cc):
                if p < 0.05:  # Christoffersen conditional coverage rejected
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4, "✕",
                            ha="center", va="bottom", fontsize=9, color=CB_RED, fontweight="bold")
        exp = d[d.model == "Uncond"].set_index("regime").loc[regs].expected
        for xi, e in zip(x, exp):
            ax.hlines(e, xi - 0.42, xi + 0.42, color=CB_BLACK, lw=1.6, ls="--", zorder=4)
        ax.set_xticks(x, ["Low-Vol", "Med-Vol", "High-Vol"])
        ax.set_title(mk, fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(d.violations.max(), exp.max()) * 1.25)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("VaR$_{99}$ violations")
    handles = [mpatches.Patch(color=CB_GREY, label="Unconditional VaR$_{99}$"),
               mpatches.Patch(color=CB_BLUE, label="Regime-conditional EVT-VaR$_{99}$"),
               plt.Line2D([], [], color=CB_BLACK, ls="--", lw=1.6, label="Expected (1%)"),
               plt.Line2D([], [], color=CB_RED, marker="$✕$", ls="", ms=7,
                          label="Conditional coverage rejected ($p_{cc}<0.05$)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig_var_coverage")


# ------------------------------------------------------------------ R1-Q6
def fig_execution_lag():
    RF, HOLD, COST = 0.04 / 252, 5, 5
    pred = pd.read_csv(ART / "sp500_oos_dated_predictions.csv", parse_dates=["date"]).set_index("date")
    ret = pd.read_csv(ART / "sp500_full_daily_returns.csv", parse_dates=["Date"]).set_index("Date").iloc[:, 0]
    ret = ret[(ret.index >= pred.index.min()) & (ret.index <= pred.index.max())]
    s = pred["pred_moe"].reindex(ret.index).fillna(0).values

    def run(lag):
        hedged = np.zeros(len(ret))
        for i in np.where(s == 1)[0]:
            hedged[i + lag:i + lag + HOLD] = 1
        r = np.where(hedged == 1, RF, ret.values) - np.abs(np.diff(np.r_[0, hedged])) * COST / 1e4
        return np.cumprod(1 + r), hedged

    bh = np.cumprod(1 + ret.values)
    w1, h1 = run(1)
    w2, _ = run(2)
    idx = ret.index
    dd = lambda w: 100 * (w / np.maximum.accumulate(w) - 1)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1.2]})
    # hedged (in-cash) periods of the lag-1 overlay, light solid shading (no alpha, EPS-safe)
    edges = np.flatnonzero(np.diff(np.r_[0, h1, 0]))
    for st, en in zip(edges[::2], edges[1::2]):
        for ax in (a1, a2):
            ax.axvspan(idx[st], idx[min(en, len(idx) - 1)], color="#DCEBF7", lw=0, zorder=0)
    series = [("Buy-and-Hold", bh, CB_BLACK, 2.0, "-"),
              ("RC-MoE overlay, lag $k=1$ (5 bp)", w1, CB_BLUE, 2.2, "-"),
              ("RC-MoE overlay, lag $k=2$ (5 bp)", w2, CB_GREEN, 1.6, "--")]
    for name, w, col, lw, ls in series:
        a1.plot(idx, 100 * (w - 1), color=col, lw=lw, ls=ls, label=name, zorder=3)
        a2.plot(idx, dd(w), color=col, lw=lw * 0.8, ls=ls, label=name, zorder=3)
    a1.axhline(0, color=CB_BLACK, lw=0.6)
    a1.set_ylabel("Cumulative return (%)")
    a1.set_title("(a) Cumulative returns with explicit execution lag (Dec 2018 to Dec 2025)",
                 fontsize=11, fontweight="bold")
    shade = mpatches.Patch(color="#DCEBF7", label="In cash (lag-1 overlay)")
    a1.legend(handles=a1.get_legend_handles_labels()[0] + [shade], fontsize=8.5,
              loc="upper left", framealpha=1.0)
    a2.set_ylabel("Drawdown (%)")
    a2.set_title("(b) Drawdown", fontsize=11, fontweight="bold")
    a2.set_xlabel("Date")
    a2.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout()
    save(fig, "fig_execution_lag")


if __name__ == "__main__":
    fig_sensitivity()
    fig_coverage()
    fig_execution_lag()
