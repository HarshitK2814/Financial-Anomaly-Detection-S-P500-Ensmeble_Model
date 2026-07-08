"""
Generate the two new risk figures (journal-quality, 300 DPI, colorblind-safe):
  fig_evt_tail.png   - GPD tail fit per regime + EVT-VaR99 markers
  fig_basel_capital.png - Basel III capital: unconditional vs conditional
Run: .venv/Scripts/python.exe src/utils/make_evt_basel_figs.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evaluation.evt_risk import fit_gpd_pot, evt_var_es, compare_capital_regimes

OUT = "correct figures"
os.makedirs(OUT, exist_ok=True)
# Colorblind-safe (Wong/Tableau)
CB = {"Low": "#0072B2", "Med": "#E69F00", "High": "#D55E00"}
plt.rcParams.update({"font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 300})


def load():
    df = pd.read_csv("artifacts/sp500_full_daily_returns.csv",
                     parse_dates=["Date"])
    df = df[(df.Date >= "2016-01-01") & (df.Date <= "2025-12-31")]
    r = df["^GSPC"].values
    vol = pd.Series(r).rolling(60).std().bfill().values
    q33, q66 = np.quantile(vol, [0.33, 0.66])
    reg = np.where(vol <= q33, 0, np.where(vol <= q66, 1, 2))
    return r, reg


def fig_evt_tail(r, reg):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for k, nm in [(0, "Low"), (1, "Med"), (2, "High")]:
        ax = axes[k]
        losses = -r[reg == k]
        g = fit_gpd_pot(losses, 0.90)
        v99 = evt_var_es(g, 0.99)["VaR"]
        u = g["threshold_u"]
        exceed = losses[losses > u] - u
        # empirical vs fitted GPD survival
        xs = np.linspace(0, exceed.max(), 200)
        surv_emp = [np.mean(exceed > x) for x in xs]
        surv_gpd = stats.genpareto.sf(xs, g["xi_shape"], loc=0,
                                      scale=g["beta_scale"])
        ax.plot((xs + u) * 100, surv_emp, "o", ms=3, color=CB[nm],
                alpha=0.5, label="Empirical")
        ax.plot((xs + u) * 100, surv_gpd, "-", lw=2, color="black",
                label=f"GPD $\\hat\\xi$={g['xi_shape']:+.3f}")
        ax.axvline(v99 * 100, ls="--", color=CB[nm], lw=1.5,
                   label=f"EVT-VaR99={v99*100:.2f}%")
        ax.set_title(f"{nm}-Vol regime ($N_u$={g['n_exceed']})")
        ax.set_xlabel("Loss (%)")
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("Exceedance prob.")
    fig.suptitle("EVT Peak-Over-Threshold Tail Fit by Volatility Regime "
                 "(S&P 500, 2016--2025)", y=1.02, fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_evt_tail.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig_basel(r, reg):
    cap = compare_capital_regimes(r, reg, 0.99)
    uncond = cap["unconditional"]["capital_10d"]
    regimes = ["Low", "Med", "High"]
    cond = [cap["per_regime"][n]["capital_cond_10d"] for n in regimes]
    gaps = [cap["per_regime"][n]["reservation_gap"] for n in regimes]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(regimes))
    w = 0.38
    ax.bar(x - w/2, [uncond]*3, w, label="Unconditional VaR capital",
           color="#999999")
    ax.bar(x + w/2, cond, w, label="Regime-conditional capital",
           color=[CB[n] for n in regimes])
    for i, (c, gp) in enumerate(zip(cond, gaps)):
        tag = f"{gp:+.0%}\n{'over' if gp>0 else 'under'}"
        ax.annotate(tag, (i + w/2, max(c, uncond) + 0.01),
                    ha="center", fontsize=9,
                    color="#D55E00" if gp < 0 else "#0072B2")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}-Vol" for n in regimes])
    ax.set_ylabel("Basel III 10-day capital charge ($k$=3)")
    ax.set_title("Capital Adequacy: Unconditional VaR Over/Under-Reserves\n"
                 "by Regime (S&P 500, 2016--2025)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_basel_capital.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    r, reg = load()
    fig_evt_tail(r, reg)
    fig_basel(r, reg)
    print("DONE")
