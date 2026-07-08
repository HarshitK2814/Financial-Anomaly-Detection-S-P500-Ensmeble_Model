"""
FF/beta figure for §VI.F (Factor Exposure Analysis).
Panel (a): FF5+MOM factor loadings with 95% CI (significance).
Panel (b): rolling 60-day market beta — overlay vs buy-and-hold.
Run: .venv/Scripts/python.exe -m src.utils.make_beta_fig
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.factor_alpha import load_ff_factors

OUT = "correct figures"
plt.rcParams.update({"font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 300})
CB_BLUE, CB_ORANGE, CB_GREY = "#0072B2", "#E69F00", "#999999"


def rolling_beta(strat_excess, mkt_excess, window=60):
    s = pd.Series(strat_excess)
    m = pd.Series(mkt_excess)
    cov = s.rolling(window).cov(m)
    var = m.rolling(window).var()
    return (cov / var).values


def main():
    res = json.load(open("artifacts/ff_alpha_results.json"))
    betas, bp = res["betas"], res["beta_pvalues"]

    h = pd.read_csv("artifacts/hedge_daily_returns.csv", parse_dates=["date"])
    ff = load_ff_factors("artifacts/data/ff_factors/FF5_daily.csv",
                         "artifacts/data/ff_factors/MOM_daily.csv")
    df = h.set_index("date").join(ff, how="inner").dropna()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    # Panel (a): factor loadings
    order = ["MktRF", "SMB", "HML", "RMW", "CMA", "MOM"]
    vals = [betas[k] for k in order]
    cols = [CB_BLUE if bp[k] < 0.05 else CB_GREY for k in order]
    bars = ax1.bar(order, vals, color=cols)
    for k, b in zip(order, bars):
        star = "*" if bp[k] < 0.05 else ""
        ax1.annotate(f"{betas[k]:+.2f}{star}",
                     (b.get_x() + b.get_width()/2, b.get_height()),
                     ha="center",
                     va="bottom" if b.get_height() >= 0 else "top",
                     fontsize=9)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_ylabel("Factor loading")
    ax1.set_title(f"(a) FF5+MOM Loadings (Adj. $R^2$={res['r2_adj']:.2f})\n"
                  f"$\\alpha$={res['alpha_annualized']:+.1%} p.a. "
                  f"(p={res['alpha_pvalue']:.2f}, n.s.)")
    ax1.text(0.5, -0.30, "* = p<0.05; blue = significant",
             transform=ax1.transAxes, ha="center", fontsize=8, color="#555")

    # Panel (b): rolling 60-day market beta
    mkt = df["MktRF"].values
    bh_beta = rolling_beta(df["buy_hold"].values - df["RF"].values, mkt)
    ov_beta = rolling_beta(df["regime_aware_hedged"].values - df["RF"].values, mkt)
    x = df.index
    ax2.plot(x, bh_beta, color=CB_GREY, lw=1.5, label="Buy-and-Hold")
    ax2.plot(x, ov_beta, color=CB_ORANGE, lw=1.8, label="Regime-aware overlay")
    ax2.axhline(betas["MktRF"], ls="--", color=CB_BLUE, lw=1.2,
                label=f"Full-sample $\\beta$={betas['MktRF']:.2f}")
    ax2.set_ylabel("Rolling 60-day market $\\beta$")
    ax2.set_title("(b) Market Beta: Overlay Cuts Exposure ~37%")
    ax2.legend(fontsize=8, loc="lower left")
    ax2.tick_params(axis="x", rotation=30, labelsize=8)

    fig.tight_layout()
    p = os.path.join(OUT, "fig_factor_beta.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
