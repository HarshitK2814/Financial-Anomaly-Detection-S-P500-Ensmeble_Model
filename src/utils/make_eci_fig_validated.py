"""
Regenerate eci_chart.png with validated bootstrap CIs and null model baseline.
Run: .venv/Scripts/python.exe -m src.utils.make_eci_fig_validated
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "correct figures"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 300, "font.family": "DejaVu Sans"})

DARK_BLUE = "#1e3a5f"
MID_BLUE  = "#4a90c4"
CB = {"Low": "#0072B2", "Med": "#E69F00", "High": "#D55E00", "Global": "#555555"}
W, S = "#FFFFFF", "#475569"
N = "#0F172A"
NULL_COLOR = "#999999"


def make():
    res = json.load(open("artifacts/eci_validation_results.json"))

    labels = ["Low-Vol\n(ECI=0.646)", "Med-Vol\n(ECI=0.763)",
              "High-Vol\n(ECI=0.789)", "Global\n(ECI=0.768)"]
    keys   = ["Low", "Med", "High", "Global"]
    scores = [res[k]["cross_model_eci"] for k in keys]
    ci_lo  = [res[k]["ci_lo"] for k in keys]
    ci_hi  = [res[k]["ci_hi"] for k in keys]
    null_mean = float(np.mean([res[k]["null_mean"] for k in ["Low","Med","High"]]))

    colors = [CB[k] for k in keys]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=W)
    ax.set_facecolor(W)

    bars = ax.bar(x, scores, color=colors, edgecolor=N, linewidth=0.9, width=0.55,
                  zorder=3, alpha=0.88)
    # 95% CI error bars
    yerr_lo = [s - lo for s, lo in zip(scores, ci_lo)]
    yerr_hi = [hi - s  for s, hi in zip(scores, ci_hi)]
    ax.errorbar(x, scores, yerr=[yerr_lo, yerr_hi],
                fmt="none", ecolor=N, elinewidth=1.8, capsize=5, zorder=5)

    # Null model reference line
    ax.axhline(null_mean, color=NULL_COLOR, ls="--", lw=1.8, zorder=2,
               label=f"Null model (random rankings) $\\approx${null_mean:.2f}")
    ax.fill_between([-0.4, len(labels)-0.6], 0.42, 0.59,
                     color=NULL_COLOR, alpha=0.08, zorder=1)

    # Value labels on bars with significance stars
    sigs = {"Low": "$z\\!=\\!3.5^{**}$", "Med": "$z\\!=\\!6.3^{***}$",
            "High": "$z\\!=\\!7.1^{***}$", "Global": ""}
    for bar, val, lbl, k in zip(bars, scores, labels, keys):
        star = sigs[k]
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (ci_hi[keys.index(k)] - val) + 0.016,
                f"{val:.3f} {star}",
                ha="center", va="bottom", fontsize=10,
                fontweight="bold" if val == max(scores) else "normal",
                color=CB[k] if k != "Global" else DARK_BLUE)

    # Pairwise significance annotation
    ax.annotate("All pairwise Wilcoxon $p<0.001$",
                xy=(1.5, 0.85), fontsize=9, ha="center", color=N,
                bbox=dict(boxstyle="round,pad=0.3", fc="#F0F9FF", ec="#0072B2", lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 0.97)
    ax.set_ylabel("ECI Score (1 = perfect cross-model agreement)", fontsize=11,
                  fontweight="bold", color=N)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)

    ax.set_title("Explainability Consistency Index (ECI) by Volatility Regime\n"
                 "RF vs XGBoost vs LightGBM: bootstrap 95% CI, 2,000 replicates",
                 fontsize=13, fontweight="bold", color=N, pad=14)

    fig.text(0.12, 0.01,
             "ECI = cross-model agreement (RF vs XGB vs LGBM tree importances). "
             "Cross-method ECI (tree FI vs SHAP) = 0.85--0.88, consistently higher. "
             "** p<0.01, *** p<0.001 vs null.",
             fontsize=8, color=S, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    p = os.path.join(OUT, "eci_chart.png")
    plt.savefig(p, dpi=300, bbox_inches="tight", facecolor=W)
    plt.close()
    print("wrote", p)


if __name__ == "__main__":
    make()
