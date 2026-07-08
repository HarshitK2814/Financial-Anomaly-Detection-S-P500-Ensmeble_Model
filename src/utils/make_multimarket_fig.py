"""
Multi-market EVT/Basel replication figure.
(a) EVT-VaR99 Low vs High across markets. (b) Basel reservation gap Low/High.
Run: .venv/Scripts/python.exe -m src.utils.make_multimarket_fig
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "correct figures"
plt.rcParams.update({"font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 300})
CB_BLUE, CB_ORANGE = "#0072B2", "#D55E00"


def main():
    d = json.load(open("artifacts/multimarket_evt_basel.json"))
    mk = list(d.keys())
    x = np.arange(len(mk))
    w = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    # (a) EVT-VaR99 low vs high
    lo = [d[m]["EVT_VaR99_Low"] * 100 for m in mk]
    hi = [d[m]["EVT_VaR99_High"] * 100 for m in mk]
    ax1.bar(x - w/2, lo, w, label="Low-Vol regime", color=CB_BLUE)
    ax1.bar(x + w/2, hi, w, label="High-Vol regime", color=CB_ORANGE)
    for i, (a, b) in enumerate(zip(lo, hi)):
        ax1.annotate(f"{a:.1f}", (i - w/2, a), ha="center", va="bottom", fontsize=8)
        ax1.annotate(f"{b:.1f}", (i + w/2, b), ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x); ax1.set_xticklabels(mk, rotation=15)
    ax1.set_ylabel("EVT-VaR$_{99}$ (%)")
    ax1.set_title("(a) Tail Risk Rises Low$\\rightarrow$High in Every Market")
    ax1.legend(fontsize=9)

    # (b) Basel reservation gaps
    gl = [d[m]["Basel_gap_Low"] * 100 for m in mk]
    gh = [d[m]["Basel_gap_High"] * 100 for m in mk]
    ax2.bar(x - w/2, gl, w, label="Low-Vol (over-reserve)", color=CB_BLUE)
    ax2.bar(x + w/2, gh, w, label="High-Vol (under-reserve)", color=CB_ORANGE)
    ax2.axhline(0, color="black", lw=0.8)
    for i, (a, b) in enumerate(zip(gl, gh)):
        ax2.annotate(f"{a:+.0f}%", (i - w/2, a), ha="center", va="bottom", fontsize=8)
        ax2.annotate(f"{b:+.0f}%", (i + w/2, b), ha="center", va="top", fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels(mk, rotation=15)
    ax2.set_ylabel("Basel capital reservation gap")
    ax2.set_title("(b) Unconditional VaR Mis-Reserves in Every Market")
    ax2.legend(fontsize=8, loc="upper right")

    fig.suptitle("Multi-Market Replication of the EVT / Basel III Finding "
                 "(2016--2025)", y=1.02, fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_multimarket.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
