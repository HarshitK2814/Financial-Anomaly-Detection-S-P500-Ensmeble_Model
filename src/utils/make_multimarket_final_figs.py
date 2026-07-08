"""
Publication-ready multi-market figures from Path A v1 results.

Generates three panels:
  (a) Bar chart: Returns-only baseline vs Full RC-MoE pipeline (Global & MoE)
  (b) Learning curve: fold-by-fold event-wise F1 per market (shows data dependence)
  (c) Precision-Recall scatter: all markets + both models

Saves:
  correct figures/fig_multimarket_final.png   (3-panel, 300 DPI)
  correct figures/fig_multimarket_bar.png     (bar only, for paper insert)
  correct figures/fig_multimarket_curve.png   (learning curve only)

Run from repo root:
  .venv/Scripts/python.exe -m src.utils.make_multimarket_final_figs
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ARTIFACTS = "artifacts"
FIGURES   = "correct figures"
os.makedirs(FIGURES, exist_ok=True)

CB_BLUE   = "#0072B2"   # Global model
CB_ORANGE = "#D55E00"   # MoE / RC-MoE
CB_GREEN  = "#009E73"   # returns-only baseline
CB_GRAY   = "#666666"   # reference lines
MARKER_SIZE = 8

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MKTS    = ["SP500", "NASDAQ", "Russell2000", "Bitcoin"]
DISPLAY = ["S&P 500", "NASDAQ", "Russell 2000", "Bitcoin"]


def load_results():
    v1_path  = os.path.join(ARTIFACTS, "multimarket_path_a_results.json")
    bas_path = os.path.join(ARTIFACTS, "multimarket_detection_results.json")

    v1  = json.load(open(v1_path))  if os.path.exists(v1_path)  else {}
    bas = json.load(open(bas_path)) if os.path.exists(bas_path) else {}
    return v1, bas


# ─────────────────────────────────────────────────────────────────────────────
# Panel (a) — bar chart
# ─────────────────────────────────────────────────────────────────────────────

def panel_bar(ax, v1: dict, bas: dict) -> None:
    x = np.arange(len(MKTS))
    w = 0.22

    # Returns-only Global (point-wise, from simple baseline)
    f1_bas_g = [bas.get(m, {}).get("f1_global", 0) for m in MKTS]
    # Full pipeline Global (event-wise)
    f1_v1_g  = [v1.get(m,  {}).get("f1_global_ew", 0) for m in MKTS]
    # Full pipeline MoE (event-wise) — PRIMARY
    f1_v1_m  = [v1.get(m,  {}).get("f1_moe_ew",    0) for m in MKTS]

    b0 = ax.bar(x - w,   f1_bas_g, w, label="Returns-only Global (point-wise)",
                color=CB_GREEN, alpha=0.7, edgecolor="white", linewidth=0.5)
    b1 = ax.bar(x,       f1_v1_g,  w, label="Full pipeline — Global (event-wise, tol=5)",
                color=CB_BLUE,  alpha=0.8, edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + w,   f1_v1_m,  w, label="Full pipeline — RC-MoE (event-wise, tol=5)",
                color=CB_ORANGE, edgecolor="white", linewidth=0.5)

    # Reference: main model in-sample
    ax.axhline(0.738, color=CB_GRAY, ls="--", lw=1.2, alpha=0.7,
               label="Main model in-sample, S&P 500 (0.738)")

    # Bar value labels
    for bars in [b0, b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.007,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7.2)

    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY, fontsize=10)
    ax.set_ylabel("F1 Score")
    ax.set_title("(a) OOS Detection Performance Across Asset Classes", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", ncol=1)
    ymax = max(max(f1_bas_g + f1_v1_g + f1_v1_m), 0.738) + 0.05
    ax.set_ylim(0, min(ymax * 1.25, 1.05))


# ─────────────────────────────────────────────────────────────────────────────
# Panel (b) — learning curve (fold-by-fold)
# ─────────────────────────────────────────────────────────────────────────────

def panel_learning_curve(ax, v1: dict) -> None:
    markers = ["o", "s", "^", "D"]
    colors  = [CB_BLUE, CB_ORANGE, CB_GREEN, "#CC79A7"]
    folds   = [1, 2, 3, 4]

    for i, (mkt, disp) in enumerate(zip(MKTS, DISPLAY)):
        mkt_data = v1.get(mkt, {})
        fold_recs = mkt_data.get("folds", [])
        f1_moe   = [r.get("f1_moe", 0)    for r in fold_recs if "fold" in r]
        f1_global= [r.get("f1_global", 0) for r in fold_recs if "fold" in r]
        f_idx    = [r.get("fold", 0)       for r in fold_recs if "fold" in r]

        if not f1_moe:
            continue

        ax.plot(f_idx, f1_moe, marker=markers[i], color=colors[i],
                lw=1.8, ms=MARKER_SIZE, label=f"{disp} RC-MoE")
        ax.plot(f_idx, f1_global, marker=markers[i], color=colors[i],
                lw=1.2, ms=MARKER_SIZE, alpha=0.45, ls="--")

    ax.set_xlabel("Walk-forward Fold (training size increases left to right)")
    ax.set_ylabel("Event-wise F1 (tolerance = 5)")
    ax.set_title("(b) RC-MoE Performance vs Training Data Volume", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xticks(folds)
    ax.set_xticklabels([f"Fold {f}\n(~{f*20}% train)" for f in folds], fontsize=8)
    ax.set_ylim(0, 0.70)

    # Annotate fold 4 S&P MoE
    sp_folds = v1.get("SP500", {}).get("folds", [])
    if sp_folds:
        f4 = next((r for r in sp_folds if r.get("fold") == 4), None)
        if f4:
            ax.annotate(f"S&P MoE F1={f4['f1_moe']:.3f}\n(fold 4, 80% train)",
                        xy=(4, f4["f1_moe"]), xytext=(3.3, f4["f1_moe"] + 0.07),
                        arrowprops=dict(arrowstyle="->", color=CB_BLUE, lw=1),
                        fontsize=7.5, color=CB_BLUE)


# ─────────────────────────────────────────────────────────────────────────────
# Panel (c) — precision-recall scatter
# ─────────────────────────────────────────────────────────────────────────────

def panel_pr_scatter(ax, v1: dict, bas: dict) -> None:
    markers  = ["o", "s", "^", "D"]
    colors   = [CB_BLUE, CB_ORANGE, CB_GREEN, "#CC79A7"]

    for i, (mkt, disp) in enumerate(zip(MKTS, DISPLAY)):
        # Full pipeline MoE
        prec_m = v1.get(mkt, {}).get("prec_moe_ew", 0)
        rec_m  = v1.get(mkt, {}).get("rec_moe_ew",  0)
        # Full pipeline Global
        prec_g = v1.get(mkt, {}).get("prec_global_ew", 0)
        rec_g  = v1.get(mkt, {}).get("rec_global_ew",  0)

        ax.scatter(rec_m, prec_m, marker=markers[i], color=colors[i],
                   s=90, zorder=5, label=f"{disp}")
        ax.scatter(rec_g, prec_g, marker=markers[i], color=colors[i],
                   s=60, zorder=4, alpha=0.45, edgecolors=colors[i], facecolors="none",
                   linewidths=1.5)
        ax.annotate(disp, xy=(rec_m, prec_m),
                    xytext=(rec_m + 0.003, prec_m - 0.03),
                    fontsize=7.5, color=colors[i])

    # ISO-F1 contours
    r_ = np.linspace(0.01, 1.0, 300)
    for f1_val in [0.2, 0.3, 0.4, 0.5]:
        p_ = f1_val * r_ / (2 * r_ - f1_val + 1e-8)
        mask = (p_ >= 0) & (p_ <= 1)
        ax.plot(r_[mask], p_[mask], lw=0.7, ls=":", color=CB_GRAY, alpha=0.5)
        idx = np.argmin(np.abs(r_[mask] - 0.35))
        ax.text(r_[mask][idx], p_[mask][idx] + 0.02,
                f"F1={f1_val}", fontsize=7, color=CB_GRAY, alpha=0.7)

    # Legend entry for filled = MoE, hollow = Global
    from matplotlib.lines import Line2D
    leg_extra = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="gray", ms=8, label="RC-MoE (filled)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="w",
               markeredgecolor="gray", ms=8, label="Global (hollow)"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + leg_extra, labels + ["RC-MoE (filled)", "Global (hollow)"],
              fontsize=7.5, loc="upper right", ncol=2)

    ax.set_xlabel("Recall (event-wise, tol=5)")
    ax.set_ylabel("Precision (event-wise, tol=5)")
    ax.set_title("(c) Precision-Recall Tradeoff — Full RC-MoE Pipeline", fontsize=10)
    ax.set_xlim(0, 0.40)
    ax.set_ylim(0, 1.05)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def make_all():
    v1, bas = load_results()

    # ── 3-panel combined figure ───────────────────────────────────
    fig = plt.figure(figsize=(17, 5.5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32)

    ax_bar   = fig.add_subplot(gs[0])
    ax_curve = fig.add_subplot(gs[1])
    ax_pr    = fig.add_subplot(gs[2])

    panel_bar(ax_bar, v1, bas)
    panel_learning_curve(ax_curve, v1)
    panel_pr_scatter(ax_pr, v1, bas)

    fig.suptitle(
        "RC-MoE Multi-Market OOS Anomaly Detection (24 Economic Factors, "
        "RF+XGB+LightGBM Ensemble, 5-Fold Walk-Forward CV, 2016-2025)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    p = os.path.join(FIGURES, "fig_multimarket_final.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {p}")

    # ── Bar-only (for paper body) ─────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    panel_bar(ax2, v1, bas)
    fig2.tight_layout()
    p2 = os.path.join(FIGURES, "fig_multimarket_bar.png")
    fig2.savefig(p2, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved -> {p2}")

    # ── Learning curve only ───────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    panel_learning_curve(ax3, v1)
    fig3.tight_layout()
    p3 = os.path.join(FIGURES, "fig_multimarket_curve.png")
    fig3.savefig(p3, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved -> {p3}")

    # ── Print paper table ─────────────────────────────────────────
    print("\n" + "="*80)
    print("TABLE III  —  Multi-Market OOS Anomaly Detection  (Full RC-MoE Pipeline)")
    print("="*80)
    hdr = f"{'Market':<14} {'Win':>5} {'Anom%':>6} "
    hdr += f"{'Gbl Prec':>9} {'Gbl Rec':>8} {'Gbl EW-F1':>10} "
    hdr += f"{'MoE Prec':>9} {'MoE Rec':>8} {'MoE EW-F1':>10} {'Delta F1':>9}"
    print(hdr)
    print("-"*80)
    for mkt, disp in zip(MKTS, DISPLAY):
        r = v1.get(mkt, {})
        gf  = r.get("f1_global_ew", 0)
        mf  = r.get("f1_moe_ew",    0)
        gp  = r.get("prec_global_ew", 0)
        gr  = r.get("rec_global_ew",  0)
        mp  = r.get("prec_moe_ew",    0)
        mr  = r.get("rec_moe_ew",     0)
        win = r.get("n_windows",      0)
        anom= r.get("anomaly_rate",   0)
        d   = mf - gf
        sign= "+" if d >= 0 else ""
        print(f"{disp:<14} {win:>5} {anom*100:>5.1f}% "
              f"{gp:>9.3f} {gr:>8.3f} {gf:>10.3f} "
              f"{mp:>9.3f} {mr:>8.3f} {mf:>10.3f} {sign}{d:>8.3f}")
    print("-"*80)
    r = v1.get("pooled", {})
    gf = r.get("f1_global_ew", 0)
    mf = r.get("f1_moe_ew",    0)
    gp = r.get("prec_global_ew", 0)
    gr = r.get("rec_global_ew",  0)
    mp = r.get("prec_moe_ew",    0)
    mr = r.get("rec_moe_ew",     0)
    d  = mf - gf
    sign = "+" if d >= 0 else ""
    print(f"{'Pooled':<14} {r.get('n_total',0):>5} "
          f"{'':>6} "
          f"{gp:>9.3f} {gr:>8.3f} {gf:>10.3f} "
          f"{mp:>9.3f} {mr:>8.3f} {mf:>10.3f} {sign}{d:>8.3f}")
    print("="*80)
    print("EW = Event-wise F1 with tolerance +/-5 windows. "
          "Main model S&P 500 in-sample (oracle threshold): 0.738.")
    print("Bitcoin: Global preferred (crypto microstructure breaks regime assumption).")


if __name__ == "__main__":
    make_all()
