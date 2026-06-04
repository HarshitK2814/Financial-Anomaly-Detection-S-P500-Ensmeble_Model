"""
Fix all figures in 'correct figures' based on paper data:
1. ablation_chart_paper.png  → Fix title to "Model Comparison (Ablation Study)", add in-sample footnote
2. oos_performance_chart.png → Add Z-Score row, fix E-F1 label, verify all numbers
3. eci_chart.png             → Add subtitle clarifying "Global" = cross-regime avg (3 models)
4. global_vs_moe_comparison.png → Fix XGBoost Sharpe label typo; add improvement annotations
5. per_regime_heatmap.png    → Add note that Low-Vol 0.0 is confirmed (only 4 anomalies)
6. phase2_architecture.png   → Already correct (252-day confirmed in paper)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import os

OUT = r"c:\Users\Harshit Kumar\Downloads\Financial Anomaly Detection Journal Paper Coding Part\Financial-Anomaly-Detection-S-P500-Ensmeble_Model\correct figures"

# ─── Color palette ───────────────────────────────────────────────────────────
N  = '#0F172A'   # near-black
A  = '#1E40AF'   # deep blue
A2 = '#2563EB'   # mid blue
S  = '#475569'   # slate
W  = '#FFFFFF'
AL = '#DBEAFE'   # light blue header
DARK_BLUE = '#1e3a5f'
MID_BLUE  = '#4a90c4'
LIGHT_GREY = '#b0b8c4'
GREEN = '#16A34A'
RED   = '#DC2626'

FONT = {'family': 'DejaVu Sans'}
plt.rcParams.update({'font.family': 'DejaVu Sans'})

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Ablation / Model Comparison Chart
# Data: Table IV of paper (Section: Ablation Study)
# ═══════════════════════════════════════════════════════════════════════════════
def fix_ablation():
    models = [
        'Regime-Cond. Factor MoE\n(Proposed)',
        'Global Random Forest',
        'ConvVAE',
        'Anomaly Transformer',
        'Z-Score',
        'Std. Deviation Rule',
    ]
    f1 = [95.5, 70.3, 64.0, 58.0, 18.4, 11.8]
    diagnosis = [
        'Stable regime-dependent signatures',
        'Regime confusion (no separation)',
        'Noise-dominated representations',
        'Data-hungry; unstable regimes',
        'Global threshold; ignores clustering',
        'Fails under heteroskedasticity',
    ]

    fig, ax = plt.subplots(figsize=(14, 8), facecolor=W)
    ax.set_facecolor(W)

    colors = [DARK_BLUE] + [LIGHT_GREY] * 5
    bars = ax.barh(range(len(models)), f1, color=colors, height=0.6,
                   edgecolor=N, linewidth=0.8, zorder=3)

    # Value labels
    for i, (v, diag) in enumerate(zip(f1, diagnosis)):
        bold = i == 0
        ax.text(v + 0.8, i, f'{v}', va='center', ha='left',
                fontsize=14, fontweight='bold' if bold else 'normal', color=N)
        ax.text(v + 5.5, i, f'— {diag}', va='center', ha='left',
                fontsize=10.5, color=S, style='italic')

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=13, fontweight='normal')
    ax.yaxis.set_tick_params(pad=6)
    # Bold the proposed model label
    ax.get_yticklabels()[0].set_fontweight('bold')
    ax.get_yticklabels()[0].set_color(DARK_BLUE)

    ax.set_xlim(0, 120)
    ax.set_xlabel('F1 Score (%)', fontsize=14, fontweight='bold', color=N)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)
    ax.grid(axis='x', color='#e2e8f0', zorder=0)

    # Title with light-blue header style
    ax.set_title('Model Comparison: Ablation Study\n(In-Sample Structural Separability)',
                 fontsize=17, fontweight='bold', color=N, pad=18)

    # Footnote
    fig.text(0.12, 0.01,
             '† In-sample structural separability diagnostic — upper bound on factor-space discriminability; NOT an out-of-sample performance claim.',
             fontsize=9, color=S, style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(OUT, 'ablation_chart_paper.png'), dpi=300, bbox_inches='tight', facecolor=W)
    plt.close()
    print("[OK] ablation_chart_paper.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Out-of-Sample Classification Performance
# Data: Table II (Multi-Model Classification, 5-Fold Temporal CV, Daily)
# Fix: add Z-Score 2.0σ bar, correct E-F1 label, verify all numbers
# ═══════════════════════════════════════════════════════════════════════════════
def fix_oos():
    # From Table II (paper lines 772-779)
    models    = ['GARCH 2.0σ', 'GARCH 2.5σ', 'Z-Score 2.0σ',
                 'XGBoost\nGlobal', 'XGBoost\nMoE',
                 'LightGBM\nGlobal', 'LightGBM\nMoE']
    ef1       = [50.62, 51.43, 47.73, 45.22, 43.81, 38.53, 40.86]
    precision = [39.42, 57.45, 35.59, 45.61, 48.94, 41.18, 54.29]
    recall    = [70.69, 46.55, 72.41, 44.83, 39.66, 36.21, 32.76]

    x = np.arange(len(models))
    w = 0.27
    fig, ax = plt.subplots(figsize=(16, 8), facecolor=W)
    ax.set_facecolor(W)

    b1 = ax.bar(x - w, precision, w, label='Precision', color=DARK_BLUE, edgecolor=N, lw=0.7, zorder=3)
    b2 = ax.bar(x,     recall,    w, label='Recall',    color=LIGHT_GREY, edgecolor=N, lw=0.7, zorder=3)

    # Event-F1 line
    ax.plot(x, ef1, 'o-', color=N, lw=2.2, ms=8, zorder=5, label='Event-F1 (±3d tolerance)')
    for i, (xi, v) in enumerate(zip(x, ef1)):
        bold = i == 6  # LightGBM MoE best precision
        ax.text(xi, v + 1.5, f'{v:.1f}', ha='center', fontsize=11,
                fontweight='bold' if bold else 'normal', color=N)

    # Value labels on bars
    for bars, vals in [(b1, precision), (b2, recall)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, color=S)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold', color=N)
    ax.set_xlabel('Model', fontsize=14, fontweight='bold', color=N)
    ax.set_ylim(0, 88)
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#e2e8f0', zorder=0)
    ax.tick_params(labelsize=11)

    ax.set_title('Out-of-Sample Classification Performance\n(5-Fold Temporal CV, 2,455 Daily Observations, 76 Anomalies)',
                 fontsize=16, fontweight='bold', color=N, pad=14)

    # Note: Removed Best Precision annotation per user request.
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'oos_performance_chart.png'), dpi=300, bbox_inches='tight', facecolor=W)
    plt.close()
    print("[OK] oos_performance_chart.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — ECI Chart
# Data: Table V (ECI section, paper lines 946-953)
# Fix: Add subtitle clarifying "Global" = cross-regime ECI (3 models)
# ═══════════════════════════════════════════════════════════════════════════════
def fix_eci():
    labels = ['Global\n(3 models)', 'Low Vol', 'Med Vol', 'High Vol']
    scores = [0.659, 0.605, 0.735, 0.650]
    colors = [MID_BLUE, LIGHT_GREY, DARK_BLUE, '#6b8cba']

    fig, ax = plt.subplots(figsize=(10, 7), facecolor=W)
    ax.set_facecolor(W)

    bars = ax.bar(labels, scores, color=colors, edgecolor=N, linewidth=0.9,
                  width=0.55, zorder=3)

    for bar, val, lbl in zip(bars, scores, labels):
        bold = val == max(scores)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005, f'{val:.3f}',
                ha='center', va='bottom', fontsize=16,
                fontweight='bold' if bold else 'normal',
                color=DARK_BLUE if bold else S)

    ax.set_ylim(0, 0.85)
    ax.set_ylabel('ECI Score\n(1 = perfect cross-model agreement)', fontsize=13,
                  fontweight='bold', color=N)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#e2e8f0', zorder=0, alpha=0.8)
    ax.tick_params(labelsize=13)

    ax.set_title('Explainability Consistency Index (ECI)\nAcross Volatility Regimes — RF, XGBoost, LightGBM',
                 fontsize=15, fontweight='bold', color=N, pad=16)

    # Removed regime interpretation arrows/annotations per user request

    fig.text(0.12, 0.01,
             'ECI = 1 − avg. pairwise Kendall τ distance between SHAP, saliency & tree attribution rankings.',
             fontsize=9, color=S, style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(OUT, 'eci_chart.png'), dpi=300, bbox_inches='tight', facecolor=W)
    plt.close()
    print("[OK] eci_chart.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Global vs MoE Comparison
# Data: Table II (ablation global vs MoE, paper lines 803-809)
# Fix: verified numbers, add improvement % annotations, fix axis labels
# ═══════════════════════════════════════════════════════════════════════════════
def fix_global_vs_moe():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor=W)
    fig.patch.set_facecolor(W)

    # Data from paper Table II (ablation Global vs MoE)
    base_models = ['LightGBM', 'XGBoost']
    prec_global = [41.18, 45.61]
    prec_moe    = [54.29, 48.94]
    sharpe_global = [0.717, 0.643]
    sharpe_moe    = [0.772, 0.748]

    improvements_prec   = ['+32%', '+7.3%']
    improvements_sharpe = ['+7.7%', '+16.3%']

    for ax, (global_vals, moe_vals, ylabel, title, impvs, fmt) in zip(
        axes,
        [
            (prec_global, prec_moe, 'Precision (%)', 'Precision Improvement', improvements_prec, '{:.2f}'),
            (sharpe_global, sharpe_moe, 'Sharpe Ratio', 'Sharpe Ratio Improvement', improvements_sharpe, '{:.3f}'),
        ]
    ):
        ax.set_facecolor(W)
        x = [0, 1]
        ax.plot(x, global_vals, 'o-', color=LIGHT_GREY, lw=2.5, ms=10, label='Global', zorder=3)
        ax.plot(x, moe_vals,    'o-', color=DARK_BLUE,  lw=2.5, ms=10, label='MoE',    zorder=4)

        for xi, g, m, imp in zip(x, global_vals, moe_vals, impvs):
            ax.text(xi + 0.03, g, fmt.format(g), fontsize=12, color=S)
            ax.text(xi + 0.03, m, fmt.format(m), fontsize=13, fontweight='bold', color=DARK_BLUE)
            # improvement badge
            mid_y = (g + m) / 2
            ax.annotate(imp, xy=(xi, mid_y), xytext=(xi + 0.18, mid_y),
                        fontsize=11, color=GREEN, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.3))

        ax.set_xticks([0, 1])
        ax.set_xticklabels(base_models, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold', color=N)
        ax.set_xlabel('Base Model', fontsize=12, color=S)
        ax.set_title(title, fontsize=15, fontweight='bold', color=N, pad=12)
        ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', color='#e2e8f0', zorder=0)
        ax.tick_params(labelsize=12)

    fig.suptitle('Regime Conditioning Impact: Global vs. MoE\n(Out-of-Sample, 5-Fold Temporal CV)',
                 fontsize=16, fontweight='bold', color=N, y=1.01)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'global_vs_moe_comparison.png'), dpi=300,
                bbox_inches='tight', facecolor=W)
    plt.close()
    print("[OK] global_vs_moe_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Per-Regime Heatmap
# Data: Table III (per-regime analysis, paper lines 832-834)
# Fix: add "N=" sample size annotations, clarify 0.0 Low-Vol with footnote
# ═══════════════════════════════════════════════════════════════════════════════
def fix_per_regime_heatmap():
    regimes = ['Low Volatility\n(N=634, 4 anomalies)',
               'Medium Volatility\n(N=911, 16 anomalies)',
               'High Volatility\n(N=910, 56 anomalies)']
    metrics = ['LGBM\nPrecision', 'LGBM\nRecall', 'GARCH\nPrecision', 'GARCH\nRecall']

    # From Table III (paper lines 832-834)
    data = np.array([
        [0.0,  0.0,  33.3, 50.0],   # Low Vol
        [75.0, 21.4, 25.7, 64.3],   # Med Vol
        [51.6, 39.0, 48.4, 75.6],   # High Vol
    ])

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=W)
    ax.set_facecolor(W)
    fig.patch.set_facecolor(W)

    im = ax.imshow(data, aspect='auto', cmap='Blues', vmin=0, vmax=80)

    # Text annotations
    for i in range(3):
        for j in range(4):
            val = data[i, j]
            text_color = 'white' if val > 45 else 'black'
            bold = val == data[i].max() or val == data[:, j].max()
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=16, color=text_color,
                    fontweight='bold' if bold else 'normal')

    ax.set_xticks(range(4))
    ax.set_xticklabels(metrics, fontsize=13, fontweight='bold')
    ax.set_yticks(range(3))
    ax.set_yticklabels(regimes, fontsize=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Score (%)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)

    ax.set_title('Per-Regime Classification Performance\n(Out-of-Sample, Daily S&P 500 2016-2025)',
                 fontsize=15, fontweight='bold', color=N, pad=20)

    # Separator between LGBM and GARCH columns
    ax.axvline(1.5, color='white', lw=3)
    # Group labels via figure text below the axes
    fig.text(0.285, 0.06, 'LightGBM MoE', ha='center', fontsize=13,
             fontweight='bold', color=DARK_BLUE)
    fig.text(0.665, 0.06, 'GARCH 2.0\u03c3', ha='center', fontsize=13,
             fontweight='bold', color=S)

    fig.text(0.12, 0.01,
             '† Low Vol LGBM shows 0.0 precision/recall: only 4 anomalies exist in this regime; '
             'the model correctly avoids false alarms but cannot achieve detection at this threshold.',
             fontsize=9, color=S, style='italic', wrap=True)

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(os.path.join(OUT, 'per_regime_heatmap.png'), dpi=300,
                bbox_inches='tight', facecolor=W)
    plt.close()
    print("[OK] per_regime_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Fixing figures...\n")
    fix_ablation()
    fix_oos()
    fix_eci()
    fix_global_vs_moe()
    fix_per_regime_heatmap()
    print("\nAll done. Check 'correct figures' folder.")
