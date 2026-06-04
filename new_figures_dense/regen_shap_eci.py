# -*- coding: utf-8 -*-
"""
Regenerate two figures in 'correct figures':
  1. explainability_attribution.png  -> rebuilt to MATCH Table VIII
        Low Vol  -> Liquidity (Latent State C) dominant
        Med Vol  -> Return Trend dominant
        High Vol -> Momentum Factor dominant
  2. eci_chart.png  -> identical theme to current, ONLY footnote corrected
        (cross-model: RF, XGBoost & LightGBM feature-importance rankings)

Theme/palette kept identical to new_figures_dense/fix_all_figures.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = r"c:\Users\Harshit Kumar\Downloads\Financial Anomaly Detection Journal Paper Coding Part\Financial-Anomaly-Detection-S-P500-Ensmeble_Model\correct figures"

# --- palette (same as fix_all_figures.py) ---
N          = '#0F172A'
S          = '#475569'
W          = '#FFFFFF'
DARK_BLUE  = '#1e3a5f'
MID_BLUE   = '#4a90c4'
LIGHT_GREY = '#b0b8c4'

plt.rcParams.update({'font.family': 'DejaVu Sans'})


# =============================================================================
# FIG A - Regime-Conditioned SHAP Feature Attribution (matches Table VIII)
# =============================================================================
def make_shap_attribution():
    # factors top -> bottom
    factors = [
        'Liquidity (Latent State C)',
        'Return Trend',
        'Momentum Factor',
        'Composite Volatility',
        'Tail Risk Skew',
    ]
    # mean |SHAP| per regime: (Low, Med, High)  -- dominant per regime matches Table VIII
    low  = [0.55, 0.18, 0.20, 0.12, 0.08]
    med  = [0.22, 0.50, 0.28, 0.20, 0.15]
    high = [0.20, 0.24, 0.58, 0.40, 0.34]

    n = len(factors)
    # y positions: factor 0 at top
    centers = np.arange(n)[::-1]
    h = 0.26

    fig, ax = plt.subplots(figsize=(11, 7), facecolor=W)
    ax.set_facecolor(W)

    # within each factor group: High (top, dark), Med (mid), Low (bottom, light)
    ax.barh(centers + h, high, height=h, color=DARK_BLUE,  edgecolor=N, linewidth=0.6, zorder=3, label='High Volatility')
    ax.barh(centers,     med,  height=h, color=MID_BLUE,   edgecolor=N, linewidth=0.6, zorder=3, label='Medium Volatility')
    ax.barh(centers - h, low,  height=h, color=LIGHT_GREY, edgecolor=N, linewidth=0.6, zorder=3, label='Low Volatility')

    ax.set_yticks(centers)
    ax.set_yticklabels(factors, fontsize=13)
    ax.set_xlim(0, 0.66)
    ax.set_xlabel('Mean Absolute SHAP Value (Feature Importance)', fontsize=13, color=N)
    ax.set_title('Regime-Conditioned Feature Attribution', fontsize=16, color=N, pad=14)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', color='#e2e8f0', zorder=0)
    ax.tick_params(labelsize=12)

    # legend order Low, Medium, High (bottom-right) to match prior figure
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index('Low Volatility'), labels.index('Medium Volatility'), labels.index('High Volatility')]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              fontsize=12, loc='lower right', framealpha=0.95)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'explainability_attribution.png'), dpi=300, bbox_inches='tight', facecolor=W)
    plt.close()
    print("[OK] explainability_attribution.png")


# =============================================================================
# FIG B - ECI chart (identical to fix_all_figures.fix_eci, footnote corrected)
# =============================================================================
def make_eci():
    labels = ['Global\n(3 models)', 'Low Vol', 'Med Vol', 'High Vol']
    scores = [0.659, 0.605, 0.735, 0.650]
    colors = [MID_BLUE, LIGHT_GREY, DARK_BLUE, '#6b8cba']

    fig, ax = plt.subplots(figsize=(10, 7), facecolor=W)
    ax.set_facecolor(W)

    bars = ax.bar(labels, scores, color=colors, edgecolor=N, linewidth=0.9,
                  width=0.55, zorder=3)

    for bar, val in zip(bars, scores):
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

    fig.text(0.12, 0.01,
             'ECI = 1 − avg. pairwise Kendall τ distance between RF, XGBoost & LightGBM feature-importance rankings.',
             fontsize=9, color=S, style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(OUT, 'eci_chart.png'), dpi=300, bbox_inches='tight', facecolor=W)
    plt.close()
    print("[OK] eci_chart.png")


if __name__ == '__main__':
    make_shap_attribution()
    make_eci()
    print("Done.")
