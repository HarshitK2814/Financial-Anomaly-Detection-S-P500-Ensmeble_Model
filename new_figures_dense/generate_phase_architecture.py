import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np
import os

# Set global styles
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5

def draw_arrow(ax, start, end, color='#333333', style='-|>', lw=2):
    arrow = patches.FancyArrowPatch(
        start, end, connectionstyle="arc3,rad=0",
        color=color, arrowstyle=style, lw=lw, mutation_scale=20
    )
    ax.add_patch(arrow)

def add_box(ax, xy, width, height, label='', facecolor='#f8f9fa', edgecolor='#333333', lw=1.5, fontsize=12, text_color='black', alpha=1.0):
    box = patches.FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.05,rounding_size=0.05",
        facecolor=facecolor, edgecolor=edgecolor, lw=lw, alpha=alpha
    )
    ax.add_patch(box)
    if label:
        ax.text(
            xy[0] + width / 2, xy[1] + height / 2, label,
            ha='center', va='center', fontsize=fontsize, color=text_color, wrap=True
        )
    return box

def create_heatmap(ax, bounds, data, cmap='RdYlGn'):
    # bounds: [x0, y0, width, height]
    sub_ax = ax.inset_axes(bounds)
    sub_ax.imshow(data, cmap=cmap, aspect='auto')
    sub_ax.set_xticks([])
    sub_ax.set_yticks([])
    for spine in sub_ax.spines.values():
        spine.set_linewidth(0.5)
    return sub_ax

def create_lineplot(ax, bounds, x, y, color='black', fill_color=None):
    sub_ax = ax.inset_axes(bounds)
    sub_ax.plot(x, y, color=color, lw=1.5)
    if fill_color:
        sub_ax.fill_between(x, y, color=fill_color, alpha=0.3)
    sub_ax.set_xticks([])
    sub_ax.set_yticks([])
    for spine in sub_ax.spines.values():
        spine.set_linewidth(1.0)
    return sub_ax

def draw_phase1(output_dir):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Title
    ax.text(5, 4.8, "Phase 1: Feature Engineering & State Space Construction", 
            ha='center', va='center', fontsize=16, fontweight='bold', color='#1a365d')

    # 1. S&P 500 Returns
    add_box(ax, (0.2, 1.5), 2.0, 2.5, facecolor='white', edgecolor='#1a365d', lw=2)
    ax.text(1.2, 4.2, "S&P 500 Returns\n(Raw Data)", ha='center', va='center', fontsize=12, fontweight='bold')
    
    np.random.seed(42)
    t = np.linspace(0, 10, 100)
    returns = np.cumsum(np.random.randn(100))
    create_lineplot(ax, [0.03, 0.35, 0.18, 0.4], t, returns, color='#2c5282')

    draw_arrow(ax, (2.3, 2.75), (2.8, 2.75))

    # 2. Rolling Window
    add_box(ax, (2.9, 1.0), 1.2, 3.2, facecolor='#ebf8ff', edgecolor='#2b6cb0', lw=2)
    ax.text(3.5, 4.4, "$W_t \\in \\mathbb{R}^{128}$\n128 log-ret", ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw stacked bars inside rolling window
    for i in range(15):
        y = 1.1 + i * 0.2
        ax.add_patch(patches.Rectangle((3.0, y), 1.0, 0.15, facecolor='#3182ce', alpha=0.7))

    draw_arrow(ax, (4.2, 2.75), (4.7, 2.75))

    # 3. Factor Mapping
    add_box(ax, (4.8, 1.5), 2.2, 2.5, facecolor='#fffaf0', edgecolor='#dd6b20', lw=2)
    ax.text(5.9, 4.2, "Factor Mapping $\\Phi(W_t)$", ha='center', va='center', fontsize=12, fontweight='bold', color='#dd6b20')
    
    factors_text = (
        "1. Return_Trend\n"
        "2. Momentum_Factor\n"
        "3. Composite_Volatility\n"
        "4. Latent_State_C\n"
        "5. Tail_Kurtosis\n"
        "(+ Lags: t-1, t-5)"
    )
    ax.text(5.9, 2.75, factors_text, ha='center', va='center', fontsize=11)

    draw_arrow(ax, (7.1, 2.75), (7.6, 2.75))

    # 4. Factor Vector Heatmap
    add_box(ax, (7.7, 2.0), 2.1, 1.5, facecolor='white', edgecolor='#2f855a', lw=2)
    ax.text(8.75, 3.8, "Factor Vector $F_t \\in \\mathbb{R}^{24}$", ha='center', va='center', fontsize=12, fontweight='bold')
    
    heatmap_data = np.random.randn(5, 24)
    create_heatmap(ax, [0.78, 0.45, 0.19, 0.25], heatmap_data)
    
    # Text labels below heatmap
    ax.text(8.0, 1.8, "Ret", ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(8.5, 1.8, "Mom", ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(9.0, 1.8, "Vol", ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(9.5, 1.8, "Lat", ha='center', va='center', fontsize=10, fontweight='bold')

    # Global standard deviation box below
    add_box(ax, (7.7, 0.5), 2.1, 0.8, label="$\\sigma_t = \\text{std}(\\{r_{t-127}, \\dots, r_t\\})$", facecolor='#e2e8f0', edgecolor='#4a5568')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase1_features.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def draw_phase2(output_dir):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Title
    ax.text(5, 4.8, "Phase 2: Volatility Estimation & Regime Gating", 
            ha='center', va='center', fontsize=16, fontweight='bold', color='#1a365d')

    # 1. Distribution Plot
    add_box(ax, (0.5, 2.0), 3.0, 2.5, facecolor='white', edgecolor='#1a365d', lw=2)
    ax.text(2.0, 4.7, "$\\sigma_t$ Distribution Analysis", ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw synthetic distribution curve using numpy
    x = np.linspace(0, 5, 200)
    y = np.exp(-0.5 * ((x - 1.5) / 0.8)**2) * x  # skewed right
    
    sub_ax = ax.inset_axes([0.06, 0.45, 0.28, 0.4])
    sub_ax.plot(x, y, color='#2c5282', lw=2)
    sub_ax.fill_between(x, y, color='#ebf8ff', alpha=0.5)
    
    q33 = 1.0
    q66 = 2.2
    
    sub_ax.axvline(q33, color='#3182ce', linestyle='--', lw=1.5)
    sub_ax.axvline(q66, color='#dd6b20', linestyle='--', lw=1.5)
    sub_ax.text(q33, max(y)*0.9, "$Q_{33}$", color='#3182ce', ha='right', fontsize=10)
    sub_ax.text(q66, max(y)*0.9, "$Q_{66}$", color='#dd6b20', ha='left', fontsize=10)
    
    sub_ax.set_xticks([])
    sub_ax.set_yticks([])
    for spine in sub_ax.spines.values(): spine.set_linewidth(1.0)

    draw_arrow(ax, (3.6, 3.25), (4.5, 3.25))
    ax.text(4.05, 3.4, "$\\sigma_t$", ha='center', va='center', fontsize=12, fontweight='bold')

    # 2. Regime Gating Check
    add_box(ax, (4.5, 2.0), 2.0, 2.5, facecolor='#fff5f5', edgecolor='#c53030', lw=2)
    ax.text(5.5, 4.7, "Regime Gating Logic", ha='center', va='center', fontsize=12, fontweight='bold', color='#c53030')
    
    add_box(ax, (5.0, 2.7), 1.0, 1.0, label="Check\n$\\sigma_t$\n\nvs $Q_{33}, Q_{66}$", facecolor='#fed7d7', edgecolor='#9b2c2c')

    # 3. Branching to regimes
    # Low Vol
    draw_arrow(ax, (4.5, 2.5), (2.5, 1.2), color='#3182ce')
    ax.text(3.5, 2.0, "$\\sigma_t \\leq Q_{33}$", color='#3182ce', fontsize=11, fontweight='bold', rotation=30)
    add_box(ax, (1.0, 0.2), 2.5, 1.0, label="Low Vol ($R_t = 0$)", facecolor='#ebf8ff', edgecolor='#2b6cb0', lw=2, fontsize=12, text_color='#2b6cb0')

    # Med Vol
    draw_arrow(ax, (5.5, 2.0), (5.5, 1.2), color='#dd6b20')
    ax.text(6.1, 1.6, "$Q_{33} < \\sigma_t \\leq Q_{66}$", color='#dd6b20', fontsize=11, fontweight='bold')
    add_box(ax, (4.25, 0.2), 2.5, 1.0, label="Med Vol ($R_t = 1$)", facecolor='#fffaf0', edgecolor='#c05621', lw=2, fontsize=12, text_color='#c05621')

    # High Vol
    draw_arrow(ax, (6.5, 2.5), (8.5, 1.2), color='#c53030')
    ax.text(7.5, 2.0, "$\\sigma_t > Q_{66}$", color='#c53030', fontsize=11, fontweight='bold', rotation=-30)
    add_box(ax, (7.5, 0.2), 2.5, 1.0, label="High Vol ($R_t = 2$)", facecolor='#fff5f5', edgecolor='#9b2c2c', lw=2, fontsize=12, text_color='#9b2c2c')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase2_gating.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def draw_phase3(output_dir):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(5, 6.8, "Phase 3: Mixture of Experts (MoE) Routing & Final Decision", 
            ha='center', va='center', fontsize=16, fontweight='bold', color='#1a365d')

    # 1. Expert Models
    # E0: Low Vol
    add_box(ax, (0.5, 3.8), 2.6, 2.5, facecolor='white', edgecolor='#3182ce', lw=2)
    ax.text(1.8, 6.0, "$E_0$: Low-Vol Expert", ha='center', va='center', fontsize=12, fontweight='bold', color='#3182ce')
    ax.text(1.8, 5.7, "Dominant Driver: Latent_State_C", ha='center', va='center', fontsize=9)
    create_heatmap(ax, [0.06, 0.65, 0.24, 0.1], np.random.randn(1, 24), cmap='RdYlGn')
    # Draw simple tree network
    ax.plot([1.8, 1.3], [4.9, 4.4], color='gray', zorder=1)
    ax.plot([1.8, 2.3], [4.9, 4.4], color='gray', zorder=1)
    ax.plot([1.3, 1.0], [4.4, 4.0], color='gray', zorder=1)
    ax.plot([1.3, 1.6], [4.4, 4.0], color='gray', zorder=1)
    ax.scatter([1.8, 1.3, 2.3, 1.0, 1.6], [4.9, 4.4, 4.4, 4.0, 4.0], s=80, color='#ebf8ff', edgecolor='#3182ce', zorder=2)
    ax.text(1.8, 4.0, "$\\hat{P}(A_t=1) = E_0(F_t)$", ha='center', va='center', fontsize=10)

    # E1: Med Vol
    add_box(ax, (3.7, 3.8), 2.6, 2.5, facecolor='white', edgecolor='#dd6b20', lw=2)
    ax.text(5.0, 6.0, "$E_1$: Med-Vol Expert", ha='center', va='center', fontsize=12, fontweight='bold', color='#dd6b20')
    ax.text(5.0, 5.7, "Dominant Driver: Return_Trend", ha='center', va='center', fontsize=9)
    create_heatmap(ax, [0.38, 0.65, 0.24, 0.1], np.random.randn(1, 24), cmap='RdYlGn')
    ax.plot([5.0, 4.5], [4.9, 4.4], color='gray', zorder=1)
    ax.plot([5.0, 5.5], [4.9, 4.4], color='gray', zorder=1)
    ax.plot([5.5, 5.2], [4.4, 4.0], color='gray', zorder=1)
    ax.plot([5.5, 5.8], [4.4, 4.0], color='gray', zorder=1)
    ax.scatter([5.0, 4.5, 5.5, 5.2, 5.8], [4.9, 4.4, 4.4, 4.0, 4.0], s=80, color='#fffaf0', edgecolor='#dd6b20', zorder=2)
    ax.text(5.0, 4.0, "$\\hat{P}(A_t=1) = E_1(F_t)$", ha='center', va='center', fontsize=10)

    # E2: High Vol
    add_box(ax, (6.9, 3.8), 2.6, 2.5, facecolor='white', edgecolor='#c53030', lw=2)
    ax.text(8.2, 6.0, "$E_2$: High-Vol Expert", ha='center', va='center', fontsize=12, fontweight='bold', color='#c53030')
    ax.text(8.2, 5.7, "Dominant Driver: Momentum_Factor", ha='center', va='center', fontsize=9)
    create_heatmap(ax, [0.70, 0.65, 0.24, 0.1], np.random.randn(1, 24), cmap='RdYlGn')
    ax.plot([8.2, 7.7], [4.9, 4.4], color='gray', zorder=1)
    ax.plot([8.2, 8.7], [4.9, 4.4], color='gray', zorder=1)
    ax.plot([7.7, 7.4], [4.4, 4.0], color='gray', zorder=1)
    ax.plot([7.7, 8.0], [4.4, 4.0], color='gray', zorder=1)
    ax.scatter([8.2, 7.7, 8.7, 7.4, 8.0], [4.9, 4.4, 4.4, 4.0, 4.0], s=80, color='#fff5f5', edgecolor='#c53030', zorder=2)
    ax.text(8.2, 4.0, "$\\hat{P}(A_t=1) = E_2(F_t)$", ha='center', va='center', fontsize=10)

    # Output lines from Experts
    draw_arrow(ax, (1.8, 3.8), (1.8, 3.1), color='#3182ce')
    ax.text(1.8, 3.45, "Active $\\hat{P}$", ha='right', va='center', color='#3182ce', fontsize=10, fontweight='bold', bbox=dict(facecolor='white', edgecolor='none'))

    draw_arrow(ax, (5.0, 3.8), (5.0, 3.1), color='gray', style='-')
    ax.text(5.0, 3.45, "(Not Activated)", ha='center', va='center', color='gray', fontsize=9, style='italic', bbox=dict(facecolor='white', edgecolor='none'))

    draw_arrow(ax, (8.2, 3.8), (8.2, 3.1), color='gray', style='-')
    ax.text(8.2, 3.45, "(Not Activated)", ha='center', va='center', color='gray', fontsize=9, style='italic', bbox=dict(facecolor='white', edgecolor='none'))

    # 2. Hard Routing Decision Sequence
    add_box(ax, (0.5, 1.5), 4.5, 1.6, facecolor='#f8f9fa', edgecolor='#4a5568', lw=2)
    ax.text(2.75, 2.7, "Hard Routing Decision Sequence", ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(2.75, 2.0, "$W_t \\rightarrow R_t = f(\\sigma_t) \\rightarrow \\text{Evaluate } E_{R_t}(F_t)$", ha='center', va='center', fontsize=11)

    draw_arrow(ax, (5.0, 2.3), (5.5, 2.3))

    # Training label detail
    add_box(ax, (0.5, 0.2), 4.5, 0.8, facecolor='#fff5f5', edgecolor='#c53030', lw=1.5)
    ax.text(2.75, 0.7, "Training Label Definition", ha='center', va='center', fontsize=10, fontweight='bold', color='#c53030')
    ax.text(2.75, 0.4, "$A_t = \\mathbf{1}(|r_t| > \\tau_{\\alpha}), \\alpha=0.02$", ha='center', va='center', fontsize=10)
    ax.text(4.9, 1.2, "(Used during training only)", ha='right', va='center', fontsize=9, style='italic', color='#c53030')
    draw_arrow(ax, (2.75, 1.5), (2.75, 1.0), color='#c53030')

    # 3. Final Decision Rule
    add_box(ax, (5.5, 1.5), 2.0, 1.6, facecolor='#f0fff4', edgecolor='#2f855a', lw=2)
    ax.text(6.5, 2.8, "Final Decision Rule", ha='center', va='center', fontsize=11, fontweight='bold', color='#2f855a')
    ax.text(6.5, 2.3, "$\\hat{P} = E_{R_t}(F_t)$", ha='center', va='center', fontsize=10)
    ax.text(6.5, 1.9, "if $\\hat{P} \\geq \\tau$", ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(6.5, 1.7, "$\\tau = 0.50$ (default)", ha='center', va='center', fontsize=9)
    
    draw_arrow(ax, (7.5, 2.3), (8.2, 2.3), color='#2f855a')
    ax.text(7.85, 2.45, "Output", ha='center', va='center', color='#2f855a', fontsize=10, fontweight='bold')

    # 4. Final Output Visuals
    ax.text(9.0, 3.1, "Final Output", ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Normal Plot
    add_box(ax, (8.2, 1.8), 1.6, 1.0, facecolor='white', edgecolor='black', lw=1.0)
    ax.text(9.0, 2.9, "NORMAL ($\\hat{P}=0.012 < \\tau$)", ha='center', va='center', fontsize=9)
    np.random.seed(1)
    t = np.linspace(0, 10, 100)
    returns = np.cumsum(np.random.randn(100))
    sub_ax = create_lineplot(ax, [0.83, 0.27, 0.14, 0.12], t, returns, color='#2c5282')
    sub_ax.axvspan(7, 8, color='#48bb78', alpha=0.4) # Green highlight

    ax.text(7.8, 1.4, "OR", ha='center', va='center', fontsize=10, fontweight='bold', bbox=dict(boxstyle="round", facecolor='white'))

    # Anomaly Plot
    add_box(ax, (8.2, 0.2), 1.6, 1.0, facecolor='white', edgecolor='black', lw=1.0)
    ax.text(9.0, 1.3, "ANOMALY ($\\hat{P}=0.98 > \\tau$)", ha='center', va='center', fontsize=9, color='#c53030')
    np.random.seed(2)
    returns2 = np.cumsum(np.random.randn(100)) - t
    sub_ax2 = create_lineplot(ax, [0.83, 0.04, 0.14, 0.12], t, returns2, color='#c53030')
    sub_ax2.axvspan(8, 9, color='#ecc94b', alpha=0.5) # Yellow/Orange highlight

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase3_moe.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    output_dir = "c:/Users/Harshit Kumar/Downloads/Financial Anomaly Detection Journal Paper Coding Part/Financial-Anomaly-Detection-S-P500-Ensmeble_Model/new_figures_dense"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating Phase 1...")
    draw_phase1(output_dir)
    print("Generating Phase 2...")
    draw_phase2(output_dir)
    print("Generating Phase 3...")
    draw_phase3(output_dir)
    print(f"Figures saved to {output_dir}")
