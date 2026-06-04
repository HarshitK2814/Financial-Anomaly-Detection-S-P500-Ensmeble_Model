import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Set global styles
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.5

def draw_arrow(ax, start, end, color='#333333', style='-|>', lw=2, zorder=3):
    arrow = patches.FancyArrowPatch(
        start, end, connectionstyle="arc3,rad=0",
        color=color, arrowstyle=style, lw=lw, mutation_scale=15, zorder=zorder
    )
    ax.add_patch(arrow)

def add_box(ax, xy, width, height, label='', facecolor='#f8f9fa', edgecolor='#333333', lw=1.5, fontsize=11, text_color='black', alpha=1.0, align='center'):
    box = patches.FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.05,rounding_size=0.1",
        facecolor=facecolor, edgecolor=edgecolor, lw=lw, alpha=alpha, zorder=2
    )
    ax.add_patch(box)
    if label:
        ax.text(
            xy[0] + width / 2, xy[1] + height / 2, label,
            ha='center', va='center', fontsize=fontsize, color=text_color, wrap=True, zorder=4
        )
    return box

def draw_phase1(output_dir):
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(5.5, 5.7, "Phase 1: Extended Feature Engineering (V2 Factors)", 
            ha='center', va='center', fontsize=16, fontweight='bold', color='#1a365d')

    # 1. Raw Input
    add_box(ax, (0.2, 2.0), 2.2, 2.0, facecolor='white', edgecolor='#1a365d', lw=2)
    ax.text(1.3, 3.7, "Raw OHLCV Windows\n$W_t \\in \\mathbb{R}^{T \\times F}$", ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw candlestick-like representation
    for i in range(5):
        x = 0.5 + i*0.4
        ax.plot([x, x], [2.2, 3.2], color='black', lw=1.5, zorder=3)
        h = np.random.uniform(0.3, 0.8)
        y = np.random.uniform(2.3, 2.9)
        c = 'green' if np.random.rand() > 0.5 else 'red'
        ax.add_patch(patches.Rectangle((x-0.1, y), 0.2, h, facecolor=c, zorder=4))
        
    draw_arrow(ax, (2.4, 3.0), (3.0, 3.0))

    # 2. Factor Groupings Box
    add_box(ax, (3.0, 0.5), 4.5, 4.8, facecolor='#f0f4f8', edgecolor='#2b6cb0', lw=2)
    ax.text(5.25, 5.0, "32 Economic Factors Construction", ha='center', va='center', fontsize=12, fontweight='bold', color='#2b6cb0')
    
    # Sub-boxes for Taxonomies
    cats = [
        ("Volatility", "Composite, Garman-Klass,\nParkinson, VIX Proxy", (3.2, 3.6), '#ebf8ff', '#3182ce'),
        ("Tail Risk", "Kurtosis, Skew,\nMax Drawdown", (5.5, 3.6), '#fff5f5', '#c53030'),
        ("Trend & Momentum", "Return Drift,\nDirectional Persistence", (3.2, 2.1), '#f0fff4', '#2f855a'),
        ("Liquidity", "Volume Shock,\nAmihud Illiquidity", (5.5, 2.1), '#fffaf0', '#dd6b20'),
        ("Microstructure", "Latent States A/B/C,\nHurst Exponent", (3.2, 0.6), '#faf5ff', '#805ad5'),
    ]
    
    for title, desc, pos, fc, ec in cats:
        add_box(ax, pos, 2.0, 1.2, facecolor=fc, edgecolor=ec, lw=1.5)
        ax.text(pos[0]+1.0, pos[1]+0.9, title, ha='center', va='center', fontsize=11, fontweight='bold', color=ec)
        ax.text(pos[0]+1.0, pos[1]+0.4, desc, ha='center', va='center', fontsize=9)

    draw_arrow(ax, (7.5, 3.0), (8.2, 3.0))

    # 3. Final Factor Vector
    add_box(ax, (8.2, 1.5), 2.5, 3.0, facecolor='white', edgecolor='#2f855a', lw=2)
    ax.text(9.45, 4.2, "Factor State Space\n$F_t \\in \\mathbb{R}^{96}$", ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw vector representation
    ax.text(9.45, 3.5, "Base Factors (32)", ha='center', va='center', fontsize=10)
    ax.add_patch(patches.Rectangle((8.5, 3.2), 1.9, 0.2, facecolor='#48bb78', alpha=0.8, edgecolor='black'))
    
    ax.text(9.45, 2.8, "Lag t-1 (32)", ha='center', va='center', fontsize=10)
    ax.add_patch(patches.Rectangle((8.5, 2.5), 1.9, 0.2, facecolor='#68d391', alpha=0.6, edgecolor='black'))
    
    ax.text(9.45, 2.1, "Lag t-5 (32)", ha='center', va='center', fontsize=10)
    ax.add_patch(patches.Rectangle((8.5, 1.8), 1.9, 0.2, facecolor='#9ae6b4', alpha=0.4, edgecolor='black'))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actual_phase1_features.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

def draw_phase2(output_dir):
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    ax.text(6, 5.7, "Phase 2: Adaptive Regime Detection & Confidence Gating", 
            ha='center', va='center', fontsize=16, fontweight='bold', color='#1a365d')

    # Input
    add_box(ax, (0.2, 2.5), 1.2, 1.0, label="Realized\nVolatility\n$\\sigma_t$", facecolor='#edf2f7', edgecolor='#4a5568')
    
    draw_arrow(ax, (1.4, 3.0), (2.0, 4.5))
    draw_arrow(ax, (1.4, 3.0), (2.0, 3.0))
    draw_arrow(ax, (1.4, 3.0), (2.0, 1.5))

    # Detectors
    add_box(ax, (2.0, 4.0), 2.8, 1.0, facecolor='#fff5f5', edgecolor='#c53030')
    ax.text(3.4, 4.6, "Gaussian HMM", ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(3.4, 4.3, "3 Latent States", ha='center', va='center', fontsize=10)
    
    add_box(ax, (2.0, 2.5), 2.8, 1.0, facecolor='#ebf8ff', edgecolor='#2b6cb0')
    ax.text(3.4, 3.1, "Rolling Quantile", ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(3.4, 2.8, "Adaptive 252-day $Q_{33}/Q_{66}$", ha='center', va='center', fontsize=10)
    
    add_box(ax, (2.0, 1.0), 2.8, 1.0, facecolor='#fffff0', edgecolor='#d69e2e')
    ax.text(3.4, 1.6, "Static Quantile", ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(3.4, 1.3, "Global Baseline", ha='center', va='center', fontsize=10)

    draw_arrow(ax, (4.8, 4.5), (5.5, 3.5))
    draw_arrow(ax, (4.8, 3.0), (5.5, 3.0))
    draw_arrow(ax, (4.8, 1.5), (5.5, 2.5))

    # Fusion
    add_box(ax, (5.5, 2.0), 2.5, 2.0, facecolor='#e6fffa', edgecolor='#319795', lw=2)
    ax.text(6.75, 3.5, "Confidence-Weighted\nFusion", ha='center', va='center', fontsize=11, fontweight='bold', color='#285e61')
    ax.text(6.75, 2.6, "$\\hat{R}_t = \\arg\\max \\sum w_i \\times C_i$\n$C_{fused} = \\max(P_{soft})$", ha='center', va='center', fontsize=10)

    draw_arrow(ax, (8.0, 3.0), (8.8, 3.0))

    # Routing Decision
    poly = patches.Polygon([[9.8, 4.0], [11.3, 3.0], [9.8, 2.0], [8.3, 3.0]], 
                           closed=True, facecolor='#fdf2f8', edgecolor='#b83280', lw=2, zorder=2)
    ax.add_patch(poly)
    ax.text(9.8, 3.0, "Confidence\n$C_{fused} \\geq 0.6$?", ha='center', va='center', fontsize=11, fontweight='bold', zorder=4)

    # Branches
    draw_arrow(ax, (9.8, 4.0), (9.8, 4.8))
    ax.text(9.9, 4.4, "Yes", fontweight='bold', color='#38a169')
    add_box(ax, (8.8, 4.8), 2.0, 0.8, label="Hard Routing\n(One-Hot Weights)", facecolor='#f0fff4', edgecolor='#2f855a')

    draw_arrow(ax, (9.8, 2.0), (9.8, 1.2))
    ax.text(9.9, 1.6, "No", fontweight='bold', color='#e53e3e')
    add_box(ax, (8.5, 0.4), 2.6, 0.8, label="Soft Routing\n(Distributed to Neighbors)", facecolor='#fff5f5', edgecolor='#c53030')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actual_phase2_gating.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

def draw_phase3(output_dir):
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7)
    ax.axis('off')

    ax.text(7.5, 6.7, "Phase 3: Regime Meta-Ensemble & Anomaly Scoring", 
            ha='center', va='center', fontsize=16, fontweight='bold', color='#1a365d')

    # Inputs
    add_box(ax, (0.2, 4.5), 1.5, 1.0, label="Factor Vector\n$F_t$", facecolor='white', edgecolor='black')
    add_box(ax, (0.2, 1.5), 1.5, 1.0, label="Routing Weights\n$W_t$", facecolor='#edf2f7', edgecolor='black')

    # Regimes
    regimes = [
        ("Low-Vol Regime", 4.5, '#ebf8ff', '#3182ce'),
        ("Med-Vol Regime", 2.5, '#fffff0', '#d69e2e'),
        ("High-Vol Regime", 0.5, '#fff5f5', '#c53030')
    ]

    for title, y, fc, ec in regimes:
        # Large Regime Box
        add_box(ax, (2.5, y), 8.0, 1.8, facecolor=fc, edgecolor=ec, lw=2, alpha=0.3)
        ax.text(3.5, y + 1.5, title, ha='center', va='center', fontsize=12, fontweight='bold', color=ec)

        draw_arrow(ax, (1.7, 5.0), (2.5, y+0.9), color='gray') # F_t connects to all

        # Experts
        experts_x = 2.8
        add_box(ax, (experts_x, y+0.2), 1.2, 0.5, label="XGBoost", facecolor='white', edgecolor=ec, fontsize=9)
        add_box(ax, (experts_x, y+0.8), 1.2, 0.5, label="LightGBM", facecolor='white', edgecolor=ec, fontsize=9)
        add_box(ax, (experts_x+1.4, y+0.2), 1.4, 0.5, label="GRU Attention", facecolor='white', edgecolor=ec, fontsize=9)
        add_box(ax, (experts_x+1.4, y+0.8), 1.4, 0.5, label="Contrastive Net", facecolor='white', edgecolor=ec, fontsize=9)

        draw_arrow(ax, (experts_x+2.8, y+0.6), (6.0, y+0.9))

        # Prediction Stacking
        add_box(ax, (6.0, y+0.4), 1.6, 1.0, facecolor='white', edgecolor=ec)
        ax.text(6.8, y+0.9, "Stacked\nPredictions\nMatrix", ha='center', va='center', fontsize=9)

        draw_arrow(ax, (7.6, y+0.9), (8.2, y+0.9))

        # Meta-Learner
        add_box(ax, (8.2, y+0.4), 2.0, 1.0, facecolor='#f7fafc', edgecolor=ec, lw=1.5)
        ax.text(9.2, y+1.1, "Meta-Learner", ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(9.2, y+0.7, "Calibrated\nLogistic Reg.", ha='center', va='center', fontsize=9)

        draw_arrow(ax, (10.2, y+0.9), (11.0, 3.5), color=ec)

    # Routing Weights connection
    draw_arrow(ax, (1.7, 2.0), (11.5, 2.0), color='black', style='-')
    draw_arrow(ax, (11.5, 2.0), (11.5, 2.8), color='black')

    # Final Soft-Voting / Gating
    add_box(ax, (11.0, 2.8), 1.8, 1.4, facecolor='#e6fffa', edgecolor='#319795', lw=2)
    ax.text(11.9, 3.5, "Weighted\nFusion\n$\\hat{P} = \sum W_t^r P_t^r$", ha='center', va='center', fontsize=10, fontweight='bold')

    draw_arrow(ax, (12.8, 3.5), (13.5, 3.5))

    # Output
    add_box(ax, (13.5, 2.8), 1.3, 1.4, facecolor='white', edgecolor='black', lw=2)
    ax.text(14.15, 3.8, "Anomaly\nOutput", ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(14.15, 3.2, "Event-F1\nEval", ha='center', va='center', fontsize=10, color='blue')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actual_phase3_meta.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    output_dir = "c:/Users/Harshit Kumar/Downloads/Financial Anomaly Detection Journal Paper Coding Part/Financial-Anomaly-Detection-S-P500-Ensmeble_Model/new_figures_dense"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating Actual Phase 1...")
    draw_phase1(output_dir)
    print("Generating Actual Phase 2...")
    draw_phase2(output_dir)
    print("Generating Actual Phase 3...")
    draw_phase3(output_dir)
    print(f"Figures saved to {output_dir}")
