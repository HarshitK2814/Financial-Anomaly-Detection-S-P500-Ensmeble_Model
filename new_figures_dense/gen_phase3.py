import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.gridspec import GridSpec
import numpy as np, os

N='#0F172A'; A='#1E40AF'; A2='#2563EB'; S='#334155'; W='#FFFFFF'; BG='#F8FAFC'
AL='#DBEAFE'; SL='#E2E8F0'; GL='#F1F5F9'
R_LOW='#2563EB'; R_MED='#D97706'; R_HI='#DC2626'
GRN='#16A34A'

out=r"c:\Users\Harshit Kumar\Downloads\Financial Anomaly Detection Journal Paper Coding Part\Financial-Anomaly-Detection-S-P500-Ensmeble_Model\correct figures"

fig=plt.figure(figsize=(20,13),facecolor=BG)
gs=GridSpec(3,3,figure=fig,hspace=0.28,wspace=0.25,
            height_ratios=[0.05,0.52,0.43],
            left=0.03,right=0.97,top=0.97,bottom=0.04)

# ── TITLE ──
ax_t=fig.add_subplot(gs[0,:])
ax_t.set_xlim(0,1); ax_t.set_ylim(0,1); ax_t.axis('off')
ax_t.add_patch(FancyBboxPatch((0,0),1,1,boxstyle="round,pad=0,rounding_size=0.05",fc=AL,ec=N,lw=1.5,transform=ax_t.transAxes,zorder=1))
ax_t.text(0.5,0.5,"PHASE 3    Regime-Conditioned Meta-Ensemble",transform=ax_t.transAxes,ha='center',va='center',fontsize=24,fontweight='bold',color='#000000',zorder=5)

# ══════ TOP: ARCHITECTURE FLOW ══════
ax_a=fig.add_subplot(gs[1,:])
ax_a.set_xlim(-0.5,21); ax_a.set_ylim(-1.5,10.5); ax_a.axis('off'); ax_a.set_facecolor(BG)

def bx(ax,x,y,w,h,fc=W,ec=N,lw=1.8,rad=0.08,z=2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0.03,rounding_size={rad}",fc=fc,ec=ec,lw=lw,zorder=z))
def ar(ax,x1,y1,x2,y2,c=S,lw=2.5):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',color=c,lw=lw,mutation_scale=18,zorder=3))
def car(ax,x1,y1,x2,y2,c=S,lw=2.5,rad=0.3):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',color=c,lw=lw,mutation_scale=18,connectionstyle=f"arc3,rad={rad}",zorder=3))
def tx(ax,x,y,t,fs=14,c=N,fw='bold',ha='center',va='center'):
    ax.text(x,y,t,fontsize=fs,color=c,fontweight=fw,ha=ha,va=va,zorder=5,linespacing=1.3)

# Col 1: Factor Vector input
bx(ax_a,0,6.0,2.5,3.5,fc=W,ec=N,lw=2.5)
tx(ax_a,1.25,9.1,r"$F_t \in \mathbb{R}^{48}$",fs=17,c=A)
tx(ax_a,1.25,8.3,"Factor\nVector",fs=15,c=N)
np.random.seed(7)
for r in range(4):
    for c_ in range(6):
        v=np.random.uniform(0.2,1.0)
        ax_a.add_patch(Rectangle((0.25+c_*0.33,6.2+r*0.4),0.28,0.33,fc=plt.cm.Blues(v),ec='none',alpha=0.9,zorder=4))

# Routing Weights input
bx(ax_a,0,2.5,2.5,2.8,fc=GL,ec=N,lw=2.5)
tx(ax_a,1.25,4.8,"Routing\nWeights",fs=15,c=N)
tx(ax_a,1.25,3.5,r"$w_t^{(k)}$",fs=17,c=A)
tx(ax_a,1.25,2.8,"from Phase 2",fs=12,c=S,fw='normal')

# Col 2: Three regime expert pools
regime_data = [
    ("LOW-VOL  k=0", R_LOW, AL, 4.5, 7.2),
    ("MED-VOL  k=1", R_MED, '#FFFBEB', 4.5, 4.0),
    ("HIGH-VOL  k=2", R_HI, '#FEE2E2', 4.5, 0.8),
]

for title, col, bg, x, y in regime_data:
    bx(ax_a, x, y, 5.8, 2.7, fc=bg, ec=col, lw=2.5)
    tx(ax_a, x+2.9, y+2.35, title, fs=15, c=col)
    
    experts = ["RF","XGB","LGBM","RA-TAN","GRU"]
    for i, exp in enumerate(experts):
        ex = x + 0.2 + i*1.1
        bx(ax_a, ex, y+0.3, 0.95, 1.5, fc=W, ec=col, lw=1.3, rad=0.04)
        tx(ax_a, ex+0.475, y+1.05, exp, fs=12, c=col)

    ar(ax_a, x+5.8, y+1.35, 11.5, y+1.35, c=col, lw=2)

# Factor input arrows to regime boxes
ar(ax_a, 2.5, 8.0, 4.5, 8.55, c=N, lw=2)
ar(ax_a, 2.5, 7.5, 4.5, 5.35, c=N, lw=2)
ar(ax_a, 2.5, 6.8, 4.5, 2.15, c=N, lw=2)

# Col 3: Meta-Learner stacking
bx(ax_a, 11.5, 0.8, 3.2, 9.1, fc=W, ec=N, lw=2.5, rad=0.1)
tx(ax_a, 13.1, 9.35, "Meta-Ensemble\nStacking Layer", fs=16, c=N)

for title, col, y in [("Meta-LR (k=0)", R_LOW, 6.8),
                       ("Meta-LR (k=1)", R_MED, 4.3),
                       ("Meta-LR (k=2)", R_HI, 1.5)]:
    bx(ax_a, 11.8, y, 2.6, 1.8, fc=W, ec=col, lw=1.8)
    tx(ax_a, 13.1, y+1.3, title, fs=14, c=col)
    tx(ax_a, 13.1, y+0.55, "Calibrated\nLogistic Reg.", fs=12, c=S, fw='normal')

# Meta to fusion arrows
for y in [7.7, 5.2, 2.4]:
    ar(ax_a, 14.4, y, 16.0, 5.75, c=N, lw=2)

# Col 4: Final Fusion
bx(ax_a, 16.0, 3.5, 3.8, 4.5, fc=W, ec=N, lw=2.5, rad=0.12)
tx(ax_a, 17.9, 6.5, "Weighted\nFusion", fs=20, c=N)
tx(ax_a, 17.9, 4.5, "Anomaly\nDecision", fs=18, c=N)

# ── ROUTING ARROW: goes BELOW the regime boxes, then UP to fusion ──
# Path: from routing box bottom -> go down -> go right under everything -> go up to fusion
ax_a.plot([1.25, 1.25, 17.9, 17.9], [2.5, -0.8, -0.8, 3.5],
          color=A2, lw=2.5, ls='-', zorder=3, solid_capstyle='round')
# Arrowhead at end
ar(ax_a, 17.9, -0.3, 17.9, 3.5, c=A2, lw=2.5)
# Label on the routing path
bx(ax_a, 7.5, -1.3, 5.5, 0.7, fc=AL, ec=A2, lw=1.5, z=4)
tx(ax_a, 10.25, -0.95, r"Routing Weights  $w_t^{(k)}$  passed to Fusion", fs=13, c=A2)

# ══════ BOTTOM ROW: Methodology-appropriate visuals ══════

# LEFT: Focal Loss Curve (training methodology)
ax_fl=fig.add_subplot(gs[2,0])
ax_fl.set_facecolor(W)
p = np.linspace(0.01, 0.99, 200)
for gamma, ls, lab in [(0, '--', r'$\gamma=0$ (CE)'), (1, '-.', r'$\gamma=1$'), (2, '-', r'$\gamma=2$ (Ours)')]:
    focal = -0.75 * (1-p)**gamma * np.log(p)
    lw = 3 if gamma==2 else 1.5
    ax_fl.plot(p, focal, ls=ls, lw=lw, color=[S, A2, N][gamma], label=lab)
ax_fl.set_xlabel(r'Predicted Probability  $p_t$', fontsize=14, fontweight='bold')
ax_fl.set_ylabel(r'$\mathcal{L}_{focal}$', fontsize=16, fontweight='bold')
ax_fl.set_title('Focal Loss for Class Imbalance', fontsize=16, fontweight='bold', color=N, pad=8)
ax_fl.legend(fontsize=13, framealpha=0.9)
ax_fl.set_ylim(0, 4)
ax_fl.spines['top'].set_visible(False)
ax_fl.spines['right'].set_visible(False)

# CENTER: Regime Distribution (showing what the gating produces)
ax_rd=fig.add_subplot(gs[2,1])
ax_rd.set_facecolor(W)
np.random.seed(2024)
low_vol = np.random.normal(0.010, 0.003, 800)
med_vol = np.random.normal(0.018, 0.004, 500)
hi_vol  = np.random.normal(0.030, 0.006, 200)
ax_rd.hist(low_vol, bins=40, alpha=0.75, color=R_LOW, label='Low-Vol', edgecolor=W, lw=0.5)
ax_rd.hist(med_vol, bins=30, alpha=0.75, color=R_MED, label='Med-Vol', edgecolor=W, lw=0.5)
ax_rd.hist(hi_vol,  bins=20, alpha=0.75, color=R_HI,  label='High-Vol', edgecolor=W, lw=0.5)
ax_rd.axvline(0.0135, color=N, lw=2, ls='--')
ax_rd.axvline(0.023, color=N, lw=2, ls='--')
ax_rd.text(0.0135, ax_rd.get_ylim()[1]*0.5, r' $Q_{33}$', fontsize=14, fontweight='bold', color=N, va='center')
ax_rd.text(0.023, ax_rd.get_ylim()[1]*0.5, r' $Q_{66}$', fontsize=14, fontweight='bold', color=N, va='center')
ax_rd.set_xlabel('Realized Volatility', fontsize=14, fontweight='bold')
ax_rd.set_ylabel('Frequency', fontsize=14, fontweight='bold')
ax_rd.set_title('Regime Distribution by Volatility', fontsize=16, fontweight='bold', color=N, pad=8)
ax_rd.legend(fontsize=13, framealpha=0.9)
ax_rd.spines['top'].set_visible(False)
ax_rd.spines['right'].set_visible(False)

# RIGHT: Isotonic Calibration Concept (methodology, not results)
ax_cal=fig.add_subplot(gs[2,2])
ax_cal.set_facecolor(W)
x = np.linspace(0, 1, 200)
# Perfect calibration (diagonal)
ax_cal.plot(x, x, color=SL, lw=2, ls='--', label='Perfect calibration')
# Uncalibrated sigmoid (typical overconfident S-curve)
uncal = 1 / (1 + np.exp(-8*(x - 0.5)))
ax_cal.plot(x, uncal, color=R_HI, lw=2.5, ls='-.', label='Raw sigmoid output')
# Isotonic calibrated (stepwise monotonic, closer to diagonal)
np.random.seed(7)
iso_x = np.sort(np.random.uniform(0, 1, 15))
iso_y = np.sort(iso_x + np.random.uniform(-0.05, 0.08, 15))
iso_y = np.clip(iso_y, 0, 1)
iso_x = np.concatenate([[0], iso_x, [1]])
iso_y = np.concatenate([[0], iso_y, [1]])
# Step function for isotonic
for i in range(len(iso_x)-1):
    ax_cal.plot([iso_x[i], iso_x[i+1]], [iso_y[i], iso_y[i]], color=A2, lw=2.5, zorder=4)
    if i < len(iso_x)-2:
        ax_cal.plot([iso_x[i+1], iso_x[i+1]], [iso_y[i], iso_y[i+1]], color=A2, lw=2.5, zorder=4)
ax_cal.plot([], [], color=A2, lw=2.5, label='Isotonic calibrated')

ax_cal.fill_between(x, uncal, x, alpha=0.08, color=R_HI)
ax_cal.set_xlabel('Predicted Probability', fontsize=14, fontweight='bold')
ax_cal.set_ylabel('True Probability', fontsize=14, fontweight='bold')
ax_cal.set_title('Isotonic Calibration of Meta-Learner', fontsize=16, fontweight='bold', color=N, pad=8)
ax_cal.legend(fontsize=12.5, loc='lower right', framealpha=0.9)
ax_cal.set_xlim(0, 1); ax_cal.set_ylim(0, 1)
ax_cal.set_aspect('equal')
ax_cal.spines['top'].set_visible(False)
ax_cal.spines['right'].set_visible(False)

plt.savefig(os.path.join(out,'phase3_architecture.png'),dpi=400,bbox_inches='tight',facecolor=BG)
plt.close()
print("Done Phase 3")
