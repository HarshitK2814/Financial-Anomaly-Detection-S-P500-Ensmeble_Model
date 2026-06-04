import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np, os

# Restrained 4-color palette
N='#0F172A'; A='#1E40AF'; S='#334155'; W='#FFFFFF'; BG='#F8FAFC'
A2='#2563EB'; A3='#3B82F6'; AL='#DBEAFE'; SL='#E2E8F0'; GL='#F1F5F9'

out=r"c:\Users\Harshit Kumar\Downloads\Financial Anomaly Detection Journal Paper Coding Part\Financial-Anomaly-Detection-S-P500-Ensmeble_Model\new_figures_dense"

def bx(ax,x,y,w,h,fc=W,ec=N,lw=2.5,rad=0.06,shadow=False):
    if shadow:
        ax.add_patch(FancyBboxPatch((x+0.07,y-0.07),w,h,boxstyle=f"round,pad=0.03,rounding_size={rad}",fc='#CBD5E1',ec='none',zorder=1))
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0.03,rounding_size={rad}",fc=fc,ec=ec,lw=lw,zorder=2))
def ar(ax,x1,y1,x2,y2,c=S,lw=2.5):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',color=c,lw=lw,mutation_scale=20,zorder=3))
def txt(ax,x,y,t,fs=14,c='#000000',fw='normal',ha='center',va='center'):
    ax.text(x,y,t,fontsize=fs,color=c,fontweight=fw,ha=ha,va=va,zorder=5,linespacing=1.3,fontfamily='sans-serif')

fig,ax=plt.subplots(figsize=(20,9),facecolor=BG)
ax.set_xlim(-0.3,18.5); ax.set_ylim(-0.5,8.5); ax.set_facecolor(BG); ax.axis('off')

# ── TITLE ──
bx(ax,-0.2,7.5,18.5,0.85,fc=AL,ec=N,lw=1.5,shadow=True)
txt(ax,9.0,7.93,"PHASE 1    Factor Engineering Pipeline",fs=22,c='#000000',fw='bold')

# ══════ A: RAW DATA ══════
bx(ax,0.2,2.0,3.0,4.8,fc=W,ec=N,lw=2.5,shadow=True)
txt(ax,1.7,6.3,"Raw OHLCV",fs=18,c='#000000',fw='bold') # Increased Header Size
txt(ax,1.7,5.7,r"$W_t \in \mathbb{R}^{128 \times 5}$",fs=14,c='#000000')

# Candlestick chart
np.random.seed(42)
prices=np.cumsum(np.random.randn(20)*0.3)+50
for i in range(len(prices)-1):
    x=0.5+i*0.13
    o,c_=prices[i],prices[i+1]
    o_n=(o-47)/8*3.2+2.3; c_n=(c_-47)/8*3.2+2.3
    h_n=max(o_n,c_n)+np.random.uniform(0.05,0.15)
    l_n=min(o_n,c_n)-np.random.uniform(0.05,0.15)
    col='#16A34A' if c_n>=o_n else '#DC2626'
    ax.plot([x,x],[l_n,h_n],color=S,lw=1.0,zorder=4)
    ax.add_patch(Rectangle((x-0.04,min(o_n,c_n)),0.08,abs(c_n-o_n)+0.02,fc=col,ec='none',zorder=4))

ar(ax,3.2,4.4,4.0,4.4,c=N,lw=3)
txt(ax,3.6,4.8,r"$\phi(\cdot)$",fs=18,c='#000000')

# ══════ B: FACTOR ENGINE ══════
bx(ax,4.0,0.3,7.8,7.0,fc=GL,ec=N,lw=2.5,rad=0.1,shadow=True)
txt(ax,7.9,6.9,"Factor Extraction Engine",fs=18,c='#000000',fw='bold') # Increased Header Size

# 6 category boxes
cats=[
    ("VOLATILITY",  "Composite\nGarman-Klass\nParkinson\nVIX Proxy",        4.3, 4.6),
    ("TAIL RISK",   "Kurtosis\nSkewness\nDrawdown",                         4.3, 2.5),
    ("TREND & MOM", "Trend\nMomentum",                                      4.3, 0.6),
    ("LIQUIDITY",   "Volume Shock\nILLIQ",                                  8.1, 4.6),
    ("MICROSTRUCTURE","Latent States",                                      8.1, 2.5),
    ("PERSISTENCE", "Hurst Exponent",                                       8.1, 0.6),
]
for title,desc,x,y in cats:
    bx(ax,x,y,3.5,1.7,fc=W,ec=A,lw=1.5,rad=0.06)
    txt(ax,x+1.75,y+1.475,title,fs=16,c='#000000',fw='bold') # Increased Header Size
    ax.plot([x, x+3.5], [y+1.25, y+1.25], color=A, lw=1.2, zorder=3)
    txt(ax,x+1.75,y+0.625,desc,fs=12.5,c='#000000',va='center') # Kept layout fix

ar(ax,11.8,4.4,12.6,4.4,c=N,lw=3)

# ══════ C: TEMPORAL AUGMENTATION ══════
bx(ax,12.6,1.5,2.8,5.5,fc=W,ec=N,lw=2.5,shadow=True)
txt(ax,14.0,6.3,"Lag\nAugmentation",fs=18,c='#000000',fw='bold') # Increased Header Size

# Three bold stacked bars
for label,y,alpha in [("Current",4.8,1.0),("Lag-1",3.6,0.7),("Lag-5",2.4,0.4)]:
    ax.add_patch(Rectangle((12.9,y),2.2,0.8,fc=A2,ec=N,alpha=alpha,lw=1.5,zorder=3))
    txt(ax,14.0,y+0.4,label,fs=14,c='#000000',fw='bold')

ar(ax,15.4,4.4,16.2,4.4,c=N,lw=3)

# ══════ D: OUTPUT ══════
bx(ax,16.2,2.0,2.0,4.8,fc=AL,ec=N,lw=2.5,rad=0.1,shadow=True)
txt(ax,17.2,6.0,r"$F_t \in \mathbb{R}^{48}$",fs=18,c='#000000',fw='bold')
txt(ax,17.2,5.2,"48-D\nFeature\nVector",fs=14,c='#000000')

# Simple embedding block 
for r in range(6):
    for c_ in range(4):
        ax.add_patch(Rectangle((16.78 + c_*0.22, 2.6 + r*0.22), 0.18, 0.18, fc=A3, ec='none', zorder=4))

plt.savefig(os.path.join(out,'phase1_architecture.png'),dpi=400,bbox_inches='tight',facecolor=BG)
plt.close()
print("Done Phase 1 blue themed larger headings")
