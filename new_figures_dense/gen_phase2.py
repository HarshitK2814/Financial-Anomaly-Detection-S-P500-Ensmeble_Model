import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
from matplotlib.gridspec import GridSpec
import numpy as np, os

N='#0F172A'; A='#1E40AF'; A2='#2563EB'; S='#334155'; W='#FFFFFF'; BG='#F8FAFC'
AL='#DBEAFE'; SL='#E2E8F0'; GL='#F1F5F9'
R_LOW='#2563EB'; R_MED='#D97706'; R_HI='#DC2626'

out=r"c:\Users\Harshit Kumar\Downloads\Financial Anomaly Detection Journal Paper Coding Part\Financial-Anomaly-Detection-S-P500-Ensmeble_Model\correct figures"

fig=plt.figure(figsize=(20,11),facecolor=BG)
gs=GridSpec(3,4,figure=fig,hspace=0.35,wspace=0.3,
            height_ratios=[0.08,0.45,0.47],
            left=0.03,right=0.97,top=0.97,bottom=0.03)

# ── TITLE ──
ax_t=fig.add_subplot(gs[0,:])
ax_t.set_xlim(0,1); ax_t.set_ylim(0,1); ax_t.axis('off')
ax_t.add_patch(FancyBboxPatch((0,0),1,1,boxstyle="round,pad=0,rounding_size=0.05",fc=AL,ec=N,lw=1.5,transform=ax_t.transAxes,zorder=1))
ax_t.text(0.5,0.5,"PHASE 2    Adaptive Regime Detection & Confidence Gating",transform=ax_t.transAxes,ha='center',va='center',fontsize=24,fontweight='bold',color='#000000',zorder=5)

# ══════ TOP ROW: VOLATILITY CHART WITH REGIME COLORING ══════
ax_vol=fig.add_subplot(gs[1,:])
ax_vol.set_facecolor(W)
np.random.seed(2024)
T=500
ret=np.random.randn(T)*0.012
# Create regime-like vol clusters
vol=np.zeros(T)
for i in range(T):
    base=0.008
    if 80<i<160: base=0.025
    if 200<i<250: base=0.022
    if 320<i<400: base=0.03
    vol[i]=base+abs(np.random.randn())*0.005
vol_smooth=np.convolve(vol,np.ones(20)/20,mode='same')

# Regime thresholds
q33=np.percentile(vol_smooth,33)
q66=np.percentile(vol_smooth,66)

# Color regime bands
for i in range(T-1):
    if vol_smooth[i]<q33: c=R_LOW
    elif vol_smooth[i]<q66: c=R_MED
    else: c=R_HI
    ax_vol.axvspan(i,i+1,color=c,alpha=0.12,zorder=0)

ax_vol.plot(range(T),vol_smooth,color=N,lw=1.8,zorder=3)
ax_vol.axhline(q33,color=R_LOW,lw=2,ls='--',zorder=2,label=r'$Q_{33}$ threshold')
ax_vol.axhline(q66,color=R_HI,lw=2,ls='--',zorder=2,label=r'$Q_{66}$ threshold')

# Regime labels on chart
ax_vol.text(40,q33-0.002,"LOW",fontsize=15,fontweight='bold',color=R_LOW)
ax_vol.text(220,q33+0.003,"MEDIUM",fontsize=15,fontweight='bold',color=R_MED)
ax_vol.text(350,q66+0.003,"HIGH",fontsize=15,fontweight='bold',color=R_HI)

ax_vol.set_ylabel("Realized Volatility",fontsize=16,fontweight='bold',color=N)
ax_vol.set_xlabel("Trading Days",fontsize=14,color=S)
ax_vol.set_title("Rolling Volatility with Adaptive Regime Boundaries",fontsize=17,fontweight='bold',color=N,pad=10)
ax_vol.legend(fontsize=13,loc='upper left',framealpha=0.9)
ax_vol.spines['top'].set_visible(False)
ax_vol.spines['right'].set_visible(False)
ax_vol.tick_params(labelsize=9)

# ══════ BOTTOM ROW: DETECTOR -> FUSION -> ROUTING ══════
def bx(ax,x,y,w,h,fc=W,ec=N,lw=1.8,rad=0.06):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0.03,rounding_size={rad}",fc=fc,ec=ec,lw=lw,zorder=2))
def ar(ax,x1,y1,x2,y2,c=S,lw=2.5):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',color=c,lw=lw,mutation_scale=18,zorder=3))
def tx(ax,x,y,t,fs=14,c=N,fw='bold',ha='center',va='center'):
    ax.text(x,y,t,fontsize=fs,color=c,fontweight=fw,ha=ha,va=va,zorder=5,linespacing=1.3)

# Bottom left: 3 Detectors
ax_d=fig.add_subplot(gs[2,0:2])
ax_d.set_xlim(0,10); ax_d.set_ylim(0,6); ax_d.axis('off'); ax_d.set_facecolor(BG)

tx(ax_d,5,5.7,"Three Parallel Regime Detectors",fs=17)

# Detector boxes
dets=[
    ("Gaussian HMM","3 latent states\nBaum-Welch EM",0.3,3.2,N),
    ("Rolling Quantile","Adaptive 252-day\nwindow",3.5,3.2,A2),
    ("Static Quantile","Global historical\nbaseline",6.7,3.2,S),
]
for name,desc,x,y,ec in dets:
    bx(ax_d,x,y,2.8,2.2,fc=W,ec=ec,lw=2)
    tx(ax_d,x+1.4,y+1.7,name,fs=14,c=ec)
    tx(ax_d,x+1.4,y+0.8,desc,fs=12,c=S,fw='normal')

# Output arrows
for x in [1.7,4.9,8.1]:
    ar(ax_d,x,3.2,x,1.8,c=N)

# Merge box
bx(ax_d,0.3,0.5,9.2,1.3,fc=AL,ec=A,lw=2)
tx(ax_d,5.0,1.45,"Confidence-Weighted Majority Vote",fs=16,c=A)
tx(ax_d,5.0,0.85,r"$\hat{R}_t = \arg\max_k \sum_i C_t^{(i)} \cdot \mathbb{I}(R_t^{(i)}=k)$",fs=14,c=N,fw='normal')

# Bottom right: Routing Decision
ax_r=fig.add_subplot(gs[2,2:4])
ax_r.set_xlim(0,10); ax_r.set_ylim(0,6); ax_r.axis('off'); ax_r.set_facecolor(BG)

tx(ax_r,5,5.7,"Adaptive Routing Decision",fs=17)

# Confidence bar visual
bx(ax_r,0.5,3.5,9,1.8,fc=W,ec=N,lw=2)
tx(ax_r,5,5.0,"Fused Confidence Score",fs=15)

# Gradient bar
for i in range(100):
    x=1.0+i*0.07
    col=plt.cm.RdYlGn(i/100)
    ax_r.add_patch(Rectangle((x,4.0),0.07,0.6,fc=col,ec='none',zorder=4))

# Threshold line
thresh_x=1.0+60*0.07  # 0.6 position
ax_r.plot([thresh_x,thresh_x],[3.8,4.85],color=N,lw=3,zorder=5)
tx(ax_r,thresh_x,3.65,r"$\tau_c = 0.6$",fs=14,c=N)

# Two outcome boxes
bx(ax_r,0.5,0.5,4.0,2.5,fc=W,ec='#16A34A',lw=2.5)
tx(ax_r,2.5,2.5,"HARD ROUTING",fs=16,c='#16A34A')
tx(ax_r,2.5,1.7,r"$w_t^{(k)} = \mathbb{I}(k = \hat{R}_t)$",fs=14,c=N,fw='normal')
tx(ax_r,2.5,1.0,"One-hot\nassignment",fs=13,c=S,fw='normal')

bx(ax_r,5.5,0.5,4.0,2.5,fc=W,ec=R_HI,lw=2.5)
tx(ax_r,7.5,2.5,"SOFT ROUTING",fs=16,c=R_HI)
tx(ax_r,7.5,1.7,r"$w_t^{(k)} = P(\hat{R}_t = k)$",fs=14,c=N,fw='normal')
tx(ax_r,7.5,1.0,"Distributed\nweights",fs=13,c=S,fw='normal')

# Arrows from bar to boxes
ar(ax_r,3.5,3.5,2.5,3.0,c='#16A34A',lw=2)
tx(ax_r,2.2,3.3,r"$C_t \geq \tau_c$",fs=13,c='#16A34A')

ar(ax_r,6.5,3.5,7.5,3.0,c=R_HI,lw=2)
tx(ax_r,7.8,3.3,r"$C_t < \tau_c$",fs=13,c=R_HI)

plt.savefig(os.path.join(out,'phase2_architecture.png'),dpi=400,bbox_inches='tight',facecolor=BG)
plt.close()
print("Done Phase 2")
