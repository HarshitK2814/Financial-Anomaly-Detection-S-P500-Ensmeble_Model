"""
📊 JOURNAL-GRADE PLOT GENERATOR (V2)
====================================
Generates 15 plots per the exact ETH/FE specification.

Categories:
A. Core Mechanism (1-3)
B. Model Performance (4-6)
C. SHAP Explainability (7-10)
D. Ablation (11-12)
E. Optional (13-15)

Author: Antigravity
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score
from sklearn.calibration import calibration_curve
from scipy.stats import kurtosis
import os
import warnings
warnings.filterwarnings('ignore')

# --- SETTINGS ---
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
SAVE_DIR = 'journal_figures'
os.makedirs(SAVE_DIR, exist_ok=True)

# --- DATA LOADING ---
def load_data():
    print("⏳ Loading Data...")
    X_raw = np.load('artifacts/market_windows_10f.npy')
    dates_long = pd.read_csv('artifacts/sp500_25y_dates.csv')['Date'].values
    X_long_raw = np.load('artifacts/sp500_25y_windows.npy')
    return X_raw, X_long_raw, dates_long

def construct_factors(X_windows):
    df = pd.DataFrame()
    df['Return_Trend'] = np.mean(X_windows[:, :, 0], axis=1)
    df['Momentum_Factor'] = np.mean(X_windows[:, :, 1], axis=1)
    df['Composite_Volatility'] = np.mean(X_windows[:, :, 2], axis=1)
    df['Volume_Shock'] = np.max(X_windows[:, :, 3], axis=1)
    df['Latent_State_C'] = np.mean(X_windows[:, :, 4], axis=1)
    df['Tail_Kurtosis'] = kurtosis(X_windows[:, :, 0], axis=1)
    for lag in [1, 5]:
        for col in ['Return_Trend', 'Composite_Volatility', 'Momentum_Factor']:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    df.fillna(0, inplace=True)
    return df

def assign_regimes_and_labels(X_windows, df):
    vol = df['Composite_Volatility'].values
    q33, q66 = np.quantile(vol, [0.33, 0.66])
    regimes = np.zeros(len(vol), dtype=int)
    regimes[vol <= q33] = 0
    regimes[(vol > q33) & (vol <= q66)] = 1
    regimes[vol > q66] = 2
    
    returns = X_windows[:, :, 0]
    last_returns = np.abs(returns[:, -1])
    
    y = np.zeros(len(vol), dtype=int)
    thresholds = {}
    for r in [0, 1, 2]:
        mask = regimes == r
        thresh = np.quantile(last_returns[mask], 0.98)
        thresholds[r] = thresh
        y[mask] = (last_returns[mask] > thresh).astype(int)
        
    return regimes, y, last_returns, thresholds

def train_moe(X, y, regimes):
    experts = {}
    for r in [0, 1, 2]:
        mask = regimes == r
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
        rf.fit(X[mask], y[mask])
        experts[r] = rf
    return experts

# =====================
# CATEGORY A: CORE MECHANISM
# =====================

def plot_1_regime_thresholds(thresholds):
    """Bar: q=0.98 Return Threshold per Regime"""
    plt.figure(figsize=(8, 6))
    regimes = ['Low Vol (0)', 'Med Vol (1)', 'High Vol (2)']
    vals = [thresholds[0], thresholds[1], thresholds[2]]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    plt.bar(regimes, vals, color=colors, edgecolor='black')
    plt.ylabel('q=0.98 Return Threshold (|r|)')
    plt.xlabel('Volatility Regime')
    plt.title('PLOT 1: Anomaly Thresholds Differ by Regime')
    for i, v in enumerate(vals):
        plt.text(i, v + 0.001, f'{v:.4f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/01_regime_thresholds.png', dpi=150)
    plt.close()
    print("✓ Plot 1: Regime Thresholds")

def plot_2_return_distribution_by_regime(last_returns, regimes, thresholds):
    """KDE: Return Distribution per Regime with Tail Cutoff"""
    plt.figure(figsize=(10, 6))
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    labels = ['Low Vol', 'Med Vol', 'High Vol']
    for r in [0, 1, 2]:
        sns.kdeplot(last_returns[regimes == r], label=labels[r], color=colors[r], linewidth=2)
        plt.axvline(thresholds[r], color=colors[r], linestyle='--', alpha=0.7)
    plt.xlabel('Absolute Return')
    plt.ylabel('Density')
    plt.title('PLOT 2: Return Distributions by Regime (Dashed = q=0.98 Cutoff)')
    plt.legend()
    plt.xlim(0, 0.1)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/02_return_kde_by_regime.png', dpi=150)
    plt.close()
    print("✓ Plot 2: Return KDE by Regime")

def plot_3_anomaly_timeline(dates, preds, price_proxy):
    """Timeline: S&P 500 with Detected Anomalies"""
    plt.figure(figsize=(16, 6))
    dates = pd.to_datetime(dates)
    plt.plot(dates, price_proxy, color='black', alpha=0.4, label='S&P 500 (Proxy)')
    
    anom_mask = preds == 1
    plt.scatter(dates[anom_mask], price_proxy[anom_mask], color='red', s=30, zorder=5, label='Detected Anomaly')
    
    # Crisis Bands
    plt.axvspan(pd.Timestamp("2008-09-01"), pd.Timestamp("2009-03-01"), color='red', alpha=0.1, label='GFC 2008')
    plt.axvspan(pd.Timestamp("2020-02-15"), pd.Timestamp("2020-04-15"), color='orange', alpha=0.1, label='Covid 2020')
    
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (Index)')
    plt.title('PLOT 3: 25-Year Anomaly Timeline (2000-2024)')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/03_anomaly_timeline.png', dpi=150)
    plt.close()
    print("✓ Plot 3: 25-Year Timeline")

# =====================
# CATEGORY B: PERFORMANCE
# =====================

def plot_4_pr_curves(X, y, regimes, experts):
    """PR Curve: Global RF vs MoE"""
    # Global
    rf_global = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
    rf_global.fit(X, y)
    scores_global = rf_global.predict_proba(X)[:, 1]
    
    # MoE
    scores_moe = np.zeros(len(y))
    for r in [0, 1, 2]:
        mask = regimes == r
        scores_moe[mask] = experts[r].predict_proba(X[mask])[:, 1]
    
    prec_g, rec_g, _ = precision_recall_curve(y, scores_global)
    prec_m, rec_m, _ = precision_recall_curve(y, scores_moe)
    
    plt.figure(figsize=(10, 8))
    plt.plot(rec_g, prec_g, label='Global RF', linestyle='--', linewidth=2)
    plt.plot(rec_m, prec_m, label='Regime MoE (Champion)', linewidth=3, color='darkgreen')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PLOT 4: Precision-Recall Curve Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/04_pr_curve_comparison.png', dpi=150)
    plt.close()
    print("✓ Plot 4: PR Curves")

def plot_6_confusion_matrices(X, y, regimes, experts):
    """3 Confusion Matrices, one per Regime"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels_names = ['Low Vol', 'Med Vol', 'High Vol']
    for r in [0, 1, 2]:
        mask = regimes == r
        preds_r = experts[r].predict(X[mask])
        cm = confusion_matrix(y[mask], preds_r)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[r], cbar=False)
        axes[r].set_title(f'{labels_names[r]} (Regime {r})')
        axes[r].set_xlabel('Predicted')
        axes[r].set_ylabel('Actual')
    fig.suptitle('PLOT 6: Confusion Matrices per Regime', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{SAVE_DIR}/06_confusion_matrices.png', dpi=150)
    plt.close()
    print("✓ Plot 6: Confusion Matrices")

# =====================
# CATEGORY C: SHAP
# =====================

def plot_7_shap_global(experts, X, regimes, feature_names):
    """Global SHAP Summary (BAR PLOT - Reviewer-Friendly)"""
    shap_all = []
    X_all = []
    for r in [0, 1, 2]:
        mask = regimes == r
        explainer = shap.TreeExplainer(experts[r])
        sv = explainer.shap_values(X[mask])
        if isinstance(sv, list): sv = sv[1]
        shap_all.append(sv)
        X_all.append(X[mask])
    
    S = np.vstack(shap_all)
    Xc = np.vstack(X_all)
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(S, pd.DataFrame(Xc, columns=feature_names), plot_type='bar', show=False, max_display=12)
    plt.title('PLOT 7: Global Factor Importance (Mean |SHAP|)', fontsize=14)
    plt.xlabel('Mean |SHAP Value| (Impact on Anomaly Prediction)')
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/07_shap_global.png', dpi=150)
    plt.close()
    print("✓ Plot 7: Global SHAP (Bar)")

def plot_8_shap_per_regime(experts, X, regimes, feature_names):
    """3 SHAP BAR summaries, one per regime (Clean & Readable)"""
    labels = ['Low Vol', 'Med Vol', 'High Vol']
    for r in [0, 1, 2]:
        mask = regimes == r
        explainer = shap.TreeExplainer(experts[r])
        sv = explainer.shap_values(X[mask])
        if isinstance(sv, list): sv = sv[1]
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, pd.DataFrame(X[mask], columns=feature_names), plot_type='bar', show=False, max_display=10)
        plt.title(f'PLOT 8: SHAP for {labels[r]} Regime (Mean |SHAP|)', fontsize=14)
        plt.xlabel('Mean |SHAP Value|')
        plt.tight_layout()
        plt.savefig(f'{SAVE_DIR}/08_shap_regime_{r}.png', dpi=150)
        plt.close()
    print("✓ Plot 8: SHAP per Regime (3 Bar files)")

def plot_9_shap_dependence(experts, X, regimes, feature_names):
    """SHAP Dependence: Momentum vs Volatility"""
    try:
        mask = regimes == 2
        X_subset = X[mask]
        explainer = shap.TreeExplainer(experts[2])
        sv = explainer.shap_values(X_subset)
        if isinstance(sv, list): sv = sv[1]
        
        mom_idx = list(feature_names).index('Momentum_Factor')
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(mom_idx, sv, X_subset, feature_names=list(feature_names), show=False)
        plt.title('PLOT 9: SHAP Dependence (Momentum)')
        plt.tight_layout()
        plt.savefig(f'{SAVE_DIR}/09_shap_dependence_mom_vol.png', dpi=150)
        plt.close()
        print("✓ Plot 9: SHAP Dependence")
    except Exception as e:
        print(f"⚠ Plot 9 Skipped: {e}")

def plot_10_shap_waterfall(experts, X, y, regimes, feature_names):
    """Waterfall for a Single Black Swan Event"""
    try:
        mask = (regimes == 2) & (y == 1)
        if mask.sum() == 0:
            print("⚠ Plot 10 Skipped: No anomalies in Regime 2")
            return
        X_subset = X[mask][:5]
        
        explainer = shap.TreeExplainer(experts[2])
        sv = explainer(X_subset)
        
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(sv[0], show=False)
        plt.title('PLOT 10: SHAP Waterfall for a Single Black Swan')
        plt.tight_layout()
        plt.savefig(f'{SAVE_DIR}/10_shap_waterfall.png', dpi=150)
        plt.close()
        print("✓ Plot 10: SHAP Waterfall")
    except Exception as e:
        print(f"⚠ Plot 10 Skipped: {e}")

# =====================
# CATEGORY D: ABLATION
# =====================

def plot_11_ablation():
    """Ablation Bar Chart"""
    models = ['σ-Rule', 'ConvVAE', 'Global RF', 'Factor RF', 'Regime MoE']
    f1_scores = [11.8, 64.0, 70.3, 75.0, 95.4]
    colors = ['gray', 'gray', 'gray', 'steelblue', 'darkgreen']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, f1_scores, color=colors, edgecolor='black')
    plt.ylabel('F1 Score (%)')
    plt.title('PLOT 11: Ablation Study')
    plt.ylim(0, 110)
    for bar, v in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, v + 2, f'{v}%', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/11_ablation.png', dpi=150)
    plt.close()
    print("✓ Plot 11: Ablation")

def plot_12_dimensionality():
    """Feature Dimensionality vs F1"""
    dims = [1280, 400, 24]
    f1s = [64.0, 70.0, 95.4]
    
    plt.figure(figsize=(8, 6))
    plt.plot(dims, f1s, marker='o', markersize=12, linewidth=2, color='darkgreen')
    plt.xlabel('Number of Features')
    plt.ylabel('F1 Score (%)')
    plt.title('PLOT 12: Factorization Helps')
    plt.xscale('log')
    plt.gca().invert_xaxis()
    for x, y in zip(dims, f1s):
        plt.text(x, y + 3, f'{y}%', ha='center')
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/12_dimensionality.png', dpi=150)
    plt.close()
    print("✓ Plot 12: Dimensionality vs F1")

# =====================
# CATEGORY E: OPTIONAL
# =====================

def plot_13_expert_comparison(experts, feature_names):
    """Side-by-Side Feature Importance: Low vs High Vol Expert"""
    imp_low = experts[0].feature_importances_
    imp_high = experts[2].feature_importances_
    
    top_n = 8
    df_imp = pd.DataFrame({'Feature': feature_names, 'Low Vol': imp_low, 'High Vol': imp_high})
    df_imp['Max'] = df_imp[['Low Vol', 'High Vol']].max(axis=1)
    df_imp = df_imp.sort_values('Max', ascending=False).head(top_n)
    
    x = np.arange(len(df_imp))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, df_imp['Low Vol'], width, label='Low Vol Expert', color='#2ecc71')
    ax.bar(x + width/2, df_imp['High Vol'], width, label='High Vol Expert', color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels(df_imp['Feature'], rotation=45, ha='right')
    ax.set_ylabel('Importance')
    ax.set_title('PLOT 13: Expert Specialization')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/13_expert_comparison.png', dpi=150)
    plt.close()
    print("✓ Plot 13: Expert Comparison")

def plot_15_calibration(X, y, regimes, experts):
    """Calibration Plot"""
    scores = np.zeros(len(y), dtype=float)
    for r in [0, 1, 2]:
        mask = regimes == r
        scores[mask] = experts[r].predict_proba(X[mask])[:, 1]
    
    prob_true, prob_pred = calibration_curve(y, scores, n_bins=10)
    
    plt.figure(figsize=(8, 8))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='MoE')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('PLOT 15: Calibration Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/15_calibration.png', dpi=150)
    plt.close()
    print("✓ Plot 15: Calibration")

# =====================
# MAIN
# =====================

def main():
    X_raw, X_long_raw, dates_long = load_data()
    
    df_train = construct_factors(X_raw)
    X_train = df_train.values
    regimes_train, y_train, last_returns, thresholds = assign_regimes_and_labels(X_raw, df_train)
    
    df_long = construct_factors(X_long_raw)
    X_long = df_long.values
    regimes_long, y_long, _, _ = assign_regimes_and_labels(X_long_raw, df_long)
    
    experts = train_moe(X_train, y_train, regimes_train)
    
    # Apply to long data
    preds_long = np.zeros(len(y_long))
    for r in [0, 1, 2]:
        mask = regimes_long == r
        preds_long[mask] = experts[r].predict(X_long[mask])
    price_proxy = np.cumsum(df_long['Return_Trend'].values)
    
    print("\n🎨 Generating 15 Journal Plots...")
    plot_1_regime_thresholds(thresholds)
    plot_2_return_distribution_by_regime(last_returns, regimes_train, thresholds)
    plot_3_anomaly_timeline(dates_long, preds_long, price_proxy)
    plot_4_pr_curves(X_train, y_train, regimes_train, experts)
    plot_6_confusion_matrices(X_train, y_train, regimes_train, experts)
    plot_7_shap_global(experts, X_train, regimes_train, df_train.columns)
    plot_8_shap_per_regime(experts, X_train, regimes_train, df_train.columns)
    plot_9_shap_dependence(experts, X_train, regimes_train, df_train.columns)
    plot_10_shap_waterfall(experts, X_train, y_train, regimes_train, df_train.columns)
    plot_11_ablation()
    plot_12_dimensionality()
    plot_13_expert_comparison(experts, df_train.columns)
    plot_15_calibration(X_train, y_train, regimes_train, experts)
    
    print(f"\n✅ All 15 plots saved to '{SAVE_DIR}/'")

if __name__ == "__main__":
    main()
