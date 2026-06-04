"""
COMPLETE IN-SAMPLE + OUT-OF-SAMPLE EVALUATION
==============================================
Runs all models on all time horizons with both evaluation protocols.
"""
import numpy as np
import pandas as pd
import os, sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, '.')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              average_precision_score, precision_recall_curve)
from sklearn.model_selection import TimeSeriesSplit


# ── Event-wise F1 (paper methodology) ──
def event_wise_f1(pred, true, tol=5):
    def get_events(labels):
        events, in_evt, start = [], False, 0
        for i, v in enumerate(labels):
            if v == 1 and not in_evt: in_evt, start = True, i
            elif v == 0 and in_evt: events.append((start, i-1)); in_evt = False
        if in_evt: events.append((start, len(labels)-1))
        return events
    te, pe = get_events(true), get_events(pred)
    if not te or not pe: return 0.0, 0.0, 0.0
    mt, mp, tp = [False]*len(te), [False]*len(pe), 0
    for i,(ts,t_e) in enumerate(te):
        for j,(ps,p_e) in enumerate(pe):
            if not mp[j] and not (p_e < ts-tol or ps > t_e+tol):
                tp += 1; mt[i] = mp[j] = True; break
    fp, fn = sum(1 for m in mp if not m), sum(1 for m in mt if not m)
    rec = tp/(tp+fn) if tp+fn else 0
    prec = tp/(tp+fp) if tp+fp else 0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
    return f1, prec, rec

def opt_thresh(scores, labels, tol=5, n=300):
    mn, mx = scores.min(), scores.max()
    if mn == mx: return mn, 0, 0, 0
    ts = np.linspace(mn, mx, n)
    best = (0, ts[0], 0, 0)
    for t in ts:
        f1, p, r = event_wise_f1((scores >= t).astype(int), labels, tol)
        if f1 > best[0]: best = (f1, t, p, r)
    return best[1], best[0], best[2], best[3]

def assign_regimes(vol):
    q33, q66 = np.quantile(vol, [0.33, 0.66])
    r = np.zeros_like(vol, dtype=int)
    r[vol <= q33] = 0; r[(vol > q33) & (vol <= q66)] = 1; r[vol > q66] = 2
    return r

def make_model(mt):
    if mt == 'RF':
        return RandomForestClassifier(n_estimators=200, max_depth=5,
                                       class_weight='balanced', random_state=42)
    elif mt == 'XGBoost':
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=200, max_depth=5, scale_pos_weight=10,
                                  learning_rate=0.05, eval_metric='aucpr',
                                  use_label_encoder=False, verbosity=0, random_state=42)
    elif mt == 'LightGBM':
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=200, max_depth=5, is_unbalance=True,
                                    learning_rate=0.05, verbose=-1, random_state=42)

def construct_basic_factors(windows):
    """Build factor features from raw windows."""
    from scipy.stats import kurtosis, skew
    N = len(windows)
    factors = []
    for i in range(N):
        w = windows[i]
        close = w[:, 0]
        rets = np.diff(close) / (np.abs(close[:-1]) + 1e-10)
        f = [
            np.mean(rets),           # Return
            np.std(rets),            # Volatility
            kurtosis(rets),          # Kurtosis
            skew(rets),              # Skew
            np.mean(rets[-5:]) - np.mean(rets[:5]),  # Momentum
            np.max(np.maximum.accumulate(close) - close) / (np.max(close) + 1e-10),  # Max DD
            np.min(rets),            # Worst return
            np.max(rets),            # Best return
        ]
        # Additional features from other columns if available
        if w.shape[1] > 1:
            for col in range(1, min(w.shape[1], 6)):
                f.append(np.std(w[:, col]))
                f.append(np.mean(w[:, col]))
        factors.append(f)
    return np.array(factors)


# ── IN-SAMPLE (train-on-all, predict-on-all) ──
def eval_in_sample(X, y, regimes, mt, mode):
    if mode == 'MoE':
        proba = np.zeros(len(y))
        for r in np.unique(regimes):
            mask = regimes == r
            if mask.sum() < 5 or len(np.unique(y[mask])) < 2: continue
            m = make_model(mt); m.fit(X[mask], y[mask])
            proba[mask] = m.predict_proba(X[mask])[:, 1]
    else:
        if len(np.unique(y)) < 2: return None
        m = make_model(mt); m.fit(X, y)
        proba = m.predict_proba(X)[:, 1]
    
    _, ef1, ep, er = opt_thresh(proba, y)
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y, proba)
    f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-8)
    bi = min(np.argmax(f1_arr), len(thresh_arr)-1)
    pw_preds = (proba >= thresh_arr[bi]).astype(int)
    
    return {
        'Event_F1': ef1, 'Event_Prec': ep, 'Event_Recall': er,
        'PW_F1': f1_score(y, pw_preds, zero_division=0),
        'PW_Prec': precision_score(y, pw_preds, zero_division=0),
        'PW_Recall': recall_score(y, pw_preds, zero_division=0),
        'AUC_PR': average_precision_score(y, proba) if len(np.unique(y))>1 else 0,
    }


# ── OUT-OF-SAMPLE (temporal CV) ──
def eval_oos(X, y, regimes, mt, mode, n_folds=5):
    tscv = TimeSeriesSplit(n_splits=n_folds)
    all_proba, all_true = [], []
    
    for tri, tei in tscv.split(X):
        if mode == 'MoE':
            fp = np.zeros(len(tei))
            for r in np.unique(regimes):
                trm = regimes[tri] == r; tem = regimes[tei] == r
                if trm.sum() < 5 or tem.sum() == 0: continue
                ytr = y[tri[trm]]
                if len(np.unique(ytr)) < 2: continue
                m = make_model(mt); m.fit(X[tri[trm]], ytr)
                fp[np.where(tem)[0]] = m.predict_proba(X[tei[tem]])[:, 1]
            all_proba.extend(fp)
        else:
            if len(np.unique(y[tri])) < 2: continue
            m = make_model(mt); m.fit(X[tri], y[tri])
            all_proba.extend(m.predict_proba(X[tei])[:, 1])
        all_true.extend(y[tei])
    
    all_proba, all_true = np.array(all_proba), np.array(all_true)
    if len(all_proba) == 0: return None
    
    _, ef1, ep, er = opt_thresh(all_proba, all_true)
    
    if len(np.unique(all_true)) < 2:
        return {'Event_F1': ef1, 'Event_Prec': ep, 'Event_Recall': er,
                'PW_F1': 0, 'PW_Prec': 0, 'PW_Recall': 0, 'AUC_PR': 0}
    
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(all_true, all_proba)
    f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-8)
    bi = min(np.argmax(f1_arr), len(thresh_arr)-1)
    pw_preds = (all_proba >= thresh_arr[bi]).astype(int)
    
    return {
        'Event_F1': ef1, 'Event_Prec': ep, 'Event_Recall': er,
        'PW_F1': f1_score(all_true, pw_preds, zero_division=0),
        'PW_Prec': precision_score(all_true, pw_preds, zero_division=0),
        'PW_Recall': recall_score(all_true, pw_preds, zero_division=0),
        'AUC_PR': average_precision_score(all_true, all_proba),
    }


def main():
    print("=" * 80)
    print("   COMPLETE IN-SAMPLE + OUT-OF-SAMPLE EVALUATION")
    print("   All Models x All Horizons x Both Protocols")
    print("=" * 80)
    
    # ── Load primary dataset ──
    X_factors = np.load('artifacts/X_factors.npy')
    y_factors = np.load('artifacts/y_factors.npy')
    X_win = np.load('artifacts/market_windows_10f.npy')
    y_labels = np.load('artifacts/market_labels.npy')
    df = pd.read_csv('artifacts/journal_factors.csv')
    
    vol = np.std(X_win[:, :, 0], axis=1)
    regimes = assign_regimes(vol)
    
    min_len = min(len(X_factors), len(y_factors), len(regimes))
    X_f = X_factors[:min_len]; y_f = y_factors[:min_len]; reg = regimes[:min_len]
    
    model_types = ['RF', 'XGBoost', 'LightGBM']
    modes = ['Global', 'MoE']
    
    # ═══════════════════════════════════════════
    # TABLE 1: PRIMARY DATASET
    # ═══════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print(f"  TABLE 1: PRIMARY DATASET (N={min_len}, anomalies={int(y_f.sum())}, rate={y_f.mean()*100:.1f}%)")
    print(f"{'=' * 80}")
    
    print(f"\n  --- IN-SAMPLE (Train-all, Predict-all) ---")
    print(f"  {'Model':<12} {'Mode':<8} {'E-F1':>7} {'E-Prec':>7} {'E-Rec':>7} {'PW-F1':>7} {'PW-Prec':>8} {'PW-Rec':>7} {'AUC-PR':>7}")
    print(f"  {'-'*73}")
    
    is_results = []
    for mt in model_types:
        for mode in modes:
            r = eval_in_sample(X_f, y_f, reg, mt, mode)
            if r:
                is_results.append({'Model': mt, 'Mode': mode, 'Protocol': 'In-Sample', 'Dataset': 'Primary', **r})
                print(f"  {mt:<12} {mode:<8} {r['Event_F1']:>7.4f} {r['Event_Prec']:>7.4f} {r['Event_Recall']:>7.4f} "
                      f"{r['PW_F1']:>7.4f} {r['PW_Prec']:>8.4f} {r['PW_Recall']:>7.4f} {r['AUC_PR']:>7.4f}")
    
    print(f"\n  --- OUT-OF-SAMPLE (5-fold Temporal CV) ---")
    print(f"  {'Model':<12} {'Mode':<8} {'E-F1':>7} {'E-Prec':>7} {'E-Rec':>7} {'PW-F1':>7} {'PW-Prec':>8} {'PW-Rec':>7} {'AUC-PR':>7}")
    print(f"  {'-'*73}")
    
    oos_results = []
    for mt in model_types:
        for mode in modes:
            r = eval_oos(X_f, y_f, reg, mt, mode)
            if r:
                oos_results.append({'Model': mt, 'Mode': mode, 'Protocol': 'Out-of-Sample', 'Dataset': 'Primary', **r})
                print(f"  {mt:<12} {mode:<8} {r['Event_F1']:>7.4f} {r['Event_Prec']:>7.4f} {r['Event_Recall']:>7.4f} "
                      f"{r['PW_F1']:>7.4f} {r['PW_Prec']:>8.4f} {r['PW_Recall']:>7.4f} {r['AUC_PR']:>7.4f}")
    
    # ═══════════════════════════════════════════
    # TABLE 2: MULTI-HORIZON (5y, 10y, 25y)
    # ═══════════════════════════════════════════
    horizons = [
        ('5y', 'artifacts/sp500_5y_windows.npy'),
        ('10y', 'artifacts/sp500_10y_windows.npy'),
        ('25y', 'artifacts/sp500_25y_windows.npy'),
    ]
    
    for hz_name, hz_path in horizons:
        if not os.path.exists(hz_path):
            print(f"\n  Skipping {hz_name}: file not found")
            continue
        
        X_hz = np.load(hz_path)
        print(f"\n{'=' * 80}")
        print(f"  TABLE: {hz_name.upper()} HORIZON (N={len(X_hz)}, shape={X_hz.shape})")
        print(f"{'=' * 80}")
        
        # Build factors
        print(f"  Building factors...")
        X_hz_f = construct_basic_factors(X_hz)
        vol_hz = np.std(X_hz[:, :, 0], axis=1)
        reg_hz = assign_regimes(vol_hz)
        
        # Need labels! Check if we can derive them
        # The multi-horizon datasets may not have separate labels
        # Use extreme anomaly definition: kurtosis > 95th percentile + volatility > 95th percentile
        from scipy.stats import kurtosis as kurt_fn
        kurt_vals = np.array([kurt_fn(np.diff(X_hz[i, :, 0]) / (np.abs(X_hz[i, :-1, 0]) + 1e-10)) for i in range(len(X_hz))])
        vol_vals = vol_hz
        
        # Anomaly label: extreme tail risk (top 2-3% by composite score)
        composite_score = (kurt_vals - kurt_vals.mean()) / (kurt_vals.std() + 1e-10) + \
                          (vol_vals - vol_vals.mean()) / (vol_vals.std() + 1e-10)
        threshold_pct = np.percentile(composite_score, 97)  # ~3% anomaly rate
        y_hz = (composite_score >= threshold_pct).astype(int)
        
        print(f"  Anomalies: {int(y_hz.sum())} ({y_hz.mean()*100:.1f}%)")
        print(f"  Regimes: Low={int((reg_hz==0).sum())}, Med={int((reg_hz==1).sum())}, High={int((reg_hz==2).sum())}")
        
        # In-sample
        print(f"\n  --- IN-SAMPLE ---")
        print(f"  {'Model':<12} {'Mode':<8} {'E-F1':>7} {'E-Prec':>7} {'E-Rec':>7} {'PW-F1':>7} {'AUC-PR':>7}")
        print(f"  {'-'*55}")
        
        for mt in model_types:
            for mode in modes:
                r = eval_in_sample(X_hz_f, y_hz, reg_hz, mt, mode)
                if r:
                    is_results.append({'Model': mt, 'Mode': mode, 'Protocol': 'In-Sample', 'Dataset': hz_name, **r})
                    print(f"  {mt:<12} {mode:<8} {r['Event_F1']:>7.4f} {r['Event_Prec']:>7.4f} {r['Event_Recall']:>7.4f} "
                          f"{r['PW_F1']:>7.4f} {r['AUC_PR']:>7.4f}")
        
        # Out-of-sample
        print(f"\n  --- OUT-OF-SAMPLE (5-fold Temporal CV) ---")
        print(f"  {'Model':<12} {'Mode':<8} {'E-F1':>7} {'E-Prec':>7} {'E-Rec':>7} {'PW-F1':>7} {'AUC-PR':>7}")
        print(f"  {'-'*55}")
        
        for mt in model_types:
            for mode in modes:
                r = eval_oos(X_hz_f, y_hz, reg_hz, mt, mode)
                if r:
                    oos_results.append({'Model': mt, 'Mode': mode, 'Protocol': 'Out-of-Sample', 'Dataset': hz_name, **r})
                    print(f"  {mt:<12} {mode:<8} {r['Event_F1']:>7.4f} {r['Event_Prec']:>7.4f} {r['Event_Recall']:>7.4f} "
                          f"{r['PW_F1']:>7.4f} {r['AUC_PR']:>7.4f}")
    
    # ═══════════════════════════════════════════
    # PER-REGIME BREAKDOWN (Best model, OOS)
    # ═══════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print(f"  PER-REGIME BREAKDOWN (XGBoost MoE, Primary, OOS)")
    print(f"{'=' * 80}")
    
    regime_names = {0: 'Low Vol', 1: 'Med Vol', 2: 'High Vol'}
    tscv = TimeSeriesSplit(n_splits=5)
    all_proba, all_true, all_reg = np.array([]), np.array([]), np.array([], dtype=int)
    
    for tri, tei in tscv.split(X_f):
        fp = np.zeros(len(tei))
        for r in range(3):
            trm = reg[tri] == r; tem = reg[tei] == r
            if trm.sum() < 5 or tem.sum() == 0: continue
            ytr = y_f[tri[trm]]
            if len(np.unique(ytr)) < 2: continue
            m = make_model('XGBoost'); m.fit(X_f[tri[trm]], ytr)
            fp[np.where(tem)[0]] = m.predict_proba(X_f[tei[tem]])[:, 1]
        all_proba = np.concatenate([all_proba, fp])
        all_true = np.concatenate([all_true, y_f[tei]])
        all_reg = np.concatenate([all_reg, reg[tei]])
    
    print(f"\n  {'Regime':<12} {'N':>5} {'Anom':>5} {'E-F1':>7} {'Prec':>7} {'Recall':>7}")
    print(f"  {'-'*45}")
    for r in range(3):
        mask = all_reg == r
        if mask.sum() == 0: continue
        _, rf, rp, rr = opt_thresh(all_proba[mask], all_true[mask])
        print(f"  {regime_names[r]:<12} {int(mask.sum()):>5} {int(all_true[mask].sum()):>5} {rf:>7.4f} {rp:>7.4f} {rr:>7.4f}")
    
    # ═══════════════════════════════════════════
    # FINAL SUMMARY TABLE
    # ═══════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print(f"  FINAL COMBINED RESULTS TABLE (for paper)")
    print(f"{'=' * 80}")
    
    all_results = is_results + oos_results
    df_results = pd.DataFrame(all_results)
    
    # Pivot for paper
    for dataset in df_results['Dataset'].unique():
        print(f"\n  [{dataset}]")
        ds_df = df_results[df_results['Dataset'] == dataset]
        for proto in ['In-Sample', 'Out-of-Sample']:
            p_df = ds_df[ds_df['Protocol'] == proto]
            if p_df.empty: continue
            print(f"    {proto}:")
            for _, row in p_df.iterrows():
                print(f"      {row['Model']:<10} {row['Mode']:<8}: E-F1={row['Event_F1']:.2%}, "
                      f"PW-F1={row['PW_F1']:.2%}, Recall={row['Event_Recall']:.2%}")
    
    # Save
    df_results.to_csv('artifacts/complete_evaluation_results.csv', index=False)
    print(f"\n  Results saved to artifacts/complete_evaluation_results.csv")
    print(f"\n{'=' * 80}")


if __name__ == '__main__':
    main()
