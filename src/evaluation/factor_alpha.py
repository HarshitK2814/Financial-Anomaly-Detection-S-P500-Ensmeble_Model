"""
FAMA-FRENCH FACTOR ALPHA TEST  (Addition 2 — Quantitative Finance unlock)
=========================================================================
Tests whether the RC-MoE anomaly-hedged strategy generates ALPHA that is
NOT explained by known risk factors. This is the gold-standard test that
finance journals (Quantitative Finance, JFDS, JFQA) require before they
accept any "economic value" claim.

Regression (Newey-West HAC standard errors):
    R_hedge_t - Rf_t = alpha
                       + b1*MktRF + b2*SMB + b3*HML
                       + b4*RMW   + b5*CMA + b6*MOM + e_t

Publishable result: alpha > 0 and statistically significant  =>  the
regime-conditioned hedge adds value beyond market, size, value,
profitability, investment, and momentum exposure.

DATA
----
Kenneth French Data Library (free):
  Fama/French 5 Factors (2x3) [Daily]  ->  F-F_Research_Data_5_Factors_2x3_daily.CSV
  Momentum Factor (Mom) [Daily]        ->  F-F_Momentum_Factor_daily.CSV
Download CSVs into  data/ff_factors/  (or pass paths explicitly).
Factors are in PERCENT and include Rf; divide by 100 before regression.

USAGE
-----
    from src.evaluation.factor_alpha import run_factor_alpha
    res = run_factor_alpha(strategy_daily_returns, dates,
                           ff5_csv='data/ff_factors/FF5_daily.csv',
                           mom_csv='data/ff_factors/MOM_daily.csv')
    print(res['summary'])
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional


def _load_french_csv(path: str, skip_search: bool = True) -> pd.DataFrame:
    """Load a Kenneth French daily CSV.

    French CSVs have header/footer junk and blank lines between rows.
    Read line-by-line: keep only lines whose first token is an 8-digit date.
    """
    col_names = None
    rows = []
    with open(path, 'r', encoding='latin-1') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            tok = parts[0]
            if len(tok) == 8 and tok.isdigit():
                rows.append(parts)
            elif col_names is None and any(
                    kw in tok.upper() for kw in ['MKT', 'SMB', 'HML', 'MOM', 'RF']):
                col_names = parts
    if not rows:
        raise ValueError(f"No daily YYYYMMDD rows found in {path}; "
                         "check you downloaded the *daily* CSV.")
    df = pd.DataFrame(rows)
    df[0] = pd.to_datetime(df[0], format='%Y%m%d')
    df = df.set_index(0)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df


def load_ff_factors(ff5_csv: str, mom_csv: str) -> pd.DataFrame:
    """Return a daily DataFrame with columns
    [MktRF, SMB, HML, RMW, CMA, RF, MOM], all in DECIMAL (not percent)."""
    ff5 = _load_french_csv(ff5_csv)
    mom = _load_french_csv(mom_csv)

    # FF5 daily has 6 data columns: Mkt-RF, SMB, HML, RMW, CMA, RF
    ff5.columns = ['MktRF', 'SMB', 'HML', 'RMW', 'CMA', 'RF'][:ff5.shape[1]]
    mom.columns = ['MOM'][:mom.shape[1]]

    df = ff5.join(mom, how='inner') / 100.0  # percent -> decimal
    return df


def run_factor_alpha(strategy_returns: np.ndarray,
                     dates,
                     ff5_csv: str,
                     mom_csv: str,
                     hac_lags: int = 5) -> Dict:
    """Regress strategy EXCESS returns on FF5+MOM with Newey-West SEs.

    Parameters
    ----------
    strategy_returns : np.ndarray
        Daily TOTAL returns of the hedged strategy (decimal, not percent).
    dates : array-like of datetime64
        Dates aligned 1:1 with strategy_returns.
    ff5_csv, mom_csv : str
        Paths to the Kenneth French daily CSVs.
    hac_lags : int
        Newey-West lag length (5 is standard for daily data).

    Returns
    -------
    dict with alpha (annualized), alpha t-stat, p-value, betas, R^2,
    and a formatted summary string.
    """
    import statsmodels.api as sm

    s = pd.Series(np.asarray(strategy_returns, dtype=float),
                  index=pd.to_datetime(pd.Index(dates)), name='R')
    factors = load_ff_factors(ff5_csv, mom_csv)

    df = pd.concat([s, factors], axis=1, join='inner').dropna()
    if len(df) < 60:
        raise ValueError(f"Only {len(df)} aligned days; check date overlap "
                         "between strategy and French factors.")

    y = df['R'] - df['RF']                      # excess return
    X = df[['MktRF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit(cov_type='HAC',
                             cov_kwds={'maxlags': hac_lags})

    alpha_daily = model.params['const']
    alpha_ann = (1 + alpha_daily) ** 252 - 1
    alpha_t = model.tvalues['const']
    alpha_p = model.pvalues['const']

    betas = {k: float(model.params[k]) for k in
             ['MktRF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']}
    beta_p = {k: float(model.pvalues[k]) for k in betas}

    verdict = ("SIGNIFICANT POSITIVE ALPHA — strategy adds value beyond "
               "known risk factors") if (alpha_daily > 0 and alpha_p < 0.05) \
        else ("No significant alpha — returns are explained by factor "
              "exposure" if alpha_p >= 0.05 else
              "Significant NEGATIVE alpha — strategy underperforms factors")

    summary = (
        "FAMA-FRENCH 5 + MOMENTUM ALPHA TEST\n"
        "=" * 52 + "\n"
        f"N daily obs aligned : {len(df)}\n"
        f"Alpha (daily)       : {alpha_daily:+.6f}\n"
        f"Alpha (annualized)  : {alpha_ann:+.4%}\n"
        f"Alpha t-stat (HAC)  : {alpha_t:+.3f}\n"
        f"Alpha p-value       : {alpha_p:.4f}\n"
        f"Adj. R^2            : {model.rsquared_adj:.4f}\n"
        "Factor loadings (p):\n" +
        "\n".join(f"   {k:6s} = {betas[k]:+.3f}  (p={beta_p[k]:.3f})"
                  for k in betas) + "\n"
        "-" * 52 + "\n"
        f"VERDICT: {verdict}\n"
    )

    return {
        'alpha_daily': float(alpha_daily),
        'alpha_annualized': float(alpha_ann),
        'alpha_tstat': float(alpha_t),
        'alpha_pvalue': float(alpha_p),
        'betas': betas,
        'beta_pvalues': beta_p,
        'r2_adj': float(model.rsquared_adj),
        'n_obs': int(len(df)),
        'verdict': verdict,
        'summary': summary,
        'model': model,
    }


if __name__ == '__main__':
    # Smoke test with synthetic data (no French CSVs required).
    # Real run: supply the hedge strategy returns from economic_backtest.py
    # and the two downloaded French CSVs.
    print("factor_alpha.py — Fama-French alpha test module loaded.")
    print("Download FF5 + MOM daily CSVs from Kenneth French's library:")
    print("  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html")
    print("Then: run_factor_alpha(hedge_returns, dates, ff5_csv, mom_csv)")
