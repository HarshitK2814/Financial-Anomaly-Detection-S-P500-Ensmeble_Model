"""
Full VaR coverage table (Reviewer 1, Query 5): Kupiec + Christoffersen per regime x market.

Same setup as make_evt_basel_figs.py / multimarket_evt_basel.json:
  daily returns 2016-2025, regimes = terciles of 60-day rolling realised vol,
  GPD POT at the 90th loss percentile, VaR99.
Models compared:
  Uncond  - one empirical VaR99 from the full sample, applied on every day
  Cond    - regime-conditional EVT-VaR99 (GPD fitted within each regime)
Rows 'All' use the full timeline (the conditional VaR switches with the regime);
regime rows use that regime's days in chronological order.

Run from model root:  python src/evaluation/coverage_table.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.evt_risk import (fit_gpd_pot, evt_var_es, kupiec_pof,
                                 christoffersen_independence, christoffersen_cc)

ART = Path("artifacts")
MARKETS = {
    "S&P 500": ("sp500_full_daily_returns.csv", "^GSPC"),
    "NASDAQ": ("returns_NASDAQ.csv", "^IXIC"),
    "Russell 2000": ("returns_Russell2000.csv", None),
    "Bitcoin": ("returns_Bitcoin.csv", None),
}
NAMES = {0: "Low", 1: "Med", 2: "High"}
CONF = 0.99


def load(fname, col):
    df = pd.read_csv(ART / fname, parse_dates=["Date"])
    df = df[(df.Date >= "2016-01-01") & (df.Date <= "2025-12-31")]
    return df[col or df.columns[1]].dropna().values


def regimes_of(r):
    vol = pd.Series(r).rolling(60).std().bfill().values
    q33, q66 = np.quantile(vol, [0.33, 0.66])
    return np.where(vol <= q33, 0, np.where(vol <= q66, 1, 2))


def tests(viol):
    k = kupiec_pof(viol, CONF)
    i = christoffersen_independence(viol)
    c = christoffersen_cc(viol, CONF)
    return dict(n=k["n"], expected=round(k["expected"], 1), violations=k["violations"],
                rate=k["actual_rate"], LR_uc=k["LR_pof"], p_uc=k["p_value"],
                LR_ind=i["LR_ind"], p_ind=i["p_value"], LR_cc=c["LR_cc"], p_cc=c["p_value"])


rows = []
for mkt, (f, col) in MARKETS.items():
    r = load(f, col)
    reg = regimes_of(r)
    var_u = np.quantile(r, 1 - CONF)                        # negative return threshold
    var_c = {k: -evt_var_es(fit_gpd_pot(-r[reg == k], 0.90), CONF)["VaR"] for k in (0, 1, 2)}
    var_c_series = np.array([var_c[k] for k in reg])
    for scope in ["All", 0, 1, 2]:
        m = np.ones(len(r), bool) if scope == "All" else reg == scope
        for model, thr in [("Uncond", np.full(len(r), var_u)), ("Cond", var_c_series)]:
            viol = (r[m] < thr[m]).astype(int)
            rows.append(dict(market=mkt, regime=NAMES.get(scope, scope), model=model,
                             **tests(viol)))

df = pd.DataFrame(rows)
df.to_csv(ART / "coverage_table.csv", index=False)
pd.set_option("display.width", 200)
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
