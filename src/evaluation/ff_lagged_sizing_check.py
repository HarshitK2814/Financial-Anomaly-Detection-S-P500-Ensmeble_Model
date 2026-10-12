"""
Check (Reviewer 1, Query 6): the regime-scaled overlay in hedge_daily_returns.csv sizes the
hedge with a volatility regime that includes the hedged day's own return. This rebuilds the
overlay with the sizing regime lagged by one day (information up to t-1 only) and re-runs the
FF5+MOM regression, to compare with artifacts/ff_alpha_results.json.

Run from model root:  python -P src/evaluation/ff_lagged_sizing_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.models.regime_detector import RollingQuantileRegimeDetector
from src.evaluation.factor_alpha import run_factor_alpha

ART = Path("artifacts")
FF_DIR = ART / "ff_data" if (ART / "ff_data").exists() else ART
h = pd.read_csv(ART / "hedge_daily_returns.csv", index_col=0, parse_dates=True)
ret = h["buy_hold"].values
rf_daily = 1.04 ** (1 / 252) - 1
sig = np.isclose(h["anomaly_hedged"].values, rf_daily) & ~np.isclose(ret, rf_daily)

vol_daily = pd.Series(ret).abs().rolling(20).std().bfill().values
reg, _ = RollingQuantileRegimeDetector(window=252).detect(vol_daily)
reg = np.asarray(reg)[:len(ret)]
hedge = {0: 0.40, 1: 0.70, 2: 1.00}


def overlay(regimes):
    hp = np.array([hedge.get(int(r), 0.70) for r in regimes])
    return np.where(sig, (1 - hp) * ret + hp * rf_daily, ret)


same_day = overlay(reg)
lagged = overlay(np.r_[reg[0], reg[:-1]])
print("reproduces saved overlay:", np.allclose(same_day, h["regime_aware_hedged"].values),
      " max abs diff", np.abs(same_day - h["regime_aware_hedged"].values).max())
print("days where sizing changes:", int((np.r_[reg[0], reg[:-1]] != reg)[sig].sum()), "of", int(sig.sum()), "hedged days")

ff5 = next(p for p in [Path("artifacts/ff_data/FF5_daily.csv"), Path("data/ff/FF5_daily.csv")] + list(Path(".").rglob("FF5_daily.csv")) if p.exists())
mom = ff5.with_name("MOM_daily.csv")
out = {}
for name, r in [("same_day_sizing", same_day), ("lagged_sizing", lagged)]:
    res = run_factor_alpha(strategy_returns=r, dates=h.index, ff5_csv=str(ff5), mom_csv=str(mom), hac_lags=5)
    w = np.cumprod(1 + r)
    out[name] = dict(beta_mkt=res["betas"]["MktRF"], alpha_ann=res["alpha_annualized"],
                     alpha_p=res["alpha_pvalue"], r2_adj=res["r2_adj"], n=res["n_obs"],
                     ann_ret=w[-1] ** (252 / len(r)) - 1)
    print(name, {k: round(v, 4) for k, v in out[name].items()})
(ART / "ff_lagged_sizing_check.json").write_text(json.dumps(out, indent=2))
