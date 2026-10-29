"""
Round-2 revision: economic backtest of the detectors on the same OOS scores.

  * equity leg: S&P 500 total-return index (^SP500TR), simple daily returns
  * cash leg  : 13-week Treasury-bill yield (^IRX), accrued daily
  * alerts    : a detector alerts on day t when its score (information up to the
                close of t-1) exceeds the 95th percentile of its own scores over
                the previous 252 out-of-sample days (causal threshold; budget 5%)
  * hedge     : after an alert the portfolio holds cash for 5 days starting with
                r_t (trade at the close of t-1); 'lag +1' starts with r_{t+1}
  * costs     : per one-way switch between equity and cash
Comparisons at an equal alert budget replace the hedging-efficiency ratio.
A random-timing benchmark places the same number of 5-day hedges at random dates.
Sharpe differences use the Ledoit-Wolf (2008) studentised circular block bootstrap.
"""
import os
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.evaluation import r2_common as C
from src.evaluation.factor_alpha import load_ff_factors

HOLD = 5
BUDGET = 0.05
TRAIL = 252
MIN_TRAIL = 60
DETS = ["GARCH-FHS", "Z-score rule", "Global GBM", "GBM + regime", "Global ensemble", "RC-MoE"]
B = 2000


def load_returns():
    tr = pd.read_csv(os.path.join(C.ART, "ohlcv_SP500TR.csv"), parse_dates=["Date"], index_col="Date")["Close"]
    ret = tr.pct_change()
    irx = pd.read_csv(os.path.join(C.ART, "irx.csv"), parse_dates=["Date"], index_col="Date")["IRX"]
    irx = irx.reindex(ret.index).ffill()
    rf = (irx.clip(lower=0) / 100.0) / 252.0
    return ret, rf


def causal_alerts(score):
    s = pd.Series(score)
    thr = s.shift(1).rolling(TRAIL, min_periods=MIN_TRAIL).quantile(1 - BUDGET)
    return (s > thr).astype(int).values


def hedge_mask(alerts, lag=0):
    n = len(alerts)
    m = np.zeros(n, dtype=int)
    for t in np.where(alerts == 1)[0]:
        m[t + lag: min(t + lag + HOLD, n)] = 1
    return m


def strat_returns(eq, rf, mask, cost_bp=0.0):
    r = np.where(mask == 1, rf, eq)
    sw = np.abs(np.diff(np.r_[0, mask]))
    r = r - sw * cost_bp / 1e4
    return r, int(sw.sum())


def stats(r, rf):
    ex = r - rf
    ann = (np.prod(1 + r)) ** (252 / len(r)) - 1
    vol = r.std(ddof=1) * np.sqrt(252)
    sh = ex.mean() / ex.std(ddof=1) * np.sqrt(252)
    dn = ex[ex < 0]
    so = ex.mean() / np.sqrt((dn ** 2).sum() / len(ex)) * np.sqrt(252)
    w = np.cumprod(1 + r)
    mdd = (1 - w / np.maximum.accumulate(w)).max()
    q = np.quantile(r, 0.025)
    es = -r[r <= q].mean()
    return {"ann": ann, "vol": vol, "sharpe": sh, "sortino": so, "mdd": mdd, "es975": es}


def _sr_se(x, y, lags=5):
    """Delta-method HAC standard error of SR(x) - SR(y) (Ledoit-Wolf 2008)."""
    n = len(x)
    m1, m2 = x.mean(), y.mean()
    g1, g2 = (x ** 2).mean(), (y ** 2).mean()
    V = np.column_stack([x - m1, y - m2, x ** 2 - g1, y ** 2 - g2])
    S = V.T @ V / n
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = V[L:].T @ V[:-L] / n
        S += w * (G + G.T)
    grad = np.array([g1 / (g1 - m1 ** 2) ** 1.5, -g2 / (g2 - m2 ** 2) ** 1.5,
                     -0.5 * m1 / (g1 - m1 ** 2) ** 1.5, 0.5 * m2 / (g2 - m2 ** 2) ** 1.5])
    return np.sqrt(grad @ S @ grad / n)


def lw_test(a, b, seed=0, block=10):
    """Studentised circular block bootstrap test of H0: SR_a = SR_b."""
    rng = np.random.default_rng(seed)
    sr = lambda z: z.mean() / z.std(ddof=1)
    d0 = sr(a) - sr(b)
    se0 = _sr_se(a, b)
    n = len(a)
    tstats = []
    for _ in range(B):
        st = rng.integers(0, n, size=int(np.ceil(n / block)))
        idx = ((st[:, None] + np.arange(block)[None, :]).ravel()[:n]) % n
        aa, bb = a[idx], b[idx]
        tstats.append(abs((sr(aa) - sr(bb)) - d0) / _sr_se(aa, bb))
    p = float(np.mean(np.array(tstats) >= abs(d0) / se0))
    k = np.sqrt(252)
    return {"d_sharpe": d0 * k, "ci": [(d0 - 1.96 * se0) * k, (d0 + 1.96 * se0) * k], "p": p}


def random_timing(eq, rf, n_alerts, seed=0, cost_bp=5.0):
    rng = np.random.default_rng(seed)
    n = len(eq)
    out = []
    for _ in range(B):
        a = np.zeros(n, dtype=int)
        a[rng.choice(n - HOLD, size=n_alerts, replace=False)] = 1
        r, _ = strat_returns(eq, rf, hedge_mask(a), cost_bp)
        s = stats(r, rf)
        out.append((s["sharpe"], s["mdd"], s["es975"], hedge_mask(a).mean()))
    return np.array(out)


def ff_regression(r, rf, idx):
    ff = load_ff_factors("artifacts/data/ff_factors/FF5_daily.csv", "artifacts/data/ff_factors/MOM_daily.csv")
    df = pd.DataFrame({"ex": r - rf}, index=idx).join(ff, how="inner")
    Xf = sm.add_constant(df[["MktRF", "SMB", "HML", "RMW", "CMA", "MOM"]])
    res = sm.OLS(df["ex"], Xf).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    out = {k: {"est": float(res.params[k]), "p": float(res.pvalues[k])} for k in res.params.index}
    out["alpha_ann"] = float((1 + res.params["const"]) ** 252 - 1)
    out["adj_r2"] = float(res.rsquared_adj)
    out["n"] = int(res.nobs)
    return out


def main():
    ret, rf_all = load_returns()
    sc = pd.read_csv(os.path.join(C.ART, "oos_scores_SP500.csv"), parse_dates=["date"], index_col="date")
    idx = sc.index.intersection(ret.index)
    sc = sc.loc[idx]
    eq = ret.loc[idx].values
    rf = rf_all.loc[idx].values
    res = {"period": [str(idx.min().date()), str(idx.max().date())], "n": int(len(idx)),
           "rf_mean_ann": float(rf.mean() * 252), "strategies": {}, "lag_cost": {}, "periods": {}}
    bh = stats(eq, rf)
    res["strategies"]["Buy-and-hold"] = {**bh, "hedged": 0.0, "switches": 0, "alerts": 0}
    daily = {"Buy-and-hold": eq}
    masks = {}
    for dname in DETS:
        al = causal_alerts(sc[dname].values)
        m = hedge_mask(al)
        r, sw = strat_returns(eq, rf, m, 5.0)
        s = stats(r, rf)
        s.update({"hedged": float(m.mean()), "switches": sw, "alerts": int(al.sum()),
                  "lw_vs_bh": lw_test(r - rf, eq - rf)})
        rt = random_timing(eq, rf, int(al.sum()), seed=1)
        s["random"] = {"sharpe_med": float(np.median(rt[:, 0])), "mdd_med": float(np.median(rt[:, 1])),
                       "p_sharpe": float(np.mean(rt[:, 0] >= s["sharpe"])), "p_mdd": float(np.mean(rt[:, 1] <= s["mdd"])),
                       "p_es": float(np.mean(rt[:, 2] <= s["es975"]))}
        res["strategies"][dname] = s
        daily[dname] = r
        masks[dname] = m
        print(f"{dname:16s} ann {s['ann']:.4f} sh {s['sharpe']:.3f} mdd {s['mdd']:.3f} es {s['es975']:.4f} hedged {s['hedged']:.3f} "
              f"alerts {s['alerts']} sw {sw}  LW dSh {s['lw_vs_bh']['d_sharpe']:+.3f} p={s['lw_vs_bh']['p']:.3f}  "
              f"rand p_sh {s['random']['p_sharpe']:.3f} p_mdd {s['random']['p_mdd']:.3f}", flush=True)
    print("B&H", bh)
    # lag / cost grid for RC-MoE and GARCH-FHS
    for dname in ["RC-MoE", "GARCH-FHS"]:
        al = causal_alerts(sc[dname].values)
        for lag in [0, 1]:
            for c in [0, 5, 10, 20]:
                r, sw = strat_returns(eq, rf, hedge_mask(al, lag), c)
                s = stats(r, rf)
                s.update({"hedged": float(hedge_mask(al, lag).mean()), "switches": sw})
                res["lag_cost"][f"{dname}|{lag}|{c}"] = s
    # periods
    for nm, (a, b) in {"2016-2025": ("2016-01-01", "2025-12-31"), "2022-2025": ("2022-01-01", "2025-12-31")}.items():
        sel = (idx >= a) & (idx <= b)
        res["periods"][nm] = {k: stats(v[sel], rf[sel]) for k, v in daily.items()}
    # factor exposure of the same overlays (5 bp, no extra lag)
    res["ff"] = {k: ff_regression(daily[k], rf, idx) for k in ["RC-MoE", "GARCH-FHS"]}
    for k, v in res["ff"].items():
        print(k, "beta", round(v["MktRF"]["est"], 3), "alpha_ann", round(v["alpha_ann"], 4), "p", round(v["const"]["p"], 3), "R2", round(v["adj_r2"], 3))
    pd.DataFrame({**{k: v for k, v in daily.items()}, "rf": rf, **{f"mask_{k}": v for k, v in masks.items()}}, index=idx).to_csv(
        os.path.join(C.ART, "backtest_daily.csv"))
    json.dump(res, open(os.path.join(C.ART, "backtest.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
