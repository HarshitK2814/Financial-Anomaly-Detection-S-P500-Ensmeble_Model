"""
Execution-lag backtest (Reviewer 1, Query 6).

Uses the dated out-of-sample S&P 500 predictions of the canonical walk-forward pipeline
(artifacts/sp500_oos_dated_predictions.csv, multimarket_eventwise_canonical.py).
Convention: the signal for day t is known after the close of day t; with lag k the
position moves to cash for returns r_(t+k) ... r_(t+k+4) (5-day hold, re-signals extend).
k=1: trade at close t (first hedged return r_(t+1)); k=2: one extra day of delay.
Costs: charged per one-way switch (in or out of cash). Cash earns 4% p.a.

Run from model root:  python src/evaluation/execution_lag_backtest.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path("artifacts")
RF = 0.04 / 252
HOLD = 5

pred = pd.read_csv(ART / "sp500_oos_dated_predictions.csv", parse_dates=["date"]).set_index("date")
ret = pd.read_csv(ART / "sp500_full_daily_returns.csv", parse_dates=["Date"]).set_index("Date").iloc[:, 0]
ret = ret[(ret.index >= pred.index.min()) & (ret.index <= pred.index.max())]


def backtest(signal, lag, cost_bp):
    s = signal.reindex(ret.index).fillna(0).values
    hedged = np.zeros(len(ret))
    for i in np.where(s == 1)[0]:
        hedged[i + lag:i + lag + HOLD] = 1
    r = np.where(hedged == 1, RF, ret.values)
    switches = np.abs(np.diff(np.r_[0, hedged]))
    r = r - switches * cost_bp / 1e4
    wealth = np.cumprod(1 + r)
    ann = wealth[-1] ** (252 / len(r)) - 1
    return dict(ann_ret=100 * ann,
                sharpe=(ann - 0.04) / (r.std() * np.sqrt(252)),
                sortino=(ann - 0.04) / (r[r < 0].std() * np.sqrt(252)),
                max_dd=100 * np.max(1 - wealth / np.maximum.accumulate(wealth)),
                hedged_pct=100 * hedged.mean(), switches=int(switches.sum()))


rows = [dict(strategy="Buy-and-Hold", lag=0, cost_bp=0,
             **backtest(pd.Series(0, index=ret.index), 0, 0))]
# lag 0 = hedge from the signal day itself (not implementable: the signal is known only after
# the close); reported only to show how much of the benefit depends on same-day hedging
rows.append(dict(strategy="RC-MoE", lag=0, cost_bp=0, **backtest(pred["pred_moe"], 0, 0)))
for col, name in [("pred_moe", "RC-MoE"), ("pred_global", "Global")]:
    for lag in (1, 2):
        for cost in (0, 5, 10, 20):
            rows.append(dict(strategy=name, lag=lag, cost_bp=cost, **backtest(pred[col], lag, cost)))

df = pd.DataFrame(rows)
df.to_csv(ART / "execution_lag_backtest.csv", index=False)
print(f"Period {ret.index.min().date()} .. {ret.index.max().date()}  ({len(ret)} days)")
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


# Circular block bootstrap (b = 20 days, 2,000 draws) of the RC-MoE minus buy-and-hold
# differences in Sharpe ratio and maximum drawdown, for each lag at 5 bp per switch.
def strategy_returns(signal, lag, cost_bp):
    s = signal.reindex(ret.index).fillna(0).values
    hedged = np.zeros(len(ret))
    for i in np.where(s == 1)[0]:
        hedged[i + lag:i + lag + HOLD] = 1
    r = np.where(hedged == 1, RF, ret.values)
    return r - np.abs(np.diff(np.r_[0, hedged])) * cost_bp / 1e4


def sharpe(r):
    ann = np.prod(1 + r) ** (252 / len(r)) - 1
    return (ann - 0.04) / (r.std() * np.sqrt(252))


def max_dd(r):
    w = np.cumprod(1 + r)
    return 100 * np.max(1 - w / np.maximum.accumulate(w))


rng = np.random.default_rng(42)
n, b = len(ret), 20
bh = ret.values
boot = []
for lag in (1, 2):
    s = strategy_returns(pred["pred_moe"], lag, 5)
    d_sh, d_dd = [], []
    for _ in range(2000):
        starts = rng.integers(0, n, int(np.ceil(n / b)))
        idx = (starts[:, None] + np.arange(b)).ravel()[:n] % n
        d_sh.append(sharpe(s[idx]) - sharpe(bh[idx]))
        d_dd.append(max_dd(bh[idx]) - max_dd(s[idx]))
    d_sh, d_dd = np.array(d_sh), np.array(d_dd)
    boot.append(dict(lag=lag, cost_bp=5,
                     sharpe_diff=sharpe(s) - sharpe(bh),
                     sharpe_ci_lo=np.percentile(d_sh, 2.5), sharpe_ci_hi=np.percentile(d_sh, 97.5),
                     p_sharpe=float(np.mean(d_sh <= 0)),
                     dd_reduction=max_dd(bh) - max_dd(s),
                     dd_ci_lo=np.percentile(d_dd, 2.5), dd_ci_hi=np.percentile(d_dd, 97.5),
                     p_dd=float(np.mean(d_dd <= 0))))
bdf = pd.DataFrame(boot)
bdf.to_csv(ART / "execution_lag_bootstrap.csv", index=False)
print("\nBlock bootstrap, RC-MoE minus buy-and-hold (5 bp per switch):")
print(bdf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
