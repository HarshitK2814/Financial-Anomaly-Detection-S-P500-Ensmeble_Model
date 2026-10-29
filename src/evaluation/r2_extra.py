"""
Round-2 revision: additional experiments within the reviewer's requests
(conditional-volatility baselines, equal budgets, ES/FRTB with conditional
benchmarks, backtest with lag / costs).  All variants are fixed in advance and
all are reported.

Detection (same folds, labels, purge and ex-ante information as r2_detection)
  GBM + GARCH          global LightGBM on the 24 factors + 2 GARCH features
  GBM + regime + GARCH as above with the regime one-hot
  RC-MoE + GARCH       RC-MoE whose experts also see the 2 GARCH features
  RC-GARCH             GARCH-FHS exceedance probability recalibrated by regime
                       (logistic model on logit p, regime dummies and their
                       interactions, fitted on the training block)
  GARCH features: log(sigma_t / tau_t) and logit of the GARCH-FHS exceedance
  probability, with GARCH parameters estimated on the training block only.

Risk (same refits and backtests as r2_risk)
  RC-GARCH-FHS  GARCH(1,1)-t filter; empirical quantiles of the standardised
                losses of past days in the current regime
  RC-GARCH-EVT  as above with GPD tails fitted per regime

Backtest (same rule as r2_backtest) for the new detectors, and a robustness
grid for every detector: holding period 1 / 5 days x alert budget 5% / 10%.

Run:  python -m src.evaluation.r2_extra detect | metrics | risk | backtest
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.linear_model import LogisticRegression

from src.evaluation import r2_common as C
from src.evaluation import r2_metrics as M
from src.evaluation import r2_risk as R
from src.evaluation import r2_backtest as BT
from src.evaluation.r2_detection import folds
from src.evaluation.r2_models import RCMoE, make_lgb, garch_sigma, exceed_prob_fhs

NEW_DET = ["GBM + GARCH", "GBM + regime + GARCH", "RC-MoE + GARCH", "RC-GARCH"]
NEW_RISK = ["RC-GARCH-FHS", "RC-GARCH-EVT"]


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------- detection
def detect_market(market):
    d = pd.read_pickle(os.path.join(C.ART, f"dataset_{market}.pkl"))
    full = C.load_ohlcv(C.MARKETS[market])
    r_full = np.r_[0.0, np.diff(np.log(full["Close"].values))]
    pos_full = full.index.get_indexer(d.index)
    X = d[C.feature_cols(d)].values
    y = d["y"].values.astype(int)
    reg = d["regime"].values.astype(int)
    conf = d["conf"].values
    probs = d[[f"p{j}" for j in range(3)]].values
    tau = d["tau"].values
    onehot = np.eye(3)[reg]
    rows = []
    for k, tr, te in folds(d.index, y):
        mu, sig, z = garch_sigma(r_full, pos_full[tr[-1]] + 1)
        rows_all = np.r_[tr, te]
        s_all = sig[pos_full[rows_all]]
        G = np.full((len(d), 2), np.nan)
        G[rows_all, 0] = np.log(s_all / tau[rows_all])
        G[rows_all, 1] = _logit(exceed_prob_fhs(mu, s_all, tau[rows_all], z))
        XG = np.hstack([X, G])
        sc = {}
        sc["GBM + GARCH"] = make_lgb().fit(XG[tr], y[tr]).predict_proba(XG[te])[:, 1]
        XGR = np.hstack([XG, onehot])
        sc["GBM + regime + GARCH"] = make_lgb().fit(XGR[tr], y[tr]).predict_proba(XGR[te])[:, 1]
        moe = RCMoE().fit(XG[tr], y[tr], reg[tr])
        sc["RC-MoE + GARCH"] = moe.predict(XG[te], reg[te], conf[te], probs[te])
        Z = np.column_stack([G[:, 1], onehot[:, 1:], G[:, 1:2] * onehot[:, 1:]])
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(Z[tr], y[tr])
        sc["RC-GARCH"] = lr.predict_proba(Z[te])[:, 1]
        rows.append(pd.DataFrame(sc, index=d.index[te]))
        print(f"  {market} fold {k} done", flush=True)
    out = pd.concat(rows)
    out.index.name = "date"
    out.to_csv(os.path.join(C.ART, f"oos_extra_{market}.csv"))


def load_scores(m):
    df = pd.read_csv(os.path.join(C.ART, f"oos_scores_{m}.csv"), parse_dates=["date"], index_col="date")
    ex = pd.read_csv(os.path.join(C.ART, f"oos_extra_{m}.csv"), parse_dates=["date"], index_col="date")
    deep = os.path.join(C.ART, f"oos_deep_{m}.csv")
    df = df.join(ex, how="left")
    if os.path.exists(deep):
        df = df.join(pd.read_csv(deep, parse_dates=["date"], index_col="date"), how="left")
    return df


def metrics():
    dets_all = M.DETECTORS + NEW_DET
    res, pooled = {}, []
    for m in C.MARKETS:
        df = load_scores(m)
        dets = [c for c in dets_all if c in df.columns]
        res[m] = {}
        for ref in ["RC-MoE + GARCH", "RC-GARCH"]:
            M.REF = ref
            res[m][ref] = M.evaluate(df, dets)
        d2 = df.copy()
        d2["fold"] = m + d2["fold"].astype(str)
        pooled.append(d2)
    P = pd.concat(pooled)
    dets = [c for c in dets_all if c in P.columns and P[c].notna().all()]
    res["Pooled"] = {}
    for ref in ["RC-MoE + GARCH", "RC-GARCH"]:
        M.REF = ref
        res["Pooled"][ref] = M.evaluate(P, dets)
    json.dump(res, open(os.path.join(C.ART, "extra_detection.json"), "w"), indent=2, default=float)
    for m, rr in res.items():
        r = rr["RC-MoE + GARCH"]
        print(f"\n== {m}: n={r['n']} pos={r['pos']}")
        for c, v in r["det"].items():
            print(f"  {c:22s} AUPRC {v['auprc']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}]  P@5 {v['b5']['prec']:.3f} R@5 {v['b5']['rec']:.3f} F1@10 {v['b10']['f1']:.3f}")
        for ref in ["RC-MoE + GARCH", "RC-GARCH"]:
            for c in ["GARCH-FHS", "RC-MoE", "GBM + regime", "GBM + GARCH", "GBM + regime + GARCH"]:
                v = rr[ref]["paired"].get(c)
                if v:
                    print(f"  {ref} - {c:20s} d {v['d_auprc']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] p={v['p_boot']:.3f} DM t={v['dm_t']:+.2f} p={v['dm_p']:.3f}")


# ---------------------------------------------------------------- risk
def risk_market(market):
    rs = C.regime_series(market)
    r, reg, idx = rs["r"].values, rs["regime"].values, rs.index
    loss = -r
    t0 = idx.searchsorted(pd.Timestamp(R.EVAL_START[market]))
    t1 = idx.searchsorted(pd.Timestamp(R.EVAL_END), side="right")
    ps = [0.975, 0.99]
    F = {m: {p: np.full(len(idx), np.nan) for p in ["v99", "v975", "e975"]} for m in NEW_RISK}
    for start in range(t0, t1, R.REFIT):
        end = min(start + R.REFIT, t1)
        x = 100 * r[1:start]
        fit = arch_model(x, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(disp="off")
        mu, om, a, b = (fit.params[k] for k in ["mu", "omega", "alpha[1]", "beta[1]"])
        xx = 100 * r
        s2 = np.empty(end)
        s2[1] = np.var(x)
        for t in range(2, end):
            s2[t] = om + a * (xx[t - 1] - mu) ** 2 + b * s2[t - 1]
        sig = np.sqrt(s2)
        zl = -((xx[1:start] - mu) / sig[1:start])[50:]
        rg = reg[1:start][50:]
        q = {}
        for k in range(3):
            zk = zl[(rg == k) & np.isfinite(zl)]
            q[k] = (R.hs_var_es(zk, ps), R.gpd_var_es(zk, ps)[0])
        for t in range(start, end):
            k = reg[t]
            m_, s_ = mu / 100, sig[t] / 100
            for name, qq in (("RC-GARCH-FHS", q[k][0]), ("RC-GARCH-EVT", q[k][1])):
                F[name]["v99"][t] = -m_ + s_ * qq[0.99][0]
                F[name]["v975"][t] = -m_ + s_ * qq[0.975][0]
                F[name]["e975"][t] = -m_ + s_ * qq[0.975][1]
    sel = np.arange(t0, t1)
    old = pd.read_csv(os.path.join(C.ART, f"risk_forecasts_{market}.csv"), parse_dates=[0], index_col=0)
    for m in NEW_RISK:
        for p in ["v99", "v975", "e975"]:
            old[f"{m}|{p}"] = F[m][p][sel]
    old.to_csv(os.path.join(C.ART, f"risk_forecasts_extra_{market}.csv"))
    return old


def risk():
    R.MODELS = R.MODELS + NEW_RISK
    allres = {}
    for m in C.MARKETS:
        df = risk_market(m)
        allres[m] = R.evaluate(m, df)
        print(f"\n== {m} regimes {allres[m]['regime_n']}")
        for mod, v in allres[m]["models"].items():
            rg = " | ".join(f"{v['regime'][k]['V']}/{v['regime'][k]['E']:.1f} p={v['regime'][k]['p_uc']:.3f} z2={v['regime'][k]['z2']:+.2f} cap={100*v['regime'][k]['cap_mean']:.1f}% br={v['regime'][k]['cap_breach']}" for k in range(3))
            print(f"  {mod:12s} V={v['V']}/{v['E']:.1f} puc={v['p_uc']:.3f} pind={v['p_ind']:.3f} pcc={v['p_cc']:.3f} Z2={v['z2']:+.2f} "
                  f"cap={100*v['cap_mean']:.2f}% br={v['cap_breach']} || {rg}", flush=True)
    json.dump(allres, open(os.path.join(C.ART, "extra_risk.json"), "w"), indent=2, default=float)


# ---------------------------------------------------------------- backtest
def backtest():
    ret, rf_all = BT.load_returns()
    sc = load_scores("SP500")
    idx = sc.index.intersection(ret.index)
    sc = sc.loc[idx]
    eq, rf = ret.loc[idx].values, rf_all.loc[idx].values
    bh = BT.stats(eq, rf)
    res = {"Buy-and-hold": bh, "main": {}, "grid": {}, "ff": {}}
    print(f"B&H sharpe {bh['sharpe']:.3f} mdd {bh['mdd']:.3f} vol {bh['vol']:.3f}")
    for dname in NEW_DET:
        al = BT.causal_alerts(sc[dname].values)
        m = BT.hedge_mask(al)
        r, sw = BT.strat_returns(eq, rf, m, 5.0)
        s = BT.stats(r, rf)
        s.update({"hedged": float(m.mean()), "switches": sw, "alerts": int(al.sum()), "lw_vs_bh": BT.lw_test(r - rf, eq - rf)})
        rt = BT.random_timing(eq, rf, int(al.sum()), seed=1)
        s["random"] = {"p_sharpe": float(np.mean(rt[:, 0] >= s["sharpe"])), "p_mdd": float(np.mean(rt[:, 1] <= s["mdd"])),
                       "p_es": float(np.mean(rt[:, 2] <= s["es975"]))}
        res["main"][dname] = s
        res["ff"][dname] = BT.ff_regression(r, rf, idx)
        f = res["ff"][dname]
        print(f"{dname:22s} ann {s['ann']:.4f} vol {s['vol']:.3f} sh {s['sharpe']:.3f} mdd {s['mdd']:.3f} es {s['es975']:.4f} hedged {s['hedged']:.3f} "
              f"LW d {s['lw_vs_bh']['d_sharpe']:+.3f} p={s['lw_vs_bh']['p']:.3f} rand p {s['random']['p_sharpe']:.3f} | beta {f['MktRF']['est']:.3f} "
              f"alpha {f['alpha_ann']:+.4f} p={f['const']['p']:.3f}", flush=True)
    for dname in BT.DETS + NEW_DET:
        for hold in [1, 5]:
            for bud in [0.05, 0.10]:
                BT.HOLD, BT.BUDGET = hold, bud
                al = BT.causal_alerts(sc[dname].values)
                m = BT.hedge_mask(al)
                r, sw = BT.strat_returns(eq, rf, m, 5.0)
                s = BT.stats(r, rf)
                s.update({"hedged": float(m.mean()), "lw_vs_bh": BT.lw_test(r - rf, eq - rf)})
                res["grid"][f"{dname}|{hold}|{bud}"] = s
                print(f"  grid {dname:22s} hold {hold} budget {bud:.2f}: sh {s['sharpe']:.3f} mdd {s['mdd']:.3f} hedged {s['hedged']:.3f} p={s['lw_vs_bh']['p']:.3f}", flush=True)
    BT.HOLD, BT.BUDGET = 5, 0.05
    json.dump(res, open(os.path.join(C.ART, "extra_backtest.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "detect":
        for m in (sys.argv[2:] or list(C.MARKETS)):
            detect_market(m)
    elif what == "metrics":
        metrics()
    elif what == "risk":
        risk()
    elif what == "backtest":
        backtest()
