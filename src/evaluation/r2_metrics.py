"""
Round-2 revision: detection metrics on the out-of-sample scores.

  * AUPRC (average precision) with circular block-bootstrap 95% CIs (b = 20 days)
  * precision / recall / F1 at equal alert budgets (top 5% and 10% of days in
    each test block, so every detector raises the same number of alerts)
  * paired tests that respect serial dependence:
      - block-bootstrap distribution of the AUPRC difference (two-sided p)
      - Diebold-Mariano-type test (Newey-West variance) on the daily
        difference in true alerts at the 5% budget
These replace the McNemar test on overlapping windows.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.evaluation import r2_common as C

B = 2000
BLOCK = 20
BUDGETS = (0.05, 0.10)
DETECTORS = ["Vol rule", "Z-score rule", "GARCH-FHS", "GARCH-EVT", "ConvVAE", "Anomaly Transformer",
             "Global RF", "Global GBM", "GBM + regime", "Global ensemble", "RC-MoE"]
REF = "RC-MoE"


def cbb_indices(n, rng, b=BLOCK):
    starts = rng.integers(0, n, size=int(np.ceil(n / b)))
    idx = (starts[:, None] + np.arange(b)[None, :]).ravel()[:n] % n
    return idx


def ap(y, s):
    if y.sum() == 0:
        return np.nan
    return average_precision_score(y, s)


def budget_alerts(df, col, b):
    a = np.zeros(len(df), dtype=int)
    for _, g in df.groupby("fold"):
        k = int(np.ceil(b * len(g)))
        pos = df.index.get_indexer(g.index)
        order = np.argsort(-g[col].values, kind="stable")[:k]
        a[pos[order]] = 1
    return a


def prf(y, a):
    tp = int((a & y).sum())
    p = tp / max(a.sum(), 1)
    r = tp / max(y.sum(), 1)
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f


def newey_west_t(d, lags=None):
    d = np.asarray(d, float)
    n = len(d)
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    m = d.mean()
    e = d - m
    v = e @ e / n
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        v += 2 * w * (e[L:] @ e[:-L]) / n
    se = np.sqrt(max(v, 1e-18) / n)
    from scipy.stats import norm
    t = m / se
    return float(t), float(2 * norm.sf(abs(t)))


def evaluate(df, dets, seed=0):
    df = df.reset_index(drop=True)
    y = df["y"].values.astype(int)
    out = {"n": int(len(df)), "pos": int(y.sum()), "prev": float(y.mean()), "det": {}, "paired": {}}
    rng = np.random.default_rng(seed)
    boots = [cbb_indices(len(df), rng) for _ in range(B)]
    bs = {}
    for c in dets:
        s = df[c].fillna(df[c].min()).values
        a0 = ap(y, s)
        vals = np.array([ap(y[i], s[i]) for i in boots])
        bs[c] = vals
        rec = {"auprc": a0, "ci": [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]}
        for b in BUDGETS:
            al = budget_alerts(df, c, b)
            p, r, f = prf(y, al)
            rec[f"b{int(b*100)}"] = {"prec": p, "rec": r, "f1": f, "alerts": int(al.sum())}
        # per-regime AUPRC and 5%-budget precision / recall
        al = budget_alerts(df, c, 0.05)
        rec["regime"] = {}
        for j in range(3):
            m = df["regime"].values == j
            p, r, f = prf(y[m], al[m])
            rec["regime"][j] = {"auprc": ap(y[m], s[m]), "prec": p, "rec": r, "pos": int(y[m].sum()), "n": int(m.sum()),
                                "alerts": int(al[m].sum())}
        out["det"][c] = rec
    if REF in dets:
        a_ref = budget_alerts(df, REF, 0.05)
        for c in dets:
            if c == REF:
                continue
            diff = bs[REF] - bs[c]
            d0 = out["det"][REF]["auprc"] - out["det"][c]["auprc"]
            p_boot = float(2 * min(np.nanmean(diff <= 0), np.nanmean(diff >= 0)))
            a_c = budget_alerts(df, c, 0.05)
            t, p_dm = newey_west_t((a_ref & y) - (a_c & y))
            out["paired"][c] = {"d_auprc": d0, "ci": [float(np.nanpercentile(diff, 2.5)), float(np.nanpercentile(diff, 97.5))],
                                "p_boot": min(p_boot, 1.0), "dm_t": t, "dm_p": p_dm}
    return out


def main():
    allres, pooled = {}, []
    for m in C.MARKETS:
        f = os.path.join(C.ART, f"oos_scores_{m}.csv")
        df = pd.read_csv(f, parse_dates=["date"], index_col="date")
        deep = os.path.join(C.ART, f"oos_deep_{m}.csv")
        if os.path.exists(deep):
            dd = pd.read_csv(deep, parse_dates=["date"], index_col="date")
            df = df.join(dd, how="left")
        dets = [c for c in DETECTORS if c in df.columns]
        allres[m] = evaluate(df, dets)
        df2 = df.copy()
        df2["fold"] = m + df2["fold"].astype(str)
        pooled.append(df2)
        print(m, allres[m]["pos"], {c: round(v["auprc"], 3) for c, v in allres[m]["det"].items()}, flush=True)
    P = pd.concat(pooled)
    dets = [c for c in DETECTORS if c in P.columns and P[c].notna().all()]
    allres["Pooled"] = evaluate(P, dets)
    json.dump(allres, open(os.path.join(C.ART, "detection_metrics.json"), "w"), indent=2, default=float)
    for m, r in allres.items():
        print(f"\n== {m}: n={r['n']} pos={r['pos']} prev={r['prev']:.3f}")
        for c, v in r["det"].items():
            b5 = v["b5"]
            print(f"  {c:20s} AUPRC {v['auprc']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}]  P@5 {b5['prec']:.3f} R@5 {b5['rec']:.3f} F1@5 {b5['f1']:.3f}")
        for c, v in r["paired"].items():
            print(f"  RC-MoE - {c:18s} dAUPRC {v['d_auprc']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] p={v['p_boot']:.3f}  DM t={v['dm_t']:+.2f} p={v['dm_p']:.3f}")


if __name__ == "__main__":
    main()
