"""
Round-2 revision: out-of-sample VaR / ES backtests and FRTB capital.

Regimes are the ex-ante regimes of r2_common (known at the close of t-1).
Every model forecasts day-t risk from data up to t-1; parameters are
re-estimated every 21 trading days on an expanding window.

Models
  HS        unconditional historical simulation (expanding)
  RC-EVT    regime-conditional EVT: GPD peaks-over-threshold on the losses of
            past days in the current regime
  GARCH-t   GARCH(1,1) with Student-t innovations
  GARCH-FHS GARCH(1,1) filtered historical simulation (Barone-Adesi et al.)
  GARCH-EVT GARCH(1,1) with GPD tails of the standardised residuals (McNeil-Frey)
Backtests
  VaR99 : Kupiec on the full series and on each regime subsample (count test);
          Christoffersen independence / conditional coverage only on the
          contiguous full series.
  ES97.5: Acerbi-Szekely (2014) Z2 statistic, 5% critical value -0.70.
Capital
  FRTB IMA-style ES charge: ES97.5 scaled to 10 days, max(ES_t, m_c * avg60),
  m_c = 1.5 + plus factor from VaR99 exceptions over the past 250 days.
"""
import os
import json
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import genpareto, chi2, t as student_t

from src.evaluation import r2_common as C

REFIT = 21
EVAL_START = {"SP500": "2016-01-01", "NASDAQ": "2016-01-01", "Russell2000": "2016-01-01", "Bitcoin": "2017-01-01"}
EVAL_END = "2025-12-31"
MODELS = ["HS", "RC-EVT", "GARCH-t", "GARCH-FHS", "GARCH-EVT"]
Z2_CRIT = -0.70
PLUS = {5: 0.20, 6: 0.26, 7: 0.33, 8: 0.38, 9: 0.42}


def gpd_var_es(loss, ps, uq=0.90):
    u = np.quantile(loss, uq)
    exc = loss[loss > u] - u
    xi, _, beta = genpareto.fit(exc, floc=0)
    n, nu = len(loss), len(exc)
    out = {}
    for p in ps:
        var = u + beta / xi * (((n / nu) * (1 - p)) ** (-xi) - 1) if abs(xi) > 1e-6 else u - beta * np.log((n / nu) * (1 - p))
        es = (var + beta - xi * u) / (1 - xi)
        out[p] = (var, es)
    return out, xi, nu


def hs_var_es(loss, ps):
    out = {}
    for p in ps:
        v = np.quantile(loss, p)
        out[p] = (v, loss[loss >= v].mean())
    return out


def kupiec(v, n, p=0.01):
    x = int(v)
    if n == 0:
        return np.nan
    ph = x / n
    l0 = (n - x) * np.log(1 - p) + x * np.log(p)
    l1 = ((n - x) * np.log(1 - ph) if x < n else 0) + (x * np.log(ph) if x > 0 else 0)
    return float(chi2.sf(-2 * (l0 - l1), 1))


def christoffersen(I):
    I = np.asarray(I, int)
    a, b = I[:-1], I[1:]
    n00, n01 = int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n10, n11 = int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum())
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    def ll(q, k0, k1):
        return (k0 * np.log(1 - q) if k0 else 0) + (k1 * np.log(q) if k1 else 0)
    lr_ind = -2 * (ll(p, n00 + n10, n01 + n11) - ll(p01, n00, n01) - ll(p11, n10, n11))
    p_ind = float(chi2.sf(lr_ind, 1))
    x, n = I.sum(), len(I)
    ph = x / n
    lr_uc = -2 * ((n - x) * np.log(0.99) + x * np.log(0.01) - ((n - x) * np.log(1 - ph) if x < n else 0) - (x * np.log(ph) if x else 0))
    p_cc = float(chi2.sf(lr_uc + lr_ind, 2))
    return p_ind, p_cc


def z2(loss, var975, es975):
    I = loss > var975
    return float(1 - np.sum(loss * I / es975) / (0.025 * len(loss)))


def forecast_market(market):
    rs = C.regime_series(market)
    r = rs["r"].values
    reg = rs["regime"].values
    idx = rs.index
    loss = -r
    t0 = idx.searchsorted(pd.Timestamp(EVAL_START[market]))
    t1 = idx.searchsorted(pd.Timestamp(EVAL_END), side="right")
    ps = [0.975, 0.99]
    F = {m: {p: np.full(len(idx), np.nan) for p in ["v99", "v975", "e975"]} for m in MODELS}
    xi_log = []
    for start in range(t0, t1, REFIT):
        end = min(start + REFIT, t1)
        hist = np.arange(1, start)
        L = loss[hist]
        L = L[np.isfinite(L)]
        # HS
        hs = hs_var_es(L, ps)
        for t in range(start, end):
            F["HS"]["v99"][t], F["HS"]["v975"][t], F["HS"]["e975"][t] = hs[0.99][0], hs[0.975][0], hs[0.975][1]
        # RC-EVT
        rk = {}
        for k in range(3):
            Lk = loss[hist][(reg[hist] == k) & np.isfinite(loss[hist])]
            res, xi, nu = gpd_var_es(Lk, ps)
            rk[k] = res
            xi_log.append((str(idx[start].date()), k, float(xi), int(nu)))
        for t in range(start, end):
            k = reg[t]
            F["RC-EVT"]["v99"][t], F["RC-EVT"]["v975"][t], F["RC-EVT"]["e975"][t] = rk[k][0.99][0], rk[k][0.975][0], rk[k][0.975][1]
        # GARCH family (Student-t QMLE), filter forward with fixed parameters
        x = 100 * r[1:start]
        am = arch_model(x, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False)
        fit = am.fit(disp="off")
        mu, om, a, b, nu_t = (fit.params[k] for k in ["mu", "omega", "alpha[1]", "beta[1]", "nu"])
        xx = 100 * r
        s2 = np.empty(end)
        s2[1] = np.var(x)
        for t in range(2, end):
            s2[t] = om + a * (xx[t - 1] - mu) ** 2 + b * s2[t - 1]
        sig = np.sqrt(s2)
        z = ((xx[1:start] - mu) / sig[1:start])[50:]
        zl = -z                                   # standardised losses
        sc = np.sqrt((nu_t - 2) / nu_t)
        qs = {p: student_t.ppf(p, nu_t) * sc for p in ps}
        es_t = sc * (student_t.pdf(student_t.ppf(0.975, nu_t), nu_t) / 0.025) * (nu_t + student_t.ppf(0.975, nu_t) ** 2) / (nu_t - 1)
        fhs = hs_var_es(zl, ps)
        evt, _, _ = gpd_var_es(zl, ps)
        for t in range(start, end):
            m_, s_ = mu / 100, sig[t] / 100
            F["GARCH-t"]["v99"][t], F["GARCH-t"]["v975"][t], F["GARCH-t"]["e975"][t] = -m_ + s_ * qs[0.99], -m_ + s_ * qs[0.975], -m_ + s_ * es_t
            F["GARCH-FHS"]["v99"][t], F["GARCH-FHS"]["v975"][t], F["GARCH-FHS"]["e975"][t] = -m_ + s_ * fhs[0.99][0], -m_ + s_ * fhs[0.975][0], -m_ + s_ * fhs[0.975][1]
            F["GARCH-EVT"]["v99"][t], F["GARCH-EVT"]["v975"][t], F["GARCH-EVT"]["e975"][t] = -m_ + s_ * evt[0.99][0], -m_ + s_ * evt[0.975][0], -m_ + s_ * evt[0.975][1]
    sel = np.arange(t0, t1)
    out = pd.DataFrame({"r": r[sel], "loss": loss[sel], "regime": reg[sel]}, index=idx[sel])
    for m in MODELS:
        for p in ["v99", "v975", "e975"]:
            out[f"{m}|{p}"] = F[m][p][sel]
    out.to_csv(os.path.join(C.ART, f"risk_forecasts_{market}.csv"))
    return out, xi_log


def frtb_capital(df, m):
    es10 = df[f"{m}|e975"].values * np.sqrt(10)
    viol = (df["loss"].values > df[f"{m}|v99"].values).astype(int)
    n = len(df)
    cap = np.full(n, np.nan)
    for t in range(60, n):
        exc = viol[max(0, t - 250):t].sum()
        mc = 1.5 + (0.5 if exc >= 10 else PLUS.get(exc, 0.0))
        cap[t] = max(es10[t], mc * es10[t - 60:t].mean())
    loss10 = pd.Series(df["loss"].values[::-1]).rolling(10).sum().values[::-1]     # loss over t..t+9
    return cap, loss10


def evaluate(market, df):
    res = {"n": int(len(df)), "models": {}, "regime_n": {}}
    reg = df["regime"].values
    for k in range(3):
        res["regime_n"][k] = int((reg == k).sum())
    for m in MODELS:
        L = df["loss"].values
        I = (L > df[f"{m}|v99"].values).astype(int)
        p_ind, p_cc = christoffersen(I)
        rec = {"V": int(I.sum()), "E": 0.01 * len(I), "p_uc": kupiec(I.sum(), len(I)), "p_ind": p_ind, "p_cc": p_cc,
               "z2": z2(L, df[f"{m}|v975"].values, df[f"{m}|e975"].values), "regime": {}}
        cap, loss10 = frtb_capital(df, m)
        rec["cap_mean"] = float(np.nanmean(cap))
        ok = np.isfinite(cap) & np.isfinite(loss10)
        rec["cap_breach"] = int((loss10[ok] > cap[ok]).sum())
        for k in range(3):
            s = reg == k
            rec["regime"][k] = {"V": int(I[s].sum()), "E": 0.01 * s.sum(), "p_uc": kupiec(I[s].sum(), s.sum()),
                                "z2": z2(L[s], df[f"{m}|v975"].values[s], df[f"{m}|e975"].values[s]),
                                "v99_mean": float(np.nanmean(df[f"{m}|v99"].values[s])),
                                "cap_mean": float(np.nanmean(cap[s & np.isfinite(cap)])),
                                "cap_breach": int((loss10[s & ok] > cap[s & ok]).sum())}
        res["models"][m] = rec
    # descriptive tail statistics on the evaluation days, by ex-ante regime
    desc = {}
    for k in [-1, 0, 1, 2]:
        s = np.ones(len(df), bool) if k < 0 else reg == k
        rr = df["r"].values[s]
        q5, q1 = np.quantile(rr, 0.05), np.quantile(rr, 0.01)
        L = -rr
        evt, xi, nu = gpd_var_es(L, [0.975, 0.99])
        desc["All" if k < 0 else k] = {"N": int(s.sum()), "var95": q5, "cvar95": rr[rr <= q5].mean(), "var99": q1,
                                        "cvar99": rr[rr <= q1].mean(), "xi": float(xi), "Nu": int(nu),
                                        "evt_var99": evt[0.99][0], "evt_es975": evt[0.975][1]}
    res["desc"] = desc
    return res


def main():
    allres = {}
    for m in C.MARKETS:
        df, xi_log = forecast_market(m)
        allres[m] = evaluate(m, df)
        print(f"\n== {m} n={allres[m]['n']} regimes {allres[m]['regime_n']}")
        for mod, v in allres[m]["models"].items():
            rg = " | ".join(f"{v['regime'][k]['V']}/{v['regime'][k]['E']:.1f} p={v['regime'][k]['p_uc']:.3f}" for k in range(3))
            print(f"  {mod:10s} V={v['V']}/{v['E']:.1f} puc={v['p_uc']:.3f} pind={v['p_ind']:.3f} pcc={v['p_cc']:.3f} Z2={v['z2']:+.2f} "
                  f"cap={100*v['cap_mean']:.2f}% br={v['cap_breach']} || {rg}", flush=True)
        for k, d in allres[m]["desc"].items():
            print("   desc", k, {a: round(b, 4) if isinstance(b, float) else b for a, b in d.items()})
    json.dump(allres, open(os.path.join(C.ART, "risk.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
