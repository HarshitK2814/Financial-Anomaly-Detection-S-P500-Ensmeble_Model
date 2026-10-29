"""Round-2 revision: LaTeX rows for the risk tables, written from artifacts/r2/risk.json.
Run:  python -m src.utils.make_r2_tables  ->  artifacts/r2/tables_risk.tex"""
import json
import os

from src.evaluation import r2_common as C

R = json.load(open(os.path.join(C.ART, "risk.json")))
MK = {"SP500": "S\\&P 500", "NASDAQ": "NASDAQ", "Russell2000": "Russell 2000", "Bitcoin": "Bitcoin"}
MODELS = ["HS", "RC-EVT", "GARCH-t", "GARCH-FHS", "GARCH-EVT"]


def p(x):
    return "$<$0.001" if x < 0.001 else f"{x:.3f}"


def cov_rows():
    out = []
    for m in C.MARKETS:
        first = True
        for mod in MODELS:
            v = R[m]["models"][mod]
            reg = " & ".join(f"{v['regime'][str(k)]['V']}/{v['regime'][str(k)]['E']:.1f} ({p(v['regime'][str(k)]['p_uc'])})" for k in range(3))
            lab = MK[m] if first else ""
            out.append(f"{lab} & {mod} & {v['V']}/{v['E']:.1f} & {p(v['p_uc'])} & {p(v['p_ind'])} & {p(v['p_cc'])} & ${v['z2']:+.2f}$ & {reg} \\\\")
            first = False
        out.append("\\midrule")
    out[-1] = ""
    return "\n".join(out)


def cap_rows():
    out = []
    for mod in MODELS:
        v = R["SP500"]["models"][mod]
        caps = " & ".join(f"{100*v['regime'][str(k)]['cap_mean']:.1f}\\%" for k in range(3))
        br = "/".join(str(v["regime"][str(k)]["cap_breach"]) for k in range(3))
        out.append(f"{mod} & {caps} & {100*v['cap_mean']:.1f}\\% & {br} \\\\")
    return "\n".join(out)


def mm_rows():
    out = []
    for m in C.MARKETS:
        d, mo = R[m]["desc"], R[m]["models"]
        lo, hi = mo["RC-EVT"]["regime"]["0"]["v99_mean"], mo["RC-EVT"]["regime"]["2"]["v99_mean"]
        g = [100 * (mo["HS"]["regime"][str(k)]["cap_mean"] / mo["RC-EVT"]["regime"][str(k)]["cap_mean"] - 1) for k in (0, 2)]
        kup = " / ".join(p(mo[x]["regime"]["2"]["p_uc"]) for x in ["HS", "RC-EVT", "GARCH-EVT"])
        ind = " / ".join(p(mo[x]["p_ind"]) for x in ["RC-EVT", "GARCH-EVT"])
        out.append(f"{MK[m]} & ${d['2']['xi']:+.3f}$ & ${100*lo:.1f}\\%\\!\\rightarrow\\!{100*hi:.1f}\\%$ & {kup} & {ind} & ${g[0]:+.0f}\\%$ & ${g[1]:+.0f}\\%$ \\\\")
    return "\n".join(out)


def desc_rows():
    d = R["SP500"]["desc"]
    out = []
    for k, lab in [("All", "All days"), ("0", "Low Vol"), ("1", "Med Vol"), ("2", "High Vol")]:
        x = d[k]
        out.append(f"{lab} & ${100*x['var95']:.2f}\\%$ & ${100*x['cvar95']:.2f}\\%$ & ${100*x['var99']:.2f}\\%$ & ${100*x['cvar99']:.2f}\\%$ & {x['N']:,} \\\\".replace(",", "{,}"))
        if k == "All":
            out.append("\\midrule")
    return "\n".join(out)


def evt_rows():
    d = R["SP500"]["desc"]
    return "\n".join(f"{lab} & {d[k]['Nu']} & ${d[k]['xi']:+.3f}$ & ${100*d[k]['evt_var99']:.2f}\\%$ & ${100*d[k]['evt_es975']:.2f}\\%$ \\\\"
                     for k, lab in [("0", "Low Vol"), ("1", "Med Vol"), ("2", "High Vol")])


if __name__ == "__main__":
    s = "%% coverage\n" + cov_rows() + "\n%% capital\n" + cap_rows() + "\n%% multimarket\n" + mm_rows() + "\n%% desc\n" + desc_rows() + "\n%% evt\n" + evt_rows() + "\n"
    open(os.path.join(C.ART, "tables_risk.tex"), "w").write(s)
    print(s)
