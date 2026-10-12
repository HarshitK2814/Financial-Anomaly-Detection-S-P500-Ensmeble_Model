BS = chr(92)
L = open(r"C:\Users\Harshit Kumar\Downloads\Financial Anomaly Detection Journal Paper Coding Part\Financial-Anomaly-Detection-S-P500-Ensmeble_Model\final.tex", encoding="utf8").read().split("\n")


def ln(s):
    return [i + 1 for i, l in enumerate(L) if s in l]


for lab in ["taxonomy", "sensitivity", "oos_comparison", "regime_conditioning", "backtest",
            "multimarket", "multimarket_detection", "eci_regimes"]:
    j = ln("label{tab:%s}" % lab)[0] - 1
    b = max(i for i in range(j) if L[i].lstrip().startswith(BS + "begin{table"))
    e = next(i for i in range(j, len(L)) if L[i].lstrip().startswith(BS + "end{table"))
    print(f"{lab:24s} begin {b + 1}  end {e + 1}")
for s in ["During the complete 2016--2025 evaluation window period", "Warning concerning period dependency",
          "reports the GPD parameters per regime", "subsubsection{Basel III Capital",
          "The gap between in-sample structural separability and out-of-sample generalisation is theoretically",
          "subsection{Role of Regime", "Deployment Goal &", "Kupiec} &", "& FAIL &", "$z$-score}",
          "Global (all regimes)", "tbl{Multi-market EVT", "tbl{Regime conditioning impact"]:
    print(repr(s[:50]), ln(s))
