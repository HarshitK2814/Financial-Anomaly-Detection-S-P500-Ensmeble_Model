"""
Round-2 revision (Reviewer 1, point 1): label-feature circularity check.

Same-day variant: features of the window ending at t (so r_t is included).
Compares the regime-threshold rule 1{|r_t| > tau_t} (which reproduces the label
by construction) with RC-MoE and the global ensemble trained on same-day
features, in-sample and out-of-sample (same folds as r2_detection).
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

from src.evaluation import r2_common as C
from src.evaluation.r2_models import RCMoE
from src.evaluation.r2_detection import folds


def main():
    d = pd.read_pickle(os.path.join(C.ART, "dataset_SP500.pkl"))
    df = C.load_ohlcv(C.MARKETS["SP500"])
    F, r = C.daily_factors(df)
    same = pd.concat([F.add_prefix("f_"), F.shift(1).add_prefix("l1_"), F.shift(5).add_prefix("l5_")], axis=1).loc[d.index]
    X = same.values
    y = d["y"].values.astype(int)
    reg, conf = d["regime"].values.astype(int), d["conf"].values
    P = d[["p0", "p1", "p2"]].values
    rule = (np.abs(d["r"].values) > d["tau"].values).astype(int)
    out = {}
    ys, sm, sg, ru, ism = [], [], [], [], []
    for k, tr, te in folds(d.index, y):
        m = RCMoE().fit(X[tr], y[tr], reg[tr])
        sm.append(m.predict(X[te], reg[te], conf[te], P[te]))
        sg.append(m.glob.predict(X[te]))
        ru.append(rule[te])
        ys.append(y[te])
        if k == 5:
            ism = (y[tr], m.predict(X[tr], reg[tr], conf[tr], P[tr]))
    ys, sm, sg, ru = map(np.concatenate, (ys, sm, sg, ru))
    out["oos"] = {"rule_f1": float(f1_score(ys, ru)), "rule_auprc": float(average_precision_score(ys, ru)),
                  "moe_auprc": float(average_precision_score(ys, sm)), "glob_auprc": float(average_precision_score(ys, sg)),
                  "moe_f1_05": float(f1_score(ys, (sm >= 0.5).astype(int))), "glob_f1_05": float(f1_score(ys, (sg >= 0.5).astype(int)))}
    yt, st = ism
    out["is_last_fold"] = {"moe_f1_05": float(f1_score(yt, (st >= 0.5).astype(int))), "moe_auprc": float(average_precision_score(yt, st))}
    print(json.dumps(out, indent=2))
    json.dump(out, open(os.path.join(C.ART, "sameday.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
