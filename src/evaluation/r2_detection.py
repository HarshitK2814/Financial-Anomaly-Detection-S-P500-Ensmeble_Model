"""
Round-2 revision: the single out-of-sample detection pipeline.

5-fold expanding walk-forward over the 2016-2025 sample with a 128-day purge
between each training block and its test block.  Every detector receives the
same ex-ante information (see r2_common).  Out-of-sample scores are written to
artifacts/r2/oos_scores_<market>.csv and are the only input of the metric,
backtest and figure scripts.

Run:  python -m src.evaluation.r2_detection            (all markets)
      python -m src.evaluation.r2_detection sens        (L x K sensitivity, S&P 500)
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.evaluation import r2_common as C
from src.evaluation.r2_models import (RCMoE, StackedEnsemble, make_rf, make_lgb, garch_sigma,
                                      exceed_prob_fhs, exceed_prob_evt)

N_FOLDS = 5
PURGE = C.WINDOW


BLOCKS = [("2016-01-01", "2017-12-31"), ("2018-01-01", "2019-12-31"), ("2020-01-01", "2021-12-31"),
          ("2022-01-01", "2023-12-31"), ("2024-01-01", "2025-12-31")]
MIN_TRAIN_POS = 20


def folds(index, y):
    """Two-year test blocks over 2016-2025; each is preceded by an expanding
    training window that ends PURGE trading days before the block starts."""
    for k, (a, b) in enumerate(BLOCKS, 1):
        te = np.where((index >= a) & (index <= b))[0]
        if len(te) == 0:
            continue
        tr = np.arange(0, max(te[0] - PURGE, 0))
        if y[tr].sum() < MIN_TRAIN_POS:
            continue
        yield k, tr, te


def run_market(market, with_ml=True):
    d = pd.read_pickle(os.path.join(C.ART, f"dataset_{market}.pkl"))
    full = C.load_ohlcv(C.MARKETS[market])
    r_full = np.r_[0.0, np.diff(np.log(full["Close"].values))]
    pos_full = full.index.get_indexer(d.index)            # position of each sample day in full history
    sd20 = pd.Series(r_full).rolling(20).std().shift(1).values
    sd252 = pd.Series(r_full).rolling(252).std().shift(1).values
    rprev = np.r_[np.nan, r_full[:-1]]

    fc = C.feature_cols(d)
    X = d[fc].values
    y = d["y"].values.astype(int)
    reg = d["regime"].values.astype(int)
    conf = d["conf"].values
    probs = d[[f"p{j}" for j in range(3)]].values
    tau = d["tau"].values
    onehot = np.eye(3)[reg]

    rows = []
    for k, tr, te in folds(d.index, y):
        sc = {}
        # --- statistical baselines (causal, parameters from the training block only)
        last = pos_full[tr[-1]] + 1
        mu, sig, z = garch_sigma(r_full, last)
        s_te = sig[pos_full[te]]
        sc["GARCH-FHS"] = exceed_prob_fhs(mu, s_te, tau[te], z)
        sc["GARCH-EVT"] = exceed_prob_evt(mu, s_te, tau[te], z)
        sc["Vol rule"] = sd20[pos_full[te]] / tau[te]
        sc["Z-score rule"] = np.abs(rprev[pos_full[te]]) / sd252[pos_full[te]]
        if with_ml:
            rf = make_rf().fit(X[tr], y[tr])
            sc["Global RF"] = rf.predict_proba(X[te])[:, 1]
            gb = make_lgb().fit(X[tr], y[tr])
            sc["Global GBM"] = gb.predict_proba(X[te])[:, 1]
            gbr = make_lgb().fit(np.hstack([X[tr], onehot[tr]]), y[tr])
            sc["GBM + regime"] = gbr.predict_proba(np.hstack([X[te], onehot[te]]))[:, 1]
            moe = RCMoE().fit(X[tr], y[tr], reg[tr])
            sc["Global ensemble"] = moe.glob.predict(X[te])
            sc["RC-MoE"] = moe.predict(X[te], reg[te], conf[te], probs[te])
        f = pd.DataFrame(sc, index=d.index[te])
        f.insert(0, "fold", k)
        f.insert(1, "y", y[te])
        f.insert(2, "regime", reg[te])
        f.insert(3, "conf", conf[te])
        f.insert(4, "r", d["r"].values[te])
        f.insert(5, "tau", tau[te])
        rows.append(f)
        print(f"  {market} fold {k}: train {len(tr)} ({y[tr].sum()} pos)  test {len(te)} ({y[te].sum()} pos)", flush=True)
    out = pd.concat(rows)
    out.index.name = "date"
    out.to_csv(os.path.join(C.ART, f"oos_scores_{market}.csv"))
    return out


def run_sensitivity():
    """RC-MoE and global ensemble across factor window L and regime count K.
    Labels are held fixed at the paper definition (L=128, K=3) so every cell
    predicts the same target."""
    from sklearn.metrics import average_precision_score
    base = pd.read_pickle(os.path.join(C.ART, "dataset_SP500.pkl"))
    res = {}
    for L in [64, 96, 128, 192]:
        for K in [2, 3, 4]:
            d = C.build_dataset("SP500", L=L, k=K, sample=C.HISTORY)
            d = d.loc[d.index.intersection(base.index)]
            y = base.loc[d.index, "y"].values.astype(int)
            fc = C.feature_cols(d)
            X = d[fc].values
            reg = d["regime"].values.astype(int)
            conf = d["conf"].values
            probs = d[[f"p{j}" for j in range(K)]].values
            ys, sm, sg = [], [], []
            for k, tr, te in folds(d.index, y):
                moe = RCMoE(k=K).fit(X[tr], y[tr], reg[tr])
                sm.append(moe.predict(X[te], reg[te], conf[te], probs[te]))
                sg.append(moe.glob.predict(X[te]))
                ys.append(y[te])
            ys = np.concatenate(ys)
            a_m = average_precision_score(ys, np.concatenate(sm))
            a_g = average_precision_score(ys, np.concatenate(sg))
            res[f"L{L}_K{K}"] = {"auprc_moe": a_m, "auprc_global": a_g, "n": int(len(ys)),
                                 "pos": int(ys.sum()), "prev": float(ys.mean())}
            print(f"  L={L} K={K}: AUPRC MoE {a_m:.3f}  global {a_g:.3f}  (prev {ys.mean():.3f})", flush=True)
    # HMM BIC on log window volatility, L = 128
    from hmmlearn.hmm import GaussianHMM
    sig = np.log(base["sigma"].values).reshape(-1, 1)   # full history up to 2025
    bic = {}
    for K in [2, 3, 4]:
        m = GaussianHMM(n_components=K, covariance_type="diag", n_iter=300, random_state=0).fit(sig)
        bic[K] = float(m.bic(sig))
    res["hmm_bic"] = bic
    json.dump(res, open(os.path.join(C.ART, "sensitivity.json"), "w"), indent=2)
    print(bic)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sens":
        run_sensitivity()
    else:
        mk = sys.argv[1:] or list(C.MARKETS)
        for m in mk:
            run_market(m)
