"""
Round-2 revision (UAAI 262098302.R1): single ex-ante data / regime / label definition.

Everything that the detection, backtest, VaR and ECI scripts use comes from here,
so the whole paper rests on one anomaly definition and one out-of-sample pipeline.

Timing convention (no same-day information):
  * r_t      = log(C_t / C_{t-1})
  * sigma_t  = std of r_{t-L}, ..., r_{t-1}            (known at the close of t-1)
  * R_t      = fused regime from sigma_{<=t}           (known at the close of t-1)
  * tau_t    = 95th percentile of |r_s|, s < t, R_s = R_t   (expanding, past only)
  * A_t      = 1{|r_t| > tau_t}                        (the label)
  * X_t      = 8 core factors of the window ending t-1, plus their lag-1 and lag-5
So X_t and R_t never contain r_t; the label is the only quantity that does.

Run as module from the repo root, e.g.  python -m src.evaluation.r2_detection
"""
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

warnings.filterwarnings("ignore")

ART = os.path.join("artifacts", "r2")
WINDOW = 128          # factor / regime volatility window L
ROLL = 252            # rolling-quantile detector look-back
ANOM_Q = 0.95         # one anomaly definition for the whole paper
MIN_HIST = 20         # minimum past observations for a regime threshold
TAU_C = 0.6           # hard / soft routing confidence
HMM_REFIT = 63        # HMM re-estimated every quarter on past data only
SAMPLE = ("2016-01-01", "2025-12-31")      # evaluation period (test blocks)
HISTORY = ("2005-01-01", "2025-12-31")     # training history (past data only)

MARKETS = {
    "SP500": "GSPC",
    "NASDAQ": "IXIC",
    "Russell2000": "RUT",
    "Bitcoin": "BTCUSD",
}
FACTOR_NAMES = ["Return_Trend", "Momentum", "Composite_Vol", "Latent_A_gap",
                "Latent_B_pressure", "Latent_C_range", "Tail_Kurtosis", "Tail_Skew"]


def load_ohlcv(code):
    df = pd.read_csv(os.path.join(ART, f"ohlcv_{code}.csv"), parse_dates=["Date"], index_col="Date")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Close"])


# ---------------------------------------------------------------- regimes
def _forward_filter(x, startprob, transmat, means, variances):
    """Filtered state probabilities P(S_t | x_1..x_t) of a 1-d Gaussian HMM."""
    n, k = len(x), len(startprob)
    out = np.zeros((n, k))
    dens = np.exp(-0.5 * (x[:, None] - means[None, :]) ** 2 / variances[None, :]) / np.sqrt(2 * np.pi * variances[None, :])
    dens = np.maximum(dens, 1e-300)
    a = startprob * dens[0]
    out[0] = a / a.sum()
    for t in range(1, n):
        a = (out[t - 1] @ transmat) * dens[t]
        out[t] = a / a.sum()
    return out


def _quantile_regime(value, ref, k):
    qs = np.quantile(ref, np.linspace(0, 1, k + 1)[1:-1])
    return int(np.searchsorted(qs, value, side="left"))


def fused_regimes(sigma, k=3, seed=0):
    """Causal fusion of the three detectors of the paper (rolling quantile,
    Gaussian HMM, static quantile).  sigma[t] is already known at t-1.
    Returns regime[t], confidence[t], vote probabilities[t, k]."""
    from hmmlearn.hmm import GaussianHMM
    n = len(sigma)
    logs = np.log(np.maximum(sigma, 1e-8))
    reg = np.full(n, -1)
    conf = np.zeros(n)
    probs = np.zeros((n, k))
    hmm_p = np.full((n, k), np.nan)
    valid = np.where(np.isfinite(sigma))[0]
    first = valid[0] + ROLL
    # HMM fitted on past data only and refitted every quarter
    for start in range(first, n, HMM_REFIT):
        hist = logs[valid[0]:start]
        m = GaussianHMM(n_components=k, covariance_type="diag", n_iter=200, random_state=seed)
        m.fit(hist.reshape(-1, 1))
        order = np.argsort(m.means_.ravel())
        mu = m.means_.ravel()[order]
        var = np.array([m.covars_[i].ravel()[0] for i in order])
        tm = m.transmat_[np.ix_(order, order)]
        sp = m.startprob_[order]
        end = min(start + HMM_REFIT, n)
        filt = _forward_filter(logs[valid[0]:end], sp, tm, mu, var)
        hmm_p[start:end] = filt[start - valid[0]:]
    for t in range(first, n):
        r1 = _quantile_regime(sigma[t], sigma[t - ROLL:t], k)          # rolling quantile
        r3 = _quantile_regime(sigma[t], sigma[valid[0]:t], k)          # static (expanding) quantile
        p2 = hmm_p[t]
        r2 = int(np.argmax(p2))
        votes = np.zeros(k)
        votes[r1] += 1.0
        votes[r3] += 1.0
        votes[r2] += float(p2.max())
        pv = votes / votes.sum()
        reg[t] = int(np.argmax(pv))
        conf[t] = pv.max()
        probs[t] = pv
    return reg, conf, probs


# ---------------------------------------------------------------- factors
def daily_factors(df, L=WINDOW):
    """8 core factors for every day d using data up to and including d."""
    o, h, l, c = (df[x].values for x in ["Open", "High", "Low", "Close"])
    r = np.r_[np.nan, np.diff(np.log(c))]
    rs = pd.Series(r)
    trend = rs.rolling(L).mean().values
    pos = rs.clip(lower=0).rolling(L).sum().values
    absr = rs.abs().rolling(L).sum().values
    mom = pos / (absr + 1e-12)
    vol = rs.rolling(L).std().values
    kur = rs.rolling(L).apply(lambda x: kurtosis(x), raw=True).values
    skw = rs.rolling(L).apply(lambda x: skew(x), raw=True).values
    prev_c = np.r_[np.nan, c[:-1]]
    gap = (o - prev_c) / prev_c
    rng = h - l
    pressure = (h - o) / (rng + 1e-12)
    tr = np.nanmax(np.vstack([rng, np.abs(h - prev_c), np.abs(l - prev_c)]), axis=0)
    atr = pd.Series(tr).rolling(20).mean().values
    rangef = rng / (atr + 1e-12)
    F = np.column_stack([trend, mom, vol, gap, pressure, rangef, kur, skw])
    return pd.DataFrame(F, index=df.index, columns=FACTOR_NAMES), pd.Series(r, index=df.index)


def build_dataset(market, L=WINDOW, k=3, q=ANOM_Q, sample=SAMPLE):
    """Returns a DataFrame indexed by target day t with columns:
    r (r_t), sigma (vol of window ending t-1), regime, conf, p0..p{k-1},
    tau, y, and the 24 ex-ante features f_* (8 factors at t-1, t-2, t-6)."""
    df = load_ohlcv(MARKETS[market])
    F, r = daily_factors(df, L)
    sigma = F["Composite_Vol"].shift(1).values          # window ending t-1
    reg, conf, probs = fused_regimes(sigma, k=k)
    absr = np.abs(r.values)
    n = len(r)
    tau = np.full(n, np.nan)
    for t in range(n):
        if reg[t] < 0:
            continue
        past = absr[:t][(reg[:t] == reg[t]) & np.isfinite(absr[:t])]
        if len(past) >= MIN_HIST:
            tau[t] = np.quantile(past, q)
    y = (absr > tau).astype(float)
    y[~np.isfinite(tau)] = np.nan
    X1 = F.shift(1)            # factors of window ending t-1
    X2 = F.shift(2)            # lag-1
    X6 = F.shift(6)            # lag-5
    feats = pd.concat([X1.add_prefix("f_"), X2.add_prefix("l1_"), X6.add_prefix("l5_")], axis=1)
    out = pd.DataFrame({"r": r.values, "sigma": sigma, "regime": reg, "conf": conf,
                        "tau": tau, "y": y}, index=df.index)
    for j in range(k):
        out[f"p{j}"] = probs[:, j]
    out = pd.concat([out, feats], axis=1)
    out = out.loc[sample[0]:sample[1]]
    out = out.dropna(subset=["y", "tau"] + list(feats.columns))
    out = out[out["regime"] >= 0]
    return out


def feature_cols(df):
    return [c for c in df.columns if c[:2] in ("f_", "l1", "l5")]


def regime_series(market, L=WINDOW, k=3):
    """Ex-ante regime series over the full history (used by the VaR analysis)."""
    df = load_ohlcv(MARKETS[market])
    F, r = daily_factors(df, L)
    sigma = F["Composite_Vol"].shift(1).values
    reg, conf, probs = fused_regimes(sigma, k=k)
    return pd.DataFrame({"r": r.values, "regime": reg, "conf": conf}, index=df.index)


if __name__ == "__main__":
    for m in MARKETS:
        d = build_dataset(m, sample=HISTORY)
        d.to_pickle(os.path.join(ART, f"dataset_{m}.pkl"))
        print(m, len(d), d.index.min().date(), d.index.max().date(),
              "eval anoms", int(d.loc[SAMPLE[0]:].y.sum()), "all anoms", int(d.y.sum()), f"{100*d.y.mean():.2f}%",
              "by regime", d.groupby("regime").y.agg(["sum", "count"]).values.tolist(),
              "soft share", f"{(d.conf < TAU_C).mean():.3f}")
