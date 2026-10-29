"""
Round-2 revision: detectors used in the single out-of-sample pipeline.
All estimators use explicit n_jobs (see env note: default n_jobs hangs on this machine).
"""
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from arch import arch_model
from scipy.stats import genpareto
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

NJ = 4


def make_rf(seed=0):
    return RandomForestClassifier(n_estimators=200, max_depth=5, class_weight="balanced",
                                  random_state=seed, n_jobs=NJ)


def make_xgb(pos_weight, seed=0):
    return xgb.XGBClassifier(n_estimators=200, max_depth=5, reg_alpha=0.1, reg_lambda=1.0,
                             scale_pos_weight=pos_weight, learning_rate=0.1,
                             eval_metric="logloss", verbosity=0, random_state=seed, n_jobs=NJ)


def make_lgb(seed=0):
    return lgb.LGBMClassifier(n_estimators=200, num_leaves=31, is_unbalance=True,
                              learning_rate=0.05, verbose=-1, random_state=seed, n_jobs=NJ)


def _fit_experts(X, y, families=("rf", "xgb", "lgb")):
    pw = max((1 - y.mean()) / max(y.mean(), 1e-3), 1.0)
    ms = []
    for f in families:
        m = {"rf": make_rf, "xgb": lambda: make_xgb(pw), "lgb": make_lgb}[f]()
        m.fit(X, y)
        ms.append(m)
    return ms


def _expert_probs(ms, X):
    return np.column_stack([m.predict_proba(X)[:, 1] for m in ms])


class StackedEnsemble:
    """RF + XGBoost + LightGBM experts combined by a logistic meta-learner.
    The meta-learner is fitted on expert predictions for the last 30% of the
    training slice (experts trained on the first 70%); experts are then refitted
    on the whole slice.  With too few positives the meta-learner is replaced by
    a plain average."""

    def __init__(self, families=("rf", "xgb", "lgb")):
        self.families = families
        self.meta = None

    def fit(self, X, y):
        n = len(y)
        cut = int(0.7 * n)
        if len(self.families) > 1 and y[:cut].sum() >= 3 and y[cut:].sum() >= 3:
            inner = _fit_experts(X[:cut], y[:cut], self.families)
            P = _expert_probs(inner, X[cut:])
            self.meta = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(P, y[cut:])
        self.models = _fit_experts(X, y, self.families)
        return self

    def predict(self, X):
        P = _expert_probs(self.models, X)
        if self.meta is None:
            return P.mean(axis=1)
        return self.meta.predict_proba(P)[:, 1]


class RCMoE:
    """Regime-conditioned mixture of experts with hard / soft routing."""

    def __init__(self, k=3, families=("rf", "xgb", "lgb"), tau_c=0.6):
        self.k, self.families, self.tau_c = k, families, tau_c

    def fit(self, X, y, reg):
        self.glob = StackedEnsemble(self.families).fit(X, y)
        self.experts = {}
        for j in range(self.k):
            m = reg == j
            if m.sum() >= 10 and y[m].sum() >= 3:
                self.experts[j] = StackedEnsemble(self.families).fit(X[m], y[m])
            else:
                self.experts[j] = self.glob            # sparse-regime fallback
        return self

    def predict(self, X, reg, conf, probs):
        P = np.column_stack([self.experts[j].predict(X) for j in range(self.k)])
        hard = P[np.arange(len(reg)), reg]
        soft = (P * probs).sum(axis=1)
        return np.where(conf >= self.tau_c, hard, soft)


# ---------------------------------------------------------------- GARCH baselines
def garch_sigma(r_all, train_end_idx):
    """GARCH(1,1) fitted on returns up to train_end_idx (exclusive); returns the
    one-step-ahead conditional mean and volatility for every day (causal filter)
    and the standardised training residuals."""
    x = 100 * r_all
    am = arch_model(x[:train_end_idx], mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    res = am.fit(disp="off")
    mu, om, a, b = (res.params[k] for k in ["mu", "omega", "alpha[1]", "beta[1]"])
    n = len(x)
    s2 = np.empty(n)
    s2[0] = np.var(x[:train_end_idx])
    for t in range(1, n):
        s2[t] = om + a * (x[t - 1] - mu) ** 2 + b * s2[t - 1]
    sig = np.sqrt(s2)
    z = (x[:train_end_idx] - mu) / sig[:train_end_idx]
    return mu / 100, sig / 100, z[50:]


def _gpd_tail(z_tail_pos):
    u = np.quantile(z_tail_pos, 0.90)
    exc = z_tail_pos[z_tail_pos > u] - u
    c, _, scale = genpareto.fit(exc, floc=0)
    return u, c, scale, (z_tail_pos > u).mean()


def exceed_prob_fhs(mu, sig, tau, z):
    """P(|r_t| > tau_t) under filtered historical simulation."""
    up = (tau - mu) / sig
    lo = (-tau - mu) / sig
    zs = np.sort(z)
    p_up = 1 - np.searchsorted(zs, up, side="right") / len(zs)
    p_lo = np.searchsorted(zs, lo, side="left") / len(zs)
    return p_up + p_lo


def exceed_prob_evt(mu, sig, tau, z):
    """P(|r_t| > tau_t) with GPD tails for the standardised residuals (McNeil-Frey)."""
    def tail_prob(zz, x):
        u, c, s, pu = _gpd_tail(zz)
        p = np.where(x > u, pu * genpareto.sf(x - u, c, loc=0, scale=s), (zz[:, None] > x[None, :]).mean(axis=0))
        return p
    up = (tau - mu) / sig
    lo = (tau + mu) / sig            # P(z < (-tau-mu)/sig) = P(-z > (tau+mu)/sig)
    return tail_prob(z, up) + tail_prob(-z, lo)
