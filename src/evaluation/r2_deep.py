"""
Round-2 revision: deep baselines of the ablation, re-run inside the single
out-of-sample pipeline (same folds, same ex-ante labels).

Input: raw 128 x 5 window ending at t-1 (log return, high-low range, overnight
gap, open-to-close return, log volume change), standardised with training-block
statistics.  Both models are unsupervised and trained on normal training windows.
  * ConvVAE: 1-d convolutional VAE; score = reconstruction error of the window.
  * Anomaly Transformer (Xu et al., 2022): series/prior association with the
    minimax association-discrepancy objective; score at the last time step.
Output: artifacts/r2/oos_deep_<market>.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.evaluation import r2_common as C
from src.evaluation.r2_detection import folds

torch.set_num_threads(8)
L = C.WINDOW
EPOCHS = 15
STRIDE = 2


def raw_channels(market):
    df = C.load_ohlcv(C.MARKETS[market])
    o, h, l, c, v = (df[x].values for x in ["Open", "High", "Low", "Close", "Volume"])
    pc = np.r_[np.nan, c[:-1]]
    r = np.log(c / pc)
    hl = (h - l) / c
    gap = (o - pc) / pc
    oc = (c - o) / o
    pv = np.r_[np.nan, v[:-1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        dv = np.log(np.where((v > 0) & (pv > 0), v / pv, 1.0))
    X = np.column_stack([r, hl, gap, oc, dv])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame(X, index=df.index)


def windows(raw, dates):
    pos = raw.index.get_indexer(dates)
    A = raw.values
    return np.stack([A[p - L:p] for p in pos]).astype(np.float32)   # rows p-L..p-1 -> ends at t-1


class ConvVAE(nn.Module):
    def __init__(self, ch=5, z=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(ch, 32, 5, 2, 2), nn.ReLU(), nn.Conv1d(32, 64, 5, 2, 2), nn.ReLU(),
                                 nn.Conv1d(64, 64, 5, 2, 2), nn.ReLU())
        self.mu = nn.Linear(64 * 16, z)
        self.lv = nn.Linear(64 * 16, z)
        self.dec_in = nn.Linear(z, 64 * 16)
        self.dec = nn.Sequential(nn.ConvTranspose1d(64, 64, 4, 2, 1), nn.ReLU(), nn.ConvTranspose1d(64, 32, 4, 2, 1),
                                 nn.ReLU(), nn.ConvTranspose1d(32, ch, 4, 2, 1))

    def forward(self, x):
        h = self.enc(x.transpose(1, 2)).flatten(1)
        mu, lv = self.mu(h), self.lv(h)
        zz = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu
        out = self.dec(self.dec_in(zz).view(-1, 64, 16)).transpose(1, 2)
        return out, mu, lv


class AnomalyAttention(nn.Module):
    def __init__(self, d, win):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.sig = nn.Linear(d, 1)
        idx = torch.arange(win).float()
        self.register_buffer("dist", (idx[None, :] - idx[:, None]).abs())
        self.d = d

    def forward(self, x):
        Q, K, V = self.q(x), self.k(x), self.v(x)
        S = torch.softmax(Q @ K.transpose(1, 2) / np.sqrt(self.d), dim=-1)
        sig = torch.sigmoid(self.sig(x) * 5) + 1e-5
        sig = torch.pow(3, sig) - 1
        P = torch.exp(-self.dist[None] ** 2 / (2 * sig ** 2)) / (np.sqrt(2 * np.pi) * sig)
        P = P / P.sum(-1, keepdim=True)
        return S @ V, S, P


class AnomalyTransformer(nn.Module):
    def __init__(self, ch=5, d=32, layers=2, win=L):
        super().__init__()
        self.emb = nn.Linear(ch, d)
        self.pos = nn.Parameter(torch.randn(1, win, d) * 0.02)
        self.att = nn.ModuleList([AnomalyAttention(d, win) for _ in range(layers)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)) for _ in range(layers)])
        self.n1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.n2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.out = nn.Linear(d, ch)

    def forward(self, x):
        h = self.emb(x) + self.pos
        Ss, Ps = [], []
        for a, f, n1, n2 in zip(self.att, self.ff, self.n1, self.n2):
            o, S, P = a(h)
            h = n1(h + o)
            h = n2(h + f(h))
            Ss.append(S)
            Ps.append(P)
        return self.out(h), Ss, Ps


def _kl(p, q):
    return (p * (torch.log(p + 1e-4) - torch.log(q + 1e-4))).sum(-1)


def ass_dis(Ss, Ps, detach_p=False, detach_s=False):
    tot = 0
    for S, P in zip(Ss, Ps):
        Sx = S.detach() if detach_s else S
        Px = P.detach() if detach_p else P
        tot = tot + _kl(Px, Sx) + _kl(Sx, Px)
    return tot / len(Ss)            # (batch, win)


def train_score(model_name, Xtr, Xte, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr_t = torch.tensor(Xtr[::STRIDE])
    Xte_t = torch.tensor(Xte)
    if model_name == "ConvVAE":
        m = ConvVAE()
    else:
        m = AnomalyTransformer()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        m.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), 64):
            xb = Xtr_t[perm[i:i + 64]]
            opt.zero_grad()
            if model_name == "ConvVAE":
                out, mu, lv = m(xb)
                loss = F.mse_loss(out, xb) + 1e-3 * (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1).mean())
                loss.backward()
            else:
                out, Ss, Ps = m(xb)
                rec = F.mse_loss(out, xb)
                l1 = rec - 3.0 * ass_dis(Ss, Ps, detach_p=True).mean()     # maximise phase
                l2 = rec + 3.0 * ass_dis(Ss, Ps, detach_s=True).mean()     # minimise phase
                l1.backward(retain_graph=True)
                l2.backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        if model_name == "ConvVAE":
            out, _, _ = m(Xte_t)
            return ((out - Xte_t) ** 2).mean(dim=(1, 2)).numpy()
        out, Ss, Ps = m(Xte_t)
        rec = ((out - Xte_t) ** 2).mean(-1)                        # (n, win)
        ad = torch.softmax(-ass_dis(Ss, Ps), dim=-1)
        return (ad * rec)[:, -1].numpy()


def run_market(market):
    d = pd.read_pickle(os.path.join(C.ART, f"dataset_{market}.pkl"))
    raw = raw_channels(market)
    y = d["y"].values.astype(int)
    W = windows(raw, d.index)
    rows = []
    for k, tr, te in folds(d.index, y):
        mu = W[tr].reshape(-1, 5).mean(0)
        sd = W[tr].reshape(-1, 5).std(0) + 1e-8
        Wtr = np.clip((W[tr][y[tr] == 0] - mu) / sd, -8, 8)
        Wte = np.clip((W[te] - mu) / sd, -8, 8)
        sc = {nm: train_score(nm, Wtr, Wte) for nm in ["ConvVAE", "Anomaly Transformer"]}
        rows.append(pd.DataFrame(sc, index=d.index[te]))
        print(f"  {market} fold {k} done", flush=True)
    out = pd.concat(rows)
    out.index.name = "date"
    out.to_csv(os.path.join(C.ART, f"oos_deep_{market}.csv"))


if __name__ == "__main__":
    for m in (sys.argv[1:] or list(C.MARKETS)):
        run_market(m)
