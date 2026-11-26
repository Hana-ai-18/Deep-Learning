"""Synthetic prior over tabular classification tasks (PFN-style pre-training data).

Each dataset in a batch is drawn from one of three generators:

1. **MLP-SCM prior** (TabPFN v1/v2 style): random root causes are pushed through a
   randomly-initialised sparse MLP with random activation functions and additive
   noise. Features and the target are read off random hidden nodes, which gives
   correlated, partially irrelevant and non-linearly related columns.
2. **Tree prior**: the target is a sum of random axis-aligned depth-3 decision
   trees over the generated features (data on which GBDTs shine).
3. **Linear prior**: a sparse random linear function of the features (data on
   which logistic regression shines).

The continuous target is turned into ``C`` classes by random quantile cuts and a
random label permutation. Some features are discretised into a few levels to
mimic ordinal-encoded categoricals, and every feature is standardised with the
statistics of the *training* rows (exactly what is done for real data at
evaluation time).

Everything is vectorised over the batch and runs on the GPU in milliseconds.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

_ACTS = [
    torch.tanh,
    torch.relu,
    torch.sigmoid,
    torch.sin,
    torch.abs,
    lambda x: torch.sign(x) * (x.abs() > 0.5).float(),          # step-like
    lambda x: x,                                                  # identity
    lambda x: torch.nn.functional.leaky_relu(x, 0.1),
    lambda x: x * x * torch.sign(x) * 0.5,                        # signed square
]


@dataclass
class PriorConfig:
    min_features: int = 2
    max_features: int = 20
    max_classes: int = 10
    min_rows: int = 64
    max_rows: int = 256
    train_frac_min: float = 0.4
    train_frac_max: float = 0.9
    p_tree: float = 0.3
    p_linear: float = 0.2
    p_binary: float = 0.4          # probability that a dataset is binary
    max_cat_frac: float = 0.3      # max fraction of features made categorical


def _randint(lo, hi, gen, device):
    return int(torch.randint(lo, hi + 1, (1,), generator=gen, device=device).item())


def _uniform(shape, lo, hi, gen, device):
    return lo + (hi - lo) * torch.rand(shape, generator=gen, device=device)


@torch.no_grad()
def sample_batch(B: int, cfg: PriorConfig, gen: torch.Generator, device,
                 N: int | None = None, F: int | None = None):
    """Return dict(X=(B,N,F) float, y=(B,N) long, n_train=int, n_classes=(B,) long)."""
    if N is None:
        N = _randint(min(cfg.min_rows, cfg.max_rows), cfg.max_rows, gen, device)
    if F is None:
        F = _randint(min(cfg.min_features, cfg.max_features), cfg.max_features, gen, device)
    frac = float(_uniform((1,), cfg.train_frac_min, cfg.train_frac_max, gen, device))
    n_train = max(8, min(N - 4, int(round(N * frac))))

    # ---------------- MLP-SCM ---------------- #
    h = _randint(max(F + 1, 6), max(F + 2, min(2 * F + 4, 64)), gen, device)
    L = _randint(2, 4, gen, device)
    gauss = torch.rand(B, 1, 1, generator=gen, device=device) < 0.5
    z = torch.where(gauss, torch.randn(B, N, h, generator=gen, device=device),
                    _uniform((B, N, h), -1.7, 1.7, gen, device))
    nodes = [z]
    w_scale = _uniform((B, 1, 1), 0.5, 1.6, gen, device)
    keep_p = _uniform((B, 1, 1), 0.5, 1.0, gen, device)
    noise_std = torch.exp(_uniform((B, 1, 1), -4.6, -1.6, gen, device))    # 0.01..0.2
    for _ in range(L):
        W = torch.randn(B, h, h, generator=gen, device=device) * w_scale / h ** 0.5
        W = W * (torch.rand(B, h, h, generator=gen, device=device) < keep_p)
        b = 0.5 * torch.randn(B, 1, h, generator=gen, device=device)
        pre = torch.bmm(z, W) / keep_p.clamp_min(0.3) ** 0.5 + b
        act_id = torch.randint(0, len(_ACTS), (B,), generator=gen, device=device)
        out = torch.empty_like(pre)
        for k, fn in enumerate(_ACTS):
            m = act_id == k
            if m.any():
                out[m] = fn(pre[m])
        z = out + noise_std * torch.randn(out.shape, generator=gen, device=device)
        nodes.append(z)
    allnodes = torch.cat(nodes, dim=2)                                  # (B,N,(L+1)h)
    n_nodes = allnodes.shape[2]
    # Features are read from random nodes; nodes of the last two layers (close to
    # the target in the causal graph) are favoured so that features are informative.
    layer_of = torch.arange(n_nodes, device=device) // h
    w_layer = torch.where(layer_of >= L - 1, 3.0, 1.0)
    keys = torch.rand(B, n_nodes, generator=gen, device=device) ** (1.0 / w_layer)
    feat_idx = torch.argsort(keys, dim=1, descending=True)[:, :F]
    X = torch.gather(allnodes, 2, feat_idx[:, None, :].expand(B, N, F))
    # target = random linear combination of 1-3 nodes of the last layer
    last = nodes[-1]
    k_t = _randint(1, 3, gen, device)
    t_idx = torch.randint(0, h, (B, k_t), generator=gen, device=device)
    t_w = torch.randn(B, k_t, generator=gen, device=device)
    t_mlp = (torch.gather(last, 2, t_idx[:, None, :].expand(B, N, k_t)) * t_w[:, None, :]).sum(-1)

    # ---------------- Tree prior ---------------- #
    t_tree = torch.zeros(B, N, device=device)
    n_trees, depth = 4, 3
    ar = torch.arange(B, device=device)
    for _ in range(n_trees):
        node = torch.zeros(B, N, dtype=torch.long, device=device)
        n_int = 2 ** depth - 1
        f_sel = torch.randint(0, F, (B, n_int), generator=gen, device=device)
        r_sel = torch.randint(0, N, (B, n_int), generator=gen, device=device)
        thr = X[ar[:, None], r_sel, f_sel]                               # (B,n_int)
        leaf_val = torch.randn(B, 2 ** depth, generator=gen, device=device)
        for _d in range(depth):
            f = torch.gather(f_sel, 1, node)                               # (B,N)
            th = torch.gather(thr, 1, node)
            xv = torch.gather(X, 2, f[..., None]).squeeze(-1)
            node = 2 * node + 1 + (xv > th).long()
        leaf = node - n_int
        t_tree = t_tree + torch.gather(leaf_val, 1, leaf)
    # ---------------- (generalised) linear prior ---------------- #
    Xs = (X - X.mean(1, keepdim=True)) / (X.std(1, keepdim=True) + 1e-6)
    w_lin = torch.randn(B, 1, F, generator=gen, device=device)
    w_lin = w_lin * (torch.rand(B, 1, F, generator=gen, device=device) < 0.6)
    t_lin = (Xs * w_lin).sum(-1)

    r = torch.rand(B, 1, generator=gen, device=device)
    t = torch.where(r < cfg.p_tree, t_tree,
                    torch.where(r < cfg.p_tree + cfg.p_linear, t_lin, t_mlp))
    t = t + 0.1 * t.std(dim=1, keepdim=True) * torch.randn(B, N, generator=gen, device=device) \
        * torch.rand(B, 1, generator=gen, device=device)

    # ---------------- classes ---------------- #
    Cmax = cfg.max_classes
    C = torch.randint(3, Cmax + 1, (B,), generator=gen, device=device)
    C = torch.where(torch.rand(B, generator=gen, device=device) < cfg.p_binary,
                    torch.full_like(C, 2), C)
    u = (torch.argsort(torch.argsort(t, dim=1), dim=1).float() + 0.5) / N   # ranks in (0,1)
    cuts = torch.rand(B, Cmax - 1, generator=gen, device=device) * 0.9 + 0.05
    valid = torch.arange(Cmax - 1, device=device)[None] < (C - 1)[:, None]
    cuts = torch.where(valid, cuts, torch.full_like(cuts, 2.0)).sort(dim=1).values
    y = (u[..., None] > cuts[:, None, :]).sum(-1)                          # (B,N) in 0..C-1
    keys = torch.rand(B, Cmax, generator=gen, device=device)
    keys = torch.where(torch.arange(Cmax, device=device)[None] < C[:, None], keys, keys + 10)
    perm = torch.argsort(keys, dim=1)                                      # random relabel
    y = torch.gather(perm, 1, y)

    # ---------------- feature post-processing ---------------- #
    X = (X - X.mean(1, keepdim=True)) / (X.std(1, keepdim=True) + 1e-6)
    cat_frac = _uniform((B, 1, 1), 0.0, cfg.max_cat_frac, gen, device)
    is_cat = torch.rand(B, 1, F, generator=gen, device=device) < cat_frac
    n_lvl = torch.randint(2, 9, (B, 1, F), generator=gen, device=device).float()
    ranks = (torch.argsort(torch.argsort(X, dim=1), dim=1).float() + 0.5) / N
    lvl = torch.floor(ranks * n_lvl)
    # random order of the levels -> behaves like an arbitrary ordinal encoding
    shift = torch.randint(0, 8, (B, 1, F), generator=gen, device=device).float()
    lvl = torch.remainder(lvl + shift, n_lvl)
    X = torch.where(is_cat, lvl, X)
    mu = X[:, :n_train].mean(1, keepdim=True)
    sd = X[:, :n_train].std(1, keepdim=True)
    sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
    X = ((X - mu) / sd).clamp(-6, 6)
    X = torch.nan_to_num(X, 0.0)
    return dict(X=X.float(), y=y.long(), n_train=n_train, n_classes=C.long())
