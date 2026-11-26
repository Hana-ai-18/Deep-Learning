"""nanoTabPFN: a compact re-implementation of the TabPFN v2 architecture.

A table with N rows and F features (+1 target column) is embedded cell-wise into
a tensor of shape (B, N, F+1, d). Every layer alternates

  * **feature attention**: attention across the F+1 cells of one row,
  * **item attention**  : attention across rows within one column, where every
    row (train or test) may only attend to *training* rows,
  * a feed-forward MLP,

each wrapped in a pre-norm residual block. The prediction for a test row is read
from its target-column cell.

Two architectural options (both 2025 papers) can be switched on:

  * ``norm="dyt"``  : Dynamic Tanh, DyT(x) = gamma * tanh(alpha * x) + beta,
                      replacing every LayerNorm (Zhu, Chen, He, LeCun, Liu,
                      "Transformers without Normalization", CVPR 2025).
  * ``gated=True``  : head-specific element-wise sigmoid gate applied to the
                      scaled-dot-product-attention output, Y * sigmoid(X W_g)
                      (Qiu et al., "Gated Attention for Large Language Models",
                      NeurIPS 2025 best paper).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 6
    d_ff: int = 256
    max_classes: int = 10
    col_emb_dim: int = 16
    norm: str = "layernorm"        # "layernorm" | "dyt"
    gated: bool = False
    dyt_alpha: float = 0.5

    def to_dict(self):
        return asdict(self)


class DyT(nn.Module):
    def __init__(self, d: int, alpha0: float = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1,), float(alpha0)))
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        return torch.tanh(self.alpha * x) * self.weight + self.bias


def make_norm(cfg: ModelConfig):
    if cfg.norm == "dyt":
        return DyT(cfg.d_model, cfg.dyt_alpha)
    if cfg.norm == "layernorm":
        return nn.LayerNorm(cfg.d_model)
    raise ValueError(cfg.norm)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, self.h = cfg.d_model, cfg.n_heads
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, d) if cfg.gated else None

    def forward(self, xq, xkv):
        # xq: (S, Lq, d), xkv: (S, Lk, d)
        S, Lq, d = xq.shape
        Lk = xkv.shape[1]
        dh = d // self.h
        q = self.q(xq).view(S, Lq, self.h, dh).transpose(1, 2)
        k = self.k(xkv).view(S, Lk, self.h, dh).transpose(1, 2)
        v = self.v(xkv).view(S, Lk, self.h, dh).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).reshape(S, Lq, d)
        if self.gate is not None:                     # G1 gating of Qiu et al. (2025)
            y = y * torch.sigmoid(self.gate(xq))
        return self.o(y)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_feat, self.attn_feat = make_norm(cfg), Attention(cfg)
        self.n_item, self.attn_item = make_norm(cfg), Attention(cfg)
        self.n_ff = make_norm(cfg)
        self.ff = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(),
                                nn.Linear(cfg.d_ff, cfg.d_model))

    def forward(self, x, n_train: int):
        B, N, C, d = x.shape
        # feature attention (within each row)
        h = self.n_feat(x).reshape(B * N, C, d)
        x = x + self.attn_feat(h, h).reshape(B, N, C, d)
        # item attention (within each column; keys/values = training rows only)
        h = self.n_item(x).transpose(1, 2).reshape(B * C, N, d)
        y = self.attn_item(h, h[:, :n_train])
        x = x + y.reshape(B, C, N, d).transpose(1, 2)
        # MLP
        return x + self.ff(self.n_ff(x))


class NanoTabPFN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.x_enc = nn.Linear(1, d)
        self.col_enc = nn.Linear(cfg.col_emb_dim, d, bias=False)
        self.y_emb = nn.Embedding(cfg.max_classes, d)
        self.y_unknown = nn.Parameter(torch.zeros(d))
        self.y_col = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.y_unknown, std=0.02)
        nn.init.normal_(self.y_col, std=0.02)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.n_out = make_norm(cfg)
        self.head = nn.Linear(d, cfg.max_classes)

    def forward(self, X, y_train, n_classes=None, col_gen: torch.Generator | None = None):
        """X: (B,N,F) float (first n_train rows are training rows),
        y_train: (B,n_train) long. Returns logits (B, N-n_train, max_classes)."""
        B, N, Fdim = X.shape
        n_train = y_train.shape[1]
        cells = self.x_enc(X.unsqueeze(-1))                                    # (B,N,F,d)
        r = torch.randn(B, 1, Fdim, self.cfg.col_emb_dim, device=X.device,
                        generator=col_gen).to(cells.dtype)
        cells = cells + self.col_enc(r)          # random column identities (TabPFN v2)
        yt = self.y_emb(y_train)                                               # (B,n_tr,d)
        yq = self.y_unknown.expand(B, N - n_train, -1)
        ycol = (torch.cat([yt, yq], 1) + self.y_col).unsqueeze(2)              # (B,N,1,d)
        x = torch.cat([cells, ycol.to(cells.dtype)], dim=2)                    # (B,N,F+1,d)
        for blk in self.blocks:
            x = blk(x, n_train)
        out = self.head(self.n_out(x[:, n_train:, -1]))                        # (B,Nq,K)
        if n_classes is not None:
            mask = torch.arange(self.cfg.max_classes, device=X.device)[None, None] \
                >= n_classes.view(B, 1, 1)
            out = out.masked_fill(mask, float("-inf"))
        return out

    def param_groups_for_muon(self):
        """Split params: 2-D hidden matrices -> Muon, everything else -> AdamW."""
        muon, adam = [], []
        for name, p in self.named_parameters():
            if p.ndim == 2 and name.startswith("blocks."):
                muon.append(p)
            else:
                adam.append(p)
        return muon, adam
