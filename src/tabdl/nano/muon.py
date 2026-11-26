"""Muon optimizer (single device), following Keller Jordan's reference code.

Muon = SGD-momentum (Nesterov) whose update for each 2-D weight matrix is
orthogonalised with a quintic Newton-Schulz iteration. It is applied to the
hidden weight matrices of the transformer only; embeddings, the output head,
norm parameters and biases are optimised with AdamW, as recommended.
Reference: K. Jordan et al., "Muon: An optimizer for hidden layers in neural
networks" (2024); J. Liu et al., "Muon is Scalable for LLM Training" (2025).
"""
from __future__ import annotations

import math

import torch


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        Bm = b * A + c * A @ A
        X = a * X + Bm @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """param_groups must contain the key ``use_muon`` (bool)."""

    def __init__(self, param_groups):
        for g in param_groups:
            if g["use_muon"]:
                g.setdefault("lr", 0.02)
                g.setdefault("momentum", 0.95)
                g.setdefault("weight_decay", 0.0)
                g.setdefault("ns_steps", 5)
            else:
                g.setdefault("lr", 1e-3)
                g.setdefault("betas", (0.9, 0.95))
                g.setdefault("eps", 1e-8)
                g.setdefault("weight_decay", 0.0)
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            if g["use_muon"]:
                for p in g["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if "momentum_buffer" not in st:
                        st["momentum_buffer"] = torch.zeros_like(p)
                    buf = st["momentum_buffer"]
                    grad = p.grad
                    buf.lerp_(grad, 1 - g["momentum"])
                    upd = grad.lerp(buf, g["momentum"])            # Nesterov
                    upd = zeropower_via_newtonschulz5(upd, g["ns_steps"])
                    upd = upd * max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                    p.add_(upd, alpha=-g["lr"])
            else:
                b1, b2 = g["betas"]
                for p in g["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if "step" not in st:
                        st["step"] = 0
                        st["exp_avg"] = torch.zeros_like(p)
                        st["exp_avg_sq"] = torch.zeros_like(p)
                    st["step"] += 1
                    st["exp_avg"].lerp_(p.grad, 1 - b1)
                    st["exp_avg_sq"].lerp_(p.grad.square(), 1 - b2)
                    bc1 = 1 - b1 ** st["step"]
                    bc2 = 1 - b2 ** st["step"]
                    denom = (st["exp_avg_sq"] / bc2).sqrt().add_(g["eps"])
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                    p.addcdiv_(st["exp_avg"], denom, value=-g["lr"] / bc1)
        return loss


def cosine_lr_factor(step: int, total: int, warmup: int, final_frac: float = 0.1) -> float:
    if step < warmup:
        return (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
