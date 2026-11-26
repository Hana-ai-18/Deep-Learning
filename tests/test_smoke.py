"""Fast sanity checks (CPU, < 1 min). Run: python tests/test_smoke.py"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tabdl.ensemble import blend, caruana_weights  # noqa: E402
from tabdl.metrics import all_metrics  # noqa: E402
from tabdl.nano.evaluate import nano_predict_proba  # noqa: E402
from tabdl.nano.model import ModelConfig  # noqa: E402
from tabdl.nano.prior import PriorConfig, sample_batch  # noqa: E402
from tabdl.nano.train import VARIANTS, build_model, build_optimizer, TrainConfig, _loss  # noqa: E402


def test_prior():
    g = torch.Generator().manual_seed(0)
    b = sample_batch(4, PriorConfig(max_rows=64, max_features=6), g, "cpu")
    assert b["X"].ndim == 3 and torch.isfinite(b["X"]).all()
    assert (b["y"] < b["n_classes"][:, None]).all()


def test_variants_train_step():
    mc = ModelConfig(d_model=32, n_heads=4, n_layers=2, d_ff=64)
    pc = PriorConfig(max_rows=48, max_features=5)
    g = torch.Generator().manual_seed(1)
    batch = sample_batch(4, pc, g, "cpu")
    for v in VARIANTS:
        torch.manual_seed(0)
        m = build_model(v, mc)
        opt = build_optimizer(m, v, TrainConfig(variant=v))
        loss, _ = _loss(m, batch, amp=False, device=torch.device("cpu"))
        loss.backward()
        opt.step()
        assert torch.isfinite(loss), v
        # test rows must not influence each other: permuting test rows permutes outputs
        m.eval()
        n = batch["n_train"]
        X = batch["X"][:1]
        y = batch["y"][:1, :n]
        perm = torch.randperm(X.shape[1] - n) + n
        Xp = torch.cat([X[:, :n], X[:, perm]], 1)
        g1 = torch.Generator().manual_seed(5)
        g2 = torch.Generator().manual_seed(5)
        o1 = m(X, y, col_gen=g1)
        o2 = m(Xp, y, col_gen=g2)
        assert torch.allclose(o1[:, perm - n], o2, atol=1e-4), v


def test_nano_predict():
    mc = ModelConfig(d_model=32, n_heads=4, n_layers=2, d_ff=64)
    m = build_model("full", mc).eval()
    rng = np.random.default_rng(0)
    Xtr, ytr = rng.normal(size=(40, 7)), rng.integers(0, 3, 40)
    Xte = rng.normal(size=(300, 7))
    p = nano_predict_proba(m, Xtr, ytr, Xte, 3, n_ens=3, device="cpu", test_chunk=128)
    assert p.shape == (300, 3) and np.allclose(p.sum(1), 1, atol=1e-5)


def test_metrics_and_ensemble():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200)
    p1 = rng.dirichlet(np.ones(3), 200)
    p2 = np.eye(3)[y] * 0.6 + 0.4 / 3
    w = caruana_weights([p1, p2], y, 3)
    assert w[1] > w[0]
    m = all_metrics(y, blend([p1, p2], w, 3), 3)
    assert 0.5 < m["roc_auc"] <= 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all smoke tests passed")
