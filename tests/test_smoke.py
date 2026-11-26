"""Fast sanity checks (CPU, < 1 min). Run: python tests/test_smoke.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tabdl.augment import OOFTargetEncoder  # noqa: E402
from tabdl.data import load_sklearn_datasets, make_folds  # noqa: E402
from tabdl.ensemble import blend, caruana_weights  # noqa: E402
from tabdl.metrics import all_metrics, roc_auc  # noqa: E402
from tabdl.models import make_model  # noqa: E402


def _cat_problem(n=900, k=20, seed=1, random_labels=False):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.integers(0, k, n), rng.integers(0, k, n),
                         rng.normal(size=(n, 3))]).astype(float)
    eff = rng.normal(size=k) * 1.5
    logit = eff[X[:, 0].astype(int)] + X[:, 2]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    if random_labels:
        y = rng.integers(0, 2, n)
    return X, y


def test_target_encoder_shapes():
    X, y = _cat_problem()
    te = OOFTargetEncoder([0, 1], 2, n_folds=5)
    A = te.fit_transform(X, y)
    B = te.transform(X[:10])
    assert A.shape == (len(X), 2) and B.shape == (10, 2)
    assert np.all((A > 0) & (A < 1))


def test_no_label_leakage():
    """On random labels the augmented model must stay at chance level."""
    X, y = _cat_problem(random_labels=True)
    m = make_model("logreg_gbx_te", [0, 1], 0, 2, {"gbx_folds": 4})
    m.fit(X[:600], y[:600])
    auc = roc_auc(y[600:], m.predict_proba(X[600:]), 2)
    assert 0.38 < auc < 0.62, auc


def test_augmentation_helps_on_categorical_signal():
    X, y = _cat_problem()
    aucs = {}
    for name in ["logreg", "logreg_gbx_te"]:
        m = make_model(name, [0, 1], 0, 2, {"gbx_folds": 4})
        m.fit(X[:600], y[:600])
        aucs[name] = roc_auc(y[600:], m.predict_proba(X[600:]), 2)
    assert aucs["logreg_gbx_te"] > aucs["logreg"] + 0.05, aucs


def test_metrics_and_ensemble():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200)
    p1 = rng.dirichlet(np.ones(3), 200)
    p2 = np.eye(3)[y] * 0.6 + 0.4 / 3
    w = caruana_weights([p1, p2], y, 3)
    assert w[1] > w[0]
    m = all_metrics(y, blend([p1, p2], w, 3), 3)
    assert 0.5 < m["roc_auc"] <= 1.0


def test_data():
    ds = load_sklearn_datasets()
    assert len(ds) >= 4
    for d in ds:
        folds = make_folds(d.y, 3)
        assert all(len(np.unique(d.y[tr])) == d.n_classes for tr, _ in folds)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all smoke tests passed")
