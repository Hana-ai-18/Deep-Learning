"""Classification metrics used throughout the project."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

EPS = 1e-7


def fix_proba(p: np.ndarray, n_classes: int) -> np.ndarray:
    """Make sure probabilities have shape (n, C), are finite and sum to one."""
    p = np.asarray(p, dtype=np.float64)
    if p.ndim == 1:
        p = np.stack([1 - p, p], axis=1)
    if p.shape[1] < n_classes:                      # model never saw some class
        p = np.concatenate([p, np.zeros((p.shape[0], n_classes - p.shape[1]))], axis=1)
    p = np.nan_to_num(p, nan=1.0 / n_classes)
    p = np.clip(p, EPS, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def roc_auc(y: np.ndarray, p: np.ndarray, n_classes: int) -> float:
    p = fix_proba(p, n_classes)
    try:
        if n_classes == 2:
            return float(roc_auc_score(y, p[:, 1]))
        present = np.unique(y)
        if len(present) < 2:
            return float("nan")
        # one-vs-rest macro AUC over the classes present in the test fold
        pp = p[:, present] / p[:, present].sum(axis=1, keepdims=True)
        if len(present) == 2:
            return float(roc_auc_score((y == present[1]).astype(int), pp[:, 1]))
        return float(roc_auc_score(y, pp, multi_class="ovr", average="macro",
                                   labels=present))
    except ValueError:
        return float("nan")


def all_metrics(y: np.ndarray, p: np.ndarray, n_classes: int) -> dict:
    p = fix_proba(p, n_classes)
    return {
        "roc_auc": roc_auc(y, p, n_classes),
        "accuracy": float(accuracy_score(y, p.argmax(1))),
        "log_loss": float(log_loss(y, p, labels=list(range(n_classes)))),
    }
