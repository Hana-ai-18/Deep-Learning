"""Post-hoc ensemble selection (Caruana et al., 2004) on hold-out predictions.

Given validation-set probabilities of M base models, greedily add (with
replacement) the model that most decreases the validation log-loss of the
averaged ensemble. The resulting non-negative weights are then applied to the
test-set probabilities of the same models refit on the full training fold.
"""
from __future__ import annotations

import numpy as np

from .metrics import fix_proba


def _nll(y, p):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-7, 1.0))))


def caruana_weights(val_probas: list[np.ndarray], y_val: np.ndarray, n_classes: int,
                    n_iter: int = 25, anchor: int | None = None,
                    anchor_count: int = 0) -> np.ndarray:
    """Greedy ensemble selection with replacement.

    ``anchor``/``anchor_count``: start with ``anchor_count`` copies of member
    ``anchor`` (e.g. the original TabPFN v2). This shrinks the weights towards the
    anchor, so the ensemble only moves away from it when the data clearly say so.
    """
    P = [fix_proba(p, n_classes) for p in val_probas]
    M = len(P)
    counts = np.zeros(M)
    if anchor is not None and anchor_count > 0:
        counts[anchor] = anchor_count
        cur = P[anchor].copy()
    else:                                   # start from the best single model
        best = int(np.argmin([_nll(y_val, p) for p in P]))
        counts[best] = 1
        cur = P[best].copy()
    for _ in range(n_iter - 1):
        k = counts.sum()
        scores = [_nll(y_val, (cur * k + P[m]) / (k + 1)) for m in range(M)]
        m = int(np.argmin(scores))
        counts[m] += 1
        cur = (cur * k + P[m]) / (k + 1)
    return counts / counts.sum()


def blend(probas: list[np.ndarray], weights: np.ndarray, n_classes: int) -> np.ndarray:
    out = sum(w * fix_proba(p, n_classes) for w, p in zip(weights, probas))
    return fix_proba(out, n_classes)
