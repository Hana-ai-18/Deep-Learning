"""TabPFN-GBX: GBDT-informed in-context learning for TabPFN v2.

Motivation (observed in Part A and reported in the literature): TabPFN v2 loses
to tree ensembles mainly on datasets with many *categorical* features and on
problems with strong axis-aligned / high-order interactions. Two reasons:

1. Categorical features reach TabPFN as arbitrary integer codes. Its prior was
   generated mostly from continuous causal mechanisms, so it has no reliable way
   to learn per-category statistics from a few hundred rows.
2. Its synthetic prior gives it a smooth inductive bias; GBDTs encode a different,
   complementary bias (greedy axis-aligned splits, ordered target statistics).

TabPFN-GBX keeps the pre-trained TabPFN v2 frozen (no gradient step) and only
changes its *context*: every row gets extra columns

* **TE**  - out-of-fold, smoothed per-class target encoding of each categorical
            feature (what CatBoost does internally for categoricals);
* **GBX** - out-of-fold class log-probabilities of a CatBoost model
            (centred log-ratio, C columns; 1 logit column for binary tasks).

Training rows receive *out-of-fold* values (K-fold cross-fitting) so the context
never contains label leakage; test rows receive the average prediction of the K
fold models (same distribution as the OOF values). TabPFN then decides, in
context, how much to trust the GBDT opinion versus the raw features, i.e. it can
correct the GBDT where its own prior is stronger and defer to it where trees win.

Cost: K CatBoost fits on CPU + one TabPFN forward pass. No fine-tuning.
"""
from __future__ import annotations

import hashlib

import numpy as np
from sklearn.model_selection import StratifiedKFold

# small in-process cache so tabpfn_gbx and tabpfn_gbx_te share the CatBoost runs
_GBX_CACHE: dict = {}


def _key(*arrays, extra=""):
    h = hashlib.sha1()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    h.update(extra.encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Out-of-fold target encoding for categorical columns
# --------------------------------------------------------------------------- #
class OOFTargetEncoder:
    def __init__(self, cat_idx, n_classes, n_folds=5, smoothing=10.0, max_cols=64, seed=0):
        self.cat_idx = list(cat_idx)
        self.C = n_classes
        self.k = n_folds
        self.m = smoothing
        self.max_cols = max_cols
        self.seed = seed

    def _classes_to_encode(self, y):
        if self.C == 2:
            return [1]
        order = np.argsort(-np.bincount(y, minlength=self.C))
        per_col = max(1, self.max_cols // max(1, len(self.cat_idx)))
        return sorted(order[: min(self.C - 1, per_col)].tolist())

    @staticmethod
    def _codes(col):
        return np.where(np.isnan(col), -1, col).astype(np.int64)

    def _stats(self, codes, y, classes, prior):
        out = {}
        for v in np.unique(codes):
            msk = codes == v
            n = msk.sum()
            out[v] = np.array([(np.sum(y[msk] == c) + self.m * prior[i]) / (n + self.m)
                               for i, c in enumerate(classes)])
        return out

    def _apply(self, codes, stats, prior):
        default = prior
        return np.stack([stats.get(v, default) for v in codes])

    def fit_transform(self, X, y):
        self.classes_ = self._classes_to_encode(y)
        self.prior_ = np.array([np.mean(y == c) for c in self.classes_])
        if not self.cat_idx:
            self.full_stats_ = []
            return np.zeros((len(X), 0))
        out = np.zeros((len(X), len(self.cat_idx) * len(self.classes_)))
        skf = StratifiedKFold(self.k, shuffle=True, random_state=self.seed)
        for tr, va in skf.split(X, y):
            cols = []
            for j in self.cat_idx:
                c_tr, c_va = self._codes(X[tr, j]), self._codes(X[va, j])
                st = self._stats(c_tr, y[tr], self.classes_, self.prior_)
                cols.append(self._apply(c_va, st, self.prior_))
            out[va] = np.concatenate(cols, axis=1)
        self.full_stats_ = [self._stats(self._codes(X[:, j]), y, self.classes_, self.prior_)
                            for j in self.cat_idx]
        return out

    def transform(self, X):
        if not self.cat_idx:
            return np.zeros((len(X), 0))
        cols = [self._apply(self._codes(X[:, j]), st, self.prior_)
                for j, st in zip(self.cat_idx, self.full_stats_)]
        return np.concatenate(cols, axis=1)


# --------------------------------------------------------------------------- #
# Out-of-fold CatBoost log-probabilities
# --------------------------------------------------------------------------- #
def _clr(p, C):
    p = np.clip(p, 1e-6, 1.0)
    lp = np.log(p)
    if C == 2:
        return (lp[:, 1] - lp[:, 0])[:, None]
    return lp - lp.mean(axis=1, keepdims=True)


class OOFGBDTFeaturizer:
    def __init__(self, cat_idx, n_classes, n_folds=5, iterations=500, seed=0, model="catboost"):
        self.cat_idx = list(cat_idx)
        self.C = n_classes
        self.k = n_folds
        self.iterations = iterations
        self.seed = seed
        self.model = model

    def _new(self, seed):
        if self.model == "catboost":
            from .models import CatBoostWrapper
            m = CatBoostWrapper(self.cat_idx, seed)
            m.model.set_params(iterations=self.iterations)
            return m
        if self.model == "lgbm":                       # fast stand-in (tests)
            from lightgbm import LGBMClassifier
            from .models import SklearnWrapper
            return SklearnWrapper(LGBMClassifier(n_estimators=100, verbose=-1,
                                                 random_state=seed))
        raise ValueError(self.model)

    def _proba(self, m, X, y_seen):
        p = np.asarray(m.predict_proba(X))
        if p.shape[1] < self.C:                        # a class absent from a fold
            full = np.full((len(X), self.C), 1e-6)
            full[:, y_seen] = p
            p = full / full.sum(1, keepdims=True)
        return p

    def fit_transform(self, X, y):
        key = _key(X, y, extra=f"{self.cat_idx}{self.k}{self.iterations}{self.seed}{self.model}")
        if key in _GBX_CACHE:
            oof, self.models_ = _GBX_CACHE[key]
            return oof
        oof = np.zeros((len(X), self.C))
        self.models_ = []
        skf = StratifiedKFold(self.k, shuffle=True, random_state=self.seed)
        for i, (tr, va) in enumerate(skf.split(X, y)):
            m = self._new(self.seed + i)
            m.fit(X[tr], y[tr])
            oof[va] = self._proba(m, X[va], np.unique(y[tr]))
            self.models_.append((m, np.unique(y[tr])))
        oof = _clr(oof, self.C)
        while len(_GBX_CACHE) >= 4:                    # keep memory bounded (FIFO)
            _GBX_CACHE.pop(next(iter(_GBX_CACHE)))
        _GBX_CACHE[key] = (oof, self.models_)
        return oof

    def transform(self, X):
        p = np.mean([self._proba(m, X, seen) for m, seen in self.models_], axis=0)
        return _clr(p, self.C)


# --------------------------------------------------------------------------- #
# The augmented in-context learner
# --------------------------------------------------------------------------- #
class AugmentedICL:
    """``base_factory(cat_idx) -> model`` with fit/predict_proba (e.g. TabPFN v2)."""

    def __init__(self, base_factory, cat_idx, n_classes, use_te=True, use_gbx=True,
                 n_folds=5, gbdt_iterations=500, seed=0, gbdt_model="catboost"):
        self.base_factory = base_factory
        self.cat_idx = list(cat_idx)
        self.C = n_classes
        self.use_te = use_te and len(self.cat_idx) > 0
        self.use_gbx = use_gbx
        self.k = n_folds
        self.iters = gbdt_iterations
        self.seed = seed
        self.gbdt_model = gbdt_model

    def _augment_fit(self, X, y):
        parts = [X]
        if self.use_te:
            self.te_ = OOFTargetEncoder(self.cat_idx, self.C, self.k, seed=self.seed)
            parts.append(self.te_.fit_transform(X, y))
        if self.use_gbx:
            self.gbx_ = OOFGBDTFeaturizer(self.cat_idx, self.C, self.k, self.iters,
                                          seed=self.seed, model=self.gbdt_model)
            parts.append(self.gbx_.fit_transform(X, y))
        return np.concatenate(parts, axis=1)

    def _augment(self, X):
        parts = [X]
        if self.use_te:
            parts.append(self.te_.transform(X))
        if self.use_gbx:
            parts.append(self.gbx_.transform(X))
        return np.concatenate(parts, axis=1)

    def fit(self, X, y):
        Xa = self._augment_fit(X, y)
        self.n_extra_ = Xa.shape[1] - X.shape[1]
        self.model_ = self.base_factory(self.cat_idx)   # original cat columns unchanged
        self.model_.fit(Xa, y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self._augment(X))
