"""OOF-PHE: cross-validated post-hoc ensembling of TabPFN v2 configurations.

Diagnosis from the previous run (holdout PHE): ensemble weights were chosen on a
single 20 % inner split (often < 200 rows). The selection was noisy - e.g. the
default TabPFN v2 got weight 0.01 although it is one of the best members - which
capped the gain at ~+0.0004 AUC.

OOF-PHE fixes the *estimation* of the ensemble instead of adding more models:

1. **Out-of-fold predictions on 100 % of the training rows.** For K-fold
   cross-fitting every member predicts each training row from a model that did
   not see it. For TabPFN this costs only K extra forward passes (no training),
   so it is cheap on a T4. Weights are then selected on n rows instead of 0.2 n.
2. **Full-context test predictions.** TabPFN members predict the test rows with
   the whole training fold as context (no refit cost); CatBoost uses the average
   of its K fold models (bagging).
3. **Caruana ensemble selection on the OOF predictions**, followed by a single
   **temperature** fitted on the OOF blend (log-loss), which fixes the
   under-confidence that averaging introduces.
4. Optional GBDT-informed members: CatBoost and TabPFN-GBX. TabPFN-GBX is
   cross-fitted in a *nested* way (its own inner K' folds inside every outer
   fold) so that its OOF predictions are leakage-free.

5. **Safe mode - keep what the original does well ("do no harm").** TabPFN v2 is
   already the best model on 2/3 of the datasets, so the final variant
   (a) anchors the selection on TabPFN v2 (it starts with ``anchor_count`` copies
   of it, i.e. shrinkage towards the original), and
   (b) estimates *honestly* - by cross-validating the selection procedure itself on
   the OOF rows - whether the ensemble beats TabPFN v2 row by row. Only if the
   per-row log-loss improvement is positive and significant (one-sided paired
   t-test, z > ``gate_z``) is the ensemble used; otherwise the prediction falls
   back to the unchanged TabPFN v2. Gains are therefore spent where the original
   is weak, and its strengths are preserved where it is strong.

One ``fit`` produces several variants (for the ablation study) at no extra cost:
  oofphe_pfn      TabPFN v2 configurations only, OOF weights
  oofphe_pfn_ts   + temperature scaling
  oofphe_all      + CatBoost + TabPFN-GBX members
  oofphe_all_ts   + temperature scaling
  oofphe_safe     anchored selection + temperature + significance gate  (FINAL)
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedKFold

from .ensemble import caruana_weights
from .metrics import fix_proba

PFN_MEMBERS = ["tabpfn_v2", "pfn_onehot", "pfn_ordinal", "pfn_kdi", "pfn_poly",
               "pfn_avgsm", "pfn_wide"]


def _nll(y, p):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-7, 1.0))))


def apply_temperature(p, T):
    lp = np.log(np.clip(p, 1e-7, 1.0)) / T
    lp -= lp.max(axis=1, keepdims=True)
    q = np.exp(lp)
    return q / q.sum(axis=1, keepdims=True)


def fit_temperature(p, y):
    res = minimize_scalar(lambda T: _nll(y, apply_temperature(p, T)),
                          bounds=(0.25, 4.0), method="bounded")
    return float(res.x)


class OOFStackedPHE:
    self_validating = True       # run_benchmark: no extra inner-validation refit

    def __init__(self, cat_idx, n_classes, seed=0, cfg=None, pfn_members=None,
                 use_gbdt=True, n_folds=5, n_iter=50, gbx_inner_folds=3,
                 gbx_iterations=300, cat_iterations=500, anchor_count=10, gate_z=1.645):
        self.cat_idx = list(cat_idx)
        self.C = n_classes
        self.seed = seed
        self.cfg = cfg or {}
        self.pfn_members = pfn_members or PFN_MEMBERS
        self.use_gbdt = use_gbdt
        self.k = n_folds
        self.n_iter = n_iter
        self.gbx_inner = gbx_inner_folds
        self.gbx_iter = gbx_iterations
        self.cat_iter = cat_iterations
        self.anchor_count = anchor_count
        self.gate_z = gate_z

    # ---------------- member factories ----------------
    def _pfn(self, name):
        from .models import make_model
        if name == "tabpfn_v2" and self.cfg.get("anchor_as"):   # CPU stand-in (tests)
            name = self.cfg["anchor_as"]
        return make_model(name, self.cat_idx, self.seed, self.C, self.cfg)

    def _gbx(self):
        from .augment import AugmentedICL
        from .models import TabPFNv2Wrapper
        cfg = self.cfg

        def factory(ci):
            if cfg.get("gbx_model") == "lgbm":            # CPU stand-in for the tests
                from .models import make_baseline
                return make_baseline("logreg", ci, self.seed, self.C)
            return TabPFNv2Wrapper(ci, self.seed, n_estimators=cfg.get("n_estimators", 8),
                                   ckpt=cfg.get("ckpt", "original"))
        return AugmentedICL(factory, self.cat_idx, self.C, use_te=True, use_gbx=True,
                            n_folds=self.gbx_inner, gbdt_iterations=self.gbx_iter,
                            seed=self.seed, gbdt_model=cfg.get("gbx_model", "catboost"))

    def _cat(self, seed):
        if self.cfg.get("gbx_model") == "lgbm":          # CPU stand-in for the tests
            from lightgbm import LGBMClassifier
            from .models import SklearnWrapper
            return SklearnWrapper(LGBMClassifier(n_estimators=100, verbose=-1,
                                                 random_state=seed))
        from .models import CatBoostWrapper
        m = CatBoostWrapper(self.cat_idx, seed)
        m.model.set_params(iterations=self.cat_iter)
        return m

    def _proba(self, m, X, seen):
        p = np.asarray(m.predict_proba(X))
        if p.shape[1] < self.C:
            full = np.full((len(X), self.C), 1e-7)
            full[:, seen] = p
            p = full
        return fix_proba(p, self.C)

    # ---------------- fit ----------------
    def fit(self, X, y):
        from .utils import free_gpu
        t0 = time.time()
        skf = StratifiedKFold(self.k, shuffle=True, random_state=self.seed)
        folds = list(skf.split(X, y))
        names = list(self.pfn_members) + (["catboost", "tabpfn_gbx_te"] if self.use_gbdt else [])
        self.oof_ = {n: np.zeros((len(X), self.C)) for n in names}
        self.test_models_ = {}
        self.failed_ = []
        # TabPFN configurations: K forward passes for OOF + full-context model for test
        for n in self.pfn_members:
            try:
                for tr, va in folds:
                    m = self._pfn(n).fit(X[tr], y[tr])
                    self.oof_[n][va] = self._proba(m, X[va], np.unique(y[tr]))
                    del m
                self.test_models_[n] = self._pfn(n).fit(X, y)
            except Exception as e:  # noqa: BLE001
                self.failed_.append(f"{n}: {type(e).__name__}: {e}")
                self.oof_.pop(n, None)
            free_gpu()
        if self.use_gbdt:
            # CatBoost: the K fold models give the OOF predictions and, averaged, the test one
            cats = []
            for i, (tr, va) in enumerate(folds):
                m = self._cat(self.seed + i).fit(X[tr], y[tr])
                self.oof_["catboost"][va] = self._proba(m, X[va], np.unique(y[tr]))
                cats.append((m, np.unique(y[tr])))
            self.test_models_["catboost"] = cats
            # TabPFN-GBX, nested cross-fitting (inner folds live inside each outer fold)
            try:
                for tr, va in folds:
                    g = self._gbx().fit(X[tr], y[tr])
                    self.oof_["tabpfn_gbx_te"][va] = self._proba(g, X[va], np.unique(y[tr]))
                    del g
                    free_gpu()
                self.test_models_["tabpfn_gbx_te"] = self._gbx().fit(X, y)
            except Exception as e:  # noqa: BLE001
                self.failed_.append(f"tabpfn_gbx_te: {type(e).__name__}: {e}")
                self.oof_.pop("tabpfn_gbx_te", None)
        self.members_ = [n for n in names if n in self.oof_]
        self.oof_ll_ = {n: _nll(y, self.oof_[n]) for n in self.members_}
        # ensemble selection on OOF predictions
        pfn = [n for n in self.members_ if n in self.pfn_members]
        self.weights_ = {}
        self.temps_ = {}
        for var, mem in [("pfn", pfn), ("all", self.members_)]:
            w = caruana_weights([self.oof_[n] for n in mem], y, self.C, self.n_iter)
            self.weights_[var] = dict(zip(mem, w))
            blend = sum(wi * self.oof_[n] for n, wi in zip(mem, w))
            self.temps_[var] = fit_temperature(fix_proba(blend, self.C), y)
        self._fit_safe(y)
        self.fit_time_ = time.time() - t0
        return self

    # ---------------- safe mode ----------------
    def _select(self, mem, y, rows):
        anchor = mem.index("tabpfn_v2") if "tabpfn_v2" in mem else None
        w = caruana_weights([self.oof_[n][rows] for n in mem], y[rows], self.C, self.n_iter,
                            anchor=anchor, anchor_count=self.anchor_count)
        blend = fix_proba(sum(wi * self.oof_[n][rows] for n, wi in zip(mem, w)), self.C)
        return w, fit_temperature(blend, y[rows])

    def _fit_safe(self, y):
        mem = self.members_
        w, T = self._select(mem, y, np.arange(len(y)))
        self.weights_["safe"] = dict(zip(mem, w))
        self.temps_["safe"] = T
        self.gate_ = dict(use_ensemble=True, z=float("nan"), mean_gain=float("nan"))
        if "tabpfn_v2" not in mem:
            return
        # honest estimate: cross-validate the *selection procedure* on the OOF rows
        idx = np.arange(len(y))
        oof_sel = np.zeros((len(y), self.C))
        skf = StratifiedKFold(5, shuffle=True, random_state=self.seed + 7)
        for a_, b_ in skf.split(idx, y):
            wa, Ta = self._select(mem, y, a_)
            blend = fix_proba(sum(wi * self.oof_[n][b_] for n, wi in zip(mem, wa)), self.C)
            oof_sel[b_] = apply_temperature(blend, Ta)
        ar = np.arange(len(y))
        base_loss = -np.log(np.clip(self.oof_["tabpfn_v2"][ar, y], 1e-7, 1))
        ens_loss = -np.log(np.clip(oof_sel[ar, y], 1e-7, 1))
        d = base_loss - ens_loss                      # > 0 = ensemble better
        z = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
        self.gate_ = dict(use_ensemble=bool(z > self.gate_z), z=float(z),
                          mean_gain=float(d.mean()))

    # ---------------- predict ----------------
    def _member_test(self, n, X):
        if n == "catboost":
            return np.mean([self._proba(m, X, s) for m, s in self.test_models_[n]], axis=0)
        return self._proba(self.test_models_[n], X, np.arange(self.C))

    def predict_variants(self, X):
        P = {n: self._member_test(n, X) for n in self.members_}
        out = {}
        for var in ["pfn", "all"]:
            w = self.weights_[var]
            blend = fix_proba(sum(wi * P[n] for n, wi in w.items()), self.C)
            out[f"oofphe_{var}"] = blend
            out[f"oofphe_{var}_ts"] = apply_temperature(blend, self.temps_[var])
        if self.gate_["use_ensemble"] or "tabpfn_v2" not in P:
            w = self.weights_["safe"]
            blend = fix_proba(sum(wi * P[n] for n, wi in w.items()), self.C)
            out["oofphe_safe"] = apply_temperature(blend, self.temps_["safe"])
        else:                                          # keep the original prediction
            out["oofphe_safe"] = P["tabpfn_v2"]
        return out

    def predict_proba(self, X):
        return self.predict_variants(X)["oofphe_safe"]

    def info(self):
        return dict(weights={v: {n: float(x) for n, x in w.items()}
                             for v, w in self.weights_.items()},
                    temperature={v: float(t) for v, t in self.temps_.items()},
                    oof_logloss={n: float(x) for n, x in self.oof_ll_.items()},
                    gate=self.gate_, failed=self.failed_)
