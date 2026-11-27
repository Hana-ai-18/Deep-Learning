"""Model zoo: classical baselines, TabPFN v2 and fine-tuned TabPFN v2.

Every model exposes ``fit(X, y)`` and ``predict_proba(X)`` and accepts the raw
float matrix produced by :mod:`tabdl.data` (ordinal-coded categoricals, NaN for
missing values). Each wrapper does its own model-specific preprocessing.
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_JOBS = max(1, (os.cpu_count() or 2))


def has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Classical baselines (default hyper-parameters, as in the TabPFN v2 paper's
# "default" comparison)
# --------------------------------------------------------------------------- #
class SklearnWrapper:
    def __init__(self, est):
        self.est = est

    def fit(self, X, y):
        self.est.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.est.predict_proba(X)


class CatBoostWrapper:
    def __init__(self, cat_idx, seed, use_gpu=False):
        from catboost import CatBoostClassifier
        self.cat_idx = list(cat_idx)
        self.model = CatBoostClassifier(
            random_seed=seed, verbose=0, thread_count=N_JOBS,
            allow_writing_files=False,
            task_type="GPU" if use_gpu else "CPU")

    def _prep(self, X):
        import pandas as pd
        df = pd.DataFrame(X)
        for j in self.cat_idx:
            df[j] = np.where(np.isnan(X[:, j]), -1, X[:, j]).astype(np.int64)
        return df

    def fit(self, X, y):
        self.model.fit(self._prep(X), y, cat_features=self.cat_idx or None)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self._prep(X))


def make_baseline(name: str, cat_idx, seed: int, n_classes: int):
    if name == "logreg":
        return SklearnWrapper(make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(),
                                            LogisticRegression(max_iter=3000)))
    if name == "knn":
        return SklearnWrapper(make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(),
                                            KNeighborsClassifier(n_neighbors=15,
                                                                 weights="distance")))
    if name == "rf":
        return SklearnWrapper(make_pipeline(
            SimpleImputer(strategy="mean"),
            RandomForestClassifier(n_estimators=500, n_jobs=N_JOBS, random_state=seed)))
    if name == "xgb":
        from xgboost import XGBClassifier
        return SklearnWrapper(XGBClassifier(
            tree_method="hist", device="cuda" if has_cuda() else "cpu",
            random_state=seed, n_jobs=N_JOBS, verbosity=0))
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return SklearnWrapper(LGBMClassifier(random_state=seed, n_jobs=N_JOBS, verbose=-1))
    if name == "catboost":
        return CatBoostWrapper(cat_idx, seed)
    raise ValueError(f"unknown baseline {name}")


# --------------------------------------------------------------------------- #
# TabPFN v2
# --------------------------------------------------------------------------- #
# "original" = checkpoint released with the Nature paper (Hollmann et al. 2025)
# "default"  = default v2 checkpoint of the tabpfn package
TABPFN_V2_CKPTS = {
    "original": "tabpfn-v2-classifier.ckpt",
    "default": None,
}


def _v2_model_path(ckpt: str):
    from tabpfn.model_loading import prepend_cache_path
    fname = TABPFN_V2_CKPTS.get(ckpt, ckpt)
    return None if fname is None else prepend_cache_path(fname)


class TabPFNv2Wrapper:
    def __init__(self, cat_idx, seed, n_estimators=8, ckpt="original", device=None,
                 inference_config=None, **extra):
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        kw = dict(n_estimators=n_estimators, random_state=seed,
                  device=device or ("cuda" if has_cuda() else "cpu"),
                  categorical_features_indices=list(cat_idx) or None,
                  ignore_pretraining_limits=True, **extra)
        if inference_config:
            kw["inference_config"] = inference_config
        mp = _v2_model_path(ckpt)
        if mp is not None:
            kw["model_path"] = mp
        self.model = TabPFNClassifier.create_default_for_version(ModelVersion.V2, **kw)

    def fit(self, X, y):
        self.model.fit(X.astype(np.float32), y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X.astype(np.float32))


class FinetunedTabPFNv2Wrapper:
    """Official full fine-tuning of TabPFN v2 on one dataset (tabpfn.finetuning).

    Rubachev et al. (2025, "On Finetuning Tabular Foundation Models") found full
    fine-tuning to be the most practical adaptation strategy for TabPFN v2.
    """

    def __init__(self, cat_idx, seed, n_estimators=8, ckpt="original", device=None,
                 time_limit=60, epochs=30, learning_rate=1e-5):
        from tabpfn.constants import ModelVersion
        from tabpfn.finetuning import FinetunedTabPFNClassifier
        extra = {"categorical_features_indices": list(cat_idx) or None}
        mp = _v2_model_path(ckpt)
        if mp is not None:
            extra["model_path"] = mp
        self.model = FinetunedTabPFNClassifier(
            device=device or ("cuda" if has_cuda() else "cpu"),
            epochs=epochs, time_limit=time_limit, learning_rate=learning_rate,
            random_state=seed, early_stopping=True, early_stopping_patience=4,
            n_estimators_finetune=2, n_estimators_validation=2,
            n_estimators_final_inference=n_estimators,
            eval_metric="log_loss", model_version=ModelVersion.V2,
            extra_classifier_kwargs=extra)

    def fit(self, X, y):
        import warnings
        # the internal early-stopping split must contain every class at least twice
        n, c = len(y), len(np.unique(y))
        self.model.validation_split_ratio = float(min(0.3, max(0.1, 2.0 * c / max(n, 1))))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X.astype(np.float32), y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X.astype(np.float32))


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Portfolio of TabPFN v2 inference configurations for post-hoc ensembling (PHE).
# The Nature paper reports that post-hoc ensembling of TabPFN v2 configurations
# ("TabPFN (PHE)") improves over the default; here each member changes the
# preprocessing / decoding of the *same frozen* checkpoint.
# --------------------------------------------------------------------------- #
def _pc(name, cat, **kw):
    from tabpfn.preprocessing.configs import PreprocessorConfig
    kw.setdefault("max_features_per_estimator", 10_000)   # same as the v2 default
    return PreprocessorConfig(name, categorical_name=cat, **kw)


def pfn_portfolio_config(name: str) -> dict:
    """Return TabPFNv2Wrapper kwargs for a portfolio member."""
    if name == "pfn_onehot":        # one-hot categoricals (targets the categorical weakness)
        return dict(inference_config={"PREPROCESS_TRANSFORMS": [
            _pc("quantile_uni_coarse", "onehot", append_original="auto",
                global_transformer_name="svd", max_onehot_cardinality=12),
            _pc("none", "onehot", max_onehot_cardinality=12)]})
    if name == "pfn_ordinal":       # sorted / shuffled ordinal codes + other scalers
        return dict(inference_config={"PREPROCESS_TRANSFORMS": [
            _pc("quantile_norm", "ordinal_shuffled", append_original="auto"),
            _pc("safepower", "ordinal")]})
    if name == "pfn_kdi":           # kernel-density-integral transforms
        return dict(inference_config={"PREPROCESS_TRANSFORMS": [
            _pc("kdi_alpha_1.0_uni", "numeric", append_original=True),
            _pc("robust", "numeric")]})
    if name == "pfn_poly":          # + 50 pairwise product features
        return dict(inference_config={"POLYNOMIAL_FEATURES": 50})
    if name == "pfn_sharp":         # lower softmax temperature
        return dict(softmax_temperature=0.75)
    if name == "pfn_avgsm":         # average logits instead of probabilities
        return dict(average_before_softmax=True)
    if name == "pfn_wide":          # 4x more ensemble members
        return dict(n_estimators=32)
    raise ValueError(name)


PFN_PORTFOLIO = ["pfn_onehot", "pfn_ordinal", "pfn_kdi", "pfn_poly", "pfn_sharp",
                 "pfn_avgsm", "pfn_wide"]


# methods whose single fit yields several result variants (ablations)
VARIANT_METHODS = {
    "oofphe": ["oofphe_pfn", "oofphe_pfn_ts", "oofphe_all", "oofphe_all_ts", "oofphe_safe"],
    "oofphe_test": ["oofphe_pfn", "oofphe_pfn_ts", "oofphe_all", "oofphe_all_ts",
                    "oofphe_safe"],
}


def make_model(name: str, cat_idx, seed: int, n_classes: int, cfg: dict | None = None):
    cfg = cfg or {}
    if name in ("oofphe", "oofphe_test"):
        from .oofstack import OOFStackedPHE
        kw = dict(n_folds=cfg.get("oof_folds", 5), gbx_inner_folds=cfg.get("gbx_inner", 3),
                  gbx_iterations=cfg.get("gbx_iterations", 300),
                  cat_iterations=cfg.get("cat_iterations", 500))
        if name == "oofphe_test":           # CPU stand-in: baselines instead of TabPFN
            cfg = {**cfg, "gbx_model": "lgbm"}
            kw["pfn_members"] = ["tabpfn_v2", "knn", "rf"]
            cfg = {**cfg, "anchor_as": "logreg"}
        return OOFStackedPHE(cat_idx, n_classes, seed=seed, cfg=cfg, **kw)
    if name in PFN_PORTFOLIO:
        kw = dict(n_estimators=cfg.get("n_estimators", 8), ckpt=cfg.get("ckpt", "original"))
        kw.update(pfn_portfolio_config(name))
        return TabPFNv2Wrapper(cat_idx, seed, **kw)
    if name == "gbx_onehot":        # TabPFN-GBX on top of the one-hot configuration
        from .augment import AugmentedICL

        def factory(ci):
            kw = dict(n_estimators=cfg.get("n_estimators", 8), ckpt=cfg.get("ckpt", "original"))
            kw.update(pfn_portfolio_config("pfn_onehot"))
            return TabPFNv2Wrapper(ci, seed, **kw)
        return AugmentedICL(factory, cat_idx, n_classes, use_te=True, use_gbx=True,
                            n_folds=cfg.get("gbx_folds", 5),
                            gbdt_iterations=cfg.get("gbx_iterations", 500), seed=seed)
    if name == "tabpfn_v2":
        return TabPFNv2Wrapper(cat_idx, seed, n_estimators=cfg.get("n_estimators", 8),
                               ckpt=cfg.get("ckpt", "original"))
    if name == "tabpfn_v2_ft":
        return FinetunedTabPFNv2Wrapper(cat_idx, seed,
                                        n_estimators=cfg.get("n_estimators", 8),
                                        ckpt=cfg.get("ckpt", "original"),
                                        time_limit=cfg.get("ft_time_limit", 60),
                                        epochs=cfg.get("ft_epochs", 30),
                                        learning_rate=cfg.get("ft_lr", 1e-5))
    if name in AUGMENTED:
        from .augment import AugmentedICL
        base, te, gbx = AUGMENTED[name]
        if base == "tabpfn_v2":
            def factory(ci):
                return TabPFNv2Wrapper(ci, seed, n_estimators=cfg.get("n_estimators", 8),
                                       ckpt=cfg.get("ckpt", "original"))
            gbdt = "catboost"
        else:                                   # CPU stand-in used only by the tests
            def factory(ci):
                return make_baseline(base, ci, seed, n_classes)
            gbdt = "lgbm"
        return AugmentedICL(factory, cat_idx, n_classes, use_te=te, use_gbx=gbx,
                            n_folds=cfg.get("gbx_folds", 5),
                            gbdt_iterations=cfg.get("gbx_iterations", 500),
                            seed=seed, gbdt_model=cfg.get("gbx_model", gbdt))
    return make_baseline(name, cat_idx, seed, n_classes)


# name -> (base model, use target encoding, use OOF-GBDT logits)
AUGMENTED = {
    "tabpfn_te":       ("tabpfn_v2", True,  False),   # ablation: categorical fix only
    "tabpfn_gbx":      ("tabpfn_v2", False, True),    # ablation: GBDT logits only
    "tabpfn_gbx_te":   ("tabpfn_v2", True,  True),    # TabPFN-GBX (ours)
    "logreg_gbx_te":   ("logreg",    True,  True),    # CPU stand-in for tests
}
