"""Dataset loading, preprocessing and splitting.

Two sources are supported:
  * ``openml``  : the OpenML-CC18 benchmark suite (suite id 99), filtered by size.
                  This is the benchmark family used in the TabPFN v2 Nature paper.
                  Requires internet access (enable it in Kaggle settings).
  * ``sklearn`` : a few small datasets shipped with scikit-learn (offline).
                  Used for smoke tests and as a fallback.

Every dataset is returned as a :class:`TabularDataset` holding a float matrix
``X`` (categorical columns are ordinal-encoded, missing values stay ``NaN``),
an integer label vector ``y`` in ``0..C-1`` and the list of categorical column
indices.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

CC18_SUITE_ID = 99


@dataclass
class TabularDataset:
    name: str
    X: np.ndarray                      # (n, d) float64, NaN = missing
    y: np.ndarray                      # (n,) int64 in 0..C-1
    cat_idx: list[int] = field(default_factory=list)
    source: str = ""
    did: int | None = None

    @property
    def n_classes(self) -> int:
        return int(self.y.max()) + 1

    def describe(self) -> str:
        return (f"{self.name}: n={self.X.shape[0]}, d={self.X.shape[1]}, "
                f"C={self.n_classes}, cat={len(self.cat_idx)}, "
                f"missing={np.isnan(self.X).mean():.3f}")


# --------------------------------------------------------------------------- #
# Cleaning helpers
# --------------------------------------------------------------------------- #
def _frame_to_numeric(X: pd.DataFrame, cat_mask: list[bool] | None = None):
    """Ordinal-encode categorical / object columns, keep NaN."""
    X = X.copy()
    cat_idx: list[int] = []
    cols = []
    for j, c in enumerate(X.columns):
        col = X[c]
        is_cat = (cat_mask[j] if cat_mask is not None else False) or \
            col.dtype == object or str(col.dtype) in ("category", "bool", "string")
        if is_cat:
            codes, _ = pd.factorize(col.astype("object"), sort=True)
            codes = codes.astype(np.float64)
            codes[codes < 0] = np.nan            # factorize marks NaN as -1
            cols.append(codes)
            cat_idx.append(j)
        else:
            cols.append(pd.to_numeric(col, errors="coerce").to_numpy(np.float64))
    Xn = np.column_stack(cols) if cols else np.zeros((len(X), 0))
    return Xn, cat_idx


def clean_dataset(name, X, y, cat_idx, min_per_class: int, source: str, did=None):
    """Drop NaN targets, constant columns and very rare classes; re-index labels."""
    y = pd.Series(y)
    keep = ~y.isna().to_numpy()
    X, y = X[keep], y[keep]
    counts = y.value_counts()
    ok_classes = counts[counts >= min_per_class].index
    keep = y.isin(ok_classes).to_numpy()
    X, y = X[keep], y[keep]
    y_codes, _ = pd.factorize(y.astype("object"), sort=True)

    # drop all-NaN / constant columns (keeps cat_idx consistent)
    keep_cols = []
    for j in range(X.shape[1]):
        col = X[:, j]
        finite = col[~np.isnan(col)]
        if finite.size > 0 and np.nanstd(col) > 0:
            keep_cols.append(j)
    new_cat = [keep_cols.index(j) for j in cat_idx if j in keep_cols]
    X = X[:, keep_cols]
    return TabularDataset(name=name, X=X.astype(np.float64), y=y_codes.astype(np.int64),
                          cat_idx=new_cat, source=source, did=did)


# --------------------------------------------------------------------------- #
# OpenML-CC18
# --------------------------------------------------------------------------- #
def list_cc18_candidates(max_rows=3000, max_features=100, max_classes=10,
                         min_rows=100) -> pd.DataFrame:
    import openml
    suite = openml.study.get_suite(CC18_SUITE_ID)
    meta = openml.datasets.list_datasets(data_id=list(suite.data), output_format="dataframe")
    meta = meta.rename(columns=str)
    n_feat = meta["NumberOfFeatures"].astype(float) - 1      # minus target
    mask = ((meta["NumberOfInstances"].astype(float) <= max_rows)
            & (meta["NumberOfInstances"].astype(float) >= min_rows)
            & (n_feat <= max_features)
            & (meta["NumberOfClasses"].astype(float) <= max_classes)
            & (meta["NumberOfClasses"].astype(float) >= 2))
    cand = meta[mask].copy()
    cand = cand.sort_values(["NumberOfInstances", "did"]).reset_index(drop=True)
    return cand


def load_openml_cc18(max_rows=3000, max_features=100, max_classes=10,
                     max_datasets=None, min_per_class=10, cache_dir=None,
                     verbose=True) -> list[TabularDataset]:
    import openml
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        try:
            openml.config.set_root_cache_directory(cache_dir)
        except AttributeError:            # older openml
            openml.config.cache_directory = cache_dir
    cand = list_cc18_candidates(max_rows, max_features, max_classes)
    if max_datasets is not None:
        cand = cand.head(int(max_datasets))
    out = []
    for _, row in cand.iterrows():
        did = int(row["did"])
        try:
            ds = openml.datasets.get_dataset(did, download_data=True,
                                             download_qualities=False,
                                             download_features_meta_data=False)
            X, y, cat_mask, _ = ds.get_data(target=ds.default_target_attribute,
                                            dataset_format="dataframe")
            Xn, cat_idx = _frame_to_numeric(X, cat_mask)
            d = clean_dataset(f"{ds.name}", Xn, y, cat_idx, min_per_class,
                              source="openml-cc18", did=did)
            if d.n_classes < 2 or d.X.shape[1] == 0:
                continue
            out.append(d)
            if verbose:
                print("  [ok]", d.describe())
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"OpenML dataset {did} failed: {e}")
    return out


# --------------------------------------------------------------------------- #
# sklearn fallback (offline)
# --------------------------------------------------------------------------- #
def load_sklearn_datasets(min_per_class=10) -> list[TabularDataset]:
    from sklearn import datasets as skd
    out = []
    loaders = {
        "breast_cancer": skd.load_breast_cancer,
        "wine": skd.load_wine,
        "iris": skd.load_iris,
        "digits": skd.load_digits,
    }
    for name, fn in loaders.items():
        b = fn()
        out.append(clean_dataset(name, b.data.astype(np.float64), b.target, [],
                                 min_per_class, source="sklearn"))
    # two synthetic but non-trivial problems with categorical + missing values
    rng = np.random.default_rng(0)
    for k, (n, d) in enumerate([(800, 12), (1500, 25)]):
        Xc, yc = skd.make_classification(n_samples=n, n_features=d, n_informative=d // 2,
                                         n_redundant=2, n_classes=3, flip_y=0.05,
                                         class_sep=0.8, random_state=k)
        Xc = Xc.astype(np.float64)
        cat_cols = [0, 1]
        for j in cat_cols:
            Xc[:, j] = np.digitize(Xc[:, j], np.quantile(Xc[:, j], [0.25, 0.5, 0.75]))
        Xc[rng.random(Xc.shape) < 0.03] = np.nan
        out.append(clean_dataset(f"synthetic_{k}", Xc, yc, cat_cols, min_per_class,
                                 source="sklearn"))
    return out


def get_datasets(source="auto", **kw) -> list[TabularDataset]:
    if source in ("openml", "auto"):
        try:
            ds = load_openml_cc18(**kw)
            if ds:
                return ds
            if source == "openml":
                raise RuntimeError("No OpenML dataset could be loaded.")
        except Exception as e:  # noqa: BLE001
            if source == "openml":
                raise
            warnings.warn(f"OpenML unavailable ({e}); falling back to sklearn datasets.")
    ds = load_sklearn_datasets(min_per_class=kw.get("min_per_class", 10))
    if kw.get("max_datasets"):
        ds = ds[: int(kw["max_datasets"])]
    return ds


# --------------------------------------------------------------------------- #
# Splits and preprocessing
# --------------------------------------------------------------------------- #
def make_folds(y: np.ndarray, n_folds: int, seed: int = 0):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def holdout_split(y: np.ndarray, val_frac: float, seed: int):
    """Stratified train/val split of an index range (returns positions)."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    tr, va = train_test_split(idx, test_size=val_frac, stratify=y, random_state=seed)
    return np.sort(tr), np.sort(va)


def subsample_stratified(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Stratified subsample of at most n positions (keeps every class)."""
    if len(y) <= n:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    picked = [rng.choice(np.where(y == c)[0]) for c in classes]   # >=1 per class
    rest = np.setdiff1d(np.arange(len(y)), picked)
    # proportional fill
    extra = rng.choice(rest, size=n - len(picked), replace=False)
    return np.sort(np.concatenate([picked, extra]))


class NumericPreprocessor:
    """Mean imputation + standardisation (fitted on training rows only)."""

    def fit(self, X):
        self.mean_ = np.nanmean(X, axis=0)
        self.mean_ = np.where(np.isnan(self.mean_), 0.0, self.mean_)
        Xi = np.where(np.isnan(X), self.mean_, X)
        self.std_ = Xi.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, X):
        Xi = np.where(np.isnan(X), self.mean_, X)
        return (Xi - self.mean_) / self.std_
