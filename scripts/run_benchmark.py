#!/usr/bin/env python
"""Benchmark TabPFN v2 (and its fine-tuned variant) against classical baselines.

Part A (reproduction) and Part B-1 (TabPFN-level improvement) both use this
script; only the list of methods differs.

For every dataset x fold x method it stores
  * a row in ``<out>/metrics.jsonl`` (ROC-AUC, accuracy, log-loss, timings)
  * the predicted probabilities in ``<out>/preds/<dataset>__f<k>__<method>.npz``
With ``--val_preds`` every model is additionally fit on an inner 80 % split of
the training fold and evaluated on the remaining 20 % (needed by
``run_ensembles.py``). Runs are resumable: finished (dataset, fold, method)
triples are skipped, so a Kaggle session can be restarted safely.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tabdl.data import get_datasets, holdout_split, make_folds  # noqa: E402
from tabdl.metrics import all_metrics, fix_proba  # noqa: E402
from tabdl.models import VARIANT_METHODS, has_cuda, make_model  # noqa: E402
from tabdl.utils import append_jsonl, free_gpu, set_seed, slug  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="auto", choices=["auto", "openml", "sklearn"])
    ap.add_argument("--max_rows", type=int, default=3000)
    ap.add_argument("--max_features", type=int, default=100)
    ap.add_argument("--max_classes", type=int, default=10)
    ap.add_argument("--max_datasets", type=int, default=None)
    ap.add_argument("--n_folds", type=int, default=3)
    ap.add_argument("--methods", default="logreg,rf,xgb,lgbm,catboost,tabpfn_v2")
    ap.add_argument("--val_preds", action="store_true",
                    help="also store inner-validation predictions (for ensembles)")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--ckpt", default="original", help="TabPFN v2 checkpoint: original|default")
    ap.add_argument("--n_estimators", type=int, default=8)
    ap.add_argument("--ft_time_limit", type=int, default=60)
    ap.add_argument("--ft_epochs", type=int, default=30)
    ap.add_argument("--ft_lr", type=float, default=1e-5)
    ap.add_argument("--gbx_folds", type=int, default=5, help="K of the OOF cross-fitting")
    ap.add_argument("--gbx_iterations", type=int, default=500, help="CatBoost trees (GBX)")
    ap.add_argument("--oof_folds", type=int, default=5, help="K of OOF-PHE")
    ap.add_argument("--gbx_inner", type=int, default=3, help="inner folds of nested GBX")
    ap.add_argument("--cat_iterations", type=int, default=500, help="CatBoost trees in OOF-PHE")
    ap.add_argument("--shard", type=int, default=0, help="run datasets with index %% n_shards == shard")
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_dir", default=os.environ.get("OPENML_CACHE", "openml_cache"))
    ap.add_argument("--out", default="results/benchmark")
    return ap.parse_args()


def _align(p, ytr, n_classes):
    # map model class order -> global labels (models only know classes seen in ytr)
    seen = np.unique(ytr)
    p = np.asarray(p)
    if p.shape[1] == len(seen) and len(seen) < n_classes:
        full = np.zeros((p.shape[0], n_classes))
        full[:, seen] = p
        p = full
    return fix_proba(p, n_classes)


def fit_predict(method, Xtr, ytr, Xte, cat_idx, seed, n_classes, cfg):
    """Return ({method_or_variant: proba}, fit_time, predict_time, info)."""
    model = make_model(method, cat_idx, seed, n_classes, cfg)
    t0 = time.time()
    model.fit(Xtr, ytr)
    t1 = time.time()
    if hasattr(model, "predict_variants"):
        out = {k: _align(v, ytr, n_classes) for k, v in model.predict_variants(Xte).items()}
    else:
        out = {method: _align(model.predict_proba(Xte), ytr, n_classes)}
    t2 = time.time()
    info = model.info() if hasattr(model, "info") else None
    del model
    free_gpu()
    return out, t1 - t0, t2 - t1, info


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    a = parse()
    os.makedirs(os.path.join(a.out, "preds"), exist_ok=True)
    set_seed(a.seed)
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    cfg = dict(ckpt=a.ckpt, n_estimators=a.n_estimators, ft_time_limit=a.ft_time_limit,
               ft_epochs=a.ft_epochs, ft_lr=a.ft_lr, gbx_folds=a.gbx_folds,
               gbx_iterations=a.gbx_iterations, oof_folds=a.oof_folds, gbx_inner=a.gbx_inner,
               cat_iterations=a.cat_iterations)
    print(f"CUDA available: {has_cuda()} | methods: {methods}")

    datasets = get_datasets(a.source, max_rows=a.max_rows, max_features=a.max_features,
                            max_classes=a.max_classes, max_datasets=a.max_datasets,
                            cache_dir=a.cache_dir)
    pd.DataFrame([dict(name=d.name, did=d.did, n=d.X.shape[0], d=d.X.shape[1],
                       n_classes=d.n_classes, n_cat=len(d.cat_idx),
                       missing_frac=float(np.isnan(d.X).mean()), source=d.source)
                  for d in datasets]).to_csv(os.path.join(a.out, "datasets.csv"), index=False)
    print(f"{len(datasets)} datasets")

    metrics_path = os.path.join(a.out, "metrics.jsonl")
    for di, ds in enumerate(datasets):
        if di % a.n_shards != a.shard:
            continue
        dname = slug(ds.name if ds.did is None else f"{ds.name}_{ds.did}")
        folds = make_folds(ds.y, a.n_folds, seed=a.seed)
        for k, (tr, te) in enumerate(folds):
            Xtr, ytr, Xte, yte = ds.X[tr], ds.y[tr], ds.X[te], ds.y[te]
            if a.val_preds:
                itr, iva = holdout_split(ytr, a.val_frac, seed=a.seed + k)
            for m in methods:
                outs_names = VARIANT_METHODS.get(m, [m])
                fpath = os.path.join(a.out, "preds", f"{dname}__f{k}__{outs_names[-1]}.npz")
                if os.path.exists(fpath):
                    continue
                base = dict(dataset=dname, did=ds.did, n=int(ds.X.shape[0]),
                            d=int(ds.X.shape[1]), n_classes=ds.n_classes, fold=k)
                try:
                    seed = a.seed + 1000 * k
                    P, tfit, tpred, info = fit_predict(m, Xtr, ytr, Xte, ds.cat_idx, seed,
                                                       ds.n_classes, cfg)
                    P_val = None
                    self_val = m in VARIANT_METHODS
                    if a.val_preds and not self_val:
                        P_val, _, _, _ = fit_predict(m, Xtr[itr], ytr[itr], Xtr[iva],
                                                     ds.cat_idx, seed, ds.n_classes, cfg)
                    for name in outs_names:            # marker file is written last
                        save = dict(p_test=P[name], y_test=yte)
                        if P_val is not None:
                            save.update(p_val=P_val[name], y_val=ytr[iva])
                        row = dict(base, method=name, fit_time=tfit, predict_time=tpred,
                                   status="ok", **all_metrics(yte, P[name], ds.n_classes))
                        if info is not None:
                            row["info"] = info
                        append_jsonl(metrics_path, row)
                        np.savez_compressed(os.path.join(
                            a.out, "preds", f"{dname}__f{k}__{name}.npz"), **save)
                        print(f"[{di + 1}/{len(datasets)}] {dname} f{k} {name:14s} "
                              f"AUC={row['roc_auc']:.4f} acc={row['accuracy']:.4f} "
                              f"ll={row['log_loss']:.4f} t={tfit + tpred:.1f}s", flush=True)
                except Exception as e:  # noqa: BLE001
                    row = dict(base, method=m, status="error", error=f"{type(e).__name__}: {e}")
                    print(f"[ERR] {dname} f{k} {m}: {e}", flush=True)
                    traceback.print_exc(limit=2)
                    free_gpu()
                    append_jsonl(metrics_path, row)
    print("done ->", metrics_path)


if __name__ == "__main__":
    main()
