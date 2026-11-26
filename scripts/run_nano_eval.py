#!/usr/bin/env python
"""Part B-2: evaluate pre-trained nanoTabPFN variants on real datasets.

Protocol (identical for every method, incl. baselines):
  * datasets: OpenML-CC18 subset (same filter as Part A) or sklearn fallback,
  * stratified K-fold; the training fold is subsampled to ``--n_ctx`` rows
    (the context size seen during pre-training),
  * datasets with more than ``--feat_cap`` features use a fixed random subset of
    ``--feat_cap`` features,
  * at most ``--max_test`` test rows per fold.
Rows are appended to ``<out>/metrics.jsonl`` (resumable).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tabdl.data import get_datasets, make_folds, subsample_stratified  # noqa: E402
from tabdl.metrics import all_metrics, fix_proba  # noqa: E402
from tabdl.models import make_model  # noqa: E402
from tabdl.nano.evaluate import nano_predict_proba  # noqa: E402
from tabdl.nano.train import load_model  # noqa: E402
from tabdl.utils import append_jsonl, free_gpu, read_jsonl, slug  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt_dir", default="results/nano/ckpt")
    ap.add_argument("--source", default="auto", choices=["auto", "openml", "sklearn"])
    ap.add_argument("--max_rows", type=int, default=3000)
    ap.add_argument("--max_features", type=int, default=100)
    ap.add_argument("--max_classes", type=int, default=10)
    ap.add_argument("--max_datasets", type=int, default=None)
    ap.add_argument("--n_folds", type=int, default=3)
    ap.add_argument("--n_ctx", type=int, default=200)
    ap.add_argument("--feat_cap", type=int, default=20)
    ap.add_argument("--max_test", type=int, default=500)
    ap.add_argument("--n_ens", type=int, default=4)
    ap.add_argument("--baselines", default="logreg,knn,rf,xgb,catboost")
    ap.add_argument("--with_tabpfn", action="store_true",
                    help="also run the real TabPFN v2 on the same subsample (reference)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_dir", default=os.environ.get("OPENML_CACHE", "openml_cache"))
    ap.add_argument("--out", default="results/nano")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = sorted(glob.glob(os.path.join(a.ckpt_dir, "*.pt")))
    models = {os.path.basename(p)[:-3]: load_model(p, device).to(device) for p in ckpts}
    print(f"{len(models)} nano checkpoints: {list(models)}")
    baselines = [b for b in a.baselines.split(",") if b]
    if a.with_tabpfn:
        baselines.append("tabpfn_v2")

    datasets = get_datasets(a.source, max_rows=a.max_rows, max_features=a.max_features,
                            max_classes=a.max_classes, max_datasets=a.max_datasets,
                            cache_dir=a.cache_dir)
    out_path = os.path.join(a.out, "metrics.jsonl")
    done = {(r["dataset"], r["fold"], r["method"]) for r in read_jsonl(out_path)
            if r.get("status") == "ok"}

    for di, ds in enumerate(datasets):
        dname = slug(ds.name if ds.did is None else f"{ds.name}_{ds.did}")
        rng = np.random.default_rng(a.seed + di)
        cols = np.arange(ds.X.shape[1])
        if len(cols) > a.feat_cap:
            cols = np.sort(rng.choice(cols, a.feat_cap, replace=False))
        cat_idx = [int(np.where(cols == j)[0][0]) for j in ds.cat_idx if j in cols]
        X = ds.X[:, cols]
        for k, (tr, te) in enumerate(make_folds(ds.y, a.n_folds, seed=a.seed)):
            ctx = tr[subsample_stratified(ds.y[tr], a.n_ctx, seed=a.seed + k)]
            te = te[subsample_stratified(ds.y[te], a.max_test, seed=a.seed + k)] \
                if len(te) > a.max_test else te
            Xtr, ytr, Xte, yte = X[ctx], ds.y[ctx], X[te], ds.y[te]
            base = dict(dataset=dname, did=ds.did, n=int(ds.X.shape[0]), d=int(X.shape[1]),
                        n_classes=ds.n_classes, fold=k, n_ctx=int(len(ctx)))
            jobs = [(name, "nano") for name in models] + [(b, "baseline") for b in baselines]
            for name, kind in jobs:
                method = f"nano_{name.rsplit('_s', 1)[0]}" if kind == "nano" else name
                seed_tag = int(name.rsplit("_s", 1)[1]) if kind == "nano" else 0
                key_method = f"{method}#s{seed_tag}" if kind == "nano" else method
                if (dname, k, key_method) in done:
                    continue
                row = dict(base, method=key_method, family=method, seed=seed_tag)
                try:
                    t0 = time.time()
                    if kind == "nano":
                        p = nano_predict_proba(models[name], Xtr, ytr, Xte, ds.n_classes,
                                               n_ens=a.n_ens, seed=a.seed + k, device=device)
                    else:
                        m = make_model(name, cat_idx, a.seed + k, ds.n_classes)
                        m.fit(Xtr, ytr)
                        p = np.asarray(m.predict_proba(Xte))
                        seen = np.unique(ytr)
                        if p.shape[1] == len(seen) < ds.n_classes:
                            full = np.zeros((len(p), ds.n_classes))
                            full[:, seen] = p
                            p = full
                        del m
                    p = fix_proba(p, ds.n_classes)
                    row.update(all_metrics(yte, p, ds.n_classes), time_s=time.time() - t0,
                               status="ok")
                except Exception as e:  # noqa: BLE001
                    row.update(status="error", error=f"{type(e).__name__}: {e}")
                    print(f"[ERR] {dname} f{k} {key_method}: {e}")
                free_gpu()
                append_jsonl(out_path, row)
            print(f"[{di + 1}/{len(datasets)}] {dname} fold {k} done", flush=True)
    print("done ->", out_path)


if __name__ == "__main__":
    main()
