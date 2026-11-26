#!/usr/bin/env python
"""Build post-hoc ensembles from the cached predictions of run_benchmark.py.

Each ensemble is specified as ``name=member1+member2+...``. Weights are chosen
by greedy ensemble selection (Caruana et al., 2004) on the inner validation
split and applied to the test predictions of the members refit on the full
training fold. This step is CPU-only and takes seconds.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tabdl.ensemble import blend, caruana_weights  # noqa: E402
from tabdl.metrics import all_metrics  # noqa: E402
from tabdl.utils import append_jsonl, read_jsonl  # noqa: E402

DEFAULT_SPECS = [
    "ens_pfn_cat=tabpfn_v2+catboost",
    "ens_ft_cat=tabpfn_v2_ft+catboost",
    "ens_pfn_ft_cat=tabpfn_v2+tabpfn_v2_ft+catboost",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", default="results/benchmark")
    ap.add_argument("--specs", nargs="*", default=DEFAULT_SPECS)
    ap.add_argument("--n_iter", type=int, default=25)
    a = ap.parse_args()

    files = glob.glob(os.path.join(a.bench, "preds", "*.npz"))
    index = defaultdict(dict)                       # (dataset, fold) -> method -> path
    for f in files:
        base = os.path.basename(f)[:-4]
        dname, fold, method = base.split("__")
        index[(dname, int(fold[1:]))][method] = f

    meta = {}
    for r in read_jsonl(os.path.join(a.bench, "metrics.jsonl")):
        meta[r["dataset"]] = r
    out_path = os.path.join(a.bench, "metrics.jsonl")
    done = {(r["dataset"], r["fold"], r["method"]) for r in read_jsonl(out_path)
            if r.get("status") == "ok"}

    n_new = 0
    for spec in a.specs:
        name, members = spec.split("=")
        members = members.split("+")
        for (dname, fold), mp in sorted(index.items()):
            if (dname, fold, name) in done or not all(m in mp for m in members):
                continue
            Z = [np.load(mp[m]) for m in members]
            if not all("p_val" in z for z in Z):
                print(f"[skip] {dname} f{fold}: run_benchmark.py was not called with "
                      "--val_preds")
                continue
            n_classes = Z[0]["p_test"].shape[1]
            w = caruana_weights([z["p_val"] for z in Z], Z[0]["y_val"], n_classes, a.n_iter)
            p = blend([z["p_test"] for z in Z], w, n_classes)
            m0 = meta.get(dname, {})
            row = dict(dataset=dname, did=m0.get("did"), n=m0.get("n"), d=m0.get("d"),
                       n_classes=n_classes, fold=fold, method=name, status="ok",
                       weights={m: float(x) for m, x in zip(members, w)},
                       **all_metrics(Z[0]["y_test"], p, n_classes))
            append_jsonl(out_path, row)
            n_new += 1
    print(f"added {n_new} ensemble rows -> {out_path}")


if __name__ == "__main__":
    main()
