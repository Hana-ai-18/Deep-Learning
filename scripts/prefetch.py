#!/usr/bin/env python
"""Download the OpenML datasets and the TabPFN v2 checkpoint once, before
several processes start in parallel (avoids concurrent writes to the caches)."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tabdl.data import get_datasets  # noqa: E402
from tabdl.models import make_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="openml")
    ap.add_argument("--max_rows", type=int, default=3000)
    ap.add_argument("--cache_dir", default=os.environ.get("OPENML_CACHE", "openml_cache"))
    a = ap.parse_args()
    ds = get_datasets(a.source, max_rows=a.max_rows, cache_dir=a.cache_dir)
    print(f"{len(ds)} datasets cached")
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(60, 4)), rng.integers(0, 2, 60)
    m = make_model("tabpfn_v2", [], 0, 2)
    m.fit(X[:40], y[:40])
    m.predict_proba(X[40:])
    print("TabPFN v2 checkpoint cached")


if __name__ == "__main__":
    main()
