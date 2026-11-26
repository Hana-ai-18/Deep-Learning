#!/usr/bin/env python
"""Part B-2: pre-train nanoTabPFN variants on the synthetic prior.

Variants (see tabdl/nano/train.py):
  base       LayerNorm + standard attention + AdamW   (nanoTabPFN baseline)
  dyt        + Dynamic Tanh instead of LayerNorm        (CVPR 2025)
  gate       + gated attention                          (NeurIPS 2025 best paper)
  muon       + Muon optimizer                           (2024/2025)
  gate_muon  gated attention + Muon
  full       DyT + gated attention + Muon

All variants see exactly the same stream of synthetic datasets (fixed prior
seed) so differences come only from the architecture/optimizer. Finished runs
(``<out>/<variant>_s<seed>.pt``) are skipped -> resumable.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tabdl.nano.model import ModelConfig  # noqa: E402
from tabdl.nano.prior import PriorConfig  # noqa: E402
from tabdl.nano.train import VARIANTS, TrainConfig, pretrain  # noqa: E402
from tabdl.utils import free_gpu  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr_adam", type=float, default=1e-3)
    ap.add_argument("--lr_muon", type=float, default=0.02)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=256)
    ap.add_argument("--max_rows", type=int, default=256, help="rows per synthetic dataset")
    ap.add_argument("--max_features", type=int, default=20)
    ap.add_argument("--max_classes", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--out", default="results/nano/ckpt")
    a = ap.parse_args()

    mc = ModelConfig(d_model=a.d_model, n_heads=a.n_heads, n_layers=a.n_layers,
                     d_ff=a.d_ff, max_classes=a.max_classes)
    pc = PriorConfig(max_rows=a.max_rows, max_features=a.max_features,
                     max_classes=a.max_classes)
    for seed in [int(s) for s in a.seeds.split(",")]:
        for v in a.variants.split(","):
            v = v.strip()
            if os.path.exists(os.path.join(a.out, f"{v}_s{seed}.pt")):
                print(f"skip {v}_s{seed} (exists)")
                continue
            tc = TrainConfig(variant=v, seed=seed, steps=a.steps, batch_size=a.batch_size,
                             lr_adam=a.lr_adam, lr_muon=a.lr_muon,
                             eval_every=a.eval_every, amp=not a.no_amp)
            pretrain(v, tc, mc, pc, a.out)
            free_gpu()


if __name__ == "__main__":
    main()
