"""In-context prediction on real datasets with a pre-trained nanoTabPFN."""
from __future__ import annotations

import numpy as np
import torch

from ..data import NumericPreprocessor


@torch.no_grad()
def nano_predict_proba(model, Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray,
                       n_classes: int, n_ens: int = 4, seed: int = 0, device=None,
                       test_chunk: int = 256) -> np.ndarray:
    """Average over ``n_ens`` members that differ in label permutation, feature
    order and random column embeddings (cheap test-time ensembling, as in TabPFN)."""
    device = torch.device(device or next(model.parameters()).device)
    if n_classes > model.cfg.max_classes:
        raise ValueError(f"{n_classes} classes > model max {model.cfg.max_classes}")
    pre = NumericPreprocessor().fit(Xtr)
    A = np.clip(pre.transform(Xtr), -6, 6).astype(np.float32)
    Bt = np.clip(pre.transform(Xte), -6, 6).astype(np.float32)
    rng = np.random.default_rng(seed)
    F = A.shape[1]
    probs = np.zeros((len(Xte), n_classes))
    amp = device.type == "cuda"
    nc = torch.tensor([n_classes], device=device)
    for e in range(n_ens):
        perm = rng.permutation(n_classes) if e > 0 else np.arange(n_classes)
        fperm = rng.permutation(F) if e > 0 else np.arange(F)
        ytr_p = torch.as_tensor(perm[ytr], device=device)[None]
        ctx = torch.as_tensor(A[:, fperm], device=device)
        for s in range(0, len(Bt), test_chunk):
            q = torch.as_tensor(Bt[s:s + test_chunk][:, fperm], device=device)
            X = torch.cat([ctx, q], 0)[None]
            gen = torch.Generator(device=device).manual_seed(seed * 1000 + e)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(X, ytr_p, nc, col_gen=gen)[0].float()
            p = torch.softmax(logits[:, :n_classes], -1).cpu().numpy()
            probs[s:s + test_chunk] += p[:, perm]       # back to the original labels
    return probs / n_ens
