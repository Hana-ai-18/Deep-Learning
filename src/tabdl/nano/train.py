"""Pre-training loop for nanoTabPFN on the synthetic prior."""
from __future__ import annotations

import csv
import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .model import ModelConfig, NanoTabPFN
from .muon import MuonWithAuxAdam, cosine_lr_factor
from .prior import PriorConfig, sample_batch

# name -> (norm, gated, optimizer)
VARIANTS = {
    "base":       ("layernorm", False, "adamw"),   # nanoTabPFN baseline
    "dyt":        ("dyt",       False, "adamw"),   # + Dynamic Tanh
    "gate":       ("layernorm", True,  "adamw"),   # + gated attention
    "muon":       ("layernorm", False, "muon"),    # + Muon optimizer
    "gate_muon":  ("layernorm", True,  "muon"),    # gated attention + Muon
    "full":       ("dyt",       True,  "muon"),    # DyT + gated attention + Muon
}


@dataclass
class TrainConfig:
    variant: str = "base"
    seed: int = 0
    steps: int = 3000
    batch_size: int = 32
    lr_adam: float = 1e-3
    lr_muon: float = 0.02
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    eval_every: int = 250
    n_val_batches: int = 16
    val_seed: int = 12345
    prior_seed: int = 777          # same prior stream for every variant/seed
    amp: bool = True


def build_model(variant: str, base_cfg: ModelConfig) -> NanoTabPFN:
    norm, gated, _ = VARIANTS[variant]
    cfg = ModelConfig(**{**base_cfg.to_dict(), "norm": norm, "gated": gated})
    return NanoTabPFN(cfg)


def build_optimizer(model: NanoTabPFN, variant: str, tc: TrainConfig):
    opt_name = VARIANTS[variant][2]
    if opt_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=tc.lr_adam, betas=(0.9, 0.95),
                                weight_decay=tc.weight_decay)
    else:
        muon_p, adam_p = model.param_groups_for_muon()
        opt = MuonWithAuxAdam([
            dict(params=muon_p, use_muon=True, lr=tc.lr_muon, weight_decay=tc.weight_decay),
            dict(params=adam_p, use_muon=False, lr=tc.lr_adam, betas=(0.9, 0.95),
                 weight_decay=tc.weight_decay),
        ])
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    return opt


def _loss(model, batch, amp, device):
    n_tr = batch["n_train"]
    with torch.autocast(device_type="cuda", dtype=torch.float16,
                        enabled=amp and device.type == "cuda"):
        logits = model(batch["X"], batch["y"][:, :n_tr], batch["n_classes"])
    logits = logits.float()
    tgt = batch["y"][:, n_tr:]
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
    acc = (logits.argmax(-1) == tgt).float().mean()
    return loss, acc


@torch.no_grad()
def evaluate_prior(model, val_batches, amp, device):
    model.eval()
    ls, accs = [], []
    for b in val_batches:
        loss, acc = _loss(model, b, amp, device)
        ls.append(loss.item())
        accs.append(acc.item())
    model.train()
    return sum(ls) / len(ls), sum(accs) / len(accs)


def pretrain(variant: str, tc: TrainConfig, mc: ModelConfig, pc: PriorConfig,
             out_dir: str, device=None, log_every: int = 50) -> dict:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(tc.seed)
    model = build_model(variant, mc).to(device)
    opt = build_optimizer(model, variant, tc)
    scaler = torch.amp.GradScaler("cuda", enabled=tc.amp and device.type == "cuda")
    g_train = torch.Generator(device=device).manual_seed(tc.prior_seed)
    g_val = torch.Generator(device=device).manual_seed(tc.val_seed)
    val_batches = [sample_batch(tc.batch_size, pc, g_val, device) for _ in range(tc.n_val_batches)]
    warmup = max(1, int(tc.warmup_frac * tc.steps))
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{variant}_s{tc.seed}"
    hist_path = os.path.join(out_dir, f"{tag}_history.csv")
    hist, ema, n_bad, t0 = [], None, 0, time.time()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] params={n_params / 1e6:.2f}M device={device}", flush=True)

    for step in range(tc.steps):
        fac = cosine_lr_factor(step, tc.steps, warmup)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * fac
        batch = sample_batch(tc.batch_size, pc, g_train, device)
        loss, _ = _loss(model, batch, tc.amp, device)
        if not torch.isfinite(loss):
            n_bad += 1
            opt.zero_grad(set_to_none=True)
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        if not torch.isfinite(gn):
            n_bad += 1
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        last = step == tc.steps - 1
        if (step + 1) % tc.eval_every == 0 or last:
            vl, va = evaluate_prior(model, val_batches, tc.amp, device)
            row = dict(step=step + 1, train_loss_ema=ema, val_loss=vl, val_acc=va,
                       grad_norm=float(gn), nonfinite=n_bad, time_s=time.time() - t0)
            hist.append(row)
            print(f"[{tag}] step {step + 1}/{tc.steps} train {ema:.4f} val {vl:.4f} "
                  f"acc {va:.4f} ({row['time_s']:.0f}s)", flush=True)
        elif (step + 1) % log_every == 0 and device.type == "cpu":
            print(f"[{tag}] step {step + 1} loss {ema:.4f}", flush=True)

    with open(hist_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        w.writeheader()
        w.writerows(hist)
    ckpt = dict(variant=variant, model_cfg=model.cfg.to_dict(), train_cfg=asdict(tc),
                prior_cfg=asdict(pc), state_dict=model.state_dict(), history=hist,
                n_params=n_params, train_time_s=time.time() - t0)
    torch.save(ckpt, os.path.join(out_dir, f"{tag}.pt"))
    return ckpt


def load_model(path: str, device=None) -> NanoTabPFN:
    ck = torch.load(path, map_location=device or "cpu", weights_only=False)
    model = NanoTabPFN(ModelConfig(**ck["model_cfg"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model

