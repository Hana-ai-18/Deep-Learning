#!/usr/bin/env python
"""Aggregate all results into tables (Markdown + CSV) and figures (PNG).

Usage:
  python scripts/make_report.py --bench results/benchmark --nano results/nano \
      --out results/report
Either input may be missing; the corresponding sections are skipped.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tabdl.utils import read_jsonl  # noqa: E402

REPRO_METHODS = ["logreg", "knn", "rf", "xgb", "lgbm", "catboost", "tabpfn_v2"]
NICE = {
    "logreg": "Logistic Regression", "knn": "kNN", "rf": "Random Forest",
    "xgb": "XGBoost", "lgbm": "LightGBM", "catboost": "CatBoost",
    "tabpfn_v2": "TabPFN v2", "tabpfn_v2_ft": "TabPFN v2 + fine-tune",
    "ens_pfn_cat": "Ens(TabPFN v2, CatBoost)",
    "ens_ft_cat": "Ens(TabPFN v2-FT, CatBoost)",
    "ens_pfn_ft_cat": "Ens(TabPFN v2, TabPFN v2-FT, CatBoost)",
    "nano_base": "nanoTabPFN (baseline)", "nano_dyt": "+ DyT",
    "nano_gate": "+ Gated Attention", "nano_muon": "+ Muon",
    "nano_gate_muon": "+ Gated Attn + Muon", "nano_full": "+ DyT + Gated Attn + Muon",
}


def nice(m):
    return NICE.get(m, m)


def load_ok(path):
    rows = [r for r in read_jsonl(path) if r.get("status") == "ok"]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # keep the last successful row per (dataset, fold, method)
    return df.drop_duplicates(["dataset", "fold", "method"], keep="last")


def per_dataset(df, col="method"):
    return df.groupby(["dataset", col])[["roc_auc", "accuracy", "log_loss"]].mean().reset_index()


def complete_datasets(pdf, methods, col="method"):
    ok = pdf.groupby("dataset")[col].apply(lambda s: set(methods) <= set(s))
    return ok[ok].index.tolist()


def summary_table(pdf, methods, ref=None, col="method", time_df=None):
    dsets = complete_datasets(pdf, methods, col)
    sub = pdf[pdf["dataset"].isin(dsets) & pdf[col].isin(methods)]
    piv = sub.pivot(index="dataset", columns=col, values="roc_auc")[methods]
    ranks = piv.rank(axis=1, ascending=False).mean()
    rows = []
    for m in methods:
        s = sub[sub[col] == m]
        r = dict(Method=nice(m), key=m, AUC=s["roc_auc"].mean(), Acc=s["accuracy"].mean(),
                 LogLoss=s["log_loss"].mean(), MeanRank=ranks[m])
        if ref is not None and m != ref:
            diff = piv[m] - piv[ref]
            r["Win/Tie/Loss vs " + nice(ref)] = \
                f"{int((diff > 1e-4).sum())}/{int((diff.abs() <= 1e-4).sum())}/{int((diff < -1e-4).sum())}"
            try:
                from scipy.stats import wilcoxon
                r["p (Wilcoxon)"] = float(wilcoxon(piv[m], piv[ref]).pvalue) \
                    if (diff.abs() > 1e-12).any() else 1.0
            except Exception:  # noqa: BLE001
                r["p (Wilcoxon)"] = np.nan
        if time_df is not None and m in time_df.index:
            r["Time/fold (s)"] = time_df[m]
        rows.append(r)
    out = pd.DataFrame(rows).sort_values("MeanRank").reset_index(drop=True)
    for c in out.columns:
        if c.startswith("Win/Tie/Loss"):
            out[c] = out[c].fillna("(reference)")
    return out, piv, len(dsets)


def fmt(df):
    df = df.drop(columns=[c for c in ["key"] if c in df.columns])
    return df.to_markdown(index=False, floatfmt=".4f")


def bench_times(df):
    d = df.copy()
    if "fit_time" not in d:
        return None
    d["t"] = d["fit_time"].fillna(0) + d["predict_time"].fillna(0)
    t = d[d["t"] > 0].groupby("method")["t"].mean()
    return t


def plot_bar(values: pd.Series, title, xlabel, path, lower_better=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v = values.sort_values(ascending=lower_better)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(v) + 1.2))
    ax.barh([nice(i) for i in v.index][::-1], v.values[::-1], color="#4C72B0")
    for i, x in enumerate(v.values[::-1]):
        ax.text(x, i, f" {x:.3f}", va="center", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter(piv, a, b, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(piv[b], piv[a], s=18)
    lo = float(min(piv[a].min(), piv[b].min())) - 0.01
    ax.plot([lo, 1], [lo, 1], "k--", lw=1)
    ax.set_xlabel(f"{nice(b)} ROC-AUC")
    ax.set_ylabel(f"{nice(a)} ROC-AUC")
    ax.set_title("Per-dataset ROC-AUC")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def section_bench(bench_dirs, out, md):
    frames = [load_ok(os.path.join(b, "metrics.jsonl")) for b in bench_dirs]
    frames = [f for f in frames if len(f)]
    if not frames:
        return
    df = pd.concat(frames).drop_duplicates(["dataset", "fold", "method"], keep="last")
    df.to_csv(os.path.join(out, "benchmark_all_rows.csv"), index=False)
    pdf = per_dataset(df)
    times = bench_times(df)
    for ens, members in [("ens_pfn_cat", ["tabpfn_v2", "catboost"]),
                         ("ens_ft_cat", ["tabpfn_v2_ft", "catboost"]),
                         ("ens_pfn_ft_cat", ["tabpfn_v2", "tabpfn_v2_ft", "catboost"])]:
        if times is not None and all(m in times.index for m in members):
            times[ens] = 2 * sum(times[m] for m in members)   # members fit twice
    # ---- Part A: reproduction ----
    repro = [m for m in REPRO_METHODS if m in set(pdf["method"])]
    if "tabpfn_v2" in repro and len(repro) > 1:
        tab, piv, nd = summary_table(pdf, repro, ref="tabpfn_v2", time_df=times)
        tab.to_csv(os.path.join(out, "table_A_reproduction.csv"), index=False)
        piv.to_csv(os.path.join(out, "table_A_per_dataset_auc.csv"))
        md.append(f"## Part A - Reproduction: TabPFN v2 vs. baselines ({nd} datasets)\n")
        md.append("Mean over datasets of the fold-averaged metric; MeanRank is the "
                  "average rank by ROC-AUC (1 = best).\n")
        md.append(fmt(tab) + "\n")
        plot_bar(tab.set_index("key")["MeanRank"], "Part A: mean rank (ROC-AUC)",
                 "mean rank (lower is better)", os.path.join(out, "fig_A_rank.png"),
                 lower_better=True)
        if "catboost" in piv:
            plot_scatter(piv, "tabpfn_v2", "catboost", os.path.join(out, "fig_A_scatter.png"))
        md.append("![rank](fig_A_rank.png) ![scatter](fig_A_scatter.png)\n")
    # ---- Part B-1: TabPFN-level improvements ----
    imp = [m for m in ["catboost", "tabpfn_v2", "tabpfn_v2_ft", "ens_pfn_cat", "ens_ft_cat",
                       "ens_pfn_ft_cat"] if m in set(pdf["method"])]
    if len(imp) > 2 and "tabpfn_v2" in imp:
        tab, piv, nd = summary_table(pdf, imp, ref="tabpfn_v2", time_df=times)
        tab.to_csv(os.path.join(out, "table_B1_improvement.csv"), index=False)
        piv.to_csv(os.path.join(out, "table_B1_per_dataset_auc.csv"))
        md.append(f"## Part B-1 - Improving TabPFN v2 on each dataset ({nd} datasets)\n")
        md.append(fmt(tab) + "\n")
        plot_bar(tab.set_index("key")["LogLoss"], "Part B-1: mean log-loss",
                 "log-loss (lower is better)", os.path.join(out, "fig_B1_logloss.png"),
                 lower_better=True)
        md.append("![b1](fig_B1_logloss.png)\n")
        if "weights" in df.columns:
            w = df[df["method"].str.startswith("ens_") & df["weights"].notna()]
            if len(w):
                ww = pd.json_normalize(w["weights"]).assign(method=w["method"].values)
                mw = ww.groupby("method").mean()
                mw.to_csv(os.path.join(out, "table_B1_ensemble_weights.csv"))
                md.append("Average ensemble weights:\n\n" + mw.to_markdown(floatfmt=".3f") + "\n")


def section_nano(nano_dir, out, md):
    df = load_ok(os.path.join(nano_dir, "metrics.jsonl"))
    if not len(df):
        return
    df.to_csv(os.path.join(out, "nano_all_rows.csv"), index=False)
    if "family" not in df:
        df["family"] = df["method"]
    df["family"] = df["family"].fillna(df["method"])
    # per (dataset, family, seed) -> mean over folds
    g = df.groupby(["dataset", "family", "seed"])[["roc_auc", "accuracy", "log_loss"]] \
        .mean().reset_index()
    fams = sorted(g["family"].unique(), key=lambda m: (not m.startswith("nano"), m))
    dsets = complete_datasets(g.rename(columns={"family": "method"}), fams)
    g = g[g["dataset"].isin(dsets)]
    # seed-level means over datasets, then mean +- std over seeds
    seed_level = g.groupby(["family", "seed"])[["roc_auc", "accuracy", "log_loss"]].mean()
    agg = seed_level.groupby("family").agg(["mean", "std"])
    # ranks on seed-averaged per-dataset AUC
    pdf = g.groupby(["dataset", "family"])["roc_auc"].mean().unstack()
    ranks = pdf.rank(axis=1, ascending=False).mean()
    nano_only = [f for f in fams if f.startswith("nano")]
    nano_ranks = pdf[nano_only].rank(axis=1, ascending=False).mean() if nano_only else None
    hist = {}
    for p in glob.glob(os.path.join(nano_dir, "ckpt", "*_history.csv")):
        tag = os.path.basename(p).replace("_history.csv", "")
        v, s = tag.rsplit("_s", 1)
        h = pd.read_csv(p)
        hist.setdefault("nano_" + v, []).append(h)
    rows = []
    for f in fams:
        r = dict(Method=nice(f), key=f,
                 AUC=f"{agg.loc[f, ('roc_auc', 'mean')]:.4f} ± {np.nan_to_num(agg.loc[f, ('roc_auc', 'std')]):.4f}",
                 Acc=f"{agg.loc[f, ('accuracy', 'mean')]:.4f} ± {np.nan_to_num(agg.loc[f, ('accuracy', 'std')]):.4f}",
                 LogLoss=f"{agg.loc[f, ('log_loss', 'mean')]:.4f} ± {np.nan_to_num(agg.loc[f, ('log_loss', 'std')]):.4f}",
                 MeanRank_all=ranks[f])
        if nano_ranks is not None and f in nano_ranks:
            r["MeanRank_nano"] = nano_ranks[f]
        if f in hist:
            fin = [h["val_loss"].iloc[-1] for h in hist[f]]
            r["Prior val-loss"] = f"{np.mean(fin):.4f} ± {np.std(fin):.4f}"
            r["Train time (s)"] = float(np.mean([h["time_s"].iloc[-1] for h in hist[f]]))
            r["Non-finite steps"] = int(sum(h["nonfinite"].iloc[-1] for h in hist[f]))
        rows.append(r)
    tab = pd.DataFrame(rows).sort_values("MeanRank_all").reset_index(drop=True)
    tab.to_csv(os.path.join(out, "table_B2_nano.csv"), index=False)
    pdf.to_csv(os.path.join(out, "table_B2_per_dataset_auc.csv"))
    md.append(f"## Part B-2 - nanoTabPFN architecture/optimizer ablation ({len(dsets)} datasets)\n")
    md.append("Mean ± std over pre-training seeds. Every method sees the same context "
              "subsample and feature subset.\n")
    md.append(fmt(tab) + "\n")
    # paired test of each nano variant vs. nano_base (seed-averaged per dataset AUC)
    if "nano_base" in pdf:
        from scipy.stats import wilcoxon
        lines = []
        for f in nano_only:
            if f == "nano_base":
                continue
            d = pdf[f] - pdf["nano_base"]
            try:
                pv = wilcoxon(pdf[f], pdf["nano_base"]).pvalue
            except ValueError:
                pv = np.nan
            lines.append(dict(Variant=nice(f), **{"ΔAUC vs base": d.mean(),
                              "W/T/L": f"{(d > 1e-4).sum()}/{(d.abs() <= 1e-4).sum()}/{(d < -1e-4).sum()}",
                              "p (Wilcoxon)": pv}))
        if lines:
            t2 = pd.DataFrame(lines)
            t2.to_csv(os.path.join(out, "table_B2_vs_base.csv"), index=False)
            md.append("Paired comparison with the nanoTabPFN baseline:\n\n" +
                      t2.to_markdown(index=False, floatfmt=".4f") + "\n")
    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if hist:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for f, hs in sorted(hist.items()):
            steps = hs[0]["step"].values
            vals = np.mean([h["val_loss"].values[:len(steps)] for h in hs], axis=0)
            ax.plot(steps, vals, label=nice(f))
        ax.set_xlabel("pre-training step")
        ax.set_ylabel("prior validation loss")
        ax.set_title("Part B-2: pre-training curves (mean over seeds)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_B2_curves.png"), dpi=150)
        plt.close(fig)
        md.append("![curves](fig_B2_curves.png)\n")
    plot_bar(pdf.mean(), "Part B-2: mean ROC-AUC on real data", "ROC-AUC",
             os.path.join(out, "fig_B2_auc.png"))
    md.append("![auc](fig_B2_auc.png)\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", nargs="*", default=["results/benchmark"])
    ap.add_argument("--nano", default="results/nano")
    ap.add_argument("--out", default="results/report")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    md = ["# Results\n"]
    section_bench([b for b in a.bench if os.path.isdir(b)], a.out, md)
    if a.nano and os.path.isdir(a.nano):
        section_nano(a.nano, a.out, md)
    with open(os.path.join(a.out, "report.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    print(f"\nwritten to {a.out}/report.md")


if __name__ == "__main__":
    main()
