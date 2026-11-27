#!/usr/bin/env python
"""Aggregate benchmark results into tables (Markdown + CSV) and figures (PNG).

Usage:  python scripts/make_report.py --bench results/benchmark --out results/report
Section A is written as soon as the reproduction methods exist; section B as
soon as at least one improvement method exists.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tabdl.utils import read_jsonl, slug  # noqa: E402

BASELINES = ["logreg", "knn", "rf", "xgb", "lgbm", "catboost"]
REPRO_METHODS = BASELINES + ["tabpfn_v2"]
EXISTING = ["tabpfn_v2", "catboost", "tabpfn_v2_ft", "ens_pfn_cat", "ens_ft_cat", "ens_phe"]
OURS = ["tabpfn_te", "tabpfn_gbx", "tabpfn_gbx_te", "ens_gbx", "ens_gbx_phe", "ens_gbx_phe_ft",
        "oofphe_pfn", "oofphe_pfn_ts", "oofphe_all", "oofphe_all_ts", "oofphe_safe"]
PORTFOLIO = ["tabpfn_v2", "pfn_onehot", "pfn_ordinal", "pfn_kdi", "pfn_poly", "pfn_sharp",
             "pfn_avgsm", "pfn_wide", "tabpfn_gbx_te", "gbx_onehot"]
FINAL = "oofphe_safe"
NICE = {
    "logreg": "Logistic Regression", "knn": "kNN", "rf": "Random Forest",
    "xgb": "XGBoost", "lgbm": "LightGBM", "catboost": "CatBoost",
    "tabpfn_v2": "TabPFN v2 (original)",
    "tabpfn_v2_ft": "TabPFN v2 + fine-tuning [existing]",
    "ens_pfn_cat": "Ens(TabPFN v2, CatBoost) [existing]",
    "ens_ft_cat": "Ens(TabPFN v2-FT, CatBoost) [existing]",
    "tabpfn_te": "TabPFN v2 + OOF-TE [ours, ablation]",
    "tabpfn_gbx": "TabPFN v2 + OOF-GBDT logits [ours, ablation]",
    "tabpfn_gbx_te": "TabPFN-GBX [ours]",
    "ens_gbx": "TabPFN-GBX-Ens [ours]",
    "ens_phe": "TabPFN v2-PHE (config ensemble) [existing idea]",
    "ens_gbx_phe": "TabPFN-GBX-PHE, holdout weights [ours v1]",
    "ens_gbx_phe_ft": "TabPFN-GBX-PHE + FT, holdout weights [ours v1]",
    "oofphe_pfn": "OOF-PHE: TabPFN configs [ours v2, ablation]",
    "oofphe_pfn_ts": "OOF-PHE: TabPFN configs + TS [ours v2, ablation]",
    "oofphe_all": "OOF-PHE: + CatBoost + GBX [ours v2, ablation]",
    "oofphe_all_ts": "OOF-PHE-GBX + TS [ours v2, ablation]",
    "oofphe_safe": "Safe OOF-PHE-GBX (anchored + gate) [ours v3, FINAL]",
    "pfn_onehot": "TabPFN v2 cfg: one-hot categoricals", "pfn_ordinal": "TabPFN v2 cfg: ordinal",
    "pfn_kdi": "TabPFN v2 cfg: KDI transform", "pfn_poly": "TabPFN v2 cfg: +polynomial",
    "pfn_sharp": "TabPFN v2 cfg: T=0.75", "pfn_avgsm": "TabPFN v2 cfg: avg logits",
    "pfn_wide": "TabPFN v2 cfg: 32 estimators", "gbx_onehot": "TabPFN-GBX cfg: one-hot",
}
METRICS = ["roc_auc", "accuracy", "log_loss"]


def nice(m):
    return NICE.get(m, m)


def load_ok(path):
    rows = [r for r in read_jsonl(path) if r.get("status") == "ok"]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(["dataset", "fold", "method"], keep="last")


def per_dataset(df):
    return df.groupby(["dataset", "method"])[METRICS].mean().reset_index()


def coverage_filter(pdf, methods, min_frac=0.9):
    """Keep methods that cover >= min_frac of the datasets covered by the best one."""
    cov = pdf[pdf["method"].isin(methods)].groupby("method")["dataset"].nunique()
    if cov.empty:
        return []
    keep = [m for m in methods if m in cov and cov[m] >= min_frac * cov.max()]
    dropped = [m for m in methods if m in cov and m not in keep]
    if dropped:
        print(f"[note] low coverage, excluded from joint table: {dropped}")
    return keep


def joint(pdf, methods, value="roc_auc"):
    piv = pdf[pdf["method"].isin(methods)].pivot(index="dataset", columns="method",
                                                  values=value)
    return piv[methods].dropna()


def summary_table(pdf, methods, ref, times=None):
    piv_auc = joint(pdf, methods, "roc_auc")
    piv_ll = joint(pdf, methods, "log_loss").reindex(piv_auc.index)
    piv_acc = joint(pdf, methods, "accuracy").reindex(piv_auc.index)
    dsets = piv_auc.index
    sub = pdf[pdf["dataset"].isin(dsets) & pdf["method"].isin(methods)]
    ranks = piv_auc.rank(axis=1, ascending=False).mean()
    rows = []
    for m in methods:
        s = sub[sub["method"] == m]
        r = {"Method": nice(m), "key": m, "AUC": s["roc_auc"].mean(),
             "Acc": s["accuracy"].mean(), "LogLoss": s["log_loss"].mean(),
             "MeanRank": ranks[m]}
        if m != ref:
            d = piv_auc[m] - piv_auc[ref]
            r["ΔAUC vs ref"] = d.mean()
            r["W/T/L vs ref"] = (f"{int((d > 1e-4).sum())}/{int((d.abs() <= 1e-4).sum())}/"
                                 f"{int((d < -1e-4).sum())}")
            dl = piv_ll[m] - piv_ll[ref]
            r["W/T/L LogLoss"] = (f"{int((dl < -1e-4).sum())}/{int((dl.abs() <= 1e-4).sum())}/"
                                  f"{int((dl > 1e-4).sum())}")
            from scipy.stats import wilcoxon

            def _p(a_, b_):
                try:
                    return float(wilcoxon(a_, b_).pvalue) if (np.abs(a_ - b_) > 1e-12).any() \
                        else 1.0
                except Exception:  # noqa: BLE001
                    return np.nan
            r["Worst ΔAUC"] = float(d.min())
            r["#ΔAUC<-0.005"] = int((d < -0.005).sum())
            r["p (Wilcoxon)"] = _p(piv_auc[m], piv_auc[ref])
            r["p LogLoss"] = _p(piv_ll[m], piv_ll[ref])
            r["p Acc"] = _p(piv_acc[m], piv_acc[ref])
        else:
            r["W/T/L vs ref"] = "(reference)"
            r["W/T/L LogLoss"] = "(reference)"
        if times is not None and m in times.index:
            r["Time/fold (s)"] = times[m]
        rows.append(r)
    out = pd.DataFrame(rows).sort_values("MeanRank").reset_index(drop=True)
    order = ["Method", "key", "AUC", "Acc", "LogLoss", "MeanRank", "ΔAUC vs ref",
             "W/T/L vs ref", "p (Wilcoxon)", "W/T/L LogLoss", "p LogLoss", "p Acc",
             "Worst ΔAUC", "#ΔAUC<-0.005", "Time/fold (s)"]
    return out[[c for c in order if c in out.columns]], piv_auc


def fmt(df):
    return df.drop(columns=["key"], errors="ignore").to_markdown(index=False, floatfmt=".4f")


def method_times(df):
    if "fit_time" not in df:
        return None
    d = df.copy()
    d["t"] = d["fit_time"].fillna(0) + d["predict_time"].fillna(0)
    t = d[d["t"] > 0].groupby("method")["t"].mean()
    # an ensemble costs its members fitted twice (inner split + full fold)
    for ens, mem in [("ens_pfn_cat", ["tabpfn_v2", "catboost"]),
                     ("ens_ft_cat", ["tabpfn_v2_ft", "catboost"]),
                     ("ens_gbx", ["tabpfn_v2", "tabpfn_gbx_te", "catboost"]),
                     ("ens_phe", PORTFOLIO[:8]),
                     ("ens_gbx_phe", PORTFOLIO + ["catboost"]),
                     ("ens_gbx_phe_ft", PORTFOLIO + ["catboost", "tabpfn_v2_ft"])]:
        avail = [m for m in mem if m in t.index]
        if len(avail) >= 2:
            t[ens] = 2 * sum(t[m] for m in avail)
    return t


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_bar(values: pd.Series, title, xlabel, path, lower_better=False, highlight=()):
    plt = _plt()
    v = values.sort_values(ascending=not lower_better)
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(v) + 1.3))
    colors = ["#C44E52" if k in highlight else "#4C72B0" for k in v.index]
    ax.barh([nice(i) for i in v.index], v.values, color=colors)
    for i, x in enumerate(v.values):
        ax.text(x, i, f" {x:.4f}", va="center", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter(x, y, xlabel, ylabel, title, path, labels=None):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(x, y, s=22)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8, ls=":")
    if labels is not None:
        for xi, yi, lb in zip(x, y, labels):
            if abs(yi) > 0.004 or xi < -0.005:
                ax.annotate(lb.rsplit("_", 1)[0][:18], (xi, yi), fontsize=6)
    if len(x) > 2:
        from scipy.stats import spearmanr
        rho, p = spearmanr(x, y)
        ax.set_title(f"{title}\nSpearman ρ = {rho:.2f} (p = {p:.3f})", fontsize=9)
    else:
        ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def dataset_meta(bench):
    p = os.path.join(bench, "datasets.csv")
    if not os.path.exists(p):
        return None
    m = pd.read_csv(p)
    m["dataset"] = [slug(n if pd.isna(d) else f"{n}_{int(d)}") for n, d in zip(m["name"], m["did"])]
    return m.set_index("dataset")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", default="results/benchmark")
    ap.add_argument("--out", default="results/report")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load_ok(os.path.join(a.bench, "metrics.jsonl"))
    if df.empty:
        print("no results yet")
        return
    df.to_csv(os.path.join(a.out, "all_rows.csv"), index=False)
    pdf = per_dataset(df)
    times = method_times(df)
    meta = dataset_meta(a.bench)
    present = set(pdf["method"])
    md = ["# Results\n"]

    # ------------------------------ Part A ------------------------------ #
    repro = coverage_filter(pdf, [m for m in REPRO_METHODS if m in present])
    if "tabpfn_v2" in repro and len(repro) > 1:
        tab, piv = summary_table(pdf, repro, "tabpfn_v2", times)
        tab.to_csv(os.path.join(a.out, "table_A_reproduction.csv"), index=False)
        piv.to_csv(os.path.join(a.out, "table_A_per_dataset_auc.csv"))
        md += [f"## Part A — Reproduction: TabPFN v2 vs. baselines ({len(piv)} datasets)\n",
               "Mean over datasets of the fold-averaged metric. MeanRank = average rank by "
               "ROC-AUC (1 = best). Reference = TabPFN v2.\n", fmt(tab) + "\n"]
        plot_bar(tab.set_index("key")["MeanRank"], "Part A: mean rank by ROC-AUC",
                 "mean rank (lower is better)", os.path.join(a.out, "fig_A_rank.png"),
                 lower_better=True, highlight=("tabpfn_v2",))
        md.append("![rank](fig_A_rank.png)\n")
        # where is TabPFN v2 weak?
        base_cols = [b for b in BASELINES if b in piv]
        gap = (piv["tabpfn_v2"] - piv[base_cols].max(axis=1)).rename("tabpfn_minus_best_baseline")
        gap.to_csv(os.path.join(a.out, "table_A_weakness_gap.csv"))
        weak = gap[gap < -1e-4].sort_values()
        lines = []
        for d, g in weak.items():
            r = dict(Dataset=d, Gap=g, BestBaseline=nice(piv.loc[d, base_cols].idxmax()))
            if meta is not None and d in meta.index:
                r.update(n=int(meta.loc[d, "n"]), d=int(meta.loc[d, "d"]),
                         n_cat=int(meta.loc[d, "n_cat"]))
            lines.append(r)
        if lines:
            md += [f"**Where TabPFN v2 loses** ({len(weak)}/{len(gap)} datasets; "
                   "Gap = AUC(TabPFN v2) − AUC(best baseline)):\n",
                   pd.DataFrame(lines).to_markdown(index=False, floatfmt=".4f") + "\n"]
            if meta is not None:
                has_cat = meta.reindex(gap.index)["n_cat"] > 0
                md.append(f"Mean gap on datasets **with** categorical features: "
                          f"{gap[has_cat].mean():+.4f} ({int(has_cat.sum())} datasets); "
                          f"**without**: {gap[~has_cat].mean():+.4f} "
                          f"({int((~has_cat).sum())} datasets).\n")

    # ------------------------------ Part B ------------------------------ #
    imp = [m for m in EXISTING + OURS if m in present]
    if "tabpfn_v2" in imp and any(m in present for m in OURS + ["tabpfn_v2_ft"]):
        imp = coverage_filter(pdf, imp)
        tab, piv = summary_table(pdf, imp, "tabpfn_v2", times)
        tab.to_csv(os.path.join(a.out, "table_B_improvement.csv"), index=False)
        piv.to_csv(os.path.join(a.out, "table_B_per_dataset_auc.csv"))
        md += [f"## Part B — Improving TabPFN v2 ({len(piv)} datasets)\n",
               "Reference = original TabPFN v2. ΔAUC > 0 and W/T/L show improvement; "
               "p = two-sided Wilcoxon signed-rank test over datasets.\n", fmt(tab) + "\n"]
        dl = joint(pdf, imp, "log_loss")
        dacc = joint(pdf, imp, "accuracy")
        delta = pd.DataFrame({
            "ΔAUC": (piv.sub(piv["tabpfn_v2"], axis=0)).mean(),
            "ΔAcc": (dacc.sub(dacc["tabpfn_v2"], axis=0)).mean(),
            "ΔLogLoss": (dl.sub(dl["tabpfn_v2"], axis=0)).mean(),
        }).drop(index="tabpfn_v2")
        delta.index = [nice(i) for i in delta.index]
        delta.to_csv(os.path.join(a.out, "table_B_deltas.csv"))
        md += ["Average change w.r.t. TabPFN v2 (LogLoss: negative = better):\n",
               delta.to_markdown(floatfmt="+.4f") + "\n"]
        plot_bar((piv.sub(piv["tabpfn_v2"], axis=0)).mean().drop("tabpfn_v2"),
                 "Part B: mean ΔROC-AUC vs. TabPFN v2", "ΔAUC (higher is better)",
                 os.path.join(a.out, "fig_B_delta_auc.png"), highlight=tuple(OURS))
        plot_bar((dl.sub(dl["tabpfn_v2"], axis=0)).mean().drop("tabpfn_v2"),
                 "Part B: mean Δlog-loss vs. TabPFN v2", "Δlog-loss (lower is better)",
                 os.path.join(a.out, "fig_B_delta_logloss.png"), lower_better=True,
                 highlight=tuple(OURS))
        md.append("![dauc](fig_B_delta_auc.png) ![dll](fig_B_delta_logloss.png)\n")

        # -------- does the gain land where TabPFN v2 was weak? -------- #
        base_cols = [b for b in BASELINES if b in set(pdf["method"])]
        pa = pdf.pivot(index="dataset", columns="method", values="roc_auc")
        if base_cols and "tabpfn_v2" in pa:
            gap = (pa["tabpfn_v2"] - pa[base_cols].max(axis=1)).reindex(piv.index)
            groups = {"TabPFN v2 behind best baseline": gap < -1e-4,
                      "TabPFN v2 ahead": gap >= -1e-4}
            if meta is not None:
                ncat = meta.reindex(piv.index)["n_cat"].fillna(0)
                groups["with categorical features"] = ncat > 0
                groups["numerical only"] = ncat == 0
            rows = []
            for gname, mask in groups.items():
                if mask.sum() == 0:
                    continue
                r = {"Subset": gname, "#datasets": int(mask.sum())}
                for m in imp:
                    if m != "tabpfn_v2":
                        r[nice(m)] = (piv.loc[mask, m] - piv.loc[mask, "tabpfn_v2"]).mean()
                rows.append(r)
            sg = pd.DataFrame(rows)
            sg.to_csv(os.path.join(a.out, "table_B_subgroups.csv"), index=False)
            md += ["Mean ΔAUC vs. TabPFN v2 by subset:\n",
                   sg.to_markdown(index=False, floatfmt="+.4f") + "\n"]
            ours = next((m for m in [FINAL, "tabpfn_gbx_te"] if m in piv), None)
            if ours:
                y = piv[ours] - piv["tabpfn_v2"]
                plot_scatter(gap.values, y.values, "AUC(TabPFN v2) − AUC(best baseline)",
                             f"AUC({nice(ours)}) − AUC(TabPFN v2)",
                             f"Gain of {nice(ours)} vs. weakness of TabPFN v2",
                             os.path.join(a.out, "fig_B_gain_vs_weakness.png"),
                             labels=list(piv.index))
                md.append("![gain](fig_B_gain_vs_weakness.png)\n")
        # -------- individual portfolio members -------- #
        port = coverage_filter(pdf, [m for m in PORTFOLIO if m in present])
        if len(port) > 2:
            tp, _ = summary_table(pdf, port, "tabpfn_v2", times)
            tp.to_csv(os.path.join(a.out, "table_B_portfolio_members.csv"), index=False)
            md += ["Single members of the post-hoc ensemble portfolio:\n", fmt(tp) + "\n"]
        if "weights" in df.columns:
            w = df[df["method"].str.startswith("ens_") & df["weights"].notna()]
            if len(w):
                ww = pd.json_normalize(w["weights"]).assign(method=w["method"].values)
                mw = ww.groupby("method").mean()
                mw.index = [nice(i) for i in mw.index]
                mw.to_csv(os.path.join(a.out, "table_B_ensemble_weights.csv"))
                md += ["Average ensemble weights (chosen on the inner validation split):\n",
                       mw.to_markdown(floatfmt=".3f") + "\n"]

    if "info" in df.columns:
        rows = df[(df["method"] == FINAL) & df["info"].notna()]
        if len(rows) and "gate" in rows["info"].iloc[0]:
            g = pd.DataFrame([dict(dataset=d, fold=k, **i["gate"]) for d, k, i in
                              zip(rows["dataset"], rows["fold"], rows["info"])])
            g.to_csv(os.path.join(a.out, "table_B_gate_decisions.csv"), index=False)
            frac = g.groupby("dataset")["use_ensemble"].mean()
            lines = [f"**Safe gate**: the ensemble replaced TabPFN v2 in "
                     f"{g['use_ensemble'].mean():.0%} of the {len(g)} (dataset, fold) cases; "
                     f"otherwise the original TabPFN v2 prediction was kept unchanged."]
            pa_ = pdf.pivot(index="dataset", columns="method", values="roc_auc")
            bcols = [b for b in BASELINES if b in pa_]
            if bcols and "tabpfn_v2" in pa_:
                gap_ = (pa_["tabpfn_v2"] - pa_[bcols].max(axis=1)).reindex(frac.index)
                lines.append(f"Share of folds using the ensemble: "
                             f"{frac[gap_ < -1e-4].mean():.0%} on datasets where TabPFN v2 "
                             f"is behind the best baseline vs. {frac[gap_ >= -1e-4].mean():.0%} "
                             f"where it is ahead.")
            md += [" ".join(lines) + "\n"]
        if len(rows):
            W = pd.DataFrame([r["weights"]["safe" if "safe" in r["weights"] else "all"]
                              for r in rows["info"]]).fillna(0).mean()
            T = np.mean([r["temperature"].get("safe", r["temperature"]["all"])
                         for r in rows["info"]])
            W.index = [nice(i) for i in W.index]
            W.to_csv(os.path.join(a.out, "table_B_oofphe_weights.csv"))
            md += [f"{nice(FINAL)}: average member weights (selected on 5-fold OOF predictions); "
                   f"mean fitted temperature T = {T:.3f}:\n",
                   W.rename("weight").to_frame().to_markdown(floatfmt=".3f") + "\n"]
    with open(os.path.join(a.out, "report.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    print(f"\nwritten to {a.out}/report.md")


if __name__ == "__main__":
    main()
