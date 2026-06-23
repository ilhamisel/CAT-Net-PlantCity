"""Paper-ready figures combining the 5-fold CV summary and the latency bench.

Inputs:
  results/tables/fruit_cv_summary.csv     (5-fold mean/std per model)
  results/tables/fruit_latency.csv         (CPU/GPU latency per model)
  results/logs/fruit_cv/fold*_*_results.json   (raw per-fold metrics)

Outputs:
  results/figures/fruit_cv_f1_bars.png        mean ± std F1 bar chart
  results/figures/fruit_cv_scatter.png        per-fold F1 dots + mean line
  results/figures/fruit_latency_vs_f1.png     CPU-batch1 latency vs CV F1
                                              (bubble size = params, log x-axis)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


PRETTY = {
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenetv3_large": "MobileNetV3-L",
    "vit_b16": "ViT-B/16",
    "catnet": "CAT-Net",
}
ORDER = ["resnet50", "densenet121", "efficientnet_b0", "mobilenetv3_large",
         "vit_b16", "catnet"]
COLORS = {
    "resnet50": "#4c72b0",
    "densenet121": "#dd8452",
    "efficientnet_b0": "#55a868",
    "mobilenetv3_large": "#c44e52",
    "vit_b16": "#8172b3",
    "catnet": "#000000",
}


def load_cv_summary() -> pd.DataFrame:
    return pd.read_csv(config.TABLE_DIR / "fruit_cv_summary.csv")


def load_latency() -> pd.DataFrame:
    return pd.read_csv(config.TABLE_DIR / "fruit_latency.csv")


def load_per_fold():
    rows = []
    for mn in ORDER:
        for fi in range(5):
            p = config.LOG_DIR / "fruit_cv" / f"fold{fi}_{mn}_results.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text(encoding="utf-8"))
            rows.append({
                "model": mn,
                "fold": fi,
                "accuracy": r["test_metrics"]["accuracy"] * 100,
                "f1_macro": r["test_metrics"]["f1_macro"] * 100,
            })
    return pd.DataFrame(rows)


def cv_bars(summary: pd.DataFrame):
    summary = summary.set_index("model").reindex([PRETTY[m] for m in ORDER])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(summary))
    means = summary["f1_macro_mean"].values
    stds = summary["f1_macro_std"].values
    colors = [COLORS[m] for m in ORDER]
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85,
           edgecolor="black", linewidth=0.7)
    for xi, (m, s) in enumerate(zip(means, stds)):
        ax.text(xi, m + s + 0.1, f"{m:.2f}±{s:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(summary.index, rotation=15)
    ax.set_ylabel("Macro F1 (%)")
    ax.set_ylim(93, 97.5)
    ax.set_title("Fruit-32 benchmark — 5-fold cross-validation F1 (mean ± std)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = config.FIG_DIR / "fruit_cv_f1_bars.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def cv_scatter(per_fold: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, mn in enumerate(ORDER):
        sub = per_fold[per_fold.model == mn]
        ax.scatter([i] * len(sub), sub["f1_macro"], s=60,
                   color=COLORS[mn], alpha=0.75, edgecolor="black", linewidth=0.5,
                   zorder=3)
        m = sub["f1_macro"].mean()
        ax.plot([i - 0.25, i + 0.25], [m, m], color=COLORS[mn], lw=2.5, zorder=4)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([PRETTY[m] for m in ORDER], rotation=15)
    ax.set_ylabel("Macro F1 (%)")
    ax.set_title("Fruit-32 — per-fold F1 distribution (5 folds per model)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(92, 97.5)
    fig.tight_layout()
    out = config.FIG_DIR / "fruit_cv_scatter.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def latency_vs_f1(summary: pd.DataFrame, latency: pd.DataFrame):
    cpu1 = latency[(latency.device == "cpu") & (latency.batch == 1)].copy()
    # Reverse the PRETTY mapping
    name_to_key = {v: k for k, v in PRETTY.items()}
    cpu1["mn"] = cpu1["model"].map(name_to_key)
    cv = summary.copy()
    cv["mn"] = cv["model"].map(name_to_key)
    merged = pd.merge(cpu1, cv, on="mn")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for _, row in merged.iterrows():
        mn = row["mn"]
        ax.scatter(row["mean_ms"], row["f1_macro_mean"],
                   s=row["params_M"] * 25 + 60,
                   color=COLORS[mn], alpha=0.75,
                   edgecolor="black", linewidth=0.8, zorder=3)
        ax.errorbar(row["mean_ms"], row["f1_macro_mean"],
                    yerr=row["f1_macro_std"],
                    color=COLORS[mn], alpha=0.6, capsize=3, zorder=2)
        ax.annotate(PRETTY[mn], (row["mean_ms"], row["f1_macro_mean"]),
                    xytext=(8, 4), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("CPU latency, batch=1 (ms, log scale)")
    ax.set_ylabel("5-fold mean macro F1 (%)")
    ax.set_title("Fruit-32 — accuracy vs CPU latency (bubble area ∝ params)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = config.FIG_DIR / "fruit_latency_vs_f1.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    summary = load_cv_summary()
    latency = load_latency()
    per_fold = load_per_fold()
    cv_bars(summary)
    cv_scatter(per_fold)
    latency_vs_f1(summary, latency)


if __name__ == "__main__":
    main()
