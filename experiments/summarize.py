"""Post-hoc summary: robustness bar chart and combined results table.

Builds:
- results/figures/robustness_bars.png  (model x corruption mean accuracy)
- results/tables/robustness_summary.md  (mean and worst-case acc per model)
- results/tables/headline.md            (overall headline table)
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
    "catnet": "CAT-Net (ours)",
}

ORDER = ["resnet50", "densenet121", "efficientnet_b0", "mobilenetv3_large",
         "vit_b16", "catnet"]


def main():
    rob = pd.read_csv(config.TABLE_DIR / "robustness.csv")
    rob_corr = rob[rob.corruption != "clean"]
    mean_acc = rob_corr.groupby("model")["accuracy"].mean()
    worst_acc = rob_corr.groupby("model")["accuracy"].min()
    clean_acc = rob[rob.corruption == "clean"].set_index("model")["accuracy"]

    # ---- bar chart: mean robust accuracy per model, with corruption stacks ----
    pivot = rob_corr.groupby(["model", "corruption"])["accuracy"].mean().unstack()
    pivot = pivot.reindex(ORDER)
    pivot.to_csv(config.TABLE_DIR / "robustness_mean_per_corruption.csv")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(ORDER))
    width = 0.15
    colors = plt.get_cmap("Set2")
    for i, col in enumerate(pivot.columns):
        ax.bar(x + i * width, pivot[col].values, width, label=col, color=colors(i))
    ax.set_xticks(x + width * (len(pivot.columns) - 1) / 2)
    ax.set_xticklabels([PRETTY[m] for m in ORDER], rotation=12)
    ax.set_ylabel("Mean accuracy across severities")
    ax.set_title("Robustness across corruption types")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="grey", lw=0.5, ls=":")
    ax.legend(fontsize=9, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = config.FIG_DIR / "robustness_bars.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("[figure]", out)

    # ---- summary table ----
    summary = pd.DataFrame({
        "Model": [PRETTY[m] for m in ORDER],
        "Clean Acc": [clean_acc[m] for m in ORDER],
        "Mean Robust Acc": [mean_acc[m] for m in ORDER],
        "Worst-case Acc": [worst_acc[m] for m in ORDER],
    })
    summary["Clean Acc"] = (summary["Clean Acc"] * 100).round(2)
    summary["Mean Robust Acc"] = (summary["Mean Robust Acc"] * 100).round(2)
    summary["Worst-case Acc"] = (summary["Worst-case Acc"] * 100).round(2)
    out_md = config.TABLE_DIR / "robustness_summary.md"
    with open(out_md, "w") as f:
        f.write(summary.to_markdown(index=False))
    print("[table]", out_md)

    # ---- headline table merging main + robustness ----
    main_csv = config.TABLE_DIR / "main_results_clean.csv"
    head_csv = config.TABLE_DIR / "headline.csv"
    df = pd.read_csv(main_csv)
    df["Mean Robust Acc"] = summary["Mean Robust Acc"].values
    df["Worst-case Acc"] = summary["Worst-case Acc"].values
    df.to_csv(head_csv, index=False)
    with open(config.TABLE_DIR / "headline.md", "w") as f:
        f.write(df.to_markdown(index=False))
    with open(config.TABLE_DIR / "headline.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format=lambda x: f"{x:.2f}"))
    print("[table] headline.{csv,md,tex}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
