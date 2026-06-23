"""Generate publication figures and tables from JSON result files."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config

PRETTY = {
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenetv3_large": "MobileNetV3-L",
    "vit_b16": "ViT-B/16",
    "catnet": "CAT-Net (ours)",
}

MODEL_ORDER = ["resnet50", "densenet121", "efficientnet_b0", "mobilenetv3_large",
               "vit_b16", "catnet"]


def load_results(split_mode: str = "clean"):
    out = {}
    for name in MODEL_ORDER:
        p = config.LOG_DIR / f"{name}_{split_mode}_results.json"
        if p.exists():
            with open(p) as f:
                out[name] = json.load(f)
    return out


def build_main_table(results: dict, split_mode: str) -> pd.DataFrame:
    rows = []
    for name in MODEL_ORDER:
        if name not in results:
            continue
        r = results[name]
        m = r["final_metrics"]
        rows.append({
            "Model": PRETTY[name],
            "#Params (M)": round(r["num_params"] / 1e6, 2),
            "Accuracy": round(m["accuracy"] * 100, 2),
            "Precision (macro)": round(m["precision_macro"] * 100, 2),
            "Recall (macro)": round(m["recall_macro"] * 100, 2),
            "F1 (macro)": round(m["f1_macro"] * 100, 2),
            "AUC (OvR)": round(m["auc_ovr_macro"], 4),
            "Epochs": r["epochs_trained"],
        })
    df = pd.DataFrame(rows)
    csv = config.TABLE_DIR / f"main_results_{split_mode}.csv"
    df.to_csv(csv, index=False)
    with open(config.TABLE_DIR / f"main_results_{split_mode}.md", "w") as f:
        f.write(df.to_markdown(index=False))
    with open(config.TABLE_DIR / f"main_results_{split_mode}.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"[table] {csv}")
    return df


def build_per_class_table(results: dict, split_mode: str) -> pd.DataFrame:
    rows = []
    for name in MODEL_ORDER:
        if name not in results:
            continue
        m = results[name]["final_metrics"]
        for ci, cls in enumerate(config.APPLE_CLASSES):
            rows.append({
                "Model": PRETTY[name],
                "Class": cls,
                "Precision": round(m["per_class_precision"][ci] * 100, 2),
                "Recall": round(m["per_class_recall"][ci] * 100, 2),
                "F1": round(m["per_class_f1"][ci] * 100, 2),
                "Support": int(m["per_class_support"][ci]),
            })
    df = pd.DataFrame(rows)
    csv = config.TABLE_DIR / f"per_class_{split_mode}.csv"
    df.to_csv(csv, index=False)
    with open(config.TABLE_DIR / f"per_class_{split_mode}.md", "w") as f:
        f.write(df.to_markdown(index=False))
    print(f"[table] {csv}")
    return df


def plot_training_curves(results: dict, split_mode: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(MODEL_ORDER):
        if name not in results:
            continue
        log = results[name]["epoch_log"]
        ep = [r["epoch"] for r in log]
        axes[0].plot(ep, [r["val_acc"] for r in log], color=cmap(i),
                     label=PRETTY[name], lw=2 if name == "catnet" else 1.5,
                     linestyle="-" if name != "catnet" else "--")
        axes[1].plot(ep, [r["val_f1_macro"] for r in log], color=cmap(i),
                     label=PRETTY[name], lw=2 if name == "catnet" else 1.5,
                     linestyle="-" if name != "catnet" else "--")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Validation Accuracy"); axes[0].set_title("Validation Accuracy"); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation F1 (macro)"); axes[1].set_title("Validation F1 (macro)"); axes[1].grid(alpha=0.3)
    axes[1].legend(loc="lower right", fontsize=9)
    fig.suptitle(f"Training Curves ({split_mode} split)")
    fig.tight_layout()
    out = config.FIG_DIR / f"training_curves_{split_mode}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {out}")


def plot_confusion_matrices(results: dict, split_mode: str):
    n = len(MODEL_ORDER)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.9))
    axes = axes.flatten()
    for i, name in enumerate(MODEL_ORDER):
        ax = axes[i]
        if name not in results:
            ax.axis("off"); continue
        cm = np.array(results[name]["final_metrics"]["confusion_matrix"])
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["BS", "N", "BK"], yticklabels=["BS", "N", "BK"], ax=ax,
                    vmin=0, vmax=1)
        ax.set_title(PRETTY[name])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Confusion Matrices ({split_mode} split)  BS=Brown_spot  N=Normal  BK=black_spot")
    fig.tight_layout()
    out = config.FIG_DIR / f"confusion_matrices_{split_mode}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {out}")


def plot_params_vs_f1(results: dict, split_mode: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in MODEL_ORDER:
        if name not in results:
            continue
        r = results[name]
        x = r["num_params"] / 1e6
        y = r["final_metrics"]["f1_macro"] * 100
        marker = "*" if name == "catnet" else "o"
        size = 320 if name == "catnet" else 130
        ax.scatter(x, y, s=size, marker=marker, label=PRETTY[name],
                   edgecolor="black", linewidths=0.7)
        ax.annotate(PRETTY[name], (x, y), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (M, log)")
    ax.set_ylabel("F1 (macro), %")
    ax.set_title(f"Accuracy-Efficiency Trade-off ({split_mode} split)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = config.FIG_DIR / f"params_vs_f1_{split_mode}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {out}")


def main(split_mode: str = "clean"):
    results = load_results(split_mode)
    if not results:
        print(f"No results found for split={split_mode}")
        return
    print(f"Found {len(results)} model results for split={split_mode}: {list(results)}")
    df = build_main_table(results, split_mode)
    print("\nMain results:\n", df.to_string(index=False))
    build_per_class_table(results, split_mode)
    plot_training_curves(results, split_mode)
    plot_confusion_matrices(results, split_mode)
    plot_params_vs_f1(results, split_mode)


if __name__ == "__main__":
    import sys
    split = sys.argv[1] if len(sys.argv) > 1 else "clean"
    main(split)
