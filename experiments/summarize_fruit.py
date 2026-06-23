"""Consolidate fruit benchmark results into publication tables and figures.

Inputs: per-model JSON logs in results/logs/fruit_<model>_clean_results.json,
        robustness CSV in results/tables/fruit_robustness.csv (optional).

Outputs (all prefixed with `fruit_`):
- headline.{csv,md,tex}      6 models x {params, acc, macro F1, AUC, mean robust, worst robust}
- per_fruit_accuracy.{csv,md} 6 models x 8 fruits (rows aggregated over each fruit's classes)
- per_class_clean.csv         per-class precision/recall/F1 for every model
- confusion_matrices.png      6-panel grid (one per model) on the 32-class test set
- robustness_bars.png         clean vs mean robust vs worst-case bar chart (if robustness ran)
- params_vs_f1.png            params (log) vs macro F1 scatter
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


def load_results():
    out = {}
    for m in config.MODELS:
        p = config.LOG_DIR / f"fruit_{m}_clean_results.json"
        if not p.exists():
            print(f"[warn] missing {p}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            out[m] = json.load(f)
    return out


def load_robustness():
    p = config.TABLE_DIR / "fruit_robustness.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def headline_table(results, robust):
    rows = []
    for m, r in results.items():
        fm = r["final_metrics"]
        row = {
            "model": PRETTY[m],
            "params_M": round(r["num_params"] / 1e6, 2),
            "accuracy": round(fm["accuracy"] * 100, 2),
            "precision_macro": round(fm["precision_macro"] * 100, 2),
            "recall_macro": round(fm["recall_macro"] * 100, 2),
            "f1_macro": round(fm["f1_macro"] * 100, 2),
            "auc_ovr": round(fm["auc_ovr_macro"], 4),
            "epochs": r["epochs_trained"],
        }
        if robust is not None:
            sub = robust[(robust.model == m) & (robust.corruption != "clean")]
            if len(sub) > 0:
                row["mean_robust_acc"] = round(sub["accuracy"].mean() * 100, 2)
                row["worst_robust_acc"] = round(sub["accuracy"].min() * 100, 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(config.TABLE_DIR / "fruit_headline.csv", index=False)
    with open(config.TABLE_DIR / "fruit_headline.md", "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))
    with open(config.TABLE_DIR / "fruit_headline.tex", "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, float_format="%.2f"))
    print("[table] fruit_headline.{csv,md,tex}")
    return df


def per_class_table(results):
    rows = []
    classes = config.FRUIT_CLASSES
    for m, r in results.items():
        fm = r["final_metrics"]
        for ci, cls in enumerate(classes):
            rows.append({
                "model": m,
                "class": cls,
                "fruit": config.FRUIT_OF[cls],
                "precision": fm["per_class_precision"][ci],
                "recall": fm["per_class_recall"][ci],
                "f1": fm["per_class_f1"][ci],
                "support": fm["per_class_support"][ci],
            })
    df = pd.DataFrame(rows)
    df.to_csv(config.TABLE_DIR / "fruit_per_class_clean.csv", index=False)
    print("[table] fruit_per_class_clean.csv")
    return df


def per_fruit_accuracy(results, per_class_df):
    """Per-fruit accuracy = correctly classified examples of that fruit's classes
    / total support of that fruit's classes.  Uses the confusion matrix to be
    exact (not just averaging per-class recall)."""
    classes = config.FRUIT_CLASSES
    fruits = sorted(set(config.FRUIT_OF.values()))
    rows = []
    for m, r in results.items():
        cm = np.array(r["final_metrics"]["confusion_matrix"])
        row = {"model": PRETTY[m]}
        for fruit in fruits:
            idx = [i for i, c in enumerate(classes) if config.FRUIT_OF[c] == fruit]
            # correct = sum of diagonal entries on rows belonging to this fruit
            correct = int(cm[idx, :][:, idx].diagonal().sum())  # only same-fruit predictions count as correct?
            # ^^ But for paper we want the accuracy on those test images regardless of
            # which fruit they were predicted as. So use the full row sum on diagonal:
            correct = int(sum(cm[i, i] for i in idx))
            support = int(cm[idx, :].sum())
            row[fruit] = round(correct / max(1, support) * 100, 2)
            row[f"{fruit}_n"] = support
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(config.TABLE_DIR / "fruit_per_fruit_accuracy.csv", index=False)
    with open(config.TABLE_DIR / "fruit_per_fruit_accuracy.md", "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))
    print("[table] fruit_per_fruit_accuracy.{csv,md}")
    return df


def confusion_matrices_figure(results):
    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    if rows == 1:
        axes = np.array([axes])
    for ax, (m, r) in zip(axes.flatten(), results.items()):
        cm = np.array(r["final_metrics"]["confusion_matrix"], dtype=np.float32)
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{PRETTY[m]}  (acc={r['final_metrics']['accuracy']*100:.2f}%)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(len(config.FRUIT_CLASSES)))
        ax.set_yticks(range(len(config.FRUIT_CLASSES)))
        ax.set_xticklabels([c[:14] for c in config.FRUIT_CLASSES], rotation=90, fontsize=6)
        ax.set_yticklabels([c[:14] for c in config.FRUIT_CLASSES], fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes.flatten()[len(results):]:
        ax.axis("off")
    fig.tight_layout()
    out = config.FIG_DIR / "fruit_confusion_matrices.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {out}")


def robustness_bars(results, robust):
    if robust is None:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    models = [m for m in config.MODELS if m in results]
    clean_acc = [results[m]["final_metrics"]["accuracy"] * 100 for m in models]
    mean_rob = []
    worst_rob = []
    for m in models:
        sub = robust[(robust.model == m) & (robust.corruption != "clean")]
        mean_rob.append(sub["accuracy"].mean() * 100 if len(sub) else np.nan)
        worst_rob.append(sub["accuracy"].min() * 100 if len(sub) else np.nan)

    x = np.arange(len(models))
    w = 0.27
    ax.bar(x - w, clean_acc, w, label="Clean", color="#3a86ff")
    ax.bar(x, mean_rob, w, label="Mean robust", color="#fb8500")
    ax.bar(x + w, worst_rob, w, label="Worst-case", color="#d62828")
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[m] for m in models], rotation=15)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Fruit-32 benchmark: clean vs corruption robustness")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = config.FIG_DIR / "fruit_robustness_bars.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {out}")


def params_vs_f1(results):
    fig, ax = plt.subplots(figsize=(7, 5))
    for m, r in results.items():
        p = r["num_params"] / 1e6
        f1 = r["final_metrics"]["f1_macro"] * 100
        ax.scatter(p, f1, s=120 if m == "catnet" else 80,
                    marker="*" if m == "catnet" else "o",
                    label=PRETTY[m])
        ax.annotate(PRETTY[m], (p, f1), xytext=(5, 5),
                     textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (M, log scale)")
    ax.set_ylabel("Macro F1 (%)")
    ax.set_title("Fruit-32: params vs macro F1")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = config.FIG_DIR / "fruit_params_vs_f1.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {out}")


def main():
    results = load_results()
    if not results:
        print("No fruit results found; train first.")
        return
    robust = load_robustness()
    headline_table(results, robust)
    pcdf = per_class_table(results)
    per_fruit_accuracy(results, pcdf)
    confusion_matrices_figure(results)
    robustness_bars(results, robust)
    params_vs_f1(results)
    print("Done.")


if __name__ == "__main__":
    main()
