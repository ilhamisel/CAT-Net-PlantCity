"""Aggregate 5-fold CV results: mean ± std per model + paired McNemar.

Inputs:  results/logs/fruit_cv/fold{0..4}_<model>_results.json
         (written by cv5_fold.py — each contains per-sample test_predictions,
          test_labels, test_paths, and test_metrics for one fold.)

Outputs:
  results/tables/fruit_cv_summary.{csv,md}
      mean ± std of accuracy, F1-macro, precision-macro, recall-macro,
      AUC-OvR-macro, and epochs_trained over 5 folds.

  results/tables/fruit_cv_mcnemar.{csv,md}
      Pairwise McNemar chi-squared p-values on the *concatenated*
      out-of-fold predictions (each sample appears exactly once because
      StratifiedKFold partitions the originals).  Uses the continuity-
      corrected statistic, with the exact binomial fallback when b+c ≤ 25.

Run after cv5_fold.py finishes all (fold × model) jobs.
"""
from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, chi2

import config


CV_LOG_DIR = config.LOG_DIR / "fruit_cv"
TBL_DIR = config.TABLE_DIR
MODELS = config.MODELS
MODEL_DISPLAY = {
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenetv3_large": "MobileNetV3-L",
    "vit_b16": "ViT-B/16",
    "catnet": "CAT-Net",
}
METRICS = ["accuracy", "f1_macro", "precision_macro", "recall_macro", "auc_ovr_macro"]


def load_fold(model_name: str, fold_idx: int):
    p = CV_LOG_DIR / f"fold{fold_idx}_{model_name}_results.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_metrics():
    rows = []
    per_model_paths_preds = {}  # mn -> (paths concatenated, preds concatenated, labels)
    for mn in MODELS:
        per_fold = []
        for fi in range(5):
            r = load_fold(mn, fi)
            if r is None:
                print(f"missing fold{fi} {mn}")
                continue
            per_fold.append(r)
        if not per_fold:
            continue
        row = {"model": MODEL_DISPLAY[mn]}
        for m in METRICS:
            vals = np.array([r["test_metrics"].get(m, np.nan) for r in per_fold],
                            dtype=float)
            row[f"{m}_mean"] = float(np.nanmean(vals))
            row[f"{m}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        epochs = np.array([r["epochs_trained"] for r in per_fold], dtype=float)
        row["epochs_mean"] = float(epochs.mean())
        row["epochs_std"] = float(epochs.std(ddof=1)) if len(epochs) > 1 else 0.0
        row["n_folds"] = len(per_fold)
        rows.append(row)

        paths = []
        preds = []
        labels = []
        for r in per_fold:
            paths.extend(r["test_paths"])
            preds.extend(r["test_predictions"])
            labels.extend(r["test_labels"])
        per_model_paths_preds[mn] = (paths, np.array(preds), np.array(labels))
    return rows, per_model_paths_preds


def mcnemar_pair(a_correct: np.ndarray, b_correct: np.ndarray):
    """Standard McNemar with continuity correction; exact binomial if b+c small."""
    b = int(np.sum(a_correct & ~b_correct))   # A right, B wrong
    c = int(np.sum(~a_correct & b_correct))   # A wrong, B right
    n_disc = b + c
    if n_disc == 0:
        return {"b": b, "c": c, "stat": 0.0, "p_value": 1.0, "test": "n/a"}
    if n_disc <= 25:
        p = binomtest(min(b, c), n_disc, p=0.5, alternative="two-sided").pvalue
        return {"b": b, "c": c, "stat": float("nan"),
                "p_value": float(p), "test": "exact"}
    stat = (abs(b - c) - 1.0) ** 2 / n_disc
    p = float(1.0 - chi2.cdf(stat, df=1))
    return {"b": b, "c": c, "stat": float(stat),
            "p_value": float(p), "test": "chi2_cc"}


def pairwise_mcnemar(per_model):
    if not per_model:
        return []
    # Align all models on a common set of paths.  Use the intersection of paths
    # available across every model and reorder predictions accordingly.
    common_paths = None
    for mn, (paths, _, _) in per_model.items():
        sp = set(paths)
        common_paths = sp if common_paths is None else common_paths & sp
    if not common_paths:
        print("WARN no common paths across models — McNemar skipped")
        return []
    ordered = sorted(common_paths)
    label_check = None
    aligned = {}
    for mn, (paths, preds, labels) in per_model.items():
        idx = {p: i for i, p in enumerate(paths)}
        order = np.array([idx[p] for p in ordered], dtype=int)
        aligned[mn] = (preds[order], labels[order])
        if label_check is None:
            label_check = labels[order]
        else:
            assert np.array_equal(label_check, labels[order]), (
                f"label mismatch for {mn} — folds were not sample-aligned")
    rows = []
    names = list(per_model.keys())
    for ma, mb in combinations(names, 2):
        a_pred, y = aligned[ma]
        b_pred, _ = aligned[mb]
        ac = a_pred == y
        bc = b_pred == y
        out = mcnemar_pair(ac, bc)
        rows.append({
            "model_A": MODEL_DISPLAY[ma],
            "model_B": MODEL_DISPLAY[mb],
            "A_right_B_wrong": out["b"],
            "A_wrong_B_right": out["c"],
            "test": out["test"],
            "stat": round(out["stat"], 4) if not np.isnan(out["stat"]) else "—",
            "p_value": f"{out['p_value']:.4g}",
            "n_samples": len(ordered),
        })
    return rows


def write_md(path: Path, rows: list[dict], cols: list[str]):
    if not rows:
        print(f"WARN no rows for {path.name}")
        return
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    header = "| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |"
    sep = "|" + "|".join(("-" * (widths[c] + 2)) for c in cols) + "|"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]).ljust(widths[c]) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def write_csv(path: Path, rows: list[dict], cols: list[str]):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main():
    rows, per_model = aggregate_metrics()

    summary_cols = ["model", "n_folds"]
    for m in METRICS:
        summary_cols += [f"{m}_mean", f"{m}_std"]
    summary_cols += ["epochs_mean", "epochs_std"]
    pretty = []
    for r in rows:
        out = {"model": r["model"], "n_folds": r["n_folds"]}
        for m in METRICS:
            out[f"{m}_mean"] = round(r[f"{m}_mean"] * 100, 2) if m != "auc_ovr_macro" \
                               else round(r[f"{m}_mean"], 4)
            out[f"{m}_std"] = round(r[f"{m}_std"] * 100, 2) if m != "auc_ovr_macro" \
                              else round(r[f"{m}_std"], 4)
        out["epochs_mean"] = round(r["epochs_mean"], 1)
        out["epochs_std"] = round(r["epochs_std"], 1)
        pretty.append(out)
    write_md(TBL_DIR / "fruit_cv_summary.md", pretty, summary_cols)
    write_csv(TBL_DIR / "fruit_cv_summary.csv", pretty, summary_cols)

    mc_rows = pairwise_mcnemar(per_model)
    cols = ["model_A", "model_B", "A_right_B_wrong", "A_wrong_B_right",
            "test", "stat", "p_value", "n_samples"]
    write_md(TBL_DIR / "fruit_cv_mcnemar.md", mc_rows, cols)
    write_csv(TBL_DIR / "fruit_cv_mcnemar.csv", mc_rows, cols)


if __name__ == "__main__":
    main()
