"""Per-fruit failure-mode analysis for the 32-class fruit benchmark.

Inputs : results/logs/fruit_<model>_clean_results.json  (one per model)
         → uses `final_metrics.confusion_matrix` (32x32) and `classes`.

Outputs:
  results/tables/fruit_within_fruit_confusion.md
      For every (model, fruit) pair, the top 3 (true → pred) intra-fruit
      mistakes with counts.  Reveals which disease vs. normal (or disease vs.
      disease) the model struggles with inside a given fruit.

  results/tables/fruit_cross_fruit_errors.md
      For every model, the number of predictions that landed in a *different*
      fruit family than the true label, broken down by source fruit.
      A leaf-shape-recognition sanity check.

  results/tables/fruit_failure_summary.md
      Compact one-row-per-(model, fruit) view:
          accuracy, within-fruit errors, cross-fruit errors,
          dominant intra-fruit confusion as "A→B (n)".
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import config


LOG_DIR = config.LOG_DIR
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


def load_model(mn):
    p = LOG_DIR / f"fruit_{mn}_clean_results.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def short_class(name: str, fruit: str) -> str:
    """Strip the fruit prefix from the class label for tighter tables."""
    s = name
    for prefix in (fruit, fruit.lower(), "Lokat", "lokat"):
        if s.startswith(prefix + " ") or s.startswith(prefix + "_"):
            s = s[len(prefix) + 1:].strip()
            break
    return s.strip(" _") or name


def main():
    fruit_of = config.FRUIT_OF
    classes_to_fruit = [fruit_of[c] for c in config.FRUIT_CLASSES]
    fruit_to_class_idx = defaultdict(list)
    for i, c in enumerate(config.FRUIT_CLASSES):
        fruit_to_class_idx[fruit_of[c]].append(i)
    fruits = sorted({fruit_of[c] for c in config.FRUIT_CLASSES})

    within_rows = []
    cross_rows = []
    summary_rows = []

    for mn in MODELS:
        try:
            res = load_model(mn)
        except FileNotFoundError:
            print(f"skip {mn} (no log)")
            continue
        classes = res["classes"]
        assert classes == config.FRUIT_CLASSES, f"class order mismatch for {mn}"
        cm = np.asarray(res["final_metrics"]["confusion_matrix"], dtype=int)
        n_total_err = 0
        n_cross = 0

        for fruit in fruits:
            idxs = fruit_to_class_idx[fruit]
            block = cm[np.ix_(idxs, idxs)]
            support = cm[idxs].sum(axis=1).sum()
            block_correct = int(np.trace(block))
            block_total = int(block.sum())
            within_errors = block_total - block_correct
            cross_errors = int(cm[idxs].sum()) - block_total
            acc = (block_correct + 0.0) / support * 100 if support else float("nan")

            # Top 3 intra-fruit (true -> pred) mistakes
            mistakes = []
            for li, gi in enumerate(idxs):
                for lj, gj in enumerate(idxs):
                    if gi == gj:
                        continue
                    n = int(block[li, lj])
                    if n > 0:
                        mistakes.append((n, classes[gi], classes[gj]))
            mistakes.sort(reverse=True)
            top3 = mistakes[:3]

            for n, tc, pc in top3:
                within_rows.append({
                    "model": MODEL_DISPLAY[mn],
                    "fruit": fruit,
                    "true": short_class(tc, fruit),
                    "pred": short_class(pc, fruit),
                    "count": n,
                })

            cross_rows.append({
                "model": MODEL_DISPLAY[mn],
                "fruit": fruit,
                "support": int(support),
                "within_fruit_errors": within_errors,
                "cross_fruit_errors": cross_errors,
            })

            dom = (f"{short_class(top3[0][1], fruit)}→{short_class(top3[0][2], fruit)} "
                   f"({top3[0][0]})") if top3 else "—"
            summary_rows.append({
                "model": MODEL_DISPLAY[mn],
                "fruit": fruit,
                "support": int(support),
                "acc_%": round(acc, 2),
                "within_err": within_errors,
                "cross_err": cross_errors,
                "top_intra_confusion": dom,
            })
            n_total_err += within_errors + cross_errors
            n_cross += cross_errors

        print(f"{MODEL_DISPLAY[mn]}: total errors={n_total_err}  cross-fruit={n_cross}  "
              f"({n_cross / max(n_total_err, 1):.1%} of errors)")

    def write_md(path, rows, cols):
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

    write_md(TBL_DIR / "fruit_within_fruit_confusion.md", within_rows,
             ["model", "fruit", "true", "pred", "count"])
    write_md(TBL_DIR / "fruit_cross_fruit_errors.md", cross_rows,
             ["model", "fruit", "support", "within_fruit_errors", "cross_fruit_errors"])
    write_md(TBL_DIR / "fruit_failure_summary.md", summary_rows,
             ["model", "fruit", "support", "acc_%", "within_err", "cross_err",
              "top_intra_confusion"])


if __name__ == "__main__":
    main()
