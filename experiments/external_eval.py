"""External-domain evaluation of the trained apple checkpoints.

Usage:
    /c/Users/ilhami/anaconda3/python.exe external_eval.py \
        --data_dir <path with subfolders per class> \
        --mapping  <json mapping external_class -> APPLE_CLASS or "skip">

Why a mapping file: PlantVillage names diseases differently (Apple_scab vs.
Apple Brown_spot, etc.) and may include classes we did not train on
(Cedar_apple_rust).  The mapping resolves names; "skip" filters out classes
without a counterpart in PlantCity.

Output: results/tables/external_eval_<dataset_name>.{csv,md}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import config
from dataset import build_transforms, AppleLeafDataset
from models import build_model

MODEL_DISPLAY = {
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenetv3_large": "MobileNetV3-L",
    "vit_b16": "ViT-B/16",
    "catnet": "CAT-Net",
}


def collect_samples(data_dir: Path, mapping: dict[str, str]):
    samples = []
    for sub in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        target = mapping.get(sub.name)
        if target == "skip" or target is None:
            continue
        if target not in config.APPLE_CLASSES:
            raise ValueError(f"Mapping target {target!r} not in APPLE_CLASSES")
        y = config.APPLE_CLASSES.index(target)
        for f in sub.iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                samples.append((str(f), y))
    return samples


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    logits_all = []
    y_all = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        logits_all.append(logits.cpu())
        y_all.append(y)
    logits = torch.cat(logits_all)
    y = torch.cat(y_all).numpy()
    preds = logits.argmax(1).numpy()
    probs = F.softmax(logits, 1).numpy()
    acc = accuracy_score(y, preds)
    cm = confusion_matrix(y, preds, labels=list(range(len(config.APPLE_CLASSES))))
    report = classification_report(
        y, preds, target_names=config.APPLE_CLASSES,
        labels=list(range(len(config.APPLE_CLASSES))), zero_division=0, output_dict=True,
    )
    return {"accuracy": float(acc), "confusion_matrix": cm.tolist(), "report": report,
            "n_samples": int(len(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--mapping", required=True,
                    help="JSON: {external_class: APPLE_CLASS or 'skip'}")
    ap.add_argument("--dataset_name", default="external")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    if config.CROP != "apple":
        raise SystemExit("This evaluator targets the 3-class APPLE checkpoints. "
                         "Run without CROP=fruit or unset the variable.")

    data_dir = Path(args.data_dir)
    mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
    samples = collect_samples(data_dir, mapping)
    print(f"Loaded {len(samples)} samples across {len(set(y for _,y in samples))} "
          f"mapped classes from {data_dir}")
    if not samples:
        raise SystemExit("No samples found after mapping — check folder names.")

    classes = config.APPLE_CLASSES
    ds = AppleLeafDataset(samples, classes, build_transforms(False))
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_to_run = args.models or config.MODELS

    rows = []
    for mn in models_to_run:
        ckpt = config.CKPT_DIR / f"{mn}_clean_best.pt"  # apple namespace = ""
        if not ckpt.exists():
            print(f"skip {mn} (no checkpoint at {ckpt})")
            continue
        model = build_model(mn, len(classes)).to(device)
        state = torch.load(ckpt, map_location=device)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state)
        metrics = evaluate(model, loader, device)
        rows.append({
            "model": MODEL_DISPLAY[mn],
            "n_samples": metrics["n_samples"],
            "accuracy": round(metrics["accuracy"] * 100, 2),
            "f1_macro": round(metrics["report"]["macro avg"]["f1-score"] * 100, 2),
            "precision_macro": round(metrics["report"]["macro avg"]["precision"] * 100, 2),
            "recall_macro": round(metrics["report"]["macro avg"]["recall"] * 100, 2),
        })
        print(f"{MODEL_DISPLAY[mn]}: acc={metrics['accuracy']:.4f}", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("No models evaluated.")

    cols = ["model", "n_samples", "accuracy", "precision_macro", "recall_macro", "f1_macro"]
    out_csv = config.TABLE_DIR / f"external_eval_{args.dataset_name}.csv"
    out_md = config.TABLE_DIR / f"external_eval_{args.dataset_name}.md"

    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    header = "| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |"
    sep = "|" + "|".join(("-" * (widths[c] + 2)) for c in cols) + "|"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]).ljust(widths[c]) for c in cols) + " |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out_csv}\nWrote {out_md}")


if __name__ == "__main__":
    main()
