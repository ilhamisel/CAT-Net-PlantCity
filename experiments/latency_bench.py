"""Inference latency + throughput benchmark for the 6 models.

Reported on (CPU, GPU) × (batch=1, batch=32):
  - mean/median/p95/p99 latency in ms over `n_iter` timed runs after
    `n_warmup` warmup runs
  - throughput in images/sec
  - parameter count and on-disk checkpoint size (MB)

CPU runs are forced via `torch.device("cpu")`; cuDNN benchmark mode is enabled
on GPU.  For batch=1 GPU timing we use `torch.cuda.synchronize` around every
forward pass — necessary because launches are otherwise asynchronous and would
give absurdly low numbers.

Output: results/tables/fruit_latency.{csv,md}
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

import config
from models import build_model


MODEL_DISPLAY = {
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenetv3_large": "MobileNetV3-L",
    "vit_b16": "ViT-B/16",
    "catnet": "CAT-Net",
}


def measure(model, x, n_warmup, n_iter, device):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(n_iter):
            if device == "cuda":
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(x)
                torch.cuda.synchronize()
            else:
                t0 = time.perf_counter()
                _ = model(x)
            times.append((time.perf_counter() - t0) * 1000.0)
    return times


def checkpoint_size_mb(model_name: str) -> float:
    # Fruit-namespaced clean checkpoint produced by run_all.py / run_catnet_only.py
    p = config.CKPT_DIR / f"fruit_{model_name}_clean_best.pt"
    if not p.exists():
        return float("nan")
    return p.stat().st_size / (1024 * 1024)


def run_model(mn, num_classes, device, batches, n_warmup, n_iter):
    model = build_model(mn, num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    results = {}
    for bs in batches:
        x = torch.randn(bs, 3, config.IMG_SIZE, config.IMG_SIZE, device=device)
        ts = measure(model, x, n_warmup, n_iter, device)
        arr = np.array(ts)
        results[bs] = {
            "mean_ms": float(arr.mean()),
            "median_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "std_ms": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "imgs_per_sec": float(bs * 1000.0 / arr.mean()),
        }
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return n_params, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--devices", nargs="*", default=["cpu", "cuda"])
    ap.add_argument("--batches", nargs="*", type=int, default=[1, 32])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    models_to_run = args.models or config.MODELS
    num_classes = len(config.FRUIT_CLASSES)
    devices = [d for d in args.devices
               if d == "cpu" or (d == "cuda" and torch.cuda.is_available())]

    rows = []
    for mn in models_to_run:
        ckpt_mb = checkpoint_size_mb(mn)
        for dev in devices:
            print(f"\n--- {MODEL_DISPLAY[mn]} on {dev} ---", flush=True)
            n_params, res = run_model(mn, num_classes, dev,
                                      args.batches, args.warmup, args.iters)
            for bs, m in res.items():
                rows.append({
                    "model": MODEL_DISPLAY[mn],
                    "device": dev,
                    "batch": bs,
                    "mean_ms": round(m["mean_ms"], 3),
                    "median_ms": round(m["median_ms"], 3),
                    "p95_ms": round(m["p95_ms"], 3),
                    "p99_ms": round(m["p99_ms"], 3),
                    "std_ms": round(m["std_ms"], 3),
                    "imgs_per_sec": round(m["imgs_per_sec"], 1),
                    "params_M": round(n_params / 1e6, 2),
                    "ckpt_MB": round(ckpt_mb, 2) if ckpt_mb == ckpt_mb else "—",
                })
                print(f"  bs={bs:>2}  mean={m['mean_ms']:.2f} ms  "
                      f"median={m['median_ms']:.2f} ms  "
                      f"thr={m['imgs_per_sec']:.1f} img/s", flush=True)

    cols = ["model", "device", "batch", "mean_ms", "median_ms", "p95_ms",
            "p99_ms", "std_ms", "imgs_per_sec", "params_M", "ckpt_MB"]
    out_csv = config.TABLE_DIR / "fruit_latency.csv"
    out_md = config.TABLE_DIR / "fruit_latency.md"

    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

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
