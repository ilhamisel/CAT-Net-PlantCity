"""Robustness evaluation under image corruptions for trained models.

For each trained checkpoint (clean split, best validation F1), we re-evaluate
the held-out test set under 5 corruption types at 3 severity levels each,
producing a per-model robustness profile and a corruption-vs-accuracy curve.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import config
from dataset import _enumerate_folder, stratified_split
from models import build_model


SEVERITIES = [1, 2, 3]


def _to_uint8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def gaussian_noise(img: np.ndarray, severity: int) -> np.ndarray:
    sigma = {1: 12, 2: 25, 3: 45}[severity]
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return _to_uint8(img.astype(np.float32) + noise)


def motion_blur(img: np.ndarray, severity: int) -> np.ndarray:
    k = {1: 5, 2: 9, 3: 15}[severity]
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)


def brightness(img: np.ndarray, severity: int) -> np.ndarray:
    factor = {1: 0.65, 2: 0.45, 3: 1.55}[severity]
    return _to_uint8(img.astype(np.float32) * factor)


def jpeg_compression(img: np.ndarray, severity: int) -> np.ndarray:
    q = {1: 35, 2: 20, 3: 10}[severity]
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def defocus_blur(img: np.ndarray, severity: int) -> np.ndarray:
    k = {1: 3, 2: 5, 3: 9}[severity]
    return cv2.GaussianBlur(img, (k, k), 0)


CORRUPTIONS: Dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "gaussian_noise": gaussian_noise,
    "motion_blur": motion_blur,
    "brightness": brightness,
    "jpeg_compression": jpeg_compression,
    "defocus_blur": defocus_blur,
}


def model_forward(model, x):
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def eval_under_corruption(model, samples, fn, severity, device):
    pre_pil = transforms.Compose([
        transforms.Resize(int(config.IMG_SIZE * 1.15)),
        transforms.CenterCrop(config.IMG_SIZE),
    ])
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])
    correct, total = 0, 0
    for path, label in samples:
        pil = Image.open(path).convert("RGB")
        pil = pre_pil(pil)
        arr = np.asarray(pil, dtype=np.uint8)
        corrupted = fn(arr, severity)
        x = to_tensor(Image.fromarray(corrupted)).unsqueeze(0).to(device)
        logits = model_forward(model, x)
        pred = logits.argmax(1).item()
        correct += int(pred == label)
        total += 1
    return correct / max(1, total)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    originals = _enumerate_folder(config.TEST_DIR, config.ACTIVE_CLASSES)
    _, _, test_samples = stratified_split(originals, val_ratio=0.15, test_ratio=0.15, seed=config.SEED)

    rows = []
    for model_name in config.MODELS:
        ckpt = config.CKPT_DIR / f"{config.NAMESPACE}{model_name}_clean_best.pt"
        if not ckpt.exists():
            print(f"[skip] no checkpoint for {model_name}")
            continue
        model = build_model(model_name, config.NUM_CLASSES).to(device).eval()
        model.load_state_dict(torch.load(ckpt, map_location=device))

        # baseline (no corruption)
        clean_pre = transforms.Compose([
            transforms.Resize(int(config.IMG_SIZE * 1.15)),
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ])
        c, t = 0, 0
        with torch.no_grad():
            for path, label in test_samples:
                x = clean_pre(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
                pred = model_forward(model, x).argmax(1).item()
                c += int(pred == label); t += 1
        clean_acc = c / max(1, t)
        rows.append({"model": model_name, "corruption": "clean", "severity": 0, "accuracy": clean_acc})

        for cname, fn in CORRUPTIONS.items():
            for sev in SEVERITIES:
                np.random.seed(config.SEED)  # determinise noise
                acc = eval_under_corruption(model, test_samples, fn, sev, device)
                rows.append({"model": model_name, "corruption": cname, "severity": sev, "accuracy": acc})
                print(f"{model_name:20s} {cname:18s} sev={sev}  acc={acc:.4f}")
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out_csv = config.TABLE_DIR / f"{config.NAMESPACE}robustness.csv"
    df.to_csv(out_csv, index=False)
    print("[table]", out_csv)

    # pivot: model x corruption (mean over severities)
    means = df.groupby(["model", "corruption"])["accuracy"].mean().unstack()
    means.to_csv(config.TABLE_DIR / f"{config.NAMESPACE}robustness_mean.csv")
    with open(config.TABLE_DIR / f"{config.NAMESPACE}robustness_mean.md", "w") as f:
        f.write(means.round(4).to_markdown())

    # plot: per-corruption curves (severity vs acc)
    fig, axes = plt.subplots(1, len(CORRUPTIONS), figsize=(4 * len(CORRUPTIONS), 4), sharey=True)
    if len(CORRUPTIONS) == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab10")
    for ai, cname in enumerate(CORRUPTIONS):
        ax = axes[ai]
        for i, mname in enumerate(config.MODELS):
            sub = df[(df.model == mname) & (df.corruption == cname)].sort_values("severity")
            if sub.empty:
                continue
            base = df[(df.model == mname) & (df.corruption == "clean")]["accuracy"]
            base = base.iloc[0] if len(base) else None
            xs = [0] + sub.severity.tolist()
            ys = ([base] if base is not None else []) + sub.accuracy.tolist()
            ax.plot(xs, ys, marker="o", color=cmap(i),
                    lw=2 if mname == "catnet" else 1.5,
                    linestyle="--" if mname == "catnet" else "-",
                    label=mname)
        ax.set_title(cname); ax.set_xlabel("Severity"); ax.grid(alpha=0.3)
        if ai == 0:
            ax.set_ylabel("Accuracy")
    axes[-1].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    out = config.FIG_DIR / f"{config.NAMESPACE}robustness_curves.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("[figure]", out)


if __name__ == "__main__":
    main()
