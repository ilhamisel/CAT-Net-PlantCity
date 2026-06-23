"""Few-shot evaluation: re-train each model using only N samples per class.

The val/test sets are kept identical to the main benchmark. Only the training
portion of the stratified split is subsampled to N images per class.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

import config
from dataset import (
    _enumerate_folder, stratified_split, AppleLeafDataset, build_transforms,
)
from engine import set_seed, evaluate, warmup_cosine_schedule, compute_loss
from models import build_model

from torch.utils.data import DataLoader, WeightedRandomSampler


SHOTS = [10, 30, 60]


def subsample_per_class(samples: List[Tuple[Path, int]], n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    by_class = {}
    for p, y in samples:
        by_class.setdefault(y, []).append((p, y))
    out = []
    for y, lst in by_class.items():
        if len(lst) <= n:
            out.extend(lst)
            continue
        idx = rng.choice(len(lst), size=n, replace=False)
        out.extend([lst[i] for i in idx])
    return out


def train_few_shot(model_name: str, n_per_class: int, device: str = "cuda"):
    set_seed(config.SEED)
    originals = _enumerate_folder(config.TEST_DIR, config.APPLE_CLASSES)
    train_samples, val_samples, test_samples = stratified_split(
        originals, val_ratio=0.15, test_ratio=0.15, seed=config.SEED,
    )
    train_sub = subsample_per_class(train_samples, n_per_class, seed=config.SEED)

    train_ds = AppleLeafDataset(train_sub, config.APPLE_CLASSES, build_transforms(True))
    val_ds = AppleLeafDataset(val_samples, config.APPLE_CLASSES, build_transforms(False))
    test_ds = AppleLeafDataset(test_samples, config.APPLE_CLASSES, build_transforms(False))

    counts = train_ds.class_counts()
    class_weights = counts.sum() / (len(counts) * np.maximum(counts, 1))
    sample_weights = np.array([class_weights[y] for _, y in train_ds.samples], dtype=np.float64)
    sampler = WeightedRandomSampler(torch.from_numpy(sample_weights).double(), len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=min(config.BATCH_SIZE, len(train_ds)),
                              sampler=sampler, num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, num_workers=2, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, num_workers=2, pin_memory=True, persistent_workers=True)

    model = build_model(model_name, config.NUM_CLASSES).to(device)
    cw = torch.tensor(class_weights, dtype=torch.float32, device=device)

    epochs = 20
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = warmup_cosine_schedule(optimizer, steps_per_epoch * 2, steps_per_epoch * epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_f1, best_state, patience = -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.float16):
                out = model(x)
                loss = compute_loss(out, y, cw, config.LABEL_SMOOTHING)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
        m = evaluate(model, val_loader, device, config.NUM_CLASSES)
        if m["f1_macro"] > best_f1:
            best_f1 = m["f1_macro"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    final = evaluate(model, test_loader, device, config.NUM_CLASSES)
    return {
        "model": model_name,
        "n_per_class": n_per_class,
        "train_size": len(train_sub),
        "test_accuracy": final["accuracy"],
        "test_f1_macro": final["f1_macro"],
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for n in SHOTS:
        for name in config.MODELS:
            print(f"\n--- few-shot {name} N={n} ---")
            try:
                r = train_few_shot(name, n, device=device)
                rows.append(r)
                print(f"  acc={r['test_accuracy']:.4f}  F1={r['test_f1_macro']:.4f}")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback; traceback.print_exc()
                rows.append({"model": name, "n_per_class": n, "error": str(e)})
            torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    out = config.TABLE_DIR / "few_shot.csv"
    df.to_csv(out, index=False)
    print("[table]", out)

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(config.MODELS):
        sub = df[df.model == name].sort_values("n_per_class")
        if "test_f1_macro" not in sub.columns or sub.empty:
            continue
        ax.plot(sub.n_per_class, sub.test_f1_macro * 100, marker="o",
                color=cmap(i), lw=2 if name == "catnet" else 1.5,
                linestyle="--" if name == "catnet" else "-",
                label=name)
    ax.set_xlabel("Training samples per class"); ax.set_ylabel("Test F1 (macro), %")
    ax.set_title("Few-shot training behaviour"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    out_p = config.FIG_DIR / "few_shot.png"
    fig.savefig(out_p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("[figure]", out_p)


if __name__ == "__main__":
    main()
