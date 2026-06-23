"""Perceptual-hash leakage audit for the 32-class fruit subset.

For each fruit class we compute a 16x16 grayscale perceptual hash of every
image in Images/train/<class> and Images/test/<class>, then for every test
original we count how many train images have a hash distance below a tight
threshold.  We report the duplicate ratio (mean train copies per test image)
per class so the same leakage finding from the apple study can be confirmed
fruit-wide and cited in the paper.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

import config


HASH_SIZE = 16
THRESHOLD = 8  # max sum-abs-diff to consider two images near-duplicates


def phash(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L").resize((HASH_SIZE, HASH_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.flatten()


def hashes_for_folder(folder: Path) -> tuple[list[Path], np.ndarray]:
    paths, vecs = [], []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            paths.append(p)
            vecs.append(phash(p))
    if not vecs:
        return paths, np.empty((0, HASH_SIZE * HASH_SIZE), dtype=np.float32)
    return paths, np.stack(vecs)


def audit_class(cls: str) -> dict:
    tr_dir = config.TRAIN_DIR / cls
    te_dir = config.TEST_DIR / cls
    tr_paths, tr_vecs = hashes_for_folder(tr_dir)
    te_paths, te_vecs = hashes_for_folder(te_dir)

    if len(tr_vecs) == 0 or len(te_vecs) == 0:
        return {
            "class": cls,
            "train_count": len(tr_paths),
            "test_count": len(te_paths),
            "mean_copies_per_test": float("nan"),
            "test_with_match": 0,
            "match_ratio": float("nan"),
        }

    # L1 distance between every test hash and every train hash (small enough at 16x16)
    # te_vecs: (T, D), tr_vecs: (N, D). Diff (T, N, D) is too big when T*N large;
    # but per-fruit class sizes are <= 1500 each so memory is fine in chunks.
    chunk = 64
    copies_per_test = np.zeros(len(te_vecs), dtype=np.int64)
    for s in range(0, len(te_vecs), chunk):
        block = te_vecs[s:s + chunk]  # (b, D)
        diff = np.abs(block[:, None, :] - tr_vecs[None, :, :]).sum(axis=-1)  # (b, N)
        copies_per_test[s:s + chunk] = (diff <= THRESHOLD).sum(axis=1)

    return {
        "class": cls,
        "train_count": len(tr_paths),
        "test_count": len(te_paths),
        "mean_copies_per_test": float(copies_per_test.mean()),
        "test_with_match": int((copies_per_test > 0).sum()),
        "match_ratio": float((copies_per_test > 0).mean()),
    }


def main():
    # Force the fruit class list regardless of CROP env var.
    classes = config.FRUIT_CLASSES
    print(f"Auditing {len(classes)} fruit classes for train/test leakage...")
    rows = []
    for cls in classes:
        try:
            r = audit_class(cls)
        except FileNotFoundError as e:
            print(f"  [missing] {cls}: {e}")
            continue
        rows.append(r)
        print(f"  {cls:42s}  test={r['test_count']:4d}  train={r['train_count']:5d}  "
              f"mean_copies={r['mean_copies_per_test']:5.2f}  match_ratio={r['match_ratio']:.3f}")

    out_csv = config.TABLE_DIR / "fruit_leakage_audit.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWritten {out_csv}")

    overall = {
        "n_classes": len(rows),
        "mean_copies_per_test_overall": float(np.mean([r["mean_copies_per_test"] for r in rows])),
        "min_copies": float(min(r["mean_copies_per_test"] for r in rows)),
        "max_copies": float(max(r["mean_copies_per_test"] for r in rows)),
        "mean_match_ratio": float(np.mean([r["match_ratio"] for r in rows])),
    }
    print(f"\nOverall: mean copies/test = {overall['mean_copies_per_test_overall']:.2f}  "
          f"(range {overall['min_copies']:.1f}-{overall['max_copies']:.1f})  "
          f"match_ratio = {overall['mean_match_ratio']:.3f}")


if __name__ == "__main__":
    main()
