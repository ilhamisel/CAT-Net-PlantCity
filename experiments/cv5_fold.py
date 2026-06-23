"""Stratified 5-fold CV on fruit-32 originals for all 6 models.

For each fold:
  test  = fold_i samples (~20% of 6194 originals, stratified)
  train+val = remaining ~80% -> further 85/15 stratified split for early stopping

Outputs per fold:
  results/logs/fruit_cv/fold{i}_<model>_results.json   (metrics + predictions)

Per-sample predictions are stored so cv5_aggregate.py can run paired McNemar.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

import config
config.NUM_WORKERS = 2  # safer for long unattended runs
from dataset import (
    AppleLeafDataset, _enumerate_folder, build_transforms,
)
from engine import set_seed, warmup_cosine_schedule, compute_loss, model_outputs
from models import build_model


CV_LOG_DIR = config.LOG_DIR / "fruit_cv"
CV_LOG_DIR.mkdir(parents=True, exist_ok=True)


def make_loaders(train_s, val_s, test_s, classes, batch_size):
    train_ds = AppleLeafDataset(train_s, classes, build_transforms(True))
    val_ds = AppleLeafDataset(val_s, classes, build_transforms(False))
    test_ds = AppleLeafDataset(test_s, classes, build_transforms(False))
    counts = train_ds.class_counts()
    class_weights = counts.sum() / (len(counts) * np.maximum(counts, 1))
    sample_w = np.array([class_weights[y] for _, y in train_ds.samples], dtype=np.float64)
    sampler = WeightedRandomSampler(torch.from_numpy(sample_w).double(),
                                     num_samples=len(train_ds), replacement=True)
    tr = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                     num_workers=config.NUM_WORKERS, pin_memory=True,
                     persistent_workers=(config.NUM_WORKERS > 0))
    va = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                     num_workers=config.NUM_WORKERS, pin_memory=True,
                     persistent_workers=(config.NUM_WORKERS > 0))
    te = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                     num_workers=config.NUM_WORKERS, pin_memory=True,
                     persistent_workers=(config.NUM_WORKERS > 0))
    return tr, va, te, torch.tensor(class_weights, dtype=torch.float32)


@torch.no_grad()
def collect_preds(model, loader, device, num_classes):
    model.eval()
    logits_all, y_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        logits_all.append(logits.detach().cpu())
        y_all.append(y)
    logits = torch.cat(logits_all)
    y = torch.cat(y_all)
    preds = logits.argmax(1).numpy()
    probs = F.softmax(logits, 1).numpy()
    yn = y.numpy()
    metrics = {
        "accuracy": float(accuracy_score(yn, preds)),
        "f1_macro": float(precision_recall_fscore_support(yn, preds, average="macro", zero_division=0)[2]),
        "precision_macro": float(precision_recall_fscore_support(yn, preds, average="macro", zero_division=0)[0]),
        "recall_macro": float(precision_recall_fscore_support(yn, preds, average="macro", zero_division=0)[1]),
        "confusion_matrix": confusion_matrix(yn, preds, labels=list(range(num_classes))).tolist(),
    }
    try:
        metrics["auc_ovr_macro"] = float(roc_auc_score(yn, probs, multi_class="ovr",
                                                       average="macro",
                                                       labels=list(range(num_classes))))
    except ValueError:
        metrics["auc_ovr_macro"] = float("nan")
    return metrics, preds.tolist(), yn.tolist()


def train_eval(model_name, train_s, val_s, test_s, classes, device, fold_idx):
    set_seed(config.SEED + fold_idx)  # different seed per fold for trainer noise
    bs = 16 if model_name == "catnet" else config.BATCH_SIZE
    tr, va, te, cw = make_loaders(train_s, val_s, test_s, classes, bs)
    cw = cw.to(device)
    model = build_model(model_name, len(classes)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    steps = max(1, len(tr))
    total = steps * config.EPOCHS
    warm = steps * config.WARMUP_EPOCHS
    sch = warmup_cosine_schedule(opt, warm, total)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    best_f1, best_state, patience, epoch_log = -1.0, None, 0, []
    for ep in range(1, config.EPOCHS + 1):
        model.train()
        t0 = time.time()
        for x, y in tr:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.float16):
                out = model(x)
                loss = compute_loss(out, y, cw, config.LABEL_SMOOTHING)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
        vmet, _, _ = collect_preds(model, va, device, len(classes))
        elapsed = time.time() - t0
        epoch_log.append({"epoch": ep, "val_acc": vmet["accuracy"],
                           "val_f1_macro": vmet["f1_macro"], "time_sec": elapsed})
        print(f"[fold{fold_idx} {model_name}] ep{ep:02d}  "
              f"vaA={vmet['accuracy']:.4f}  vF1={vmet['f1_macro']:.4f}  t={elapsed:.1f}s",
              flush=True)
        if vmet["f1_macro"] > best_f1:
            best_f1 = vmet["f1_macro"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.EARLY_STOP_PATIENCE:
                print(f"[fold{fold_idx} {model_name}] early stop at epoch {ep}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    tmet, test_preds, test_y = collect_preds(model, te, device, len(classes))
    return {
        "model": model_name,
        "fold": fold_idx,
        "params": int(n_params),
        "epochs_trained": len(epoch_log),
        "best_val_f1": float(best_f1),
        "test_metrics": tmet,
        "test_predictions": test_preds,
        "test_labels": test_y,
        "test_paths": [str(p) for p, _ in test_s],
        "classes": list(classes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*", type=int, default=None,
                    help="Subset of fold indices (0..4). Defaults to all 5.")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Subset of model names. Defaults to all 6 from config.")
    args = ap.parse_args()

    classes = config.FRUIT_CLASSES
    originals = _enumerate_folder(config.TEST_DIR, classes)
    labels = np.array([y for _, y in originals])
    paths = list(originals)
    print(f"Originals: {len(originals)}  classes: {len(classes)}", flush=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    fold_indices = list(skf.split(np.zeros(len(originals)), labels))

    folds_to_run = args.folds if args.folds is not None else list(range(5))
    models_to_run = args.models if args.models is not None else config.MODELS
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for fi in folds_to_run:
        tr_val_idx, te_idx = fold_indices[fi]
        tr_idx, val_idx = train_test_split(
            tr_val_idx, test_size=0.15, random_state=config.SEED,
            stratify=labels[tr_val_idx],
        )
        train_s = [paths[i] for i in tr_idx]
        val_s = [paths[i] for i in val_idx]
        test_s = [paths[i] for i in te_idx]
        print(f"\n========== FOLD {fi}  train={len(train_s)}  val={len(val_s)}  "
              f"test={len(test_s)} ==========", flush=True)
        for mn in models_to_run:
            print(f"\n---- fold{fi} model={mn} ----", flush=True)
            try:
                res = train_eval(mn, train_s, val_s, test_s, classes, device, fi)
            except Exception as e:
                print(f"ERROR fold{fi} {mn}: {e}", flush=True)
                import traceback; traceback.print_exc()
                continue
            out = CV_LOG_DIR / f"fold{fi}_{mn}_results.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            print(f"[fold{fi} {mn}] saved -> {out}", flush=True)
            print(f"[fold{fi} {mn}] test acc={res['test_metrics']['accuracy']:.4f}  "
                  f"F1={res['test_metrics']['f1_macro']:.4f}  "
                  f"AUC={res['test_metrics']['auc_ovr_macro']:.4f}", flush=True)
            del res
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
