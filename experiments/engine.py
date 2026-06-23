"""Training / evaluation engine for apple leaf disease classification."""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score,
)
from torch.optim.lr_scheduler import LambdaLR

import config
from dataset import build_dataloaders
from models import build_model


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_cosine_schedule(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def model_outputs(model, x):
    out = model(x)
    if isinstance(out, tuple):
        return out[0], out  # logits, raw_out (may include aux)
    return out, out


def compute_loss(out, target, class_weights, label_smoothing: float, aux_weight: float = 0.3):
    if isinstance(out, tuple):
        main_logits, aux_logits = out[0], out[1]
        l_main = F.cross_entropy(main_logits, target, weight=class_weights,
                                  label_smoothing=label_smoothing)
        l_aux = F.cross_entropy(aux_logits, target, weight=class_weights,
                                 label_smoothing=label_smoothing)
        return l_main + aux_weight * l_aux
    return F.cross_entropy(out, target, weight=class_weights, label_smoothing=label_smoothing)


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int):
    model.eval()
    all_logits, all_targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, _ = model_outputs(model, x)
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())
    logits = torch.cat(all_logits, 0)
    targets = torch.cat(all_targets, 0)
    probs = F.softmax(logits, dim=1).numpy()
    preds = logits.argmax(1).numpy()
    targets_np = targets.numpy()

    acc = accuracy_score(targets_np, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        targets_np, preds, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        targets_np, preds, average="weighted", zero_division=0
    )
    per_class = precision_recall_fscore_support(
        targets_np, preds, average=None, labels=list(range(num_classes)), zero_division=0
    )
    cm = confusion_matrix(targets_np, preds, labels=list(range(num_classes)))
    try:
        ovr_auc = roc_auc_score(targets_np, probs, multi_class="ovr", average="macro",
                                 labels=list(range(num_classes)))
    except ValueError:
        ovr_auc = float("nan")

    return {
        "accuracy": float(acc),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f1_w),
        "per_class_precision": per_class[0].tolist(),
        "per_class_recall": per_class[1].tolist(),
        "per_class_f1": per_class[2].tolist(),
        "per_class_support": per_class[3].tolist(),
        "confusion_matrix": cm.tolist(),
        "auc_ovr_macro": float(ovr_auc),
        "classification_report": classification_report(
            targets_np, preds, target_names=config.ACTIVE_CLASSES,
            digits=4, zero_division=0,
        ),
    }


def train_one(model_name: str, device: str = "cuda", split_mode: str = "clean") -> dict:
    set_seed(config.SEED)
    train_loader, val_loader, test_loader, split_info, class_weights = build_dataloaders(
        mode=split_mode, use_sampler=True, seed=config.SEED,
    )
    class_weights = class_weights.to(device)

    model = build_model(model_name, config.NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR,
                                   weight_decay=config.WEIGHT_DECAY)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * config.EPOCHS
    warmup_steps = steps_per_epoch * config.WARMUP_EPOCHS
    scheduler = warmup_cosine_schedule(optimizer, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    best_f1 = -1.0
    best_state = None
    patience = 0
    epoch_log = []
    ckpt_path = config.CKPT_DIR / f"{config.NAMESPACE}{model_name}_{split_mode}_best.pt"

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_correct = 0
        n_total = 0
        t0 = time.time()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.float16):
                out = model(x)
                loss = compute_loss(out, y, class_weights, config.LABEL_SMOOTHING)
                logits = out[0] if isinstance(out, tuple) else out
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            with torch.no_grad():
                preds = logits.argmax(1)
                n_correct += (preds == y).sum().item()
                n_total += y.size(0)
                epoch_loss += loss.item() * y.size(0)

        train_acc = n_correct / max(1, n_total)
        train_loss = epoch_loss / max(1, n_total)

        metrics = evaluate(model, val_loader, device, config.NUM_CLASSES)
        elapsed = time.time() - t0
        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": metrics["accuracy"],
            "val_f1_macro": metrics["f1_macro"],
            "val_auc": metrics["auc_ovr_macro"],
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": elapsed,
        }
        epoch_log.append(log_row)
        print(f"[{model_name}] ep{epoch:02d}  loss={train_loss:.4f}  "
              f"trA={train_acc:.4f}  vaA={metrics['accuracy']:.4f}  "
              f"vF1={metrics['f1_macro']:.4f}  vAUC={metrics['auc_ovr_macro']:.4f}  "
              f"lr={log_row['lr']:.2e}  t={elapsed:.1f}s")

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_path)
            patience = 0
        else:
            patience += 1
            if patience >= config.EARLY_STOP_PATIENCE:
                print(f"[{model_name}] early stop at epoch {epoch}")
                break

    # final evaluation with best weights on held-out test set
    if best_state is not None:
        model.load_state_dict(best_state)
    final = evaluate(model, test_loader, device, config.NUM_CLASSES)

    result = {
        "model": model_name,
        "split_mode": split_mode,
        "split_info": split_info,
        "num_params": int(n_params),
        "best_val_f1_macro": float(best_f1),
        "final_metrics": final,
        "epoch_log": epoch_log,
        "epochs_trained": len(epoch_log),
        "classes": list(config.ACTIVE_CLASSES),
        "checkpoint": str(ckpt_path),
    }
    out_path = config.LOG_DIR / f"{config.NAMESPACE}{model_name}_{split_mode}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[{model_name}] saved -> {out_path}")
    return result
