"""Comprehensive figure + statistical pipeline for the Apple paper v2.

Runs from the experiments/ directory so it can import the existing config,
dataset, and models packages. Produces ~25 figures, ~12 tables, and a JSON
dump of every statistical result under experiments/paper_v2/.

Usage:
    cd experiments
    /c/Users/ilhami/anaconda3/python.exe paper_v2/scripts/run_full_analysis.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    cohen_kappa_score,
)
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Make sibling modules (config, dataset, models, external_eval) importable
# when this file is launched from any cwd.
# ---------------------------------------------------------------------------
EXP_DIR = Path(__file__).resolve().parents[2]   # experiments/
sys.path.insert(0, str(EXP_DIR))

import config
from dataset import build_dataloaders, build_transforms, AppleLeafDataset
from external_eval import collect_samples
from models import build_model
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
PV_DIR = (config.ROOT / "archive" / "plantvillage dataset" / "color")
MAPPING_PATH = EXP_DIR / "plantvillage_mapping.json"

OUT_DIR = EXP_DIR / "paper_v2"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
STAT_DIR = OUT_DIR / "stats"
CACHE_DIR = OUT_DIR / "cache"
for d in (FIG_DIR, TBL_DIR, STAT_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

MODELS = config.MODELS
MODEL_DISPLAY = {
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenetv3_large": "MobileNetV3-L",
    "vit_b16": "ViT-B/16",
    "catnet": "CAT-Net",
}
MODEL_COLOR = {
    "resnet50": "#4C72B0",
    "densenet121": "#55A868",
    "efficientnet_b0": "#C44E52",
    "mobilenetv3_large": "#8172B2",
    "vit_b16": "#CCB974",
    "catnet": "#E24A33",  # highlight
}
CLASSES = config.APPLE_CLASSES
CLASS_SHORT = ["Brown_spot", "Normal", "black_spot"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RNG = np.random.default_rng(42)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# ---------------------------------------------------------------------------
# 1. Inference cache
# ---------------------------------------------------------------------------

def load_model(name: str):
    model = build_model(name, len(CLASSES))
    ckpt = config.CKPT_DIR / f"{name}_clean_best.pt"
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval().to(DEVICE)
    return model


@torch.no_grad()
def run_inference(model, loader, capture_features: bool = False):
    """Return logits [N,K], probs, preds, labels, optional penultimate [N,D]."""
    logits_all, y_all, feats_all = [], [], []

    hook_handle = None
    feat_buffer = []
    if capture_features:
        # Use the input to the final Linear head as the penultimate feature.
        head = _find_head(model)
        def hook(_m, inp, _out):
            x = inp[0]
            feat_buffer.append(x.detach().float().cpu())
        hook_handle = head.register_forward_hook(hook)

    try:
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            logits_all.append(logits.float().cpu())
            y_all.append(y)
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    logits = torch.cat(logits_all).numpy()
    labels = torch.cat(y_all).numpy()
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    preds = probs.argmax(1)
    feats = torch.cat(feat_buffer).numpy() if feat_buffer else None
    return logits, probs, preds, labels, feats


def _find_head(model):
    """Locate the final Linear layer for feature hooks across architectures."""
    name = type(model).__name__
    if name == "CATNet":
        return model.head[-1]
    # timm models commonly use .head (ViT), .classifier (densenet, efficientnet),
    # or .fc (resnet). Find the last Linear by depth-first scan.
    last_lin = None
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            last_lin = m
    if last_lin is None:
        raise RuntimeError("No Linear head found")
    return last_lin


def get_clean_test_loader():
    _, _, test_loader, info, _ = build_dataloaders(mode="clean", use_sampler=False)
    return test_loader, info


def get_external_loader():
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    samples = collect_samples(PV_DIR, mapping)
    ds = AppleLeafDataset(samples, CLASSES, build_transforms(False))
    return DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True), len(samples)


def build_cache():
    print("\n[1/8] Building inference cache...")
    test_loader, _ = get_clean_test_loader()
    ext_loader, ext_n = get_external_loader()
    print(f"  clean test: {len(test_loader.dataset)} images")
    print(f"  external PV: {ext_n} images")

    for mn in MODELS:
        cache_path = CACHE_DIR / f"preds_{mn}.npz"
        if cache_path.exists():
            print(f"  cached → {mn}")
            continue
        t0 = time.time()
        model = load_model(mn)
        cl = run_inference(model, test_loader, capture_features=True)
        ex = run_inference(model, ext_loader, capture_features=True)
        np.savez(cache_path,
                 clean_logits=cl[0], clean_probs=cl[1], clean_preds=cl[2],
                 clean_labels=cl[3], clean_feats=cl[4],
                 ext_logits=ex[0], ext_probs=ex[1], ext_preds=ex[2],
                 ext_labels=ex[3], ext_feats=ex[4])
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print(f"  {mn}: {time.time()-t0:.1f}s")


def load_cache(mn):
    return np.load(CACHE_DIR / f"preds_{mn}.npz", allow_pickle=False)


# ---------------------------------------------------------------------------
# 2. Statistical helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(y_true, y_pred, metric_fn, n=2000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(42)
    N = len(y_true)
    point = metric_fn(y_true, y_pred)
    scores = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, N, size=N)
        scores[i] = metric_fn(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, lo, hi, scores.std()


def mcnemar_exact(c01, c10):
    """Exact binomial McNemar p-value for discordant pairs (c01, c10)."""
    from scipy.stats import binomtest
    n = c01 + c10
    if n == 0:
        return 1.0
    k = min(c01, c10)
    return binomtest(k, n, p=0.5, alternative="two-sided").pvalue


def holm_bonferroni(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty_like(p)
    prev = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        val = max(val, prev)
        adj[idx] = min(val, 1.0)
        prev = adj[idx]
    return adj


def expected_calibration_error(probs, preds, labels, n_bins=15):
    confidences = probs.max(axis=1)
    accuracies = (preds == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    per_bin = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            per_bin.append((lo, hi, 0, np.nan, np.nan))
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = accuracies[mask].mean()
        weight = mask.sum() / len(confidences)
        ece += weight * abs(bin_conf - bin_acc)
        per_bin.append((lo, hi, int(mask.sum()), float(bin_conf), float(bin_acc)))
    return float(ece), per_bin


# ---------------------------------------------------------------------------
# 3. Headline tables (clean + external) with bootstrap CIs
# ---------------------------------------------------------------------------

def acc_metric(y_true, y_pred): return accuracy_score(y_true, y_pred)
def f1m_metric(y_true, y_pred): return f1_score(y_true, y_pred, average="macro", zero_division=0)


def headline_with_ci():
    print("\n[2/8] Headline tables with 2000-resample bootstrap 95% CIs...")
    rows_clean, rows_ext = [], []
    for mn in MODELS:
        c = load_cache(mn)
        # Clean
        acc_p, acc_lo, acc_hi, _ = bootstrap_ci(c["clean_labels"], c["clean_preds"], acc_metric)
        f1_p, f1_lo, f1_hi, _ = bootstrap_ci(c["clean_labels"], c["clean_preds"], f1m_metric)
        kappa = cohen_kappa_score(c["clean_labels"], c["clean_preds"])
        ece, _ = expected_calibration_error(c["clean_probs"], c["clean_preds"], c["clean_labels"])
        rows_clean.append({
            "model": MODEL_DISPLAY[mn],
            "accuracy": round(acc_p * 100, 2),
            "acc_ci": f"[{acc_lo*100:.2f}, {acc_hi*100:.2f}]",
            "f1_macro": round(f1_p * 100, 2),
            "f1_ci": f"[{f1_lo*100:.2f}, {f1_hi*100:.2f}]",
            "kappa": round(kappa, 4),
            "ece": round(ece, 4),
        })
        # External
        acc_p, acc_lo, acc_hi, _ = bootstrap_ci(c["ext_labels"], c["ext_preds"], acc_metric)
        f1_p, f1_lo, f1_hi, _ = bootstrap_ci(c["ext_labels"], c["ext_preds"], f1m_metric)
        kappa = cohen_kappa_score(c["ext_labels"], c["ext_preds"])
        ece, _ = expected_calibration_error(c["ext_probs"], c["ext_preds"], c["ext_labels"])
        rows_ext.append({
            "model": MODEL_DISPLAY[mn],
            "accuracy": round(acc_p * 100, 2),
            "acc_ci": f"[{acc_lo*100:.2f}, {acc_hi*100:.2f}]",
            "f1_macro": round(f1_p * 100, 2),
            "f1_ci": f"[{f1_lo*100:.2f}, {f1_hi*100:.2f}]",
            "kappa": round(kappa, 4),
            "ece": round(ece, 4),
        })

    _save_table(rows_clean, TBL_DIR / "headline_clean_ci")
    _save_table(rows_ext, TBL_DIR / "headline_external_ci")
    return rows_clean, rows_ext


def _save_table(rows, stem):
    import csv
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    header = "| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |"
    sep = "|" + "|".join("-" * (widths[c] + 2) for c in cols) + "|"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]).ljust(widths[c]) for c in cols) + " |")
    Path(f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 4. Per-class metrics
# ---------------------------------------------------------------------------

def per_class_metrics():
    print("\n[3/8] Per-class metrics (clean + external)...")
    for split in ("clean", "ext"):
        rows = []
        for mn in MODELS:
            c = load_cache(mn)
            y, p = c[f"{split}_labels"], c[f"{split}_preds"]
            pr, rc, f1, sup = precision_recall_fscore_support(
                y, p, labels=list(range(len(CLASSES))), zero_division=0)
            for i, cname in enumerate(CLASS_SHORT):
                rows.append({
                    "model": MODEL_DISPLAY[mn],
                    "class": cname,
                    "precision": round(pr[i] * 100, 2),
                    "recall": round(rc[i] * 100, 2),
                    "f1": round(f1[i] * 100, 2),
                    "support": int(sup[i]),
                })
        _save_table(rows, TBL_DIR / f"per_class_{split}")


# ---------------------------------------------------------------------------
# 5. McNemar pairwise
# ---------------------------------------------------------------------------

def mcnemar_pairwise():
    print("\n[4/8] McNemar pairwise tests with Holm-Bonferroni correction...")
    for split in ("clean", "ext"):
        labels = load_cache(MODELS[0])[f"{split}_labels"]
        correct = {mn: (load_cache(mn)[f"{split}_preds"] == labels).astype(int) for mn in MODELS}
        K = len(MODELS)
        pvals = np.full((K, K), 1.0)
        b_mat = np.zeros((K, K), dtype=int)
        c_mat = np.zeros((K, K), dtype=int)
        pairs = []
        for i in range(K):
            for j in range(i + 1, K):
                ci, cj = correct[MODELS[i]], correct[MODELS[j]]
                c01 = int(((ci == 0) & (cj == 1)).sum())
                c10 = int(((ci == 1) & (cj == 0)).sum())
                p = mcnemar_exact(c01, c10)
                pvals[i, j] = pvals[j, i] = p
                b_mat[i, j] = c01; b_mat[j, i] = c10
                c_mat[i, j] = c10; c_mat[j, i] = c01
                pairs.append((i, j, c01, c10, p))
        raw_p = [p for *_, p in pairs]
        adj_p = holm_bonferroni(raw_p)
        rows = []
        for (i, j, c01, c10, p), padj in zip(pairs, adj_p):
            rows.append({
                "A": MODEL_DISPLAY[MODELS[i]],
                "B": MODEL_DISPLAY[MODELS[j]],
                "b (A wrong, B right)": c01,
                "c (A right, B wrong)": c10,
                "p_raw": f"{p:.4g}",
                "p_holm": f"{padj:.4g}",
                "sig_holm": "*" if padj < 0.05 else "ns",
            })
        _save_table(rows, TBL_DIR / f"mcnemar_{split}")

        # heatmap
        adj_grid = np.ones((K, K))
        k = 0
        for i in range(K):
            for j in range(i + 1, K):
                adj_grid[i, j] = adj_p[k]
                adj_grid[j, i] = adj_p[k]
                k += 1
        np.fill_diagonal(adj_grid, np.nan)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(-np.log10(adj_grid + 1e-12), cmap="viridis", vmin=0, vmax=4)
        for i in range(K):
            for j in range(K):
                if i == j: continue
                v = adj_grid[i, j]
                txt = "ns" if v >= 0.05 else f"{v:.2g}"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if -np.log10(v + 1e-12) > 2 else "black", fontsize=8)
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODELS], rotation=30, ha="right")
        ax.set_yticklabels([MODEL_DISPLAY[m] for m in MODELS])
        ax.set_title(f"McNemar Holm-adjusted p-values ({split})")
        plt.colorbar(im, ax=ax, label="-log10(p_adj)")
        plt.savefig(FIG_DIR / f"mcnemar_heatmap_{split}.png")
        plt.close()


# ---------------------------------------------------------------------------
# 6. Confusion matrices (panel of 6 per split)
# ---------------------------------------------------------------------------

def confusion_panels():
    print("\n[5/8] Confusion matrix panels...")
    for split in ("clean", "ext"):
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6))
        for ax, mn in zip(axes.ravel(), MODELS):
            c = load_cache(mn)
            cm = confusion_matrix(c[f"{split}_labels"], c[f"{split}_preds"],
                                  labels=list(range(len(CLASSES))))
            cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
            for i in range(len(CLASSES)):
                for j in range(len(CLASSES)):
                    ax.text(j, i, f"{cm[i,j]}\n{cm_norm[i,j]:.2f}",
                            ha="center", va="center",
                            color="white" if cm_norm[i, j] > 0.5 else "black",
                            fontsize=8)
            ax.set_xticks(range(len(CLASSES))); ax.set_yticks(range(len(CLASSES)))
            ax.set_xticklabels(CLASS_SHORT, rotation=20, ha="right")
            ax.set_yticklabels(CLASS_SHORT)
            ax.set_title(MODEL_DISPLAY[mn])
            if ax is axes[1, 0]:
                ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        fig.suptitle(f"Confusion matrices — {'leakage-free split' if split == 'clean' else 'PlantVillage external'}",
                     fontsize=13)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"confusion_panel_{split}.png")
        plt.close()


# ---------------------------------------------------------------------------
# 7. ROC + PR curves (per model, one-vs-rest macro)
# ---------------------------------------------------------------------------

def roc_pr_curves():
    print("\n[6/8] ROC + PR curves...")
    for split in ("clean", "ext"):
        fig_roc, ax_roc = plt.subplots(figsize=(6.4, 5.4))
        fig_pr, ax_pr = plt.subplots(figsize=(6.4, 5.4))
        rows_auc = []
        for mn in MODELS:
            c = load_cache(mn)
            y, probs = c[f"{split}_labels"], c[f"{split}_probs"]
            K = len(CLASSES)
            y_oh = np.eye(K)[y]
            # Macro-average ROC
            fprs, tprs = [], []
            aucs = []
            for k in range(K):
                fpr, tpr, _ = roc_curve(y_oh[:, k], probs[:, k])
                fprs.append(fpr); tprs.append(tpr); aucs.append(auc(fpr, tpr))
            all_fpr = np.unique(np.concatenate(fprs))
            mean_tpr = np.zeros_like(all_fpr)
            for k in range(K):
                mean_tpr += np.interp(all_fpr, fprs[k], tprs[k])
            mean_tpr /= K
            macro_auc = auc(all_fpr, mean_tpr)
            ax_roc.plot(all_fpr, mean_tpr, label=f"{MODEL_DISPLAY[mn]} (AUC={macro_auc:.3f})",
                        color=MODEL_COLOR[mn], lw=2)
            # Macro PR
            ap_list = []
            recs, precs = [], []
            for k in range(K):
                p, r, _ = precision_recall_curve(y_oh[:, k], probs[:, k])
                recs.append(r); precs.append(p)
                ap_list.append(average_precision_score(y_oh[:, k], probs[:, k]))
            # interpolate to common recall grid
            grid = np.linspace(0, 1, 200)
            avg_p = np.zeros_like(grid)
            for k in range(K):
                # PR curves come reversed by sklearn (recall decreasing); fix order
                r = recs[k][::-1]; p = precs[k][::-1]
                avg_p += np.interp(grid, r, p)
            avg_p /= K
            macro_ap = float(np.mean(ap_list))
            ax_pr.plot(grid, avg_p, label=f"{MODEL_DISPLAY[mn]} (AP={macro_ap:.3f})",
                       color=MODEL_COLOR[mn], lw=2)
            rows_auc.append({
                "model": MODEL_DISPLAY[mn],
                "macro_auc": round(macro_auc, 4),
                "macro_ap": round(macro_ap, 4),
                **{f"auc_{CLASS_SHORT[k]}": round(aucs[k], 4) for k in range(K)},
                **{f"ap_{CLASS_SHORT[k]}": round(ap_list[k], 4) for k in range(K)},
            })
        ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax_roc.set_xlabel("False positive rate"); ax_roc.set_ylabel("True positive rate")
        ax_roc.set_title(f"Macro-avg ROC ({split})")
        ax_roc.legend(loc="lower right", fontsize=8); ax_roc.grid(alpha=0.25)
        fig_roc.savefig(FIG_DIR / f"roc_macro_{split}.png"); plt.close(fig_roc)

        ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
        ax_pr.set_title(f"Macro-avg PR ({split})")
        ax_pr.legend(loc="lower left", fontsize=8); ax_pr.grid(alpha=0.25)
        fig_pr.savefig(FIG_DIR / f"pr_macro_{split}.png"); plt.close(fig_pr)
        _save_table(rows_auc, TBL_DIR / f"auc_ap_{split}")


# ---------------------------------------------------------------------------
# 8. Calibration / reliability diagrams + confidence histograms
# ---------------------------------------------------------------------------

def calibration_plots():
    print("\n[7/8] Calibration + confidence histograms...")
    for split in ("clean", "ext"):
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6))
        rows_ece = []
        for ax, mn in zip(axes.ravel(), MODELS):
            c = load_cache(mn)
            ece, bins = expected_calibration_error(c[f"{split}_probs"], c[f"{split}_preds"],
                                                   c[f"{split}_labels"])
            xs, ys, ws = [], [], []
            for lo, hi, n, conf, acc in bins:
                if n == 0: continue
                xs.append((lo + hi) / 2); ys.append(acc); ws.append(n)
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
            ax.bar(xs, ys, width=(1 / 15) * 0.9, alpha=0.7, color=MODEL_COLOR[mn], edgecolor="black")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
            ax.set_title(f"{MODEL_DISPLAY[mn]}  ECE={ece:.3f}")
            ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
            ax.grid(alpha=0.25)
            rows_ece.append({"model": MODEL_DISPLAY[mn], "ece": round(ece, 4)})
        fig.suptitle(f"Reliability diagrams ({split})", fontsize=13)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"reliability_{split}.png")
        plt.close()
        _save_table(rows_ece, TBL_DIR / f"ece_{split}")

        # Confidence histograms (correct vs wrong)
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6))
        for ax, mn in zip(axes.ravel(), MODELS):
            c = load_cache(mn)
            conf = c[f"{split}_probs"].max(axis=1)
            correct = c[f"{split}_preds"] == c[f"{split}_labels"]
            ax.hist(conf[correct], bins=20, alpha=0.6, color="#2A9D8F", label="correct", range=(0, 1))
            ax.hist(conf[~correct], bins=20, alpha=0.6, color="#E76F51", label="wrong", range=(0, 1))
            ax.set_title(MODEL_DISPLAY[mn])
            ax.set_xlabel("Max softmax"); ax.set_ylabel("Count")
            ax.legend(fontsize=8); ax.grid(alpha=0.25)
        fig.suptitle(f"Confidence histograms ({split})", fontsize=13)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"confidence_hist_{split}.png")
        plt.close()


# ---------------------------------------------------------------------------
# 9. Bootstrap CI bar chart
# ---------------------------------------------------------------------------

def bootstrap_bars():
    print("\n[8/8] Bootstrap CI bars, radar, delta, latency, t-SNE, architecture...")
    for split in ("clean", "ext"):
        means, lo, hi = [], [], []
        for mn in MODELS:
            c = load_cache(mn)
            p, l, h, _ = bootstrap_ci(c[f"{split}_labels"], c[f"{split}_preds"], f1m_metric)
            means.append(p * 100); lo.append(l * 100); hi.append(h * 100)
        means = np.array(means); lo = np.array(lo); hi = np.array(hi)
        err = np.vstack([means - lo, hi - means])
        order = np.argsort(-means)
        x = np.arange(len(MODELS))
        fig, ax = plt.subplots(figsize=(8, 4.6))
        colors = [MODEL_COLOR[MODELS[i]] for i in order]
        ax.bar(x, means[order], yerr=err[:, order], color=colors,
               edgecolor="black", capsize=6)
        for i, idx in enumerate(order):
            ax.text(i, means[idx] + 1.0, f"{means[idx]:.1f}", ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels([MODEL_DISPLAY[MODELS[i]] for i in order],
                                            rotation=15, ha="right")
        ax.set_ylabel("F1-macro (%)"); ax.set_ylim(0, 105)
        ax.set_title(f"F1-macro with 95% bootstrap CI ({split})")
        ax.grid(axis="y", alpha=0.25)
        plt.savefig(FIG_DIR / f"bootstrap_f1_{split}.png")
        plt.close()


# ---------------------------------------------------------------------------
# 10. Per-class F1 radar
# ---------------------------------------------------------------------------

def f1_radar():
    for split in ("clean", "ext"):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)
        angles = np.linspace(0, 2 * np.pi, len(CLASSES), endpoint=False).tolist()
        angles += angles[:1]
        for mn in MODELS:
            c = load_cache(mn)
            _, _, f1, _ = precision_recall_fscore_support(
                c[f"{split}_labels"], c[f"{split}_preds"],
                labels=list(range(len(CLASSES))), zero_division=0)
            vals = (f1 * 100).tolist() + [f1[0] * 100]
            ax.plot(angles, vals, lw=2, label=MODEL_DISPLAY[mn], color=MODEL_COLOR[mn])
            ax.fill(angles, vals, alpha=0.08, color=MODEL_COLOR[mn])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(CLASS_SHORT)
        ax.set_ylim(0, 105)
        ax.set_title(f"Per-class F1 (%) — {split}")
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
        plt.savefig(FIG_DIR / f"radar_f1_{split}.png")
        plt.close()


# ---------------------------------------------------------------------------
# 11. Clean -> External degradation plot
# ---------------------------------------------------------------------------

def clean_vs_external():
    rows = []
    for mn in MODELS:
        c = load_cache(mn)
        f_clean = f1_score(c["clean_labels"], c["clean_preds"], average="macro", zero_division=0) * 100
        f_ext = f1_score(c["ext_labels"], c["ext_preds"], average="macro", zero_division=0) * 100
        a_clean = accuracy_score(c["clean_labels"], c["clean_preds"]) * 100
        a_ext = accuracy_score(c["ext_labels"], c["ext_preds"]) * 100
        rows.append({"model": MODEL_DISPLAY[mn], "f1_clean": round(f_clean, 2),
                     "f1_ext": round(f_ext, 2), "f1_delta": round(f_clean - f_ext, 2),
                     "acc_clean": round(a_clean, 2), "acc_ext": round(a_ext, 2),
                     "acc_delta": round(a_clean - a_ext, 2)})
    _save_table(rows, TBL_DIR / "clean_vs_external")

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    xs = [0, 1]
    for r, mn in zip(rows, MODELS):
        ax.plot(xs, [r["f1_clean"], r["f1_ext"]], "o-", lw=2, markersize=8,
                color=MODEL_COLOR[mn], label=r["model"])
        ax.text(1.02, r["f1_ext"], r["model"], va="center", fontsize=8, color=MODEL_COLOR[mn])
    ax.set_xticks(xs); ax.set_xticklabels(["Clean (in-domain)", "External (PlantVillage)"])
    ax.set_ylabel("F1-macro (%)")
    ax.set_title("Domain transfer: F1-macro Clean → External")
    ax.set_ylim(0, 105); ax.grid(alpha=0.25)
    plt.savefig(FIG_DIR / "domain_transfer_lines.png")
    plt.close()


# ---------------------------------------------------------------------------
# 12. Latency benchmark
# ---------------------------------------------------------------------------

def latency_bench():
    rows = []
    sizes = (1, 32)
    for mn in MODELS:
        model = load_model(mn)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        disk_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
        result = {"model": MODEL_DISPLAY[mn], "params_M": round(params / 1e6, 2),
                  "disk_MB": round(disk_mb, 1)}
        for bs in sizes:
            x = torch.randn(bs, 3, 224, 224, device=DEVICE)
            # warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = model(x)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            n = 20
            for _ in range(n):
                with torch.no_grad():
                    _ = model(x)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            dt = (time.time() - t0) / n * 1000
            result[f"gpu_bs{bs}_ms"] = round(dt, 2)
        # CPU latency (single-image only — cheaper)
        model_cpu = model.cpu()
        x = torch.randn(1, 3, 224, 224)
        for _ in range(2):
            with torch.no_grad():
                _ = model_cpu(x)
        t0 = time.time()
        n = 8
        for _ in range(n):
            with torch.no_grad():
                _ = model_cpu(x)
        result["cpu_bs1_ms"] = round((time.time() - t0) / n * 1000, 2)
        rows.append(result)
        del model, model_cpu
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print(f"  {mn}: gpu_bs1={result['gpu_bs1_ms']}ms gpu_bs32={result['gpu_bs32_ms']}ms "
              f"cpu_bs1={result['cpu_bs1_ms']}ms")
    _save_table(rows, TBL_DIR / "latency")

    # latency vs F1 (clean)
    f1_clean = {}
    for mn in MODELS:
        c = load_cache(mn)
        f1_clean[mn] = f1_score(c["clean_labels"], c["clean_preds"], average="macro", zero_division=0) * 100
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for r, mn in zip(rows, MODELS):
        ax.scatter(r["cpu_bs1_ms"], f1_clean[mn], s=80 + r["params_M"] * 25,
                   color=MODEL_COLOR[mn], edgecolor="black")
        ax.text(r["cpu_bs1_ms"] * 1.05, f1_clean[mn] - 0.6,
                f"{r['model']}\n({r['params_M']}M)", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("CPU latency, batch=1 (ms, log)")
    ax.set_ylabel("F1-macro on clean test (%)")
    ax.set_title("Efficiency: CPU latency vs. accuracy")
    ax.grid(True, which="both", alpha=0.25)
    plt.savefig(FIG_DIR / "cpu_latency_vs_f1.png")
    plt.close()

    # params vs F1
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for r, mn in zip(rows, MODELS):
        ax.scatter(r["params_M"], f1_clean[mn], s=120, color=MODEL_COLOR[mn],
                   edgecolor="black")
        ax.text(r["params_M"] * 1.05, f1_clean[mn] - 0.4, r["model"], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (M, log)")
    ax.set_ylabel("F1-macro on clean test (%)")
    ax.set_title("Efficiency: parameter count vs. accuracy")
    ax.grid(True, which="both", alpha=0.25)
    plt.savefig(FIG_DIR / "params_vs_f1_v2.png")
    plt.close()


# ---------------------------------------------------------------------------
# 13. t-SNE of penultimate features
# ---------------------------------------------------------------------------

def tsne_features():
    for split in ("clean", "ext"):
        n_show = min(800, sum(load_cache(MODELS[0])[f"{split}_labels"].shape[:1]))
        idx = np.arange(load_cache(MODELS[0])[f"{split}_labels"].shape[0])
        if len(idx) > n_show:
            idx = RNG.choice(idx, size=n_show, replace=False)
            idx.sort()
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6))
        for ax, mn in zip(axes.ravel(), MODELS):
            c = load_cache(mn)
            feats = c[f"{split}_feats"][idx]
            y = c[f"{split}_labels"][idx]
            ts = TSNE(n_components=2, perplexity=min(30, len(feats) // 4),
                      init="pca", random_state=42, learning_rate="auto")
            emb = ts.fit_transform(feats)
            for k, name in enumerate(CLASS_SHORT):
                m = y == k
                ax.scatter(emb[m, 0], emb[m, 1], s=12, alpha=0.7, label=name)
            ax.set_title(MODEL_DISPLAY[mn])
            ax.set_xticks([]); ax.set_yticks([])
            if ax is axes[0, 0]:
                ax.legend(fontsize=8, loc="best")
        fig.suptitle(f"t-SNE of penultimate features ({split})", fontsize=13)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"tsne_{split}.png")
        plt.close()


# ---------------------------------------------------------------------------
# 14. Architecture diagram (matplotlib)
# ---------------------------------------------------------------------------

def architecture_diagram():
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis("off")

    def box(x, y, w, h, label, color):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                           linewidth=1.5, facecolor=color, edgecolor="black")
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=10, fontweight="bold")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))

    box(0.1, 2.0, 1.6, 1.0, "Input\n3×224×224", "#EFEFEF")
    box(2.0, 2.0, 2.2, 1.0, "EfficientNet-B0\nstem\n(320×7×7)", "#A8DADC")
    box(4.5, 2.0, 1.4, 1.0, "CBAM\n(Ch+Sp)", "#F4A261")
    box(6.2, 2.0, 1.4, 1.0, "1×1 conv\n→ 256", "#E9C46A")
    box(7.9, 2.0, 2.0, 1.0, "49 tokens +\nCLS + pos", "#F1FAEE")
    box(10.2, 2.0, 2.2, 1.0, "3× Transformer\nblock (4-head)", "#457B9D")
    # heads
    box(12.5, 3.4, 0.4, 0.7, "main\nhead", "#E63946")
    box(12.5, 0.9, 0.4, 0.7, "aux\nhead", "#E63946")

    arrow(1.7, 2.5, 2.0, 2.5)
    arrow(4.2, 2.5, 4.5, 2.5)
    arrow(5.9, 2.5, 6.2, 2.5)
    arrow(7.6, 2.5, 7.9, 2.5)
    arrow(9.9, 2.5, 10.2, 2.5)
    arrow(12.4, 2.7, 12.5, 3.55)   # cls -> main
    arrow(5.2, 2.0, 5.2, 1.4)
    arrow(5.2, 1.4, 12.5, 1.4)     # cnn -> aux
    arrow(12.4, 1.4, 12.5, 1.25)

    ax.text(6.5, 4.4, "CAT-Net architecture (5.29 M params)",
            ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(8.9, 0.45, "CE on CLS + 0.3 × CE on aux",
            ha="center", va="center", fontsize=9, style="italic")

    plt.savefig(FIG_DIR / "architecture_catnet.png")
    plt.close()


# ---------------------------------------------------------------------------
# 15. Class distribution + sample grid
# ---------------------------------------------------------------------------

def dataset_overview():
    # Class counts
    from dataset import _enumerate_folder
    samples = _enumerate_folder(config.TEST_DIR, CLASSES)
    counts = np.zeros(len(CLASSES), dtype=int)
    for _, y in samples:
        counts[y] += 1
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    colors = ["#E63946", "#2A9D8F", "#264653"]
    ax.bar(CLASS_SHORT, counts, color=colors, edgecolor="black")
    for i, n in enumerate(counts):
        ax.text(i, n + 6, str(n), ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Number of originals (Images/test)")
    ax.set_title("PlantCity apple subset — class distribution (573 originals)")
    ax.grid(axis="y", alpha=0.25)
    plt.savefig(FIG_DIR / "class_distribution.png")
    plt.close()

    # Sample grid: 3 classes × 4 samples
    from PIL import Image
    fig, axes = plt.subplots(3, 4, figsize=(11, 8.5))
    for ci, cls in enumerate(CLASSES):
        files = sorted((config.TEST_DIR / cls).iterdir())[:4]
        for j, p in enumerate(files):
            img = Image.open(p).convert("RGB")
            axes[ci, j].imshow(img)
            axes[ci, j].axis("off")
            if j == 0:
                axes[ci, j].set_ylabel(CLASS_SHORT[ci], fontsize=12, rotation=0,
                                       ha="right", va="center")
                axes[ci, j].text(-0.05, 0.5, CLASS_SHORT[ci], transform=axes[ci, j].transAxes,
                                ha="right", va="center", fontsize=13, fontweight="bold")
    fig.suptitle("PlantCity apple — representative samples", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "sample_grid_plantcity.png")
    plt.close()

    # PlantVillage sample grid (3 classes × 4)
    pv_classes = ["Apple___Apple_scab", "Apple___healthy", "Apple___Black_rot"]
    fig, axes = plt.subplots(3, 4, figsize=(11, 8.5))
    for ci, cls in enumerate(pv_classes):
        files = sorted((PV_DIR / cls).iterdir())[:4]
        for j, p in enumerate(files):
            img = Image.open(p).convert("RGB")
            axes[ci, j].imshow(img)
            axes[ci, j].axis("off")
            if j == 0:
                axes[ci, j].text(-0.05, 0.5, cls.replace("Apple___", ""),
                                transform=axes[ci, j].transAxes,
                                ha="right", va="center", fontsize=12, fontweight="bold")
    fig.suptitle("PlantVillage apple — representative samples", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "sample_grid_plantvillage.png")
    plt.close()


# ---------------------------------------------------------------------------
# 16. Leakage perceptual-hash plot
# ---------------------------------------------------------------------------

def leakage_phash():
    from PIL import Image
    print("  computing pHash leakage...")
    def dhash(img, size=16):
        img = img.convert("L").resize((size + 1, size))
        a = np.asarray(img, dtype=np.int16)
        diff = a[:, 1:] > a[:, :-1]
        return diff.flatten()

    def hash_dist(h1, h2):
        return int(np.sum(h1 != h2))

    train_root = config.TRAIN_DIR
    test_root = config.TEST_DIR
    sample_per_class = 30
    matched, unmatched = [], []
    rng = np.random.default_rng(0)
    for cls in CLASSES:
        tests = list((test_root / cls).iterdir())[:sample_per_class]
        trains = list((train_root / cls).iterdir())
        if not trains:
            continue
        train_hashes = [dhash(Image.open(p)) for p in trains]
        train_paths = trains
        # matched: nearest neighbour in same class
        for tp in tests:
            th = dhash(Image.open(tp))
            dists = [hash_dist(th, x) for x in train_hashes]
            matched.append(min(dists))
        # unmatched: random pair within same class but not nearest
        for _ in range(sample_per_class):
            i, j = rng.integers(0, len(train_hashes), size=2)
            if i == j: continue
            unmatched.append(hash_dist(train_hashes[i], train_hashes[j]))

    fig, ax = plt.subplots(figsize=(8, 4.6))
    bins = np.linspace(0, 256, 60)
    ax.hist(matched, bins=bins, alpha=0.7, label=f"test ↔ nearest train ({len(matched)})",
            color="#E63946", edgecolor="black")
    ax.hist(unmatched, bins=bins, alpha=0.7, label=f"random train ↔ train ({len(unmatched)})",
            color="#264653", edgecolor="black")
    ax.set_xlabel("Hamming distance on 16×16 dHash (max 256)")
    ax.set_ylabel("Count")
    ax.set_title("Leakage signature: test images have near-zero distance to train copies")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.savefig(FIG_DIR / "leakage_phash_histogram.png")
    plt.close()


# ---------------------------------------------------------------------------
# 17. Compile master JSON
# ---------------------------------------------------------------------------

def dump_master_stats():
    out = {}
    for mn in MODELS:
        c = load_cache(mn)
        d = {}
        for split in ("clean", "ext"):
            y = c[f"{split}_labels"]; p = c[f"{split}_preds"]; pr = c[f"{split}_probs"]
            d[split] = {
                "accuracy": float(accuracy_score(y, p)),
                "f1_macro": float(f1_score(y, p, average="macro", zero_division=0)),
                "kappa": float(cohen_kappa_score(y, p)),
                "ece": expected_calibration_error(pr, p, y)[0],
                "report": classification_report(y, p, target_names=CLASS_SHORT,
                                                labels=list(range(len(CLASSES))),
                                                zero_division=0, output_dict=True),
                "n_samples": int(len(y)),
            }
        out[MODEL_DISPLAY[mn]] = d
    (STAT_DIR / "master_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  wrote {STAT_DIR / 'master_metrics.json'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")
    print(f"Models: {MODELS}")
    print(f"Output root: {OUT_DIR}")

    build_cache()
    headline_with_ci()
    per_class_metrics()
    mcnemar_pairwise()
    confusion_panels()
    roc_pr_curves()
    calibration_plots()
    bootstrap_bars()
    f1_radar()
    clean_vs_external()
    latency_bench()
    tsne_features()
    architecture_diagram()
    dataset_overview()
    leakage_phash()
    dump_master_stats()
    print("\nDone.")


if __name__ == "__main__":
    main()
