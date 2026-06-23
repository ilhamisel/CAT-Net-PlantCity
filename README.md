# CAT-Net-PlantCity

> **A Leakage-Free Benchmark and a Compact Convolutional-Attention-Transformer (CAT-Net) for Apple Leaf Disease Classification on PlantCity.**

[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-draft-brightgreen.svg)](experiments/paper_v2/paper.md)

This repository contains the full code, trained-checkpoint loader, statistical pipeline, and reproducible figures for our benchmark of six architectures on the apple subset of the [PlantCity dataset](https://doi.org/10.1016/j.dib.2025.112130) (Khan et al., *Data in Brief*, 2025). We:

1. **Document a previously unreported data-leakage pattern** in PlantCity's recommended train/test split (every test image has 3–4 augmented near-duplicates inside the train folder; perceptual-hash distance ≤ 12 vs. 75–160 for random within-class pairs).
2. **Re-establish a leakage-free 70/15/15 stratified split** of 573 originals.
3. **Propose CAT-Net** — a 5.29 M-parameter CNN-Attention-Transformer hybrid (EfficientNet-B0 stem + CBAM + 3 Transformer blocks + deep-supervised dual head).
4. **Benchmark six architectures** across four axes: clean accuracy, corruption robustness, few-shot data efficiency and **external-domain transfer to PlantVillage** (2 896 OOD images, no fine-tuning).
5. **Release a one-script pipeline** that reproduces 26 figures, 24 tables, and full statistical machinery (2 000-resample bootstrap CIs, pairwise McNemar with Holm–Bonferroni, Cohen's κ, ROC/PR, ECE).

---

## Headline results

### Leakage-free split (86-image test set)

| Model | Params (M) | Acc | F1-macro | κ | ECE |
|---|---:|---:|---:|---:|---:|
| ResNet-50 | 23.51 | 100.00 | 100.00 | 1.000 | 0.039 |
| DenseNet-121 | 6.96 | 100.00 | 100.00 | 1.000 | 0.054 |
| EfficientNet-B0 | 4.01 | 97.67 | 96.95 | 0.963 | 0.040 |
| MobileNetV3-L | 4.21 | 100.00 | 100.00 | 1.000 | 0.062 |
| ViT-B/16 | 85.80 | 98.84 | 99.04 | 0.982 | 0.045 |
| **CAT-Net (ours)** | **5.29** | **100.00** | **100.00** | **1.000** | 0.054 |

### External-domain transfer — PlantVillage (2 896 images, no fine-tuning)

| Model | Acc | F1-macro | κ | ECE |
|---|---:|---:|---:|---:|
| ResNet-50 | 32.39 | 26.96 | 0.007 | 0.329 |
| DenseNet-121 | 37.78 | 29.20 | 0.078 | 0.337 |
| **EfficientNet-B0** | **58.39** | **45.16** | 0.305 | 0.248 |
| MobileNetV3-L | 57.63 | 31.30 | 0.126 | 0.255 |
| ViT-B/16 | 53.07 | 27.77 | 0.081 | 0.320 |
| **CAT-Net (ours)** | 57.08 | **41.32** | **0.227** | 0.302 |

**Key finding.** Every model that saturated the in-domain test set collapses on PlantVillage *except* EfficientNet-B0 and CAT-Net, the only architectures that retain ≥ 41 % macro-F1 across all three classes. CAT-Net is statistically better than ResNet-50, DenseNet-121 and ViT-B/16 (Holm-adjusted McNemar p < 10⁻⁶) and is the only compact model that (i) saturates the clean split, (ii) holds 100 % F1 at N = 10 training images per class, and (iii) transfers to PlantVillage with κ > 0.2 — at **1/16 the parameter count of ViT-B/16**.

---

## Repository layout

```
.
├── README.md                       ← you are here
├── LICENSE                         ← MIT
├── CITATION.cff                    ← academic citation
├── requirements.txt                ← pip dependencies
├── environment.yml                 ← optional conda spec
├── .gitignore
├── CLAUDE.md                       ← project development log
├── 1-s2.0-S2352340925008510-main.pdf   ← PlantCity dataset paper
└── experiments/
    ├── config.py                   ← paths + hyperparameters
    ├── dataset.py                  ← AppleLeafDataset, split builders
    ├── engine.py                   ← train / evaluate
    ├── models/
    │   ├── baselines.py            ← 5 timm wrappers
    │   └── catnet.py               ← CAT-Net implementation
    ├── run_all.py                  ← train orchestrator: "clean" | "standard" | "both"
    ├── robustness.py               ← 5 corruptions × 3 severities
    ├── few_shot.py                 ← N ∈ {10, 30, 60} retrains
    ├── xai.py                      ← Grad-CAM grid
    ├── external_eval.py            ← PlantVillage OOD evaluation
    ├── plantvillage_mapping.json   ← class mapping
    ├── paper_draft.md              ← v1 paper draft (apple)
    ├── fruit_paper_draft.md        ← companion paper (9 fruits / 32 classes)
    ├── paper_v2/                   ← v2 paper + comprehensive analysis
    │   ├── paper.md                ←   8-section draft (this is the headline)
    │   ├── figures/   (26 PNG)
    │   ├── tables/    (24 .md/.csv)
    │   ├── stats/master_metrics.json
    │   ├── cache/     ← per-model predictions for downstream analysis
    │   └── scripts/run_full_analysis.py
    └── results/
        ├── checkpoints/            ← *.pt files (downloaded separately, see "Checkpoints")
        ├── logs/                   ← per-model JSON training logs
        ├── tables/                 ← legacy v1 tables
        └── figures/                ← legacy v1 figures
```

---

## Setup

```bash
git clone https://github.com/<your-user>/CAT-Net-PlantCity.git
cd CAT-Net-PlantCity
pip install -r requirements.txt
```

Requires a CUDA-capable GPU for training (we used an RTX 3090, 24 GB). Inference and the paper pipeline run on CPU but are ~10× slower.

### Datasets

Neither dataset is bundled with this repo.

- **PlantCity (apple subset).** Download from the [Data in Brief article](https://doi.org/10.1016/j.dib.2025.112130) and place at `Images/{train,test}/<class>/`. Only `Images/test/` is used to build the leakage-free split.
- **PlantVillage (for the external test).** Use the `color/` mirror; place at `archive/plantvillage dataset/color/Apple___*/`. Class mapping to PlantCity is in `experiments/plantvillage_mapping.json`.

### Checkpoints

Checkpoints (six `.pt` files, ≈ 520 MB total) are too large for the git tree. We release them via [GitHub Releases](https://github.com/<your-user>/CAT-Net-PlantCity/releases) — download and place under `experiments/results/checkpoints/`.

---

## Reproduce

```bash
cd experiments

# 1. Train every model on the leakage-free split (skip if using released checkpoints)
/path/to/python run_all.py clean

# 2. Optional: train on the leaky split as an anti-benchmark
/path/to/python run_all.py standard

# 3. Robustness, few-shot, Grad-CAM
/path/to/python robustness.py
/path/to/python few_shot.py
/path/to/python xai.py

# 4. External-domain test on PlantVillage
/path/to/python external_eval.py \
    --data_dir "/abs/path/to/archive/plantvillage dataset/color" \
    --mapping plantvillage_mapping.json \
    --dataset_name plantvillage

# 5. The comprehensive paper-v2 pipeline (all stats + 26 figures + 24 tables)
/path/to/python paper_v2/scripts/run_full_analysis.py
```

The paper-v2 script is idempotent: a `paper_v2/cache/preds_*.npz` file is written per model after the first run, so subsequent runs only regenerate figures and tables.

---

## CAT-Net at a glance

```
Input 3×224×224
  → EfficientNet-B0 stem (320×7×7)
  → CBAM (channel + spatial attention)
  → 1×1 conv → 256-d
  → 49 spatial tokens + CLS + learnable pos
  → 3× Transformer encoder block (4-head, MLP ratio 2.0)
  → main head on CLS token
  + aux head on global-pooled CNN feats (deep supervision, weight 0.3)
```

| Property | Value |
|---|---|
| Parameters | 5.29 M |
| Disk | 21.1 MB |
| CPU latency (batch=1) | 14.4 ms |
| GPU latency (batch=32) | 11.2 ms |
| Smartphone-deployable | ✓ (≤ 20 ms / frame ceiling) |

Architecture and implementation: [`experiments/models/catnet.py`](experiments/models/catnet.py).

---

## Figures (selected)

All figures live under `experiments/paper_v2/figures/`. A few highlights:

- `architecture_catnet.png` — CAT-Net block diagram
- `leakage_phash_histogram.png` — bimodal split that proves the leakage
- `confusion_panel_ext.png` — six confusion matrices on PlantVillage
- `domain_transfer_lines.png` — clean → external F1 slope per model
- `mcnemar_heatmap_ext.png` — Holm-adjusted significance matrix
- `bootstrap_f1_ext.png` — 95 % CI for macro-F1 (n = 2 896)
- `tsne_ext.png` — t-SNE of penultimate features under domain shift

---

## Companion 32-class fruit benchmark

`experiments/fruit_paper_draft.md` documents a separate paper on the 9-fruit / 32-class subset of the same dataset (5-fold CV, paired McNemar, CPU latency, per-fruit failure analysis). Reproduce with `CROP=fruit /path/to/python {cv5_fold,cv5_aggregate,per_fruit_confusion,latency_bench}.py`.

---

## Citation

If you use this code or the leakage-free split, please cite both PlantCity and our paper. A `CITATION.cff` is included for tooling.

```bibtex
@misc{catnet-plantcity-2026,
  title  = {A Leakage-Free Benchmark and a Compact Convolutional-Attention-Transformer
            (CAT-Net) for Apple Leaf Disease Classification on PlantCity},
  author = {<author list to be filled>},
  year   = {2026},
  howpublished = {\url{https://github.com/<your-user>/CAT-Net-PlantCity}},
}

@article{khan2025plantcity,
  title   = {PlantCity: A comprehensive image based on multi crop leaves in Pakistan},
  author  = {Khan and Nisa and Ahmad and Zubair and Alshammari},
  journal = {Data in Brief},
  volume  = {63},
  pages   = {112130},
  year    = {2025},
  doi     = {10.1016/j.dib.2025.112130}
}
```

---

## License

Released under the MIT License. The bundled PlantCity dataset paper PDF (`1-s2.0-S2352340925008510-main.pdf`) remains © Elsevier — please refer to the original publication's terms.

---

## Acknowledgements

We thank the authors of the PlantCity dataset for releasing it under a permissive licence, and the PlantVillage team for the external-domain test corpus.
