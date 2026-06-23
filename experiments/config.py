"""Central configuration for PlantCity leaf disease classification experiments.

The pipeline supports two crops:
- "apple" (default, 3 classes): the original benchmark documented in CLAUDE.md.
- "fruit"  (32 classes):       a unified 9-fruit / 32-class benchmark
                               (Persimmon dropped — only one class available).

The active crop is selected by the `CROP` environment variable:

    CROP=apple   (default)
    CROP=fruit   python run_all.py clean

All output filenames are prefixed with NAMESPACE so the two studies coexist
without overwriting checkpoints, logs, tables, or figures.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "Images"
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"

RESULTS_DIR = ROOT / "experiments" / "results"
CKPT_DIR = RESULTS_DIR / "checkpoints"
LOG_DIR = RESULTS_DIR / "logs"
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
for _d in (CKPT_DIR, LOG_DIR, FIG_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

APPLE_CLASSES = ["Apple Brown_spot", "Apple Normal", "Apple black_spot"]

# 32-class fruit benchmark.  Persimmon is excluded (single class only).
# Order: fruits grouped together, then alphabetical within fruit.
FRUIT_CLASSES = [
    # Apple (3)
    "Apple Brown_spot",
    "Apple Normal",
    "Apple black_spot",
    # Apricot (3)
    "Apricot Normal",
    "Apricot blight leaf disease",
    "Apricot shot_hole",
    # Cherry (5)
    "Cherry Leaf Scorch",
    "Cherry Normal leaf",
    "Cherry brown_spot",
    "Cherry purple leaf spot",
    "Cherry_shot hole disease",
    # Fig (4)
    "Fig Blight_leaf disease",
    "Fig Brown spot",
    "Fig normal leaf",
    "Fig_rust leaf",
    # Grape (7)
    "Grape Anthracnose leaf",
    "Grape Brown spot leaf",
    "Grape Downy mildew leaf",
    "Grape Mites_leaf disease",
    "Grape Normal_leaf",
    "Grape Powdery_mildew leaf",
    "Grape shot hole leaf disease",
    # Loquat / Lokat (2)
    "Lokat Normal leaf",
    "lokat Leaf_spot",
    # Pear (3)
    "Pear Black spot _ leaf disease",
    "Pear Normal _leaf",
    "Pear fire blight",
    # Walnut (5)
    "Walnut Anthracnose_leaf disease",
    "Walnut Blotch_leaf disease",
    "Walnut Normal_leaf",
    "Walnut Shot_hole",
    "Walnut leaf gall mite",
]

# Mapping each fruit class to its parent fruit (for per-fruit aggregation tables).
FRUIT_OF = {
    "Apple Brown_spot": "Apple", "Apple Normal": "Apple", "Apple black_spot": "Apple",
    "Apricot Normal": "Apricot", "Apricot blight leaf disease": "Apricot",
    "Apricot shot_hole": "Apricot",
    "Cherry Leaf Scorch": "Cherry", "Cherry Normal leaf": "Cherry",
    "Cherry brown_spot": "Cherry", "Cherry purple leaf spot": "Cherry",
    "Cherry_shot hole disease": "Cherry",
    "Fig Blight_leaf disease": "Fig", "Fig Brown spot": "Fig",
    "Fig normal leaf": "Fig", "Fig_rust leaf": "Fig",
    "Grape Anthracnose leaf": "Grape", "Grape Brown spot leaf": "Grape",
    "Grape Downy mildew leaf": "Grape", "Grape Mites_leaf disease": "Grape",
    "Grape Normal_leaf": "Grape", "Grape Powdery_mildew leaf": "Grape",
    "Grape shot hole leaf disease": "Grape",
    "Lokat Normal leaf": "Loquat", "lokat Leaf_spot": "Loquat",
    "Pear Black spot _ leaf disease": "Pear", "Pear Normal _leaf": "Pear",
    "Pear fire blight": "Pear",
    "Walnut Anthracnose_leaf disease": "Walnut", "Walnut Blotch_leaf disease": "Walnut",
    "Walnut Normal_leaf": "Walnut", "Walnut Shot_hole": "Walnut",
    "Walnut leaf gall mite": "Walnut",
}

CROP = os.environ.get("CROP", "apple").lower()
if CROP == "apple":
    ACTIVE_CLASSES = APPLE_CLASSES
    NAMESPACE = ""           # keep legacy apple filenames untouched
elif CROP == "fruit":
    ACTIVE_CLASSES = FRUIT_CLASSES
    NAMESPACE = "fruit_"
else:
    raise ValueError(f"Unknown CROP={CROP!r}. Use 'apple' or 'fruit'.")

NUM_CLASSES = len(ACTIVE_CLASSES)

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 25
LR = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2
LABEL_SMOOTHING = 0.05
EARLY_STOP_PATIENCE = 6
SEED = 42

MODELS = [
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "mobilenetv3_large",
    "vit_b16",
    "catnet",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def ns(name: str) -> str:
    """Prefix a filename stem with the active namespace (empty for apple)."""
    return f"{NAMESPACE}{name}"
