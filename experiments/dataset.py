"""Apple leaf disease dataset for PlantCity (3-class subset).

Two split modes are supported:
- "standard": the train/test folders as provided by the dataset authors. The
  train folder contains the augmented copies of the test originals, so this
  split contains data leakage and only serves as a comparison point.
- "clean":    a stratified 70/15/15 split of the original (non-augmented)
  images located in the test folder. Augmentation is applied online to the
  training portion only. This is the split we report as the main benchmark.
"""
from pathlib import Path
from typing import List, Tuple, Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

import config


class AppleLeafDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], classes: Sequence[str], transform=None):
        self.samples = list(samples)
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def class_counts(self):
        counts = np.zeros(len(self.classes), dtype=np.int64)
        for _, y in self.samples:
            counts[y] += 1
        return counts


def _enumerate_folder(root: Path, classes: Sequence[str]) -> List[Tuple[Path, int]]:
    out = []
    for ci, cls in enumerate(classes):
        cdir = root / cls
        if not cdir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {cdir}")
        for p in sorted(cdir.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                out.append((p, ci))
    return out


def build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize(int(config.IMG_SIZE * 1.15)),
            transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(config.IMG_SIZE * 1.15)),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def stratified_split(samples, val_ratio=0.15, test_ratio=0.15, seed=42):
    labels = [y for _, y in samples]
    idx = np.arange(len(samples))
    train_idx, temp_idx, _, temp_labels = train_test_split(
        idx, labels, test_size=val_ratio + test_ratio, random_state=seed,
        stratify=labels,
    )
    rel_test = test_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=rel_test, random_state=seed,
        stratify=temp_labels,
    )
    return (
        [samples[i] for i in train_idx],
        [samples[i] for i in val_idx],
        [samples[i] for i in test_idx],
    )


def build_dataloaders(mode: str = "clean", use_sampler: bool = True, seed: int = 42):
    classes = config.ACTIVE_CLASSES
    if mode == "standard":
        train_samples = _enumerate_folder(config.TRAIN_DIR, classes)
        test_samples = _enumerate_folder(config.TEST_DIR, classes)
        val_samples = test_samples  # only used for early stop tracking
        split_info = {
            "mode": "standard",
            "warning": "Standard split contains data leakage: train holds augmented copies of test originals.",
        }
    elif mode == "clean":
        originals = _enumerate_folder(config.TEST_DIR, classes)  # 573 originals
        train_samples, val_samples, test_samples = stratified_split(
            originals, val_ratio=0.15, test_ratio=0.15, seed=seed,
        )
        split_info = {
            "mode": "clean",
            "source": "originals from Images/test (non-augmented)",
            "ratios": "70/15/15 stratified",
            "seed": seed,
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")

    train_ds = AppleLeafDataset(train_samples, classes, build_transforms(True))
    val_ds = AppleLeafDataset(val_samples, classes, build_transforms(False))
    test_ds = AppleLeafDataset(test_samples, classes, build_transforms(False))

    counts = train_ds.class_counts()
    class_weights = counts.sum() / (len(counts) * np.maximum(counts, 1))
    sample_weights = np.array([class_weights[y] for _, y in train_ds.samples], dtype=np.float64)

    if use_sampler:
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=config.BATCH_SIZE, sampler=sampler,
            num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
            num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )

    info = {
        **split_info,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "train_per_class": train_ds.class_counts().tolist(),
        "val_per_class": val_ds.class_counts().tolist(),
        "test_per_class": test_ds.class_counts().tolist(),
        "classes": classes,
        "class_weights": class_weights.tolist(),
    }
    return train_loader, val_loader, test_loader, info, torch.tensor(class_weights, dtype=torch.float32)
