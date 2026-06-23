"""Grad-CAM visualisations for all models on apple leaf disease test samples."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

import config
from dataset import build_dataloaders, _enumerate_folder, stratified_split
from models import build_model

PRETTY = {
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenetv3_large": "MobileNetV3-L",
    "vit_b16": "ViT-B/16",
    "catnet": "CAT-Net (ours)",
}


class CATNetWrapper(torch.nn.Module):
    """Strip the aux output so Grad-CAM sees a single tensor."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, tuple):
            return out[0]
        return out


def reshape_vit(tensor):
    """ViT-B/16 target reshape: tokens -> (B, C, H, W). Drops CLS token."""
    # tensor shape: (B, 197, 768) -> spatial 14x14 + 1 CLS
    n_tokens = tensor.shape[1]
    grid = int((n_tokens - 1) ** 0.5)
    result = tensor[:, 1:, :].reshape(tensor.shape[0], grid, grid, tensor.shape[2])
    return result.permute(0, 3, 1, 2)


def get_target_layer(model_name: str, model: torch.nn.Module):
    if model_name == "resnet50":
        return [model.layer4[-1]], None
    if model_name == "densenet121":
        return [model.features.norm5], None
    if model_name == "efficientnet_b0":
        return [model.blocks[-1]], None
    if model_name == "mobilenetv3_large":
        return [model.blocks[-1]], None
    if model_name == "vit_b16":
        return [model.blocks[-1].norm1], reshape_vit
    if model_name == "catnet":
        # take CBAM output (CNN spatial map) — best for CAM visualisation
        return [model.model.cbam], None
    raise ValueError(model_name)


def load_test_originals(seed: int):
    originals = _enumerate_folder(config.TEST_DIR, config.APPLE_CLASSES)
    _, _, test_samples = stratified_split(originals, val_ratio=0.15, test_ratio=0.15, seed=seed)
    return test_samples


def preprocess_image(path: Path):
    pil = Image.open(path).convert("RGB")
    pre = transforms.Compose([
        transforms.Resize(int(config.IMG_SIZE * 1.15)),
        transforms.CenterCrop(config.IMG_SIZE),
    ])
    pil = pre(pil)
    rgb = np.asarray(pil, dtype=np.float32) / 255.0
    tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])(pil).unsqueeze(0)
    return rgb, tensor


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_samples = load_test_originals(config.SEED)
    np.random.seed(0)
    # one representative example per class
    by_class = {}
    for p, y in test_samples:
        by_class.setdefault(y, []).append(p)
    examples = [(by_class[y][0], y) for y in sorted(by_class) if y in by_class]
    print("XAI examples:")
    for p, y in examples:
        print(f"  class={y} ({config.APPLE_CLASSES[y]}) -> {p.name}")

    n_models = len(config.MODELS)
    n_examples = len(examples)
    fig, axes = plt.subplots(n_examples, n_models + 1, figsize=((n_models + 1) * 2.5, n_examples * 2.6))
    if n_examples == 1:
        axes = axes[None, :]

    # Original column first
    for ri, (path, y) in enumerate(examples):
        rgb, _ = preprocess_image(path)
        axes[ri, 0].imshow(rgb)
        axes[ri, 0].set_title("Original" if ri == 0 else "")
        axes[ri, 0].set_ylabel(config.APPLE_CLASSES[y], fontsize=10)
        axes[ri, 0].set_xticks([]); axes[ri, 0].set_yticks([])

    for mi, model_name in enumerate(config.MODELS):
        ckpt = config.CKPT_DIR / f"{model_name}_clean_best.pt"
        if not ckpt.exists():
            print(f"[skip] no checkpoint for {model_name} ({ckpt})")
            for ri in range(n_examples):
                axes[ri, mi + 1].axis("off")
            continue
        model = build_model(model_name, config.NUM_CLASSES).to(device).eval()
        model.load_state_dict(torch.load(ckpt, map_location=device))
        wrapped = CATNetWrapper(model) if model_name == "catnet" else model
        target_layers, reshape = get_target_layer(model_name, wrapped)

        cam = GradCAM(model=wrapped, target_layers=target_layers, reshape_transform=reshape)

        for ri, (path, y) in enumerate(examples):
            rgb, x = preprocess_image(path)
            x = x.to(device)
            with torch.no_grad():
                logits = wrapped(x)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                pred = int(probs.argmax())
            try:
                cam_map = cam(input_tensor=x, targets=None)[0]
                vis = show_cam_on_image(rgb, cam_map, use_rgb=True, image_weight=0.55)
            except Exception as e:
                print(f"[warn] cam failed {model_name} on {path.name}: {e}")
                vis = (rgb * 255).astype(np.uint8)
            axes[ri, mi + 1].imshow(vis)
            mark = "OK" if pred == y else "X"
            axes[ri, mi + 1].set_title(
                f"{PRETTY[model_name]}\n{mark} pred={config.APPLE_CLASSES[pred].split()[-1]}  p={probs[pred]:.2f}"
                if ri == 0 else f"{mark} pred={config.APPLE_CLASSES[pred].split()[-1]}  p={probs[pred]:.2f}",
                fontsize=8,
            )
            axes[ri, mi + 1].set_xticks([]); axes[ri, mi + 1].set_yticks([])

        del cam, model, wrapped
        torch.cuda.empty_cache()

    fig.suptitle("Grad-CAM visualisations on held-out apple leaf test samples (clean split)")
    fig.tight_layout()
    out = config.FIG_DIR / "gradcam_grid.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {out}")


if __name__ == "__main__":
    main()
