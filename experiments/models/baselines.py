"""Baseline CNN and ViT models constructed via timm for ImageNet-pretrained backbones."""
import timm
import torch.nn as nn

_NAME_MAP = {
    "resnet50": "resnet50",
    "densenet121": "densenet121",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3_large": "mobilenetv3_large_100",
    "vit_b16": "vit_base_patch16_224",
}


def build_baseline(name: str, num_classes: int) -> nn.Module:
    if name not in _NAME_MAP:
        raise ValueError(f"Unknown baseline model: {name}")
    return timm.create_model(_NAME_MAP[name], pretrained=True, num_classes=num_classes)
