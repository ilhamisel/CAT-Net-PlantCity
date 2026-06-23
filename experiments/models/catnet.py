"""CAT-Net: Convolutional-Attention-Transformer Network for apple leaf disease.

Design rationale:
- EfficientNet-B0 backbone provides parameter-efficient local texture features.
- CBAM (channel + spatial attention) refines the CNN feature map to highlight
  disease lesions before global reasoning.
- A shallow Transformer encoder reasons over CNN feature tokens to capture
  long-range relations between lesion regions on the leaf.
- A deep-supervised auxiliary head on the CNN branch stabilises training.
"""
from __future__ import annotations

import math
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x):  # x: B, C, H, W
        b, c, _, _ = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        w = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(spatial_kernel)

    def forward(self, x):
        return self.spatial(self.channel(x))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 2.0,
                 dropout: float = 0.1, attn_dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=attn_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None, need_weights: bool = False):
        h = self.norm1(x)
        attn_out, attn_w = self.attn(h, h, h, need_weights=need_weights, average_attn_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        if need_weights:
            return x, attn_w
        return x, None


class CATNet(nn.Module):
    """CNN (EfficientNet-B0) -> CBAM -> Transformer encoder -> dual classifier."""

    def __init__(self, num_classes: int = 3, embed_dim: int = 256,
                 depth: int = 3, heads: int = 4, aux_weight: float = 0.3,
                 dropout: float = 0.1):
        super().__init__()
        self.aux_weight = aux_weight

        backbone = timm.create_model("efficientnet_b0", pretrained=True, features_only=True)
        self.backbone = backbone
        feat_channels = backbone.feature_info.channels()[-1]  # 320 for EfficientNet-B0

        self.cbam = CBAM(feat_channels, reduction=16, spatial_kernel=7)
        self.proj = nn.Conv2d(feat_channels, embed_dim, kernel_size=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 50, embed_dim))  # 49 spatial + 1 CLS (7x7=49)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, heads=heads, mlp_ratio=2.0, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, num_classes))
        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(feat_channels, num_classes),
        )

    def forward_features(self, x, return_attn: bool = False):
        feats = self.backbone(x)[-1]
        feats = self.cbam(feats)
        cnn_feats = feats
        tokens = self.proj(feats)
        b, c, h, w = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        if tokens.shape[1] != self.pos_embed.shape[1]:
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2), size=tokens.shape[1], mode="linear"
            ).transpose(1, 2)
        else:
            pos = self.pos_embed
        tokens = tokens + pos
        attn_maps = []
        for blk in self.blocks:
            tokens, attn_w = blk(tokens, need_weights=return_attn)
            if return_attn:
                attn_maps.append(attn_w)
        tokens = self.norm(tokens)
        return cnn_feats, tokens, attn_maps

    def forward(self, x, return_attn: bool = False):
        cnn_feats, tokens, attn_maps = self.forward_features(x, return_attn=return_attn)
        cls_out = tokens[:, 0]
        main_logits = self.head(cls_out)
        aux_logits = self.aux_head(cnn_feats)
        if return_attn:
            return main_logits, aux_logits, attn_maps
        return main_logits, aux_logits

    @torch.no_grad()
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_catnet(num_classes: int) -> CATNet:
    return CATNet(num_classes=num_classes)
