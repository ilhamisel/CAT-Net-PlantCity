"""Modern-style CAT-Net architecture diagram for the paper.

Produces `paper_v2/figures/architecture_catnet_v2.png` (and .pdf) with:
- Top row: end-to-end pipeline with tensor shapes annotated on arrows.
- Lower-left panel: expanded CBAM module (channel + spatial attention).
- Lower-right panel: expanded Transformer encoder block (MHA + MLP + residuals).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Color palette (modern, muted, print-friendly)
# ---------------------------------------------------------------------------
C_INPUT     = "#ECEFF4"
C_BACKBONE  = "#A8DADC"
C_CBAM      = "#F4A261"
C_PROJ      = "#E9C46A"
C_TOKEN     = "#F1FAEE"
C_TRANS     = "#457B9D"
C_NORM      = "#B5BFC9"
C_HEAD_MAIN = "#E63946"
C_HEAD_AUX  = "#C9938F"
C_CLASS     = "#264653"

C_PANEL_BG  = "#F7F8FA"
C_PANEL_EDGE = "#9AA3AE"

C_OP_POOL   = "#CDE7F0"
C_OP_MLP    = "#D8E2DC"
C_OP_CONV   = "#FFE5D9"
C_OP_ATTN   = "#A8DADC"
C_OP_NORM   = "#E2E2E2"
C_OP_GELU   = "#FFD6A5"
C_OP_SIG    = "#FBD8DD"

FONT_TITLE  = dict(fontsize=15, fontweight="bold")
FONT_SUB    = dict(fontsize=11, style="italic", color="#333333")
FONT_PANEL  = dict(fontsize=12, fontweight="bold")
FONT_BOX    = dict(fontsize=10, fontweight="bold")
FONT_BOX_S  = dict(fontsize=9, fontweight="bold")
FONT_SHAPE  = dict(fontsize=8.5, style="italic", color="#333333")
FONT_CAP    = dict(fontsize=9, color="#333333")

ARROW_STYLE = dict(arrowstyle="-|>", lw=1.5, color="#222222",
                   mutation_scale=15, shrinkA=2, shrinkB=2)
ARROW_DASHED = dict(arrowstyle="-|>", lw=1.3, color="#555555",
                    mutation_scale=13, linestyle=(0, (4, 2)),
                    shrinkA=2, shrinkB=2)


def box(ax, x, y, w, h, label, color, font=FONT_BOX, edge="black", lw=1.4,
        text_color=None):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=lw, facecolor=color, edgecolor=edge,
    )
    ax.add_patch(patch)
    f = dict(font)
    if text_color is not None:
        f["color"] = text_color
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", **f)
    return patch


def arrow(ax, x1, y1, x2, y2, style=None):
    s = style or ARROW_STYLE
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), **s))


def shape_label(ax, x1, y1, x2, y2, label, side="above"):
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    if side == "above":
        ax.text(mx, my + 0.20, label, ha="center", va="bottom", **FONT_SHAPE)
    else:
        ax.text(mx, my - 0.20, label, ha="center", va="top", **FONT_SHAPE)


def panel(ax, x, y, w, h, title):
    rect = Rectangle((x, y), w, h, linewidth=1.3, linestyle="--",
                     facecolor=C_PANEL_BG, edgecolor=C_PANEL_EDGE)
    ax.add_patch(rect)
    ax.text(x + 0.25, y + h - 0.35, title, ha="left", va="center",
            **FONT_PANEL)


def circ(ax, x, y, r, label, color="#FFFFFF", edge="black"):
    c = mpatches.Circle((x, y), r, facecolor=color, edgecolor=edge,
                        linewidth=1.4)
    ax.add_patch(c)
    ax.text(x, y, label, ha="center", va="center", fontsize=11,
            fontweight="bold")


def build_figure():
    fig, ax = plt.subplots(figsize=(16.5, 10.0))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis("off")

    # -----------------------------------------------------------------------
    # Title
    # -----------------------------------------------------------------------
    ax.text(
        9.0, 10.55,
        "CAT-Net : Convolutional-Attention-Transformer Network  (5.30 M params)",
        ha="center", va="center", **FONT_TITLE,
    )
    ax.text(
        9.0, 10.15,
        "EfficientNet-B0 backbone   +   CBAM attention   +   3-layer Transformer encoder   +   deep-supervised dual head",
        ha="center", va="center", **FONT_SUB,
    )

    # -----------------------------------------------------------------------
    # Top row: end-to-end pipeline
    # -----------------------------------------------------------------------
    y_top = 7.65
    h_top = 1.10

    # Input
    box(ax, 0.35, y_top, 1.70, h_top, "Leaf image\n(RGB)", C_INPUT)

    # EfficientNet-B0
    box(ax, 2.55, y_top, 2.55, h_top,
        "EfficientNet-B0\nbackbone\n(MBConv stages 1-7)", C_BACKBONE,
        font=dict(fontsize=9.5, fontweight="bold"))

    # CBAM
    box(ax, 5.60, y_top, 1.60, h_top, "CBAM\n(channel +\nspatial)", C_CBAM)

    # 1x1 projection
    box(ax, 7.65, y_top, 1.55, h_top, "1×1 Conv\nproj. → 256", C_PROJ)

    # Tokenize
    box(ax, 9.65, y_top, 2.10, h_top, "Tokenize +\n[CLS] +\nPos. embed",
        C_TOKEN)

    # Transformer
    box(ax, 12.20, y_top, 2.60, h_top,
        "Transformer encoder\n× 3   (4 heads)", C_TRANS,
        font=dict(fontsize=10, fontweight="bold"), text_color="white")

    # LayerNorm
    box(ax, 15.25, y_top, 1.20, h_top, "LayerNorm", C_NORM)

    # Arrows top row with shape labels above
    arrow(ax, 2.05, y_top + h_top / 2, 2.55, y_top + h_top / 2)
    shape_label(ax, 2.05, y_top + h_top / 2, 2.55, y_top + h_top / 2,
                "3 × 224 × 224")

    arrow(ax, 5.10, y_top + h_top / 2, 5.60, y_top + h_top / 2)
    shape_label(ax, 5.10, y_top + h_top / 2, 5.60, y_top + h_top / 2,
                "320 × 7 × 7")

    arrow(ax, 7.20, y_top + h_top / 2, 7.65, y_top + h_top / 2)
    shape_label(ax, 7.20, y_top + h_top / 2, 7.65, y_top + h_top / 2,
                "320 × 7 × 7")

    arrow(ax, 9.20, y_top + h_top / 2, 9.65, y_top + h_top / 2)
    shape_label(ax, 9.20, y_top + h_top / 2, 9.65, y_top + h_top / 2,
                "256 × 7 × 7")

    arrow(ax, 11.75, y_top + h_top / 2, 12.20, y_top + h_top / 2)
    shape_label(ax, 11.75, y_top + h_top / 2, 12.20, y_top + h_top / 2,
                "50 × 256")

    arrow(ax, 14.80, y_top + h_top / 2, 15.25, y_top + h_top / 2)
    shape_label(ax, 14.80, y_top + h_top / 2, 15.25, y_top + h_top / 2,
                "50 × 256")

    # -----------------------------------------------------------------------
    # Heads
    # -----------------------------------------------------------------------
    head_x = 17.00
    # Main head (from CLS via LayerNorm)
    box(ax, head_x, y_top + 0.70, 0.85, 0.55, "Main\nhead", C_HEAD_MAIN,
        font=dict(fontsize=9, fontweight="bold"), text_color="white")
    # Aux head (from GAP of CBAM CNN features)
    box(ax, head_x, y_top - 0.05, 0.85, 0.55, "Aux\nhead", C_HEAD_AUX,
        font=dict(fontsize=9, fontweight="bold"), text_color="white")
    # 32 classes (9 fruits)
    box(ax, head_x, y_top + 1.50, 0.85, 0.55, "32 classes\n(9 fruits)", C_CLASS,
        font=dict(fontsize=8.5, fontweight="bold"), text_color="white")

    # Main path arrow: LN -> Main head
    arrow(ax, 16.45, y_top + h_top / 2 + 0.10,
          head_x, y_top + 0.97)
    ax.text((16.45 + head_x) / 2, y_top + 0.99 + 0.18,
            "CLS  (1 × 256)", ha="center", va="bottom", **FONT_SHAPE)

    # Main head -> 3 classes
    arrow(ax, head_x + 0.42, y_top + 1.25, head_x + 0.42, y_top + 1.50)

    # Aux path: CBAM output -> GAP -> Aux head (dashed)
    arrow(ax, 6.40, y_top, 6.40, y_top - 0.85, style=ARROW_DASHED)
    arrow(ax, 6.40, y_top - 0.85, 17.42, y_top - 0.85, style=ARROW_DASHED)
    arrow(ax, 17.42, y_top - 0.85, 17.42, y_top - 0.05, style=ARROW_DASHED)
    ax.text(11.90, y_top - 0.75, "GAP(320 → 320)", ha="center", va="bottom",
            **FONT_SHAPE)

    # Loss caption
    ax.text(17.42, y_top - 1.30,
            "Loss  =  CE(main)  +  λ · CE(aux)\nλ = 0.3",
            ha="center", va="top", fontsize=9, style="italic", color="#222")

    # -----------------------------------------------------------------------
    # Connector lines from top-row boxes down to expanded panels
    # -----------------------------------------------------------------------
    # CBAM -> CBAM panel
    ax.add_patch(FancyArrowPatch(
        (6.40, y_top - 0.05), (4.50, 6.05),
        connectionstyle="arc3,rad=-0.25",
        arrowstyle="-", lw=1.0, color="#9AA3AE", linestyle=(0, (2, 2)),
    ))
    # Transformer -> Transformer panel
    ax.add_patch(FancyArrowPatch(
        (13.50, y_top - 0.05), (13.70, 6.05),
        connectionstyle="arc3,rad=0.10",
        arrowstyle="-", lw=1.0, color="#9AA3AE", linestyle=(0, (2, 2)),
    ))

    # -----------------------------------------------------------------------
    # Lower-left panel: CBAM expanded
    # -----------------------------------------------------------------------
    panel(ax, 0.35, 0.50, 8.45, 5.55,
          "(a)  CBAM : Convolutional Block Attention Module")

    # Channel attention sub-pipeline (upper half)
    cy = 4.20
    box(ax, 0.75, cy, 0.85, 0.75, "F", C_INPUT, font=FONT_BOX_S)
    box(ax, 1.95, cy + 0.50, 1.30, 0.55, "AvgPool", C_OP_POOL,
        font=FONT_BOX_S)
    box(ax, 1.95, cy - 0.10, 1.30, 0.55, "MaxPool", C_OP_POOL,
        font=FONT_BOX_S)
    box(ax, 3.65, cy, 1.30, 0.75, "Shared\nMLP", C_OP_MLP, font=FONT_BOX_S)
    box(ax, 5.30, cy, 0.85, 0.75, "σ", C_OP_SIG,
        font=dict(fontsize=14, fontweight="bold"))
    circ(ax, 6.65, cy + 0.40, 0.22, "×")
    box(ax, 7.10, cy, 1.00, 0.75, "F'", C_INPUT, font=FONT_BOX_S)

    # Channel arrows
    arrow(ax, 1.60, cy + 0.40, 1.95, cy + 0.77)
    arrow(ax, 1.60, cy + 0.40, 1.95, cy + 0.17)
    arrow(ax, 3.25, cy + 0.77, 3.65, cy + 0.40)
    arrow(ax, 3.25, cy + 0.17, 3.65, cy + 0.40)
    arrow(ax, 4.95, cy + 0.40, 5.30, cy + 0.40)
    arrow(ax, 6.15, cy + 0.40, 6.43, cy + 0.40)
    # Skip F -> ×
    ax.add_patch(FancyArrowPatch(
        (1.175, cy + 0.75), (6.65, cy + 0.62),
        connectionstyle="arc3,rad=-0.30",
        arrowstyle="-|>", lw=1.2, color="#555555", mutation_scale=12,
        linestyle=(0, (4, 2)),
    ))
    arrow(ax, 6.87, cy + 0.40, 7.10, cy + 0.40)

    ax.text(4.40, cy + 1.10,
            r"Channel attention   $M_c(F) \in \mathbb{R}^{C \times 1 \times 1}$",
            ha="center", va="center", fontsize=10, style="italic",
            color="#222")

    # Spatial attention sub-pipeline (lower half)
    sy = 1.75
    box(ax, 0.75, sy, 0.85, 0.75, "F'", C_INPUT, font=FONT_BOX_S)
    box(ax, 1.95, sy + 0.50, 1.55, 0.55, "Avg (chan.)", C_OP_POOL,
        font=FONT_BOX_S)
    box(ax, 1.95, sy - 0.10, 1.55, 0.55, "Max (chan.)", C_OP_POOL,
        font=FONT_BOX_S)
    box(ax, 3.85, sy, 1.40, 0.75, "Conv 7×7", C_OP_CONV, font=FONT_BOX_S)
    box(ax, 5.60, sy, 0.85, 0.75, "σ", C_OP_SIG,
        font=dict(fontsize=14, fontweight="bold"))
    circ(ax, 6.95, sy + 0.40, 0.22, "×")
    box(ax, 7.40, sy, 0.85, 0.75, "F''", C_INPUT, font=FONT_BOX_S)

    arrow(ax, 1.60, sy + 0.40, 1.95, sy + 0.77)
    arrow(ax, 1.60, sy + 0.40, 1.95, sy + 0.17)
    arrow(ax, 3.50, sy + 0.77, 3.85, sy + 0.40)
    arrow(ax, 3.50, sy + 0.17, 3.85, sy + 0.40)
    arrow(ax, 5.25, sy + 0.40, 5.60, sy + 0.40)
    arrow(ax, 6.45, sy + 0.40, 6.73, sy + 0.40)
    ax.add_patch(FancyArrowPatch(
        (1.175, sy + 0.75), (6.95, sy + 0.62),
        connectionstyle="arc3,rad=-0.30",
        arrowstyle="-|>", lw=1.2, color="#555555", mutation_scale=12,
        linestyle=(0, (4, 2)),
    ))
    arrow(ax, 7.17, sy + 0.40, 7.40, sy + 0.40)

    ax.text(4.40, sy - 0.55,
            r"Spatial attention   $M_s(F') \in \mathbb{R}^{1 \times H \times W}$",
            ha="center", va="center", fontsize=10, style="italic",
            color="#222")

    # Vertical chain F' -> F' (upper-out routed around to lower-in)
    ax.add_patch(FancyArrowPatch(
        (7.60, cy), (7.60, sy + 1.30),
        arrowstyle="-", lw=1.2, color="#222222",
    ))
    ax.add_patch(FancyArrowPatch(
        (7.60, sy + 1.30), (1.175, sy + 1.30),
        arrowstyle="-", lw=1.2, color="#222222",
    ))
    ax.add_patch(FancyArrowPatch(
        (1.175, sy + 1.30), (1.175, sy + 0.78),
        arrowstyle="-|>", lw=1.2, color="#222222", mutation_scale=12,
    ))
    ax.text(4.40, sy + 1.43, "carry  F'", ha="center", va="bottom",
            fontsize=8.5, style="italic", color="#444")

    # CBAM full equation at bottom
    ax.text(4.40, 0.95,
            r"$F'' \; = \; M_s\!\left(M_c(F) \otimes F\right) \; \otimes \; \left(M_c(F) \otimes F\right)$",
            ha="center", va="center", fontsize=10.5, color="#222")

    # -----------------------------------------------------------------------
    # Lower-right panel: Transformer encoder block expanded
    # -----------------------------------------------------------------------
    panel(ax, 9.00, 0.50, 8.85, 5.55,
          "(b)  Transformer encoder block   (stacked × 3)")

    bx = 9.35
    by_in = 4.45  # input tokens y
    by_mha = 4.45  # MHA branch y
    by_mlp = 2.65  # MLP branch y
    by_out = 0.95  # output y
    h_b = 0.85

    # Input tokens (top-left)
    box(ax, bx, by_mha, 1.20, h_b, "Tokens\n50 × 256", C_TOKEN,
        font=FONT_BOX_S)

    # MHA branch
    box(ax, bx + 1.85, by_mha, 1.25, h_b, "LayerNorm", C_OP_NORM,
        font=FONT_BOX_S)
    box(ax, bx + 3.45, by_mha, 1.95, h_b,
        "Multi-Head\nAttention   (h=4)", C_OP_ATTN, font=FONT_BOX_S)
    circ(ax, bx + 5.85, by_mha + h_b / 2, 0.25, "+")
    box(ax, bx + 6.35, by_mha, 1.25, h_b, "Z₁\n50 × 256", C_TOKEN,
        font=FONT_BOX_S)

    arrow(ax, bx + 1.20, by_mha + h_b / 2, bx + 1.85, by_mha + h_b / 2)
    arrow(ax, bx + 3.10, by_mha + h_b / 2, bx + 3.45, by_mha + h_b / 2)
    arrow(ax, bx + 5.40, by_mha + h_b / 2, bx + 5.60, by_mha + h_b / 2)
    arrow(ax, bx + 6.10, by_mha + h_b / 2, bx + 6.35, by_mha + h_b / 2)

    # Residual: input tokens -> sum node (curved dashed)
    ax.add_patch(FancyArrowPatch(
        (bx + 0.60, by_mha),
        (bx + 5.85, by_mha - 0.10),
        connectionstyle="arc3,rad=-0.32",
        arrowstyle="-|>", lw=1.3, color="#555555", mutation_scale=13,
        linestyle=(0, (4, 2)),
    ))

    # Connect Z1 down to MLP branch (vertical arrow)
    arrow(ax, bx + 6.95, by_mha, bx + 6.95, by_mlp + h_b)

    # MLP branch (Z1 in -> LN -> MLP -> + -> Z2)
    box(ax, bx, by_mlp, 1.20, h_b, "Z₁\n50 × 256", C_TOKEN, font=FONT_BOX_S)
    box(ax, bx + 1.85, by_mlp, 1.25, h_b, "LayerNorm", C_OP_NORM,
        font=FONT_BOX_S)
    box(ax, bx + 3.45, by_mlp, 1.95, h_b,
        "MLP\n(GELU,  ratio 2.0)", C_OP_GELU, font=FONT_BOX_S)
    circ(ax, bx + 5.85, by_mlp + h_b / 2, 0.25, "+")
    box(ax, bx + 6.35, by_mlp, 1.25, h_b, "Z₂\n50 × 256", C_TOKEN,
        font=FONT_BOX_S)

    # Connect upper Z1 across to lower Z1
    ax.add_patch(FancyArrowPatch(
        (bx + 6.95, by_mlp + h_b),
        (bx + 0.60, by_mlp + h_b),
        connectionstyle="arc3,rad=0.0",
        arrowstyle="-", lw=1.3, color="#222222",
    ))
    arrow(ax, bx + 0.60, by_mlp + h_b, bx + 0.60, by_mlp + h_b - 0.001)

    arrow(ax, bx + 1.20, by_mlp + h_b / 2, bx + 1.85, by_mlp + h_b / 2)
    arrow(ax, bx + 3.10, by_mlp + h_b / 2, bx + 3.45, by_mlp + h_b / 2)
    arrow(ax, bx + 5.40, by_mlp + h_b / 2, bx + 5.60, by_mlp + h_b / 2)
    arrow(ax, bx + 6.10, by_mlp + h_b / 2, bx + 6.35, by_mlp + h_b / 2)

    # Residual around MLP
    ax.add_patch(FancyArrowPatch(
        (bx + 1.20, by_mlp + h_b * 0.85),
        (bx + 5.85, by_mlp + h_b / 2 + 0.10),
        connectionstyle="arc3,rad=-0.32",
        arrowstyle="-|>", lw=1.3, color="#555555", mutation_scale=13,
        linestyle=(0, (4, 2)),
    ))

    # Z2 -> output
    arrow(ax, bx + 6.95, by_mlp, bx + 6.95, by_out + h_b)
    box(ax, bx + 6.35, by_out, 1.25, h_b, "to next\nblock", C_TOKEN,
        font=FONT_BOX_S)

    # Equations
    ax.text(bx + 0.10, by_out + 0.30,
            r"$Z_1 = Z + \mathrm{MHA}\!\left(\mathrm{LN}(Z)\right)$",
            ha="left", va="center", fontsize=11, color="#222")
    ax.text(bx + 0.10, by_out - 0.10,
            r"$Z_2 = Z_1 + \mathrm{MLP}\!\left(\mathrm{LN}(Z_1)\right)$",
            ha="left", va="center", fontsize=11, color="#222")

    # -----------------------------------------------------------------------
    # Legend (bottom-right corner)
    # -----------------------------------------------------------------------
    legend_items = [
        Line2D([0], [0], color="#222222", lw=1.6,
               marker=">", markersize=8, label="forward path"),
        Line2D([0], [0], color="#555555", lw=1.6, linestyle=(0, (4, 2)),
               marker=">", markersize=8, label="residual / aux path"),
        mpatches.Patch(facecolor=C_BACKBONE, edgecolor="black",
                       label="CNN backbone"),
        mpatches.Patch(facecolor=C_CBAM, edgecolor="black",
                       label="CBAM (attention)"),
        mpatches.Patch(facecolor=C_TRANS, edgecolor="black",
                       label="Transformer"),
        mpatches.Patch(facecolor=C_HEAD_MAIN, edgecolor="black",
                       label="classifier head"),
    ]
    leg = ax.legend(
        handles=legend_items, loc="upper right",
        bbox_to_anchor=(0.995, 0.965), frameon=True, fontsize=9,
        ncol=2, handlelength=2.0, borderpad=0.5, columnspacing=1.2,
    )
    leg.get_frame().set_edgecolor("#9AA3AE")
    leg.get_frame().set_facecolor("#FFFFFF")

    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    return fig


def main():
    fig = build_figure()
    out_png = FIG_DIR / "architecture_catnet_v2.png"
    out_pdf = FIG_DIR / "architecture_catnet_v2.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
