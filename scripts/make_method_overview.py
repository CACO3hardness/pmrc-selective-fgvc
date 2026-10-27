from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "outputs"
FIGURE_ROOT = OUTPUT_ROOT / "figures"

skill_scripts = os.environ.get("NATURE_FIGURE_SCRIPTS")
if skill_scripts:
    sys.path.insert(0, skill_scripts)
from audit_panel_alignment import (
    matplotlib_layout_manifest,
    require_matplotlib_panel_alignment,
)


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str,
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.8,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + 0.018,
        xy[1] + height - 0.05,
        title,
        ha="left",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color="#20252A",
    )
    ax.text(
        xy[0] + 0.018,
        xy[1] + height - 0.115,
        body,
        ha="left",
        va="top",
        fontsize=6.2,
        color="#343A40",
        linespacing=1.25,
    )


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#5B6570",
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=0.9,
        color=color,
        shrinkA=1,
        shrinkB=1,
    )
    ax.add_patch(arrow)


def main() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
        }
    )
    width_mm = 183
    height_mm = 88
    figure, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    lane_specs = [
        (0.69, "1  Noisy training", "#F7F8FA"),
        (0.37, "2  Clean calibration", "#F7F8FA"),
        (0.05, "3  Selective inference", "#F7F8FA"),
    ]
    for y, label, color in lane_specs:
        ax.add_patch(
            FancyBboxPatch(
                (0.008, y),
                0.984,
                0.27,
                boxstyle="round,pad=0.006,rounding_size=0.012",
                facecolor=color,
                edgecolor="#E1E5EA",
                linewidth=0.6,
            )
        )
        ax.text(
            0.02,
            y + 0.255,
            label,
            fontsize=6.8,
            fontweight="bold",
            color="#49515A",
            va="center",
        )

    add_box(
        ax,
        (0.04, 0.735),
        0.22,
        0.17,
        "Training images",
        "observed labels ỹ\nmay be noisy",
        "#EAF2F8",
        "#8AA9BE",
    )
    add_box(
        ax,
        (0.35, 0.705),
        0.24,
        0.22,
        "Frozen classifier",
        "features z = φθ(x)\nlogits and MSP",
        "#DDEBF7",
        "#5B8FB9",
    )
    add_box(
        ax,
        (0.68, 0.735),
        0.27,
        0.17,
        "Noisy-label prototypes",
        "class means from observed labels\nprototype margin m_proto",
        "#FCE8D5",
        "#D8894A",
    )
    add_arrow(ax, (0.26, 0.82), (0.35, 0.82))
    add_arrow(ax, (0.59, 0.82), (0.68, 0.82))

    add_box(
        ax,
        (0.04, 0.415),
        0.22,
        0.17,
        "Clean validation",
        "reliable labels y\ncorrectness target s",
        "#EAF2F8",
        "#8AA9BE",
    )
    add_box(
        ax,
        (0.35, 0.385),
        0.24,
        0.22,
        "Reliability features",
        "confidence, view agreement,\nfeature norm and m_proto",
        "#E8F3EE",
        "#6AA383",
    )
    add_box(
        ax,
        (0.68, 0.405),
        0.27,
        0.18,
        "Logistic calibrator",
        "fit r(x) to validation correctness\nfive-fold L2 selection",
        "#FBE3D6",
        "#D6602A",
    )
    add_arrow(ax, (0.26, 0.50), (0.35, 0.50))
    add_arrow(ax, (0.475, 0.705), (0.475, 0.605))
    add_arrow(ax, (0.59, 0.50), (0.68, 0.50))
    add_arrow(ax, (0.815, 0.700), (0.815, 0.585), color="#D6602A")

    add_box(
        ax,
        (0.04, 0.095),
        0.22,
        0.17,
        "Test image",
        "no test labels used\nfor calibration",
        "#EAF2F8",
        "#8AA9BE",
    )
    add_box(
        ax,
        (0.35, 0.065),
        0.24,
        0.22,
        "Same frozen model",
        "ŷ = arg max pθ(c|x)\nfeatures and logits",
        "#DDEBF7",
        "#5B8FB9",
    )
    add_box(
        ax,
        (0.68, 0.095),
        0.27,
        0.17,
        "Applicability gate and reorder",
        "if a_v > 0.5: rank by r(x)\nelse: rank by MSP\naccept top q; argmax fixed",
        "#E8F3EE",
        "#4F8A69",
    )
    add_arrow(ax, (0.26, 0.18), (0.35, 0.18))
    add_arrow(ax, (0.59, 0.18), (0.68, 0.18))
    add_arrow(ax, (0.815, 0.405), (0.815, 0.265), color="#D6602A")

    figure.canvas.draw()
    layout = matplotlib_layout_manifest(figure)
    (FIGURE_ROOT / "figureS2_method_overview.alignment-layout.json").write_text(
        json.dumps(layout, indent=2),
        encoding="utf-8",
    )
    require_matplotlib_panel_alignment(
        figure,
        json_out=FIGURE_ROOT / "figureS2_method_overview.alignment.json",
        overlay_svg=FIGURE_ROOT / "figureS2_method_overview.alignment.svg",
        strict=True,
    )
    stem = FIGURE_ROOT / "figureS2_method_overview"
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
