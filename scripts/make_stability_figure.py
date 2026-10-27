from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "outputs"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
RUNS_ROOT = PROJECT_ROOT / "runs"

skill_scripts = os.environ.get("NATURE_FIGURE_SCRIPTS")
if skill_scripts:
    sys.path.insert(0, skill_scripts)
from audit_panel_alignment import (
    matplotlib_layout_manifest,
    require_matplotlib_panel_alignment,
)


RUNS = [
    ("CUB40 R50 seed0", "elr_resnet50_cub_semantic40_seed0", "#0072B2"),
    ("CUB40 R50 seed1", "elr_resnet50_cub_semantic40_seed1", "#56B4E9"),
    ("Aircraft40 R50", "elr_resnet50_aircraft_semantic40_seed0", "#009E73"),
    ("Flowers38 R50", "elr_resnet50_flowers102_semantic375_seed0", "#CC79A7"),
    ("CIFAR100N R18", "elr_resnet18_cifar100n_human_seed0", "#D55E00"),
]


def main() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 5.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
        }
    )
    width_mm = 89
    height_mm = 82
    figure, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4))
    fractions = np.array([0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 1.0])
    source_rows = []
    for label, run_name, color in RUNS:
        path = RUNS_ROOT / run_name / "posthoc_stability.json"
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        means = []
        standard_deviations = []
        sample_sizes = []
        for fraction in fractions:
            key = "1.0" if fraction == 1.0 else str(fraction)
            result = payload["results"]["all"][key]
            means.append(result["aurc"]["mean"])
            standard_deviations.append(result["aurc"]["std"])
            sample_sizes.append(result["sample_size"])
            source_rows.append(
                {
                    "setting": label,
                    "run_name": run_name,
                    "fraction": fraction,
                    "sample_size": result["sample_size"],
                    "aurc_mean": result["aurc"]["mean"],
                    "aurc_std": result["aurc"]["std"],
                }
            )
        ax.errorbar(
            fractions,
            means,
            yerr=standard_deviations,
            color=color,
            marker="o",
            markersize=3,
            linewidth=1.1,
            capsize=1.5,
            label=f"{label} (n={sample_sizes[-1]})",
        )
    ax.set_xlabel("Clean calibration-set fraction")
    ax.set_ylabel("PMRC AURC (lower is better)")
    ax.set_xticks(fractions, ["5%", "10%", "20%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", color="#E5E7E9", linewidth=0.5)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        handlelength=1.8,
        labelspacing=0.35,
        columnspacing=1.0,
    )
    figure.tight_layout()
    figure.canvas.draw()
    layout = matplotlib_layout_manifest(figure)
    (FIGURE_ROOT / "figureS1_stability.alignment-layout.json").write_text(
        json.dumps(layout, indent=2),
        encoding="utf-8",
    )
    require_matplotlib_panel_alignment(
        figure,
        json_out=FIGURE_ROOT / "figureS1_stability.alignment.json",
        overlay_svg=FIGURE_ROOT / "figureS1_stability.alignment.svg",
        strict=True,
    )
    stem = FIGURE_ROOT / "figureS1_stability"
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(figure)

    import csv

    with (FIGURE_ROOT / "figureS1_stability_source_data.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "setting",
                "run_name",
                "fraction",
                "sample_size",
                "aurc_mean",
                "aurc_std",
            ],
        )
        writer.writeheader()
        writer.writerows(source_rows)


if __name__ == "__main__":
    main()
