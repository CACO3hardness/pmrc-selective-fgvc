from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if (PROJECT_ROOT / "paper").is_dir():
    FIGURE_ROOT = PROJECT_ROOT / "paper" / "figures"
    SOURCE_DATA_ROOT = PROJECT_ROOT / "source_data"
    SUMMARY_PATH = PROJECT_ROOT / "supplementary_data" / "results_summary_latest.json"
else:
    OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "outputs"
    FIGURE_ROOT = OUTPUT_ROOT / "figures"
    SOURCE_DATA_ROOT = OUTPUT_ROOT / "source_data"
    SUMMARY_PATH = OUTPUT_ROOT / "results_summary_latest.json"
SUMMARY_PATH = Path(os.environ.get("PMRC_RESULTS_JSON", SUMMARY_PATH))

skill_scripts = os.environ.get("NATURE_FIGURE_SCRIPTS")
if skill_scripts:
    sys.path.insert(0, skill_scripts)
from audit_panel_alignment import (
    matplotlib_layout_manifest,
    require_matplotlib_panel_alignment,
)


SETTINGS = [
    (
        "cub",
        "resnet50",
        "semantic",
        0.2,
        "CUB20",
    ),
    (
        "cub",
        "resnet50",
        "semantic",
        0.4,
        "CUB40",
    ),
    (
        "cub",
        "resnet50",
        "symmetric",
        0.4,
        "CUBS40",
    ),
    (
        "aircraft",
        "resnet50",
        "semantic",
        0.2,
        "Air20",
    ),
    (
        "aircraft",
        "resnet50",
        "semantic",
        0.4,
        "Air40",
    ),
    (
        "flowers102",
        "resnet50",
        "semantic",
        0.2,
        "Flow25",
    ),
    (
        "flowers102",
        "resnet50",
        "semantic",
        0.4,
        "Flow38",
    ),
    (
        "cifar100n",
        "resnet18",
        "human",
        0.402,
        "CIFAR-H",
    ),
    (
        "cub",
        "resnet18",
        "semantic",
        0.4,
        "CUB-R18",
    ),
]

BASELINE_RUNS = [
    ("ELR-R50\nCUB40", "elr_resnet50_cub_semantic40_seed0"),
    ("ELR-R18\nCUB40", "elr_resnet18_cub_semantic40_seed0"),
    ("Air-R50\n40%", "elr_resnet50_aircraft_semantic40_seed0"),
    ("Flowers-R50\n38%", "elr_resnet50_flowers102_semantic375_seed0"),
    ("CIFAR100N\nR18", "elr_resnet18_cifar100n_human_seed0"),
]

SETTING_DISPLAY_LABELS = {
    "CUB20": "CUB\n20",
    "CUB40": "CUB\n40",
    "CUBS40": "CUB-S\n40",
    "Air20": "Air\n20",
    "Air40": "Air\n40",
    "Flow25": "Flow\n25",
    "Flow38": "Flow\n38",
    "CIFAR-H": "CIFAR\nH",
    "CUB-R18": "CUB\nR18",
}


def setting_key(row: dict) -> tuple[str, str, str, float]:
    return (
        row["dataset"],
        row["model"],
        row["noise_kind"],
        float(row["requested_noise_rate"]),
    )


def load_rows() -> list[dict]:
    with SUMMARY_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["runs"]


def selected_groups(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    by_setting: dict[tuple[str, str, str, float], list[dict]] = defaultdict(list)
    for row in rows:
        by_setting[setting_key(row)].append(row)
    groups = []
    for key in SETTINGS:
        group_rows = sorted(by_setting[key[:4]], key=lambda row: row["seed"])
        if group_rows:
            groups.append((key[4], group_rows))
    return groups


def write_source_data(rows: list[dict], groups: list[tuple[str, list[dict]]]) -> None:
    path = SOURCE_DATA_ROOT / "figure1_source_data.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "panel",
                "setting",
                "run_name",
                "seed",
                "metric",
                "value",
                "ci_low",
                "ci_high",
            ],
        )
        writer.writeheader()
        for label, group_rows in groups:
            setting_label = label.replace("\n", " ")
            for row in group_rows:
                writer.writerow(
                    {
                        "panel": "a",
                        "setting": setting_label,
                        "run_name": row["run_name"],
                        "seed": row["seed"],
                        "metric": "msp_aurc",
                        "value": row["msp_aurc"],
                    }
                )
                writer.writerow(
                    {
                        "panel": "a",
                        "setting": setting_label,
                        "run_name": row["run_name"],
                        "seed": row["seed"],
                        "metric": "pmrc_aurc",
                        "value": row["pmrc_aurc"],
                    }
                )
                writer.writerow(
                    {
                        "panel": "b",
                        "setting": setting_label,
                        "run_name": row["run_name"],
                        "seed": row["seed"],
                        "metric": "risk90_reduction_percentage_points",
                        "value": 100
                        * (row["msp_risk_at_90"] - row["pmrc_risk_at_90"]),
                    }
                )
        by_name = {row["run_name"]: row for row in rows}
        methods = (
            ("temperature_scaling_aurc", "Temperature"),
            ("pnorm_aurc", "MaxLogit-pNorm"),
            ("knn_agreement_aurc", "KNN agreement"),
            ("embedding_trust_aurc", "Trust Score"),
        )
        for label, run_name in BASELINE_RUNS:
            row = by_name[run_name]
            for field, method in methods:
                value = row[field]
                if value is None:
                    continue
                writer.writerow(
                    {
                        "panel": "c",
                        "setting": label.replace("\n", " "),
                        "run_name": run_name,
                        "seed": row["seed"],
                        "metric": f"{method}_minus_pmrc_aurc",
                        "value": value - row["pmrc_aurc"],
                    }
                )
        for row in rows:
            writer.writerow(
                {
                    "panel": "d",
                    "setting": f"{row['dataset']}/{row['model']}/{row['method']}",
                    "run_name": row["run_name"],
                    "seed": row["seed"],
                    "metric": "prototype_within_msp_auc",
                    "value": row["prototype_within_msp_auc"],
                    "ci_low": row["prototype_within_msp_auc_ci95_low"],
                    "ci_high": row["prototype_within_msp_auc_ci95_high"],
                }
            )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        va="top",
        ha="left",
    )


def plot_panel_a(ax: plt.Axes, groups: list[tuple[str, list[dict]]]) -> None:
    msp_color = "#7890A3"
    pmrc_color = "#D55E00"
    positions = np.arange(len(groups))
    for position, (_, group_rows) in zip(positions, groups):
        offsets = (
            np.linspace(-0.14, 0.14, len(group_rows))
            if len(group_rows) > 1
            else np.array([0.0])
        )
        for offset, row in zip(offsets, group_rows):
            ax.plot(
                [position - 0.18, position + 0.18],
                [row["msp_aurc"], row["pmrc_aurc"]],
                color="#B8BFC5",
                linewidth=0.65,
                zorder=1,
            )
            ax.scatter(
                [position - 0.18],
                [row["msp_aurc"]],
                s=10,
                color=msp_color,
                edgecolor="white",
                linewidth=0.35,
                zorder=2,
            )
            ax.scatter(
                [position + 0.18],
                [row["pmrc_aurc"]],
                s=10,
                color=pmrc_color,
                edgecolor="white",
                linewidth=0.35,
                zorder=2,
            )
        means = [
            np.mean([row[field] for row in group_rows])
            for field in ("msp_aurc", "pmrc_aurc")
        ]
        errors = [
            (
                np.std([row[field] for row in group_rows], ddof=1)
                if len(group_rows) > 1
                else 0.0
            )
            for field in ("msp_aurc", "pmrc_aurc")
        ]
        ax.errorbar(
            [position - 0.18, position + 0.18],
            means,
            yerr=errors,
            fmt="none",
            ecolor="#222222",
            elinewidth=0.7,
            capsize=1.2,
            capthick=0.7,
            zorder=3,
        )
        ax.plot(
            [position - 0.18, position + 0.18],
            means,
            color="#222222",
            linewidth=1.4,
            marker="o",
            markersize=2.8,
            zorder=3,
        )
    ax.set_xticks(
        positions,
        [SETTING_DISPLAY_LABELS[label] for label, _ in groups],
    )
    ax.tick_params(axis="x", labelsize=6.2, pad=1.2)
    ax.set_ylabel("AURC (lower is better)")
    ax.set_ylim(0.035, 0.31)
    ax.grid(axis="y", color="#E5E7E9", linewidth=0.5)
    ax.set_title(
        "Paired selective-ranking improvement (MSP to PMRC)",
        loc="left",
        pad=9,
    )
    panel_label(ax, "a")


def plot_panel_b(ax: plt.Axes, groups: list[tuple[str, list[dict]]]) -> None:
    positions = np.arange(len(groups))
    means = []
    standard_deviations = []
    for _, group_rows in groups:
        values = [
            100 * (row["msp_risk_at_90"] - row["pmrc_risk_at_90"])
            for row in group_rows
        ]
        means.append(float(np.mean(values)))
        standard_deviations.append(
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        )
        offsets = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else [0]
        ax.scatter(
            np.full(len(values), positions[len(means) - 1]) + np.asarray(offsets),
            values,
            s=8,
            color="#555555",
            alpha=0.65,
            zorder=2,
        )
    ax.barh(
        positions,
        means,
        height=0.58,
        color="#009E73",
        alpha=0.72,
        edgecolor="none",
        zorder=1,
    )
    ax.errorbar(
        means,
        positions,
        xerr=standard_deviations,
        fmt="none",
        ecolor="#222222",
        elinewidth=0.7,
        capsize=1.2,
        capthick=0.7,
        zorder=3,
    )
    ax.axvline(0, color="#333333", linewidth=0.7)
    ax.set_yticks(positions, [label.replace("\n", " ") for label, _ in groups])
    ax.invert_yaxis()
    ax.set_xlabel("Risk reduction at 90% coverage (pp)")
    ax.grid(axis="x", color="#E5E7E9", linewidth=0.5)
    ax.set_title("High-coverage selective risk", loc="left", pad=9)
    panel_label(ax, "b")


def plot_panel_c(
    ax: plt.Axes,
    rows: list[dict],
    panel_letter: str = "c",
) -> None:
    by_name = {row["run_name"]: row for row in rows}
    field_names = [
        ("temperature_scaling_aurc", "Temperature"),
        ("pnorm_aurc", "pNorm"),
        ("knn_agreement_aurc", "KNN"),
        ("embedding_trust_aurc", "Trust"),
    ]
    values = []
    pmrc_labels = []
    for _, run_name in BASELINE_RUNS:
        row = by_name[run_name]
        pmrc_labels.append(f"{row['pmrc_aurc']:.3f}")
        values.append(
            [
                (
                    row[field] - row["pmrc_aurc"]
                    if row[field] is not None
                    else np.nan
                )
                for field, _ in field_names
            ]
        )
    matrix = np.asarray(values)
    cmap = LinearSegmentedColormap.from_list(
        "pmrc_delta",
        ["#FFF7EC", "#FDD49E", "#FC8D59", "#D7301F"],
    )
    image = ax.imshow(
        matrix,
        cmap=cmap,
        vmin=0,
        vmax=max(0.07, float(np.nanmax(matrix))),
        aspect="auto",
    )
    baseline_labels = [
        "CUB-R50",
        "CUB-R18",
        "Air-R50",
        "Flow-R50",
        "CIFAR",
    ]
    ax.set_xticks(
        np.arange(len(BASELINE_RUNS)),
        [
            label + f"\n{pmrc}"
            for label, pmrc in zip(baseline_labels, pmrc_labels)
        ],
    )
    ax.set_xlabel("PMRC AURC for each setting", labelpad=2)
    ax.set_yticks(np.arange(len(field_names)), [label for _, label in field_names])
    ax.tick_params(axis="x", labelsize=6.0, pad=1.2)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            if np.isfinite(value):
                ax.text(
                    col_index,
                    row_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6.0,
                    color="white" if value > 0.045 else "#222222",
                )
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("AURC minus PMRC AURC", fontsize=6.8)
    colorbar.ax.tick_params(labelsize=6.2)
    ax.set_title("Comparison with post-hoc baselines", loc="left", pad=9)
    panel_label(ax, panel_letter)


def plot_panel_d(
    ax: plt.Axes,
    rows: list[dict],
    panel_letter: str = "d",
) -> None:
    group_order = [
        ("CUB\nR50", lambda row: row["dataset"] == "cub" and row["model"] == "resnet50" and row["method"] == "elr" and row["noise_kind"] != "clean"),
        ("CUB\nR18", lambda row: row["dataset"] == "cub" and row["model"] == "resnet18"),
        ("CE\nR50", lambda row: row["dataset"] == "cub" and row["method"] == "ce"),
        ("Air\nR50", lambda row: row["dataset"] == "aircraft"),
        ("Flow\nR50", lambda row: row["dataset"] == "flowers102"),
        ("CIFAR\nR18", lambda row: row["dataset"] == "cifar100n"),
        ("Clean", lambda row: row["noise_kind"] == "clean"),
    ]
    for position, (label, predicate) in enumerate(group_order):
        selected = [row for row in rows if predicate(row)]
        if not selected:
            continue
        offsets = (
            np.linspace(-0.14, 0.14, len(selected))
            if len(selected) > 1
            else np.array([0.0])
        )
        for offset, row in zip(offsets, selected):
            low = row["prototype_within_msp_auc_ci95_low"]
            high = row["prototype_within_msp_auc_ci95_high"]
            value = row["prototype_within_msp_auc"]
            color = "#7F8C98" if row["noise_kind"] == "clean" else "#0072B2"
            ax.plot(
                [position + offset, position + offset],
                [low, high],
                color=color,
                linewidth=0.65,
                alpha=0.7,
            )
            ax.scatter(
                position + offset,
                value,
                s=8,
                color=color,
                edgecolor="white",
                linewidth=0.25,
                zorder=2,
            )
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=0.7)
    ax.set_xticks(
        np.arange(len(group_order)),
        [label for label, _ in group_order],
    )
    ax.tick_params(axis="x", labelsize=6.2, pad=1.5)
    ax.set_ylabel("Correctness AUROC within MSP quintile")
    ax.set_ylim(0.48, 0.86)
    ax.grid(axis="y", color="#E5E7E9", linewidth=0.5)
    ax.set_title("Prototype margin beyond confidence", loc="left", pad=9)
    panel_label(ax, panel_letter)


def main() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    groups = selected_groups(rows)
    write_source_data(rows, groups)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.2,
            "axes.titlesize": 7.4,
            "axes.labelsize": 7.1,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
        }
    )
    # Final figure contract: 160 mm wide by 124 mm high, matching the
    # manuscript text width so the exported text is not reduced again.
    # Uncertainty: dots show individual seeds; black connectors/bars show seed means.
    width_mm = 160
    height_mm = 130
    figure_width_in = width_mm / 25.4
    figure_height_in = height_mm / 25.4
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(figure_width_in, figure_height_in),
        constrained_layout=True,
    )
    plot_panel_a(axes[0, 0], groups)
    plot_panel_b(axes[0, 1], groups)
    plot_panel_c(axes[1, 0], rows)
    plot_panel_d(axes[1, 1], rows)

    figure.canvas.draw()
    layout = matplotlib_layout_manifest(figure)
    (FIGURE_ROOT / "figure1_pmrc.alignment-layout.json").write_text(
        json.dumps(layout, indent=2),
        encoding="utf-8",
    )
    require_matplotlib_panel_alignment(
        figure,
        json_out=FIGURE_ROOT / "figure1_pmrc.alignment.json",
        overlay_svg=FIGURE_ROOT / "figure1_pmrc.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    stem = FIGURE_ROOT / "figure1_pmrc"
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    figure.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)

    effect_figure, effect_axes = plt.subplots(
        1,
        2,
        figsize=(figure_width_in, 68 / 25.4),
        constrained_layout=True,
    )
    plot_panel_a(effect_axes[0], groups)
    plot_panel_b(effect_axes[1], groups)
    effect_figure.canvas.draw()
    effect_layout = matplotlib_layout_manifest(effect_figure)
    (FIGURE_ROOT / "figure1_pmrc_effect.alignment-layout.json").write_text(
        json.dumps(effect_layout, indent=2),
        encoding="utf-8",
    )
    require_matplotlib_panel_alignment(
        effect_figure,
        json_out=FIGURE_ROOT / "figure1_pmrc_effect.alignment.json",
        overlay_svg=FIGURE_ROOT / "figure1_pmrc_effect.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    effect_stem = FIGURE_ROOT / "figure1_pmrc_effect"
    effect_figure.savefig(
        effect_stem.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    effect_figure.savefig(
        effect_stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(effect_figure)

    mechanism_figure, mechanism_axes = plt.subplots(
        1,
        2,
        figsize=(figure_width_in, 72 / 25.4),
        constrained_layout=True,
    )
    plot_panel_c(mechanism_axes[0], rows, panel_letter="a")
    plot_panel_d(mechanism_axes[1], rows, panel_letter="b")
    mechanism_figure.canvas.draw()
    mechanism_layout = matplotlib_layout_manifest(mechanism_figure)
    (FIGURE_ROOT / "figure1_pmrc_mechanism.alignment-layout.json").write_text(
        json.dumps(mechanism_layout, indent=2),
        encoding="utf-8",
    )
    require_matplotlib_panel_alignment(
        mechanism_figure,
        json_out=FIGURE_ROOT / "figure1_pmrc_mechanism.alignment.json",
        overlay_svg=FIGURE_ROOT / "figure1_pmrc_mechanism.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    mechanism_stem = FIGURE_ROOT / "figure1_pmrc_mechanism"
    mechanism_figure.savefig(
        mechanism_stem.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    mechanism_figure.savefig(
        mechanism_stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(mechanism_figure)


if __name__ == "__main__":
    main()
