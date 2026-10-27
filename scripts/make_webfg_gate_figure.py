from __future__ import annotations

import csv
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
RESULT_PATH = OUTPUT_ROOT / "applicability_gate.json"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
FIGURE_STEM = FIGURE_ROOT / "figure2_webfg_gate"

skill_scripts = os.environ.get("NATURE_FIGURE_SCRIPTS")
if skill_scripts:
    sys.path.insert(0, skill_scripts)
from audit_panel_alignment import require_matplotlib_panel_alignment


SUBSET_LABELS = {
    "webfg_bird": "Bird",
    "webfg_aircraft": "Aircraft",
    "webfg_car": "Car",
}
COLORS = {
    "webfg_bird": "#8C99A6",
    "webfg_aircraft": "#D55E00",
    "webfg_car": "#0072B2",
}


def load_rows() -> list[dict]:
    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload["runs"]
        if row["dataset"] in SUBSET_LABELS
    ]
    missing_subsets = set(SUBSET_LABELS) - {row["dataset"] for row in rows}
    if missing_subsets:
        raise ValueError(f"Missing WebFG subsets: {sorted(missing_subsets)}")
    for row in rows:
        row["relative_aurc_reduction_pct"] = (
            100.0
            * (row["msp_aurc"] - row["pmrc_aurc"])
            / row["msp_aurc"]
        )
    return rows


def write_source_data(rows: list[dict]) -> None:
    path = FIGURE_ROOT / "figure2_webfg_gate_source_data.csv"
    fields = [
        "subset",
        "seed",
        "run_name",
        "msp_aurc",
        "pmrc_aurc",
        "relative_aurc_reduction_pct",
        "validation_prototype_within_msp_auc",
        "selected_method",
        "msp_risk90",
        "pmrc_risk90",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["dataset"], item["seed"])):
            writer.writerow(
                {
                    "subset": SUBSET_LABELS[row["dataset"]],
                    "seed": row["seed"],
                    "run_name": row["run_name"],
                    "msp_aurc": row["msp_aurc"],
                    "pmrc_aurc": row["pmrc_aurc"],
                    "relative_aurc_reduction_pct": row[
                        "relative_aurc_reduction_pct"
                    ],
                    "validation_prototype_within_msp_auc": row[
                        "validation_prototype_within_msp_auc"
                    ],
                    "selected_method": row["selected_method"],
                    "msp_risk90": row["msp_risk90"],
                    "pmrc_risk90": row["pmrc_risk90"],
                }
            )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="top",
    )


def plot_aurc(ax: plt.Axes, rows: list[dict]) -> None:
    subsets = list(SUBSET_LABELS)
    for x_position, subset in enumerate(subsets):
        selected = sorted(
            [row for row in rows if row["dataset"] == subset],
            key=lambda item: item["seed"],
        )
        offsets = np.linspace(-0.07, 0.07, len(selected))
        for offset, row in zip(offsets, selected):
            msp_x = x_position - 0.18 + offset
            pmrc_x = x_position + 0.18 + offset
            ax.plot(
                [msp_x, pmrc_x],
                [row["msp_aurc"], row["pmrc_aurc"]],
                color="#C5CBD1",
                linewidth=0.7,
                zorder=1,
            )
            ax.scatter(
                msp_x,
                row["msp_aurc"],
                s=16,
                facecolor="white",
                edgecolor="#6D7A86",
                linewidth=0.8,
                zorder=3,
            )
            ax.scatter(
                pmrc_x,
                row["pmrc_aurc"],
                s=16,
                color=COLORS[subset],
                edgecolor="white",
                linewidth=0.35,
                zorder=4,
            )
    ax.set_xticks(
        np.arange(len(subsets)),
        [SUBSET_LABELS[subset] for subset in subsets],
    )
    ax.set_ylabel("AURC (lower is better)")
    ax.set_ylim(0.065, 0.14)
    ax.grid(axis="y", color="#E8EBED", linewidth=0.5)
    ax.scatter([], [], s=16, facecolor="white", edgecolor="#6D7A86",
               label="MSP")
    ax.scatter([], [], s=16, color="#555555", label="PMRC")
    ax.legend(
        loc="upper left",
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.8,
        borderaxespad=0.2,
    )
    panel_label(ax, "a")


def plot_gate(ax: plt.Axes, rows: list[dict]) -> None:
    for subset, label in SUBSET_LABELS.items():
        selected = [row for row in rows if row["dataset"] == subset]
        ax.scatter(
            [
                row["validation_prototype_within_msp_auc"]
                for row in selected
            ],
            [row["relative_aurc_reduction_pct"] for row in selected],
            s=22,
            color=COLORS[subset],
            edgecolor="white",
            linewidth=0.45,
            label=label,
            zorder=3,
        )
    ax.axvline(0.5, color="#555555", linewidth=0.8, linestyle=(0, (4, 2)))
    ax.axhline(0.0, color="#8A8A8A", linewidth=0.7)
    ax.set_xlim(0.35, 0.67)
    ax.set_ylim(-3.3, 5.6)
    ax.text(
        0.475,
        5.3,
        "fallback",
        ha="right",
        va="top",
        color="#555555",
        fontsize=6,
    )
    ax.text(
        0.525,
        5.3,
        "use PMRC",
        ha="left",
        va="top",
        color="#555555",
        fontsize=6,
    )
    ax.set_xlabel("Validation prototype AUROC within MSP strata")
    ax.set_ylabel("Relative AURC reduction (%)")
    ax.legend(
        loc="lower center",
        ncol=3,
        handletextpad=0.25,
        columnspacing=0.8,
        borderaxespad=0.2,
    )
    panel_label(ax, "b")


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    write_source_data(rows)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 2.7),
        gridspec_kw={"wspace": 0.32},
    )
    plot_aurc(axes[0], rows)
    plot_gate(axes[1], rows)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.17, top=0.91)

    require_matplotlib_panel_alignment(
        fig,
        json_out=str(FIGURE_STEM) + ".alignment-layout.json",
        overlay_svg=str(FIGURE_STEM) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(FIGURE_STEM.with_suffix(".svg"))
    fig.savefig(FIGURE_STEM.with_suffix(".pdf"))
    fig.savefig(FIGURE_STEM.with_suffix(".png"), dpi=300)
    fig.savefig(FIGURE_STEM.with_suffix(".tiff"), dpi=600)
    plt.close(fig)
    print(f"Wrote {FIGURE_STEM.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
