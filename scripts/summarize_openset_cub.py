from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "outputs"


def main() -> None:
    source = OUTPUT_ROOT / "openset_cub160.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = sorted(payload.get("runs", []), key=lambda row: int(row["seed"]))
    if not rows:
        raise RuntimeError("No completed strict CUB open-set runs")
    metrics = {}
    for key in (
        "relative_gain_pct",
        "msp_aurc",
        "pmrc_aurc",
        "msp_unknown_auroc",
        "pmrc_unknown_auroc",
        "msp_risk90",
        "pmrc_risk90",
        "validation_gate_auc",
    ):
        metrics[key] = {
            "mean": float(np.mean([row[key] for row in rows])),
            "min": float(np.min([row[key] for row in rows])),
            "max": float(np.max([row[key] for row in rows])),
        }
    summary = {
        "known_classes": 160,
        "unknown_classes": 40,
        "seeds": [int(row["seed"]) for row in rows],
        "runs": rows,
        "metrics": metrics,
    }
    (OUTPUT_ROOT / "openset_cub160_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    with (OUTPUT_ROOT / "openset_cub160_summary.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Strict CUB open-set evaluation",
        "",
        "Each classifier is trained on 160 known classes only. The 40 held-out "
        "classes are excluded from training, validation, prototype construction, "
        "and calibration. Unknown test images are mixed with known test images.",
        "",
        "| Seed | Known / unknown tests | AURC gain | MSP unknown AUROC | "
        "PMRC unknown AUROC | Gate |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['known_test_count']} / "
            f"{row['unknown_test_count']} | {row['relative_gain_pct']:+.2f}% | "
            f"{row['msp_unknown_auroc']:.4f} | "
            f"{row['pmrc_unknown_auroc']:.4f} | "
            f"{row['gate_selected_method']} |"
        )
    lines.extend(
        [
            "",
            "| Metric | Mean | Min | Max |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, values in metrics.items():
        lines.append(
            f"| {key} | {values['mean']:.4f} | {values['min']:.4f} | "
            f"{values['max']:.4f} |"
        )
    lines.extend(
        [
            "",
            "This is a closed-set-trained model evaluated under open-set input "
            "prevalence. It does not claim a dedicated open-set detector.",
        ]
    )
    (OUTPUT_ROOT / "openset_cub160_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
