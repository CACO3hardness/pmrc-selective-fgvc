from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "outputs"
RESULTS_PATH = OUTPUT_ROOT / "results_summary_latest.json"
FIELDS = (
    ("temperature_scaling_aurc", "Temperature"),
    ("pnorm_aurc", "MaxLogit-pNorm"),
    ("knn_agreement_aurc", "KNN agreement"),
    ("embedding_trust_aurc", "Trust Score"),
)


def main() -> None:
    payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    rows = payload["runs"]
    controlled_rows = [
        row for row in rows if row["noise_kind"] != "web"
    ]
    groups = {
        f"Controlled {len(controlled_rows)}-model matrix": controlled_rows,
        "WebFG real web noise": [
            row for row in rows if row["noise_kind"] == "web"
        ],
    }
    output = {}
    lines = [
        "# Selective-baseline comparison",
        "",
        "PMRC lower indicates that PMRC had lower test AURC than the baseline",
        "on the same model and test samples.",
        "",
        "| Evaluation set | Method | Available runs | Mean AURC | PMRC lower |",
        "|---|---|---:|---:|---:|",
    ]
    for group_name, group_rows in groups.items():
        output[group_name] = {"n": len(group_rows), "methods": {}}
        for field, label in FIELDS:
            available = [
                row for row in group_rows if row.get(field) is not None
            ]
            values = [float(row[field]) for row in available]
            pmrc_lower = sum(
                row["pmrc_aurc"] < float(row[field]) for row in available
            )
            output[group_name]["methods"][field] = {
                "label": label,
                "available": len(available),
                "mean_aurc": float(np.mean(values)) if values else None,
                "pmrc_lower": pmrc_lower,
                "exceptions": [
                    row["run_name"]
                    for row in available
                    if row["pmrc_aurc"] >= float(row[field])
                ],
            }
            mean = (
                f"{float(np.mean(values)):.4f}" if values else "not measured"
            )
            lines.append(
                f"| {group_name} | {label} | {len(available)} | {mean} | "
                f"{pmrc_lower}/{len(available)} |"
            )
    (OUTPUT_ROOT / "selective_baseline_comparison_final.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_ROOT / "selective_baseline_comparison_final.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"Summarized baselines for {len(rows)} runs.")


if __name__ == "__main__":
    main()
