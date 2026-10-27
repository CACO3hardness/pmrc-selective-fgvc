from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_prototype_source_suite import DEFAULT_RUNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "outputs"
VARIANTS = (
    "without_prototype",
    "noisy",
    "clean_oracle",
    "pretrained",
    "shuffled",
)


def run_group(run_name: str) -> str:
    if "cub_semantic40" in run_name:
        return "CUB semantic-40"
    if "aircraft_semantic40" in run_name:
        return "Aircraft semantic-40"
    if "flowers102_semantic375" in run_name:
        return "Flowers102 semantic-37.5"
    return run_name


def main() -> None:
    rows = []
    for run_name in DEFAULT_RUNS:
        path = PROJECT_ROOT / "runs" / run_name / "prototype_source_ablation.json"
        if not path.is_file():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        row = {"run_name": run_name, "group": run_group(run_name)}
        for variant in VARIANTS:
            row[f"{variant}_aurc"] = result["prototype_variants"][variant][
                "test_aurc"
            ]
        row["noisy_minus_without_prototype"] = (
            row["noisy_aurc"] - row["without_prototype_aurc"]
        )
        row["noisy_minus_pretrained"] = (
            row["noisy_aurc"] - row["pretrained_aurc"]
        )
        row["noisy_minus_shuffled"] = (
            row["noisy_aurc"] - row["shuffled_aurc"]
        )
        row["noisy_minus_clean_oracle"] = (
            row["noisy_aurc"] - row["clean_oracle_aurc"]
        )
        rows.append(row)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    summary_rows = []
    for group, group_rows in sorted(grouped.items()):
        summary = {
            "group": group,
            "n": len(group_rows),
            "noisy_beats_without_prototype": sum(
                row["noisy_minus_without_prototype"] < 0
                for row in group_rows
            ),
            "noisy_beats_pretrained": sum(
                row["noisy_minus_pretrained"] < 0
                for row in group_rows
            ),
            "noisy_beats_shuffled": sum(
                row["noisy_minus_shuffled"] < 0
                for row in group_rows
            ),
        }
        for variant in VARIANTS:
            summary[f"{variant}_mean"] = float(
                np.mean([row[f"{variant}_aurc"] for row in group_rows])
            )
        summary_rows.append(summary)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "prototype_source_runs.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT_ROOT / "prototype_source_groups.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# Prototype-source ablation",
        "",
        "Lower AURC is better. Noisy prototypes use observed training labels.",
        "Clean-oracle prototypes use unavailable clean training labels and are a",
        "diagnostic upper bound, not a deployable method.",
        "",
        "| Setting | n | No prototype | Noisy prototype | Clean oracle | "
        "ImageNet prototype | Shuffled prototype | Noisy wins vs no/ImageNet/shuffled |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['group']} | {row['n']} | "
            f"{row['without_prototype_mean']:.4f} | "
            f"{row['noisy_mean']:.4f} | "
            f"{row['clean_oracle_mean']:.4f} | "
            f"{row['pretrained_mean']:.4f} | "
            f"{row['shuffled_mean']:.4f} | "
            f"{row['noisy_beats_without_prototype']}/{row['n']}, "
            f"{row['noisy_beats_pretrained']}/{row['n']}, "
            f"{row['noisy_beats_shuffled']}/{row['n']} |"
        )
    lines.extend(
        [
            "",
            "Paired bootstrap confidence intervals are retained in",
            "`prototype_source_summary.json`.",
        ]
    )
    (OUTPUT_ROOT / "prototype_source_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"Summarized {len(rows)} source-ablation runs.")


if __name__ == "__main__":
    main()
