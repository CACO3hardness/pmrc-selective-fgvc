from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgel.utils import read_json, write_json
from scripts.summarize_results import infer_noise


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT.parent.parent / "outputs"
GROUP_ORDER = (
    "confidence",
    "confidence_plus_view",
    "confidence_plus_prototype",
    "all",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize confidence, view, and prototype feature ablations."
    )
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for result_path in sorted(args.runs_root.glob("*/posthoc_reliability.json")):
        run_dir = result_path.parent
        config_path = run_dir / "config.json"
        if not config_path.is_file():
            continue
        result = read_json(result_path)
        groups = result.get("feature_group_results", {})
        if not all(name in groups for name in GROUP_ORDER):
            continue
        config = read_json(config_path)
        noise_kind, requested_noise_rate, _ = infer_noise(config)
        dataset = str(config.get("dataset", "unknown")).lower()
        if dataset in {"cub", "cub200"}:
            dataset = "cub"
        elif dataset.startswith("aircraft"):
            dataset = "aircraft"
        elif dataset.startswith("flowers"):
            dataset = "flowers102"
        row = {
            "run_name": run_dir.name,
            "dataset": dataset,
            "model": str(config.get("model", "unknown")),
            "method": str(config.get("method", "unknown")),
            "noise_kind": noise_kind,
            "requested_noise_rate": requested_noise_rate,
            "seed": int(config.get("seed", 0)),
        }
        for group_name in GROUP_ORDER:
            row[f"{group_name}_aurc"] = float(groups[group_name]["test_aurc"])
        row["prototype_minus_view_aurc"] = (
            row["confidence_plus_prototype_aurc"]
            - row["confidence_plus_view_aurc"]
        )
        row["all_minus_prototype_aurc"] = (
            row["all_aurc"] - row["confidence_plus_prototype_aurc"]
        )
        rows.append(row)

    grouped: dict[tuple[str, str, str, str, float], list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["method"],
            row["model"],
            row["noise_kind"],
            row["requested_noise_rate"],
        )
        grouped[key].append(row)

    summaries = []
    for key, group_rows in sorted(grouped.items()):
        dataset, method, model, noise_kind, requested_noise_rate = key
        summary = {
            "dataset": dataset,
            "method": method,
            "model": model,
            "noise_kind": noise_kind,
            "requested_noise_rate": requested_noise_rate,
            "n": len(group_rows),
            "prototype_improves_over_view": sum(
                row["prototype_minus_view_aurc"] < 0 for row in group_rows
            ),
            "all_improves_over_prototype": sum(
                row["all_minus_prototype_aurc"] < 0 for row in group_rows
            ),
        }
        for group_name in GROUP_ORDER:
            values = [row[f"{group_name}_aurc"] for row in group_rows]
            summary[f"{group_name}_mean"] = float(np.mean(values))
            summary[f"{group_name}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        summaries.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "feature_group_ablation.json",
        {"n_runs": len(rows), "runs": rows, "summaries": summaries},
    )
    with (args.output_dir / "feature_group_ablation_runs.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "feature_group_ablation_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    lines = [
        "# Feature-group ablation",
        "",
        "Lower AURC is better. The prototype columns replace only the prototype-",
        "margin descriptor while preserving all other confidence and two-view",
        "descriptors. Negative deltas indicate improvement.",
        "",
        "| Dataset | Method | Backbone | Noise | Rate | n | Confidence | +View | "
        "+Prototype | All | Prototype gain vs +View |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        improvement = (
            item["confidence_plus_view_mean"]
            - item["confidence_plus_prototype_mean"]
        )
        rate = (
            "uncontrolled"
            if item["noise_kind"] == "web"
            else f"{100.0 * item['requested_noise_rate']:.1f}%"
        )
        lines.append(
            f"| {item['dataset']} | {item['method']} | {item['model']} | "
            f"{item['noise_kind']} | "
            f"{rate} | {item['n']} | "
            f"{item['confidence_mean']:.4f} | "
            f"{item['confidence_plus_view_mean']:.4f} | "
            f"{item['confidence_plus_prototype_mean']:.4f} | "
            f"{item['all_mean']:.4f} | {improvement:+.4f} |"
        )
    (args.output_dir / "feature_group_ablation.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote feature-group ablation for {len(rows)} runs.")


if __name__ == "__main__":
    main()
