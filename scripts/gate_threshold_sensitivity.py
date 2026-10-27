from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT.parent.parent / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate-threshold sensitivity.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.40, 0.45, 0.475, 0.50, 0.525, 0.55, 0.60],
    )
    return parser.parse_args()


def summarize(rows: list[dict], scope: str, thresholds: list[float]) -> list[dict]:
    results = []
    for threshold in thresholds:
        gains = []
        for row in rows:
            use_pmrc = (
                row["validation_gate_auc"] is not None
                and row["validation_gate_auc"] > threshold
            )
            score = row["pmrc_aurc"] if use_pmrc else row["msp_aurc"]
            gains.append(
                100.0 * (row["msp_aurc"] - score) / row["msp_aurc"]
            )
        results.append(
            {
                "scope": scope,
                "threshold": threshold,
                "n": len(rows),
                "pmrc_selected": int(
                    np.mean(
                        [
                            row["validation_gate_auc"] is not None
                            and row["validation_gate_auc"] > threshold
                            for row in rows
                        ]
                    )
                    * len(rows)
                ),
                "mean_gain_pct": float(np.mean(gains)),
                "nonregression_rate": float(np.mean([gain >= 0 for gain in gains])),
                "worse_count": int(np.sum(np.asarray(gains) < 0)),
                "minimum_gain_pct": float(np.min(gains)),
            }
        )
    return results


def main() -> None:
    args = parse_args()
    main_rows = json.loads(
        (args.output_dir / "applicability_gate.json").read_text(
            encoding="utf-8"
        )
    )["runs"]
    for row in main_rows:
        row["validation_gate_auc"] = row[
            "validation_prototype_within_msp_auc"
        ]
    heldout_rows = json.loads(
        (args.output_dir / "heldout_class_calibration.json").read_text(
            encoding="utf-8"
        )
    )["rows"]
    for row in heldout_rows:
        row["run_name"] = (
            f"{row['run_name']}:{row['heldout_fraction']}:{row['repeat']}"
        )
    rows = summarize(main_rows, "original_test_split", args.thresholds)
    rows.extend(summarize(heldout_rows, "heldout_class", args.thresholds))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gate_threshold_sensitivity.json").write_text(
        json.dumps({"thresholds": args.thresholds, "results": rows}, indent=2),
        encoding="utf-8",
    )
    with (args.output_dir / "gate_threshold_sensitivity.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Gate-threshold sensitivity",
        "",
        "| Scope | Threshold | PMRC selected | Mean AURC gain | "
        "Non-regression | Regressions |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scope']} | {row['threshold']:.3f} | "
            f"{row['pmrc_selected']}/{row['n']} | "
            f"{row['mean_gain_pct']:+.2f}% | "
            f"{100 * row['nonregression_rate']:.1f}% | {row['worse_count']} |"
        )
    (args.output_dir / "gate_threshold_sensitivity.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("Wrote gate_threshold_sensitivity.{json,csv,md}")


if __name__ == "__main__":
    main()
