from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgel.data import load_manifest
from rgel.utils import read_json, write_json
from scripts.posthoc_reliability import aurc_from_scores
from scripts.prototype_contamination_diagnostic import (
    DEFAULT_RUNS,
    build_prototypes,
    extract_train_views,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Partition classes by feature concentration and prototype fidelity."
    )
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    return parser.parse_args()


def relative_gain(msp: float, pmrc: float) -> float:
    return 100.0 * (msp - pmrc) / msp


def grouped_gain(
    test_features: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    test_predictions: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float | int]:
    mask = np.isin(test_labels, classes)
    if not mask.any():
        raise ValueError("No test samples belong to requested class partition")
    correct = test_predictions[mask] == test_labels[mask]
    msp = aurc_from_scores(test_features[mask, 0], correct)
    pmrc = aurc_from_scores(test_scores[mask], correct)
    return {
        "classes": int(len(classes)),
        "test_samples": int(mask.sum()),
        "msp_aurc": float(msp),
        "pmrc_aurc": float(pmrc),
        "relative_gain_pct": relative_gain(msp, pmrc),
    }


def run_partition(run_dir: Path, device: torch.device) -> list[dict]:
    config = read_json(run_dir / "config.json")
    rows = load_manifest(config["manifest"])
    train_rows = [row for row in rows if row.split == "train"]
    num_classes = max(row.class_id for row in rows) + 1
    train_features, noisy_labels, clean_labels = extract_train_views(
        config,
        run_dir,
        train_rows,
        num_classes,
        device,
    )
    noisy_prototypes, _ = build_prototypes(
        train_features,
        noisy_labels,
        num_classes,
    )
    clean_prototypes, _ = build_prototypes(
        train_features,
        clean_labels,
        num_classes,
    )
    concentration = np.array(
        [
            np.sum(
                train_features[clean_labels == class_id]
                * clean_prototypes[class_id],
                axis=1,
            ).mean()
            for class_id in range(num_classes)
        ]
    )
    fidelity = np.sum(noisy_prototypes * clean_prototypes, axis=1)

    test_features = np.load(run_dir / "posthoc_test_features.npy")
    test_scores = np.load(run_dir / "posthoc_test_scores.npy")
    test_labels = np.load(run_dir / "posthoc_test_labels.npy")
    test_predictions = np.load(run_dir / "posthoc_test_predictions.npy")
    rows_out = []
    for factor, values in (
        ("concentration", concentration),
        ("prototype_fidelity", fidelity),
    ):
        threshold = float(np.median(values))
        low_classes = np.flatnonzero(values <= threshold)
        high_classes = np.flatnonzero(values > threshold)
        for group, classes in (("low", low_classes), ("high", high_classes)):
            metrics = grouped_gain(
                test_features,
                test_scores,
                test_labels,
                test_predictions,
                classes,
            )
            rows_out.append(
                {
                    "run_name": run_dir.name,
                    "dataset": str(config.get("dataset", "unknown")),
                    "model": str(config.get("model", "unknown")),
                    "seed": int(config.get("seed", 0)),
                    "factor": factor,
                    "group": group,
                    "median_threshold": threshold,
                    **metrics,
                }
            )
    return rows_out


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    for factor in ("concentration", "prototype_fidelity"):
        selected = [row for row in rows if row["factor"] == factor]
        by_run: dict[str, dict[str, dict]] = {}
        for row in selected:
            by_run.setdefault(row["run_name"], {})[row["group"]] = row
        low = np.array(
            [value["low"]["relative_gain_pct"] for value in by_run.values()]
        )
        high = np.array(
            [value["high"]["relative_gain_pct"] for value in by_run.values()]
        )
        differences = low - high
        rng = np.random.default_rng(17)
        bootstrap = np.array(
            [
                differences[
                    rng.integers(0, len(differences), len(differences))
                ].mean()
                for _ in range(2000)
            ]
        )
        output.append(
            {
                "factor": factor,
                "n_runs": len(by_run),
                "low_mean_gain_pct": float(low.mean()),
                "high_mean_gain_pct": float(high.mean()),
                "low_minus_high_pct": float(differences.mean()),
                "low_minus_high_ci95": [
                    float(np.quantile(bootstrap, 0.025)),
                    float(np.quantile(bootstrap, 0.975)),
                ],
                "runs_favoring_low": int(np.sum(differences > 0)),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for run_name in args.runs:
        run_dir = args.runs_root / run_name
        if not (run_dir / "best.pt").is_file():
            continue
        rows.extend(run_partition(run_dir, device))
        print(run_name, "complete", flush=True)
    summaries = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "prototype_factor_partition.json",
        {"summaries": summaries, "rows": rows},
    )
    with (args.output_dir / "prototype_factor_partition.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Prototype factor partition",
        "",
        "Classes are split within each run at the median class-level feature "
        "concentration or noisy-clean prototype fidelity. AURC is then "
        "evaluated on held-out test samples from each class group.",
        "",
        "| Factor | Low-group gain | High-group gain | Low minus high | "
        "Bootstrap 95% CI | Runs favoring low |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['factor']} | {row['low_mean_gain_pct']:+.2f}% | "
            f"{row['high_mean_gain_pct']:+.2f}% | "
            f"{row['low_minus_high_pct']:+.2f} pp | "
            f"[{row['low_minus_high_ci95'][0]:+.2f}, "
            f"{row['low_minus_high_ci95'][1]:+.2f}] | "
            f"{row['runs_favoring_low']}/{row['n_runs']} |"
        )
    (args.output_dir / "prototype_factor_partition.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
