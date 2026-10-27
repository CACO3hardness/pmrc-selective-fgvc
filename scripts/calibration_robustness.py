from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgel.utils import read_json, write_json
from scripts.posthoc_reliability import (
    aurc_from_scores,
    fit_logistic,
    risk_at_coverage,
    score_logistic,
)
from scripts.prototype_independence import stratified_auc
from scripts.summarize_results import infer_noise


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT.parent.parent / "outputs"
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stress-test PMRC calibration with corrupted or incomplete data."
    )
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--runs", nargs="+", default=None)
    parser.add_argument(
        "--label-noise",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20],
    )
    parser.add_argument(
        "--missing-classes",
        type=float,
        nargs="+",
        default=[0.10, 0.25, 0.50],
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bins", type=int, default=5)
    return parser.parse_args()


def fit_and_score(
    val_features: np.ndarray,
    val_correct: np.ndarray,
    test_features: np.ndarray,
    weight_decay: float,
    bins: int,
) -> dict[str, object] | None:
    if not val_correct.any() or val_correct.all():
        return None
    calibrator, mean, std = fit_logistic(
        val_features,
        val_correct,
        weight_decay,
    )
    pmrc_scores = score_logistic(
        calibrator,
        test_features,
        mean,
        std,
    )
    diagnostic = stratified_auc(
        val_features[:, 0],
        val_features[:, 9],
        val_correct,
        bins,
    )
    gate_auc = diagnostic["weighted_auc"]
    use_pmrc = gate_auc is not None and gate_auc > 0.5
    return {
        "pmrc_scores": pmrc_scores,
        "selected_weight_decay": weight_decay,
        "validation_gate_auc": gate_auc,
        "gate_selected_method": "pmrc" if use_pmrc else "msp",
    }


def corrupt_labels(
    labels: np.ndarray,
    rate: float,
    num_classes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    corrupted = labels.copy()
    count = int(round(rate * len(labels)))
    positions = rng.choice(len(labels), size=count, replace=False)
    for position in positions:
        alternatives = np.delete(np.arange(num_classes), labels[position])
        corrupted[position] = int(rng.choice(alternatives))
    return corrupted


def normalized_dataset(config: dict) -> str:
    dataset = str(config.get("dataset", "unknown")).lower()
    if dataset in {"cub", "cub200"}:
        return "cub"
    if dataset.startswith("aircraft"):
        return "aircraft"
    if dataset.startswith("flowers"):
        return "flowers102"
    if dataset.startswith("webfg_"):
        return dataset
    return dataset


def condition_rows(
    run_dir: Path,
    repeats: int,
    label_noise_rates: list[float],
    missing_class_rates: list[float],
    bins: int,
) -> list[dict]:
    required = (
        "config.json",
        "posthoc_val_features.npy",
        "posthoc_val_labels.npy",
        "posthoc_val_predictions.npy",
        "posthoc_test_features.npy",
        "posthoc_test_labels.npy",
        "posthoc_test_predictions.npy",
    )
    if not all((run_dir / name).is_file() for name in required):
        return []
    config = read_json(run_dir / "config.json")
    posthoc = read_json(run_dir / "posthoc_reliability.json")
    weight_decay = float(posthoc["selected_weight_decay"])
    val_features = np.load(run_dir / "posthoc_val_features.npy")
    val_labels = np.load(run_dir / "posthoc_val_labels.npy")
    val_predictions = np.load(run_dir / "posthoc_val_predictions.npy")
    test_features = np.load(run_dir / "posthoc_test_features.npy")
    test_labels = np.load(run_dir / "posthoc_test_labels.npy")
    test_predictions = np.load(run_dir / "posthoc_test_predictions.npy")
    test_correct = test_predictions == test_labels
    num_classes = int(
        max(
            len(np.unique(val_labels)),
            len(np.unique(test_labels)),
        )
    )
    noise_kind, requested_rate, _ = infer_noise(config)
    base = {
        "run_name": run_dir.name,
        "dataset": normalized_dataset(config),
        "method": str(config.get("method", "unknown")),
        "model": str(config.get("model", "unknown")),
        "noise_kind": noise_kind,
        "requested_noise_rate": requested_rate,
        "seed": int(config.get("seed", 0)),
    }
    conditions: list[tuple[str, float, int, np.ndarray | None, np.ndarray | None]] = [
        ("clean", 0.0, 0, None, None)
    ]
    for noise_rate in label_noise_rates:
        for repeat in range(repeats):
            conditions.append(("label_noise", noise_rate, repeat, None, None))
    for missing_rate in missing_class_rates:
        for repeat in range(repeats):
            conditions.append(("missing_classes", missing_rate, repeat, None, None))

    rows = []
    classes = np.unique(val_labels)
    for condition, rate, repeat, _, _ in conditions:
        condition_seed = (
            1000 * (base["seed"] + 1)
            + int(rate * 1000)
            + repeat
            + (0 if condition == "clean" else 100000 if condition == "label_noise" else 200000)
        )
        rng = np.random.default_rng(condition_seed)
        if condition == "clean":
            selected = np.arange(len(val_labels))
            calibration_labels = val_labels
        elif condition == "label_noise":
            selected = np.arange(len(val_labels))
            calibration_labels = corrupt_labels(
                val_labels,
                rate,
                num_classes,
                rng,
            )
        else:
            drop_count = int(round(rate * len(classes)))
            dropped_classes = rng.choice(classes, size=drop_count, replace=False)
            selected = np.flatnonzero(~np.isin(val_labels, dropped_classes))
            calibration_labels = val_labels[selected]

        calibration_correct = val_predictions[selected] == calibration_labels
        result = fit_and_score(
            val_features[selected],
            calibration_correct,
            test_features,
            weight_decay,
            bins,
        )
        if result is None:
            continue
        pmrc_scores = np.asarray(result["pmrc_scores"])
        msp_scores = test_features[:, 0]
        gated_scores = (
            pmrc_scores
            if result["gate_selected_method"] == "pmrc"
            else msp_scores
        )
        rows.append(
            {
                **base,
                "condition": condition,
                "perturbation_rate": rate,
                "repeat": repeat,
                "validation_size": int(len(selected)),
                "validation_positive_rate": float(calibration_correct.mean()),
                "selected_weight_decay": result["selected_weight_decay"],
                "validation_gate_auc": result["validation_gate_auc"],
                "gate_selected_method": result["gate_selected_method"],
                "msp_aurc": aurc_from_scores(msp_scores, test_correct),
                "pmrc_aurc": aurc_from_scores(pmrc_scores, test_correct),
                "gated_aurc": aurc_from_scores(gated_scores, test_correct),
                "msp_risk90": risk_at_coverage(msp_scores, test_correct, 0.9),
                "pmrc_risk90": risk_at_coverage(
                    pmrc_scores, test_correct, 0.9
                ),
                "gated_risk90": risk_at_coverage(
                    gated_scores, test_correct, 0.9
                ),
            }
        )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, float], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["condition"], row["perturbation_rate"])].append(row)
    summaries = []
    for (dataset, condition, rate), group_rows in sorted(groups.items()):
        pmrc_relative = [
            100.0 * (row["msp_aurc"] - row["pmrc_aurc"]) / row["msp_aurc"]
            for row in group_rows
        ]
        gated_relative = [
            100.0 * (row["msp_aurc"] - row["gated_aurc"]) / row["msp_aurc"]
            for row in group_rows
        ]
        summaries.append(
            {
                "dataset": dataset,
                "condition": condition,
                "perturbation_rate": rate,
                "n_evaluations": len(group_rows),
                "msp_aurc_mean": float(
                    np.mean([row["msp_aurc"] for row in group_rows])
                ),
                "pmrc_aurc_mean": float(
                    np.mean([row["pmrc_aurc"] for row in group_rows])
                ),
                "gated_aurc_mean": float(
                    np.mean([row["gated_aurc"] for row in group_rows])
                ),
                "pmrc_relative_improvement_pct_mean": float(
                    np.mean(pmrc_relative)
                ),
                "gated_relative_improvement_pct_mean": float(
                    np.mean(gated_relative)
                ),
                "pmrc_success_rate": float(
                    np.mean([value > 0 for value in pmrc_relative])
                ),
                "gated_success_rate": float(
                    np.mean([value > 0 for value in gated_relative])
                ),
                "pmrc_nonregression_rate": float(
                    np.mean([value >= 0 for value in pmrc_relative])
                ),
                "gated_nonregression_rate": float(
                    np.mean([value >= 0 for value in gated_relative])
                ),
                "gate_pmrc_selection_rate": float(
                    np.mean(
                        [row["gate_selected_method"] == "pmrc" for row in group_rows]
                    )
                ),
            }
        )
    return summaries


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    run_dirs = []
    if args.runs:
        run_dirs = [args.runs_root / name for name in args.runs]
    else:
        run_dirs = sorted(
            [path for path in args.runs_root.iterdir() if path.is_dir()]
        )
    rows = []
    for index, run_dir in enumerate(run_dirs, start=1):
        if not (run_dir / "posthoc_reliability.json").is_file():
            continue
        new_rows = condition_rows(
            run_dir,
            args.repeats,
            args.label_noise,
            args.missing_classes,
            args.bins,
        )
        rows.extend(new_rows)
        print(f"[{index}/{len(run_dirs)}] {run_dir.name}: {len(new_rows)} conditions", flush=True)

    summaries = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "calibration_robustness.json",
        {
            "label_noise": args.label_noise,
            "missing_classes": args.missing_classes,
            "repeats": args.repeats,
            "n_rows": len(rows),
            "rows": rows,
            "summaries": summaries,
        },
    )
    with (args.output_dir / "calibration_robustness.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} condition evaluations.")


if __name__ == "__main__":
    main()
