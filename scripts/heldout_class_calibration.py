from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgel.utils import read_json, write_json
from scripts.calibration_robustness import (
    fit_and_score,
    normalized_dataset,
)
from scripts.posthoc_reliability import aurc_from_scores, risk_at_coverage
from scripts.summarize_results import infer_noise


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT.parent.parent / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate calibration and gate transfer to unseen classes."
    )
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--runs", nargs="+", default=None)
    parser.add_argument(
        "--heldout-fractions",
        type=float,
        nargs="+",
        default=[0.2, 0.4],
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bins", type=int, default=5)
    return parser.parse_args()


def evaluate_run(
    run_dir: Path,
    heldout_fractions: list[float],
    repeats: int,
    bins: int,
) -> list[dict]:
    required = (
        "config.json",
        "posthoc_reliability.json",
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
    val_correct = val_predictions == val_labels
    classes = np.unique(np.concatenate((val_labels, test_labels)))
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

    rows = []
    for heldout_fraction in heldout_fractions:
        for repeat in range(repeats):
            rng = np.random.default_rng(
                300000
                + 1000 * base["seed"]
                + int(heldout_fraction * 1000)
                + repeat
            )
            heldout_count = max(1, int(round(heldout_fraction * len(classes))))
            heldout_classes = rng.choice(
                classes,
                size=heldout_count,
                replace=False,
            )
            calibration = np.flatnonzero(~np.isin(val_labels, heldout_classes))
            evaluation = np.flatnonzero(np.isin(test_labels, heldout_classes))
            if len(calibration) == 0 or len(evaluation) == 0:
                continue
            calibration_correct = val_correct[calibration]
            evaluation_correct = (
                test_predictions[evaluation] == test_labels[evaluation]
            )
            result = fit_and_score(
                val_features[calibration],
                calibration_correct,
                test_features[evaluation],
                weight_decay,
                bins,
            )
            if result is None:
                continue
            pmrc_scores = np.asarray(result["pmrc_scores"])
            msp_scores = test_features[evaluation, 0]
            gated_scores = (
                pmrc_scores
                if result["gate_selected_method"] == "pmrc"
                else msp_scores
            )
            rows.append(
                {
                    **base,
                    "heldout_fraction": heldout_fraction,
                    "heldout_classes": int(len(heldout_classes)),
                    "repeat": repeat,
                    "validation_size": int(len(calibration)),
                    "evaluation_size": int(len(evaluation)),
                    "validation_gate_auc": result["validation_gate_auc"],
                    "gate_selected_method": result["gate_selected_method"],
                    "msp_aurc": aurc_from_scores(msp_scores, evaluation_correct),
                    "pmrc_aurc": aurc_from_scores(pmrc_scores, evaluation_correct),
                    "gated_aurc": aurc_from_scores(gated_scores, evaluation_correct),
                    "msp_risk90": risk_at_coverage(
                        msp_scores, evaluation_correct, 0.9
                    ),
                    "pmrc_risk90": risk_at_coverage(
                        pmrc_scores, evaluation_correct, 0.9
                    ),
                    "gated_risk90": risk_at_coverage(
                        gated_scores, evaluation_correct, 0.9
                    ),
                }
            )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["heldout_fraction"])].append(row)
    summaries = []
    for (dataset, fraction), group_rows in sorted(groups.items()):
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
                "heldout_fraction": fraction,
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
    run_dirs = (
        [args.runs_root / name for name in args.runs]
        if args.runs
        else sorted(path for path in args.runs_root.iterdir() if path.is_dir())
    )
    rows = []
    for index, run_dir in enumerate(run_dirs, start=1):
        if not (run_dir / "posthoc_reliability.json").is_file():
            continue
        run_rows = evaluate_run(
            run_dir,
            args.heldout_fractions,
            args.repeats,
            args.bins,
        )
        rows.extend(run_rows)
        print(f"[{index}/{len(run_dirs)}] {run_dir.name}: {len(run_rows)} splits", flush=True)

    summaries = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "heldout_class_calibration.json",
        {
            "heldout_fractions": args.heldout_fractions,
            "repeats": args.repeats,
            "n_rows": len(rows),
            "rows": rows,
            "summaries": summaries,
        },
    )
    with (args.output_dir / "heldout_class_calibration.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} class-holdout evaluations.")


if __name__ == "__main__":
    main()
