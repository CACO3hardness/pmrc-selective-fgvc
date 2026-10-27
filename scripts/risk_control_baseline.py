from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgel.utils import read_json, write_json
from scripts.risk_constrained_coverage import (
    evaluate_threshold,
    select_threshold,
)
from scripts.summarize_results import infer_noise


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT.parent.parent / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split Learn-then-Test risk-control baseline."
    )
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target-risks",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15],
    )
    parser.add_argument("--delta", type=float, default=0.10)
    return parser.parse_args()


def ltt_threshold(
    scores: np.ndarray,
    correct: np.ndarray,
    target_risk: float,
    delta: float,
) -> dict[str, float]:
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_correct = correct[order]
    cumulative_errors = np.cumsum(~sorted_correct)
    group_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    sample_sizes = group_ends + 1
    errors = cumulative_errors[group_ends]
    candidate_count = len(group_ends)
    family_confidence = 1.0 - delta / candidate_count
    upper_bounds = beta.ppf(
        family_confidence,
        errors + 1,
        sample_sizes - errors,
    )
    feasible = np.flatnonzero(upper_bounds <= target_risk)
    if len(feasible) == 0:
        return {
            "threshold": float("inf"),
            "calibration_coverage": 0.0,
            "calibration_risk": None,
            "calibration_upper_bound": None,
            "n_candidates": float(candidate_count),
        }
    end = int(feasible[-1])
    return {
        "threshold": float(sorted_scores[end]),
        "calibration_coverage": float(sample_sizes[end] / len(scores)),
        "calibration_risk": float(errors[end] / sample_sizes[end]),
        "calibration_upper_bound": float(upper_bounds[end]),
        "n_candidates": float(candidate_count),
    }


def main() -> None:
    args = parse_args()
    rows = []
    for posthoc_path in sorted(args.runs_root.glob("*/posthoc_reliability.json")):
        run_dir = posthoc_path.parent
        required = (
            "config.json",
            "posthoc_val_features.npy",
            "posthoc_val_labels.npy",
            "posthoc_val_predictions.npy",
            "posthoc_test_features.npy",
            "posthoc_test_labels.npy",
            "posthoc_test_predictions.npy",
            "posthoc_test_scores.npy",
        )
        if not all((run_dir / name).is_file() for name in required):
            continue
        config = read_json(run_dir / "config.json")
        val_features = np.load(run_dir / "posthoc_val_features.npy")
        val_labels = np.load(run_dir / "posthoc_val_labels.npy")
        val_predictions = np.load(run_dir / "posthoc_val_predictions.npy")
        test_features = np.load(run_dir / "posthoc_test_features.npy")
        test_labels = np.load(run_dir / "posthoc_test_labels.npy")
        test_predictions = np.load(run_dir / "posthoc_test_predictions.npy")
        pmrc_scores = np.load(run_dir / "posthoc_test_scores.npy")
        val_correct = val_predictions == val_labels
        test_correct = test_predictions == test_labels
        noise_kind, requested_rate, _ = infer_noise(config)
        for target_risk in args.target_risks:
            for method, val_scores, test_scores in (
                ("msp", val_features[:, 0], test_features[:, 0]),
                ("pmrc", None, pmrc_scores),
            ):
                if method == "pmrc":
                    selected_decay = float(
                        read_json(posthoc_path)["selected_weight_decay"]
                    )
                    from scripts.posthoc_reliability import (
                        fit_logistic,
                        score_logistic,
                    )

                    calibrator, mean, std = fit_logistic(
                        val_features,
                        val_correct,
                        selected_decay,
                    )
                    val_scores = score_logistic(
                        calibrator,
                        val_features,
                        mean,
                        std,
                    )
                assert val_scores is not None
                ltt = ltt_threshold(
                    val_scores,
                    val_correct,
                    target_risk,
                    args.delta,
                )
                _, test_coverage, test_risk = evaluate_threshold(
                    test_scores,
                    test_correct,
                    ltt["threshold"],
                )
                empirical_threshold, empirical_calibration_coverage, empirical_calibration_risk = (
                    select_threshold(
                        val_scores,
                        val_correct,
                        target_risk,
                    )
                )
                _, empirical_test_coverage, empirical_test_risk = evaluate_threshold(
                    test_scores,
                    test_correct,
                    empirical_threshold,
                )
                if test_coverage == 0.0:
                    test_risk = None
                if empirical_test_coverage == 0.0:
                    empirical_test_risk = None
                rows.append(
                    {
                        "run_name": run_dir.name,
                        "dataset": str(config.get("dataset", "unknown")),
                        "method": str(config.get("method", "unknown")),
                        "model": str(config.get("model", "unknown")),
                        "noise_kind": noise_kind,
                        "requested_noise_rate": requested_rate,
                        "seed": int(config.get("seed", 0)),
                        "target_risk": target_risk,
                        "delta": args.delta,
                        "score": method,
                        "ltt_threshold": ltt["threshold"],
                        "ltt_calibration_coverage": ltt["calibration_coverage"],
                        "ltt_calibration_risk": ltt["calibration_risk"],
                        "ltt_calibration_upper_bound": ltt[
                            "calibration_upper_bound"
                        ],
                        "ltt_test_coverage": test_coverage,
                        "ltt_test_risk": test_risk,
                        "empirical_calibration_threshold": empirical_threshold,
                        "empirical_calibration_coverage": empirical_calibration_coverage,
                        "empirical_calibration_risk": empirical_calibration_risk,
                        "empirical_test_coverage": empirical_test_coverage,
                        "empirical_test_risk": empirical_test_risk,
                    }
                )
        print(f"{run_dir.name}: complete", flush=True)

    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["target_risk"])].append(row)
    summaries = []
    for (dataset, target_risk), group_rows in sorted(groups.items()):
        for score in ("msp", "pmrc"):
            selected = [row for row in group_rows if row["score"] == score]
            ltt_risks = [
                row["ltt_test_risk"]
                for row in selected
                if row["ltt_test_risk"] is not None
            ]
            empirical_risks = [
                row["empirical_test_risk"]
                for row in selected
                if row["empirical_test_risk"] is not None
            ]
            summaries.append(
                {
                    "dataset": dataset,
                    "target_risk": target_risk,
                    "score": score,
                    "n": len(selected),
                    "ltt_test_coverage_mean": float(
                        np.mean([row["ltt_test_coverage"] for row in selected])
                    ),
                    "ltt_test_risk_mean": (
                        float(np.mean(ltt_risks)) if ltt_risks else None
                    ),
                    "empirical_test_coverage_mean": float(
                        np.mean(
                            [row["empirical_test_coverage"] for row in selected]
                        )
                    ),
                    "empirical_test_risk_mean": (
                        float(np.mean(empirical_risks))
                        if empirical_risks
                        else None
                    ),
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "risk_control_baseline.json",
        {
            "target_risks": args.target_risks,
            "delta": args.delta,
            "n_rows": len(rows),
            "rows": rows,
            "summaries": summaries,
        },
    )
    with (args.output_dir / "risk_control_baseline.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote risk-control results for {len(rows) // 6} runs.")


if __name__ == "__main__":
    main()
