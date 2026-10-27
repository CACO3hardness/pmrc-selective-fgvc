from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rgel.utils import read_json, write_json
from scripts.posthoc_reliability import aurc_from_scores


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT.parent.parent / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MSP and PMRC coverage under risk constraints."
    )
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target-risks",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15],
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def select_threshold(
    scores: np.ndarray,
    correct: np.ndarray,
    target_risk: float,
) -> tuple[float, float, float]:
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_correct = correct[order]
    cumulative_risk = np.cumsum(~sorted_correct) / np.arange(
        1, len(correct) + 1
    )

    group_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    feasible = group_ends[cumulative_risk[group_ends] <= target_risk]
    if len(feasible) == 0:
        return float("inf"), 0.0, 0.0

    end = int(feasible[-1])
    threshold = float(sorted_scores[end])
    accepted = scores >= threshold
    coverage = float(accepted.mean())
    risk = float(1.0 - correct[accepted].mean()) if accepted.any() else 0.0
    return threshold, coverage, risk


def evaluate_threshold(
    scores: np.ndarray,
    correct: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, float, float]:
    if np.isinf(threshold):
        accepted = np.zeros(len(scores), dtype=bool)
        return accepted, 0.0, 0.0
    accepted = scores >= threshold
    coverage = float(accepted.mean())
    risk = float(1.0 - correct[accepted].mean()) if accepted.any() else 0.0
    return accepted, coverage, risk


def paired_coverage_ci(
    accepted_msp: np.ndarray,
    accepted_pmrc: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> list[float]:
    differences = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sample = rng.integers(0, len(accepted_pmrc), len(accepted_pmrc))
        differences[index] = (
            accepted_pmrc[sample].mean() - accepted_msp[sample].mean()
        )
    low, high = np.quantile(differences, [0.025, 0.975])
    return [float(low), float(high)]


def load_run(run_dir: Path) -> dict | None:
    required = (
        "config.json",
        "posthoc_reliability.json",
        "posthoc_val_features.npy",
        "posthoc_test_features.npy",
        "posthoc_val_scores.npy",
        "posthoc_test_scores.npy",
        "posthoc_val_predictions.npy",
        "posthoc_test_predictions.npy",
        "posthoc_val_labels.npy",
        "posthoc_test_labels.npy",
    )
    if not all((run_dir / name).is_file() for name in required):
        return None

    config = read_json(run_dir / "config.json")
    posthoc = read_json(run_dir / "posthoc_reliability.json")
    val_correct = (
        np.load(run_dir / "posthoc_val_predictions.npy")
        == np.load(run_dir / "posthoc_val_labels.npy")
    )
    test_correct = (
        np.load(run_dir / "posthoc_test_predictions.npy")
        == np.load(run_dir / "posthoc_test_labels.npy")
    )
    scores = {
        "msp": (
            np.load(run_dir / "posthoc_val_features.npy")[:, 0],
            np.load(run_dir / "posthoc_test_features.npy")[:, 0],
        ),
        "pmrc": (
            np.load(run_dir / "posthoc_val_scores.npy"),
            np.load(run_dir / "posthoc_test_scores.npy"),
        ),
    }
    return {
        "run_name": run_dir.name,
        "dataset": str(config.get("dataset", "unknown")),
        "seed": int(config.get("seed", 0)),
        "test_accuracy": float(posthoc["test_accuracy"]),
        "val_correct": val_correct,
        "test_correct": test_correct,
        "scores": scores,
        "test_aurc": {
            "msp": aurc_from_scores(scores["msp"][1], test_correct),
            "pmrc": aurc_from_scores(scores["pmrc"][1], test_correct),
        },
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict], targets: list[float]) -> None:
    lines = [
        "# Risk-constrained selective coverage",
        "",
        "Thresholds are selected on the clean calibration split. Test coverage and",
        "risk are then measured on the untouched test split. Coverage gain is PMRC",
        "minus MSP; its CI resamples test items with both thresholds held fixed.",
        "",
        "| Run | Target risk | MSP coverage | MSP risk | PMRC coverage | PMRC risk | "
        "Coverage gain [95% CI] |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for target in targets:
        selected = [row for row in rows if row["target_risk"] == target]
        for row in sorted(selected, key=lambda item: item["run_name"]):
            lines.append(
                f"| {row['run_name']} | {100 * target:.0f}% | "
                f"{100 * row['msp_test_coverage']:.1f}% | "
                f"{100 * row['msp_test_risk']:.1f}% | "
                f"{100 * row['pmrc_test_coverage']:.1f}% | "
                f"{100 * row['pmrc_test_risk']:.1f}% | "
                f"{100 * row['pmrc_minus_msp_coverage']:+.1f} pp "
                f"[{100 * row['coverage_gain_ci95'][0]:+.1f}, "
                f"{100 * row['coverage_gain_ci95'][1]:+.1f}] |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(17)
    rows = []
    missing = []
    for run_dir in sorted(args.runs_root.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "posthoc_reliability.json").is_file():
            continue
        run = load_run(run_dir)
        if run is None:
            missing.append(run_dir.name)
            continue
        for target in args.target_risks:
            thresholds = {}
            thresholds["msp"] = select_threshold(
                run["scores"]["msp"][0],
                run["val_correct"],
                target,
            )
            thresholds["pmrc"] = select_threshold(
                run["scores"]["pmrc"][0],
                run["val_correct"],
                target,
            )
            accepted = {}
            coverage = {}
            test_risk = {}
            for method in ("msp", "pmrc"):
                accepted[method], coverage[method], test_risk[method] = (
                    evaluate_threshold(
                        run["scores"][method][1],
                        run["test_correct"],
                        thresholds[method][0],
                    )
                )
            rows.append(
                {
                    "run_name": run["run_name"],
                    "dataset": run["dataset"],
                    "seed": run["seed"],
                    "target_risk": float(target),
                    "msp_val_threshold": thresholds["msp"][0],
                    "msp_val_coverage": thresholds["msp"][1],
                    "msp_val_risk": thresholds["msp"][2],
                    "pmrc_val_threshold": thresholds["pmrc"][0],
                    "pmrc_val_coverage": thresholds["pmrc"][1],
                    "pmrc_val_risk": thresholds["pmrc"][2],
                    "msp_test_coverage": coverage["msp"],
                    "msp_test_risk": test_risk["msp"],
                    "pmrc_test_coverage": coverage["pmrc"],
                    "pmrc_test_risk": test_risk["pmrc"],
                    "pmrc_minus_msp_coverage": coverage["pmrc"] - coverage["msp"],
                    "coverage_gain_ci95": paired_coverage_ci(
                        accepted["msp"],
                        accepted["pmrc"],
                        rng,
                        args.bootstrap,
                    ),
                    "msp_test_aurc": run["test_aurc"]["msp"],
                    "pmrc_test_aurc": run["test_aurc"]["pmrc"],
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    serializable = []
    for row in rows:
        item = dict(row)
        coverage_ci = item.pop("coverage_gain_ci95")
        item["coverage_gain_ci95_low"] = coverage_ci[0]
        item["coverage_gain_ci95_high"] = coverage_ci[1]
        serializable.append(item)
    write_json(
        args.output_dir / "risk_constrained_coverage.json",
        {
            "target_risks": args.target_risks,
            "n_runs": len(serializable) // len(args.target_risks),
            "missing_runs": missing,
            "rows": serializable,
        },
    )
    write_csv(args.output_dir / "risk_constrained_coverage.csv", serializable)
    write_markdown(
        args.output_dir / "risk_constrained_coverage.md",
        rows,
        args.target_risks,
    )
    print(
        f"Wrote {len(rows)} risk constraints from "
        f"{len(serializable) // len(args.target_risks)} runs."
    )


if __name__ == "__main__":
    main()
