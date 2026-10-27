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
from scripts.posthoc_reliability import aurc_from_scores, risk_at_coverage
from scripts.summarize_results import infer_noise


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT.parent.parent / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decision-cost analysis at different rejection costs."
    )
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--rejection-costs",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5, 1.0],
    )
    return parser.parse_args()


def choose_cost_threshold(
    scores: np.ndarray,
    correct: np.ndarray,
    rejection_cost: float,
) -> tuple[float, float, float, float]:
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_correct = correct[order]
    cumulative_errors = np.cumsum(~sorted_correct)
    coverage = np.arange(1, len(scores) + 1) / len(scores)
    risk = cumulative_errors / np.arange(1, len(scores) + 1)
    cost = (1.0 - coverage) * rejection_cost + coverage * risk
    min_cost = float(cost.min())
    candidates = np.flatnonzero(np.isclose(cost, min_cost))
    index = int(candidates[-1])
    return (
        float(sorted_scores[index]),
        float(coverage[index]),
        float(risk[index]),
        min_cost,
    )


def evaluate_cost(
    scores: np.ndarray,
    correct: np.ndarray,
    threshold: float,
    rejection_cost: float,
) -> tuple[float, float, float]:
    accepted = scores >= threshold
    coverage = float(accepted.mean())
    risk = float(1.0 - correct[accepted].mean()) if accepted.any() else 0.0
    cost = (1.0 - coverage) * rejection_cost + coverage * risk
    return coverage, risk, cost


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
            "posthoc_val_scores.npy",
            "posthoc_test_scores.npy",
        )
        if not all((run_dir / name).is_file() for name in required):
            continue
        config = read_json(run_dir / "config.json")
        val_labels = np.load(run_dir / "posthoc_val_labels.npy")
        val_predictions = np.load(run_dir / "posthoc_val_predictions.npy")
        test_features = np.load(run_dir / "posthoc_test_features.npy")
        test_labels = np.load(run_dir / "posthoc_test_labels.npy")
        test_predictions = np.load(run_dir / "posthoc_test_predictions.npy")
        val_scores = {
            "msp": np.load(run_dir / "posthoc_val_features.npy")[:, 0],
            "pmrc": np.load(run_dir / "posthoc_val_scores.npy"),
        }
        test_scores = {
            "msp": test_features[:, 0],
            "pmrc": np.load(run_dir / "posthoc_test_scores.npy"),
        }
        val_correct = val_predictions == val_labels
        test_correct = test_predictions == test_labels
        noise_kind, requested_rate, _ = infer_noise(config)
        for rejection_cost in args.rejection_costs:
            for method in ("msp", "pmrc"):
                threshold, val_coverage, val_risk, val_cost = choose_cost_threshold(
                    val_scores[method],
                    val_correct,
                    rejection_cost,
                )
                test_coverage, test_risk, test_cost = evaluate_cost(
                    test_scores[method],
                    test_correct,
                    threshold,
                    rejection_cost,
                )
                rows.append(
                    {
                        "run_name": run_dir.name,
                        "dataset": str(config.get("dataset", "unknown")),
                        "method": str(config.get("method", "unknown")),
                        "model": str(config.get("model", "unknown")),
                        "noise_kind": noise_kind,
                        "requested_noise_rate": requested_rate,
                        "seed": int(config.get("seed", 0)),
                        "rejection_cost": rejection_cost,
                        "score": method,
                        "threshold": threshold,
                        "validation_coverage": val_coverage,
                        "validation_risk": val_risk,
                        "validation_cost": val_cost,
                        "test_coverage": test_coverage,
                        "test_risk": test_risk,
                        "test_cost": test_cost,
                        "test_aurc": aurc_from_scores(
                            test_scores[method],
                            test_correct,
                        ),
                        "test_risk90": risk_at_coverage(
                            test_scores[method],
                            test_correct,
                            0.9,
                        ),
                    }
                )
        print(f"{run_dir.name}: complete", flush=True)

    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["rejection_cost"])].append(row)
    summaries = []
    for (dataset, rejection_cost), group_rows in sorted(groups.items()):
        msp = [row for row in group_rows if row["score"] == "msp"]
        pmrc = [row for row in group_rows if row["score"] == "pmrc"]
        by_name = {row["run_name"]: row for row in msp}
        differences = [
            by_name[row["run_name"]]["test_cost"] - row["test_cost"]
            for row in pmrc
        ]
        summaries.append(
            {
                "dataset": dataset,
                "rejection_cost": rejection_cost,
                "n": len(pmrc),
                "msp_cost_mean": float(
                    np.mean([row["test_cost"] for row in msp])
                ),
                "pmrc_cost_mean": float(
                    np.mean([row["test_cost"] for row in pmrc])
                ),
                "mean_cost_reduction": float(np.mean(differences)),
                "pmrc_lower_count": int(np.sum(np.asarray(differences) > 0)),
                "msp_coverage_mean": float(
                    np.mean([row["test_coverage"] for row in msp])
                ),
                "pmrc_coverage_mean": float(
                    np.mean([row["test_coverage"] for row in pmrc])
                ),
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "decision_cost_analysis.json",
        {
            "rejection_costs": args.rejection_costs,
            "n_rows": len(rows),
            "rows": rows,
            "summaries": summaries,
        },
    )
    with (args.output_dir / "decision_cost_analysis.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Decision-cost analysis",
        "",
        "Error cost is normalized to one. Rejection cost is the cost of abstaining",
        "relative to one classification error. Thresholds are selected on",
        "calibration data and evaluated on fixed test data.",
        "",
        "| Rejection cost | Evaluations | Mean MSP cost | Mean PMRC cost | "
        "PMRC lower | MSP coverage | PMRC coverage |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    overall = []
    for rejection_cost in args.rejection_costs:
        selected = [
            row for row in rows if row["rejection_cost"] == rejection_cost
        ]
        msp = {row["run_name"]: row for row in selected if row["score"] == "msp"}
        pmrc = {
            row["run_name"]: row for row in selected if row["score"] == "pmrc"
        }
        differences = [
            msp[name]["test_cost"] - pmrc[name]["test_cost"]
            for name in sorted(set(msp) & set(pmrc))
        ]
        overall.append(
            {
                "rejection_cost": rejection_cost,
                "msp_cost_mean": float(
                    np.mean([row["test_cost"] for row in msp.values()])
                ),
                "pmrc_cost_mean": float(
                    np.mean([row["test_cost"] for row in pmrc.values()])
                ),
                "mean_cost_reduction": float(np.mean(differences)),
                "pmrc_lower_count": int(np.sum(np.asarray(differences) > 0)),
                "n": len(differences),
                "msp_coverage_mean": float(
                    np.mean([row["test_coverage"] for row in msp.values()])
                ),
                "pmrc_coverage_mean": float(
                    np.mean([row["test_coverage"] for row in pmrc.values()])
                ),
            }
        )
        lines.append(
            f"| {rejection_cost:.2f} | {len(differences)} | "
            f"{overall[-1]['msp_cost_mean']:.4f} | "
            f"{overall[-1]['pmrc_cost_mean']:.4f} | "
            f"{overall[-1]['pmrc_lower_count']}/{len(differences)} | "
            f"{overall[-1]['msp_coverage_mean']:.1%} | "
            f"{overall[-1]['pmrc_coverage_mean']:.1%} |"
        )
    (args.output_dir / "decision_cost_analysis.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} decision-cost evaluations.")


if __name__ == "__main__":
    main()
