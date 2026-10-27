from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.posthoc_reliability import (
    aurc_from_scores,
    fit_logistic,
    risk_at_coverage,
    score_logistic,
)
from rgel.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test PMRC calibration stability.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    val_features = np.load(run_dir / "posthoc_val_features.npy")
    test_features = np.load(run_dir / "posthoc_test_features.npy")
    val_labels = np.load(run_dir / "posthoc_val_labels.npy")
    val_predictions = np.load(run_dir / "posthoc_val_predictions.npy")
    test_labels = np.load(run_dir / "posthoc_test_labels.npy")
    test_predictions = np.load(run_dir / "posthoc_test_predictions.npy")
    val_correct = val_labels == val_predictions
    test_correct = test_labels == test_predictions

    feature_groups = {
        "confidence_plus_prototype": [0, 1, 2, 3, 4, 9],
        "all": list(range(test_features.shape[1])),
    }
    fractions = (0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 1.0)
    rng = np.random.default_rng(2026)
    results: dict[str, dict] = {
        "msp": {
            "aurc": aurc_from_scores(test_features[:, 0], test_correct),
            "risk_at_90": risk_at_coverage(test_features[:, 0], test_correct, 0.9),
        },
        "prototype_margin": {
            "aurc": aurc_from_scores(test_features[:, 9], test_correct),
            "risk_at_90": risk_at_coverage(test_features[:, 9], test_correct, 0.9),
        },
    }
    for group_name, selected_features in feature_groups.items():
        selected_features = list(selected_features)
        group_results: dict[str, dict] = {}
        for fraction in fractions:
            aurcs: list[float] = []
            risks_90: list[float] = []
            repetitions = 1 if fraction == 1.0 else args.repeats
            sample_size = int(round(fraction * len(val_correct)))
            for _ in range(repetitions):
                indices = rng.choice(len(val_correct), size=sample_size, replace=False)
                if val_correct[indices].sum() == 0 or (~val_correct[indices]).sum() == 0:
                    continue
                calibrator, mean, std = fit_logistic(
                    val_features[indices][:, selected_features],
                    val_correct[indices],
                    args.weight_decay,
                )
                scores = score_logistic(
                    calibrator,
                    test_features[:, selected_features],
                    mean,
                    std,
                )
                aurcs.append(aurc_from_scores(scores, test_correct))
                risks_90.append(risk_at_coverage(scores, test_correct, 0.9))
            group_results[str(fraction)] = {
                "sample_size": sample_size,
                "samples": len(aurcs),
                "aurc": summarize(aurcs),
                "risk_at_90": summarize(risks_90),
            }
        results[group_name] = group_results

    output = {
        "run_dir": str(run_dir),
        "validation_size": int(len(val_correct)),
        "test_size": int(len(test_correct)),
        "results": results,
    }
    write_json(run_dir / "posthoc_stability.json", output)
    print(output)


if __name__ == "__main__":
    main()
