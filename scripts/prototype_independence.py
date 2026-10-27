from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rgel.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether prototype margin adds information beyond MSP."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def roc_auc(scores: np.ndarray, correct: np.ndarray) -> float:
    correct = correct.astype(bool)
    positives = int(correct.sum())
    negatives = int((~correct).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = ranks[correct].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def stratified_auc(
    confidence: np.ndarray,
    scores: np.ndarray,
    correct: np.ndarray,
    bins: int,
) -> dict:
    edges = np.quantile(confidence, np.linspace(0.0, 1.0, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    details = []
    weighted_sum = 0.0
    total = 0
    for index in range(bins):
        mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        count = int(mask.sum())
        auc = roc_auc(scores[mask], correct[mask]) if count else float("nan")
        if np.isfinite(auc):
            weighted_sum += auc * count
            total += count
        details.append(
            {
                "bin": index,
                "low": float(edges[index]),
                "high": float(edges[index + 1]),
                "n": count,
                "accuracy": float(correct[mask].mean()) if count else None,
                "auc": float(auc) if np.isfinite(auc) else None,
            }
        )
    return {
        "weighted_auc": float(weighted_sum / total) if total else None,
        "samples_used": total,
        "bins": details,
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    features = np.load(run_dir / "posthoc_test_features.npy")
    labels = np.load(run_dir / "posthoc_test_labels.npy")
    predictions = np.load(run_dir / "posthoc_test_predictions.npy")
    correct = predictions == labels
    msp = features[:, 0]
    prototype_margin = features[:, 9]

    result = {
        "run_dir": str(run_dir),
        "n": len(correct),
        "accuracy": float(correct.mean()),
        "overall_auc": {
            "msp": roc_auc(msp, correct),
            "prototype_margin": roc_auc(prototype_margin, correct),
        },
        "stratified_by_msp": stratified_auc(
            msp,
            prototype_margin,
            correct,
            args.bins,
        ),
        "pearson_correlation": {
            "msp_vs_prototype_margin": float(np.corrcoef(msp, prototype_margin)[0, 1]),
            "prototype_margin_vs_correct": float(
                np.corrcoef(prototype_margin, correct.astype(float))[0, 1]
            ),
        },
    }

    rng = np.random.default_rng(7)
    bootstrap_values = []
    for _ in range(args.bootstrap):
        sample = rng.integers(0, len(correct), len(correct))
        value = stratified_auc(
            msp[sample],
            prototype_margin[sample],
            correct[sample],
            args.bins,
        )["weighted_auc"]
        if value is not None:
            bootstrap_values.append(value)
    result["stratified_by_msp"]["weighted_auc_ci95"] = [
        float(np.quantile(bootstrap_values, 0.025)),
        float(np.quantile(bootstrap_values, 0.975)),
    ]
    write_json(run_dir / "prototype_independence.json", result)
    print(result)


if __name__ == "__main__":
    main()
