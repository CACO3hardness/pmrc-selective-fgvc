from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rgel.utils import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare rejection of semantic-neighbor and other errors."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def rejection_mask(scores: np.ndarray, coverage: float = 0.9) -> np.ndarray:
    accepted = max(0, min(len(scores), int(np.ceil(coverage * len(scores)))))
    order = np.argsort(-scores, kind="stable")
    rejected = np.zeros(len(scores), dtype=bool)
    rejected[order[accepted:]] = True
    return rejected


def fraction(mask: np.ndarray) -> float:
    return float(mask.mean()) if len(mask) else float("nan")


def main() -> None:
    args = parse_args()
    config = read_json(args.run_dir / "config.json")
    noise_csv = config.get("noise_csv")
    if not noise_csv:
        raise ValueError("Semantic-error analysis requires a semantic noise CSV")
    metadata_path = Path(str(noise_csv)).with_suffix(".json")
    metadata = read_json(metadata_path)
    neighbor_map = metadata.get("neighbor_map")
    if not neighbor_map:
        raise ValueError("Noise metadata has no neighbor_map")

    features = np.load(args.run_dir / "posthoc_test_features.npy")
    labels = np.load(args.run_dir / "posthoc_test_labels.npy")
    predictions = np.load(args.run_dir / "posthoc_test_predictions.npy")
    pmrc_scores = np.load(args.run_dir / "posthoc_test_scores.npy")
    msp_scores = features[:, 0]
    correct = predictions == labels
    errors = ~correct
    semantic_errors = np.zeros(len(labels), dtype=bool)
    for index_where_error in np.flatnonzero(errors):
        semantic_errors[index_where_error] = int(predictions[index_where_error]) in set(
            int(value) for value in neighbor_map.get(str(int(labels[index_where_error])), [])
        )
    other_errors = errors & ~semantic_errors

    msp_rejected_90 = rejection_mask(msp_scores, 0.9)
    pmrc_rejected_90 = rejection_mask(pmrc_scores, 0.9)
    output = {
        "run_dir": str(args.run_dir),
        "n_test": int(len(labels)),
        "n_errors": int(errors.sum()),
        "n_semantic_neighbor_errors": int(semantic_errors.sum()),
        "semantic_error_share": float(semantic_errors.sum() / max(errors.sum(), 1)),
        "mean_prototype_margin": {
            "correct": float(features[correct, 9].mean()),
            "all_errors": float(features[errors, 9].mean()),
            "semantic_errors": float(features[semantic_errors, 9].mean())
            if semantic_errors.any()
            else None,
            "other_errors": float(features[other_errors, 9].mean())
            if other_errors.any()
            else None,
        },
        "at_90_coverage": {
            "msp": {
                "rejected_all_errors": fraction(msp_rejected_90[errors]),
                "rejected_semantic_errors": fraction(
                    msp_rejected_90[semantic_errors]
                ),
                "rejected_other_errors": fraction(msp_rejected_90[other_errors]),
                "semantic_errors_rejected": int(
                    (msp_rejected_90 & semantic_errors).sum()
                ),
                "other_errors_rejected": int((msp_rejected_90 & other_errors).sum()),
            },
            "pmrc": {
                "rejected_all_errors": fraction(pmrc_rejected_90[errors]),
                "rejected_semantic_errors": fraction(
                    pmrc_rejected_90[semantic_errors]
                ),
                "rejected_other_errors": fraction(pmrc_rejected_90[other_errors]),
                "semantic_errors_rejected": int(
                    (pmrc_rejected_90 & semantic_errors).sum()
                ),
                "other_errors_rejected": int(
                    (pmrc_rejected_90 & other_errors).sum()
                ),
            },
        },
    }
    write_json(args.run_dir / "semantic_error_analysis.json", output)
    print(output)


if __name__ == "__main__":
    main()
