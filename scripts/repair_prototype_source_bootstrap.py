from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.prototype_source_ablation import paired_bootstrap_aurc_difference
from scripts.run_prototype_source_suite import DEFAULT_RUNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute paired-bootstrap direction frequencies from saved scores."
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = PROJECT_ROOT / "runs"
    summary = []
    for run_name in DEFAULT_RUNS:
        run_dir = runs_root / run_name
        result_path = run_dir / "prototype_source_ablation.json"
        score_path = run_dir / "prototype_source_scores.npz"
        if not result_path.is_file() or not score_path.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        scores = np.load(score_path)
        correct = scores["correct"]
        rng = np.random.default_rng(7)
        rng.shuffle(np.arange(len(scores["noisy_val"])))
        repaired = {}
        for comparison in (
            "without_prototype",
            "clean_oracle",
            "pretrained",
            "shuffled",
        ):
            repaired[f"noisy_minus_{comparison}"] = (
                paired_bootstrap_aurc_difference(
                    scores["noisy_test"],
                    scores[f"{comparison}_test"],
                    correct,
                    rng,
                    args.bootstrap,
                )
            )
        result["paired_bootstrap"] = repaired
        result_path.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        summary.append(
            {
                "run_name": run_name,
                "variants": result["prototype_variants"],
                "paired_bootstrap": repaired,
            }
        )
        print(f"Repaired {run_name}", flush=True)

    output_path = PROJECT_ROOT.parent.parent / "outputs" / "prototype_source_summary.json"
    output_path.write_text(
        json.dumps(
            {
                "requested_runs": list(DEFAULT_RUNS),
                "completed_runs": [item["run_name"] for item in summary],
                "failures": [],
                "results": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Updated {output_path}", flush=True)


if __name__ == "__main__":
    main()
