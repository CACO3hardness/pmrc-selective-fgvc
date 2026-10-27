from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgel.utils import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "outputs"


def main() -> None:
    selected = []
    relative_spreads = []
    for path in sorted(RUNS_ROOT.glob("*/posthoc_reliability.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        selected.append(str(payload["selected_weight_decay"]))
        values = np.asarray(list(payload["cv_aurc"].values()), dtype=np.float64)
        relative_spreads.append(float((values.max() - values.min()) / values.mean()))
    counts = Counter(selected)
    output = {
        "n_runs": len(selected),
        "selected_weight_decay_counts": dict(sorted(counts.items())),
        "relative_cv_aurc_span": {
            "median": float(np.median(relative_spreads)),
            "mean": float(np.mean(relative_spreads)),
            "p90": float(np.quantile(relative_spreads, 0.9)),
        },
    }
    write_json(OUTPUT_ROOT / "cv_stability_summary.json", output)
    lines = [
        "# Five-fold calibration stability",
        "",
        f"Across {output['n_runs']} completed runs, selected L2 strengths were:",
        "",
        "| Weight decay | Runs |",
        "|---:|---:|",
    ]
    for value, count in output["selected_weight_decay_counts"].items():
        lines.append(f"| {value} | {count} |")
    lines.extend(
        [
            "",
            "The median relative span between the best and worst candidate CV "
            f"AURC was {100 * output['relative_cv_aurc_span']['median']:.2f}% "
            f"(90th percentile {100 * output['relative_cv_aurc_span']['p90']:.2f}%).",
        ]
    )
    (OUTPUT_ROOT / "cv_stability_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
