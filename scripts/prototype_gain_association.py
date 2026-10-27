from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgel.utils import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "outputs"


def main() -> None:
    contamination = json.loads(
        (OUTPUT_ROOT / "prototype_contamination.json").read_text(
            encoding="utf-8"
        )
    )
    results = json.loads(
        (OUTPUT_ROOT / "results_summary_latest.json").read_text(
            encoding="utf-8"
        )
    )
    by_run = {row["run_name"]: row for row in results["runs"]}
    rows = []
    for row in contamination["runs"]:
        result = by_run[row["run_name"]]
        rows.append(
            {
                "run_name": row["run_name"],
                "mean_prototype_cosine": row["mean_prototype_cosine"],
                "relative_pmrc_gain_pct": result[
                    "relative_aurc_improvement_pct"
                ],
            }
        )
    cosine = np.asarray([row["mean_prototype_cosine"] for row in rows])
    gain = np.asarray([row["relative_pmrc_gain_pct"] for row in rows])
    pearson = pearsonr(cosine, gain)
    spearman = spearmanr(cosine, gain)
    output = {
        "n_runs": len(rows),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "rows": rows,
    }
    write_json(OUTPUT_ROOT / "prototype_gain_association.json", output)
    lines = [
        "# Prototype displacement and PMRC gain",
        "",
        f"Across {len(rows)} representative runs, prototype cosine versus PMRC "
        f"relative gain has Pearson r={pearson.statistic:.3f} "
        f"(p={pearson.pvalue:.3f}) and Spearman rho="
        f"{spearman.statistic:.3f} (p={spearman.pvalue:.3f}).",
        "",
        "The association is exploratory and confounded by dataset. It should "
        "not be interpreted as evidence for a causal mechanism.",
        "",
        "| Run | Prototype cosine | PMRC gain |",
        "|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run_name']} | {row['mean_prototype_cosine']:.4f} | "
            f"{row['relative_pmrc_gain_pct']:+.2f}% |"
        )
    (OUTPUT_ROOT / "prototype_gain_association.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
