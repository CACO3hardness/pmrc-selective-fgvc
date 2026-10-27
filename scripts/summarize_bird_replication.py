from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgel.utils import read_json
from scripts.prototype_independence import stratified_auc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "outputs"
SEEDS = tuple(range(6))


def summarize_group(row_group: list[dict]) -> dict:
    relative = [
        100.0 * (row["msp_aurc"] - row["pmrc_aurc"]) / row["msp_aurc"]
        for row in row_group
    ]
    return {
        "n": len(row_group),
        "msp_aurc_mean": float(np.mean([row["msp_aurc"] for row in row_group])),
        "pmrc_aurc_mean": float(
            np.mean([row["pmrc_aurc"] for row in row_group])
        ),
        "msp_risk90_mean": float(
            np.mean([row["msp_risk90"] for row in row_group])
        ),
        "pmrc_risk90_mean": float(
            np.mean([row["pmrc_risk90"] for row in row_group])
        ),
        "risk90_reduction_pp_mean": float(
            np.mean(
                [
                    100.0 * (row["msp_risk90"] - row["pmrc_risk90"])
                    for row in row_group
                ]
            )
        ),
        "relative_improvement_pct_mean": float(np.mean(relative)),
        "relative_improvement_pct_std": (
            float(np.std(relative, ddof=1)) if len(relative) > 1 else 0.0
        ),
        "relative_improvement_pct_min": float(np.min(relative)),
        "relative_improvement_pct_max": float(np.max(relative)),
        "seeds_improved": int(sum(value > 0 for value in relative)),
        "gate_pmrc_selected": int(
            sum(row["gate_selected_method"] == "pmrc" for row in row_group)
        ),
    }


def main() -> None:
    rows = []
    for seed in SEEDS:
        run_dir = RUNS_ROOT / f"elr_resnet50_webfg_bird_seed{seed}"
        result_path = run_dir / "posthoc_reliability.json"
        required = (
            result_path,
            run_dir / "prototype_independence.json",
            run_dir / "posthoc_val_features.npy",
            run_dir / "posthoc_val_labels.npy",
            run_dir / "posthoc_val_predictions.npy",
        )
        if not all(path.is_file() for path in required):
            continue
        result = read_json(result_path)
        independence = read_json(run_dir / "prototype_independence.json")
        val_features = np.load(run_dir / "posthoc_val_features.npy")
        val_labels = np.load(run_dir / "posthoc_val_labels.npy")
        val_predictions = np.load(run_dir / "posthoc_val_predictions.npy")
        val_auc = stratified_auc(
            val_features[:, 0],
            val_features[:, 9],
            val_predictions == val_labels,
            5,
        )["weighted_auc"]
        msp = result["test_results"]["msp"]
        pmrc = result["test_results"]["learned_reliability"]
        rows.append(
            {
                "seed": seed,
                "test_accuracy": float(result["test_accuracy"]),
                "msp_aurc": float(msp["aurc"]),
                "pmrc_aurc": float(pmrc["aurc"]),
                "relative_improvement_pct": float(
                    100.0 * (msp["aurc"] - pmrc["aurc"]) / msp["aurc"]
                ),
                "msp_risk90": float(msp["risk_at_90"]),
                "pmrc_risk90": float(pmrc["risk_at_90"]),
                "validation_within_msp_auc": float(val_auc),
                "test_within_msp_auc": float(
                    independence["stratified_by_msp"]["weighted_auc"]
                ),
                "gate_selected_method": "pmrc" if val_auc > 0.5 else "msp",
                "difference_ci95_low": float(
                    result["learned_minus_msp_aurc_ci95"][0]
                ),
                "difference_ci95_high": float(
                    result["learned_minus_msp_aurc_ci95"][1]
                ),
            }
        )

    original = [row for row in rows if row["seed"] < 3]
    replication = [row for row in rows if row["seed"] >= 3]
    groups = {}
    if original:
        groups["original_seed0_2"] = summarize_group(original)
    if replication:
        groups["replication_seed3_5"] = summarize_group(replication)
    if rows:
        groups["all_available"] = summarize_group(rows)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "webfg_bird_replication_runs.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "available_seeds": [row["seed"] for row in rows],
        "runs": rows,
        "groups": groups,
    }
    (OUTPUT_ROOT / "webfg_bird_replication.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# WebFG bird replication",
        "",
        "All completed seeds are reported. The gate uses validation within-MSP",
        "prototype AUROC, while the final column reports the held-out test value.",
        "",
        "| Seed | Acc | MSP AURC | PMRC AURC | Relative change | "
        "MSP risk@90 | PMRC risk@90 | Validation AUC | Test AUC | Gate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {100 * row['test_accuracy']:.2f}% | "
            f"{row['msp_aurc']:.5f} | {row['pmrc_aurc']:.5f} | "
            f"{row['relative_improvement_pct']:+.2f}% | "
            f"{row['msp_risk90']:.5f} | {row['pmrc_risk90']:.5f} | "
            f"{row['validation_within_msp_auc']:.4f} | "
            f"{row['test_within_msp_auc']:.4f} | "
            f"{row['gate_selected_method']} |"
        )
    lines.extend(["", "## Group summaries", ""])
    for name, summary in groups.items():
        lines.append(
            f"- `{name}`: n={summary['n']}, mean seed-wise relative change "
            f"{summary['relative_improvement_pct_mean']:+.2f}% "
            f"± {summary['relative_improvement_pct_std']:.2f}% "
            f"({summary['relative_improvement_pct_min']:+.2f}% to "
            f"{summary['relative_improvement_pct_max']:+.2f}%), "
            f"{summary['seeds_improved']}/{summary['n']} seeds improved, "
            f"gate selected PMRC in {summary['gate_pmrc_selected']}/{summary['n']}, "
            f"mean 90% risk reduction "
            f"{summary['risk90_reduction_pp_mean']:+.2f} pp."
        )
    (OUTPUT_ROOT / "webfg_bird_replication.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"Summarized {len(rows)} completed bird seeds.")


if __name__ == "__main__":
    main()
