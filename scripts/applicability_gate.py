from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgel.utils import read_json, write_json
from scripts.posthoc_reliability import aurc_from_scores, risk_at_coverage
from scripts.prototype_independence import stratified_auc
from scripts.summarize_results import infer_noise


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT.parent.parent / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a validation-only prototype-signal applicability gate."
    )
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bins", type=int, default=5)
    return parser.parse_args()


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

        validation = stratified_auc(
            val_features[:, 0],
            val_features[:, 9],
            val_correct,
            args.bins,
        )
        use_pmrc = bool(
            validation["weighted_auc"] is not None
            and validation["weighted_auc"] > args.threshold
        )
        selected_scores = pmrc_scores if use_pmrc else test_features[:, 0]
        noise_kind, _, _ = infer_noise(config)
        dataset = str(config.get("dataset", "unknown")).lower()
        if dataset in {"cub", "cub200"}:
            dataset = "cub"
        elif dataset.startswith("aircraft"):
            dataset = "aircraft"
        elif dataset.startswith("flowers"):
            dataset = "flowers102"
        rows.append(
            {
                "run_name": run_dir.name,
                "dataset": dataset,
                "method": str(config.get("method", "unknown")),
                "model": str(config.get("model", "unknown")),
                "noise_kind": noise_kind,
                "seed": int(config.get("seed", 0)),
                "validation_prototype_within_msp_auc": float(
                    validation["weighted_auc"]
                ),
                "selected_method": "pmrc" if use_pmrc else "msp",
                "msp_aurc": aurc_from_scores(test_features[:, 0], test_correct),
                "pmrc_aurc": aurc_from_scores(pmrc_scores, test_correct),
                "gated_aurc": aurc_from_scores(selected_scores, test_correct),
                "msp_risk90": risk_at_coverage(
                    test_features[:, 0], test_correct, 0.9
                ),
                "pmrc_risk90": risk_at_coverage(pmrc_scores, test_correct, 0.9),
                "gated_risk90": risk_at_coverage(
                    selected_scores, test_correct, 0.9
                ),
            }
        )

    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["dataset"],
                row["method"],
                row["model"],
                row["noise_kind"],
            )
        ].append(row)
    summaries = []
    for key, group_rows in sorted(grouped.items()):
        dataset, method, model, noise_kind = key
        gated_relative = [
            100.0 * (row["msp_aurc"] - row["gated_aurc"]) / row["msp_aurc"]
            for row in group_rows
        ]
        summaries.append(
            {
                "dataset": dataset,
                "method": method,
                "model": model,
                "noise_kind": noise_kind,
                "n": len(group_rows),
                "pmrc_selected": sum(
                    row["selected_method"] == "pmrc" for row in group_rows
                ),
                "msp_aurc_mean": float(
                    np.mean([row["msp_aurc"] for row in group_rows])
                ),
                "pmrc_aurc_mean": float(
                    np.mean([row["pmrc_aurc"] for row in group_rows])
                ),
                "gated_aurc_mean": float(
                    np.mean([row["gated_aurc"] for row in group_rows])
                ),
                "gated_relative_improvement_pct_mean": float(
                    np.mean(gated_relative)
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "applicability_gate.json",
        {
            "threshold": args.threshold,
            "gate_source": "clean validation split only",
            "n_runs": len(rows),
            "runs": rows,
            "summaries": summaries,
        },
    )
    with (args.output_dir / "applicability_gate.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Validation-only applicability gate",
        "",
        f"PMRC is selected only when prototype-margin AUROC within five MSP",
        f"validation bins exceeds `{args.threshold:.2f}`. This diagnostic uses",
        "only the clean calibration split; no test label or test feature is",
        "used to choose the method.",
        "",
        "| Dataset | Method | Backbone | Noise | n | PMRC selected | "
        "MSP AURC | PMRC AURC | Gated AURC | Gated relative improvement |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['dataset']} | {item['method']} | {item['model']} | "
            f"{item['noise_kind']} | {item['n']} | "
            f"{item['pmrc_selected']}/{item['n']} | "
            f"{item['msp_aurc_mean']:.4f} | "
            f"{item['pmrc_aurc_mean']:.4f} | "
            f"{item['gated_aurc_mean']:.4f} | "
            f"{item['gated_relative_improvement_pct_mean']:.1f}% |"
        )
    gains = [
        100.0 * (row["msp_aurc"] - row["gated_aurc"]) / row["msp_aurc"]
        for row in rows
    ]
    lines.extend(
        [
            "",
            f"Across {len(rows)} runs, the gate improves {sum(value > 1e-12 for value in gains)} and ",
            f"leaves {sum(abs(value) <= 1e-12 for value in gains)} unchanged, with "
            f"{sum(value < -1e-12 for value in gains)} test regressions.",
        ]
    )
    (args.output_dir / "applicability_gate.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote applicability gate for {len(rows)} runs; "
        f"PMRC selected in {sum(row['selected_method'] == 'pmrc' for row in rows)}."
    )


if __name__ == "__main__":
    main()
