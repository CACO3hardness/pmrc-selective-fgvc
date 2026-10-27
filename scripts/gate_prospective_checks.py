from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgel.utils import read_json, write_json
from scripts.posthoc_reliability import aurc_from_scores
from scripts.prototype_independence import stratified_auc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leave-one-subset-out and lower-bound checks for the PMRC gate."
    )
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap", type=int, default=500)
    return parser.parse_args()


def web_subset(run_dir: Path) -> str | None:
    prefix = "elr_resnet50_webfg_"
    if not run_dir.name.startswith(prefix):
        return None
    return run_dir.name.removeprefix(prefix).rsplit("_seed", 1)[0]


def load_web_run(run_dir: Path) -> dict:
    subset = web_subset(run_dir)
    if subset is None:
        raise ValueError("Not a WebFG run")
    config = read_json(run_dir / "config.json")
    val_features = np.load(run_dir / "posthoc_val_features.npy")
    val_scores = np.load(run_dir / "posthoc_val_scores.npy")
    val_labels = np.load(run_dir / "posthoc_val_labels.npy")
    val_predictions = np.load(run_dir / "posthoc_val_predictions.npy")
    test_features = np.load(run_dir / "posthoc_test_features.npy")
    test_scores = np.load(run_dir / "posthoc_test_scores.npy")
    test_labels = np.load(run_dir / "posthoc_test_labels.npy")
    test_predictions = np.load(run_dir / "posthoc_test_predictions.npy")
    val_correct = val_predictions == val_labels
    test_correct = test_predictions == test_labels
    diagnostic = stratified_auc(
        val_features[:, 0],
        val_features[:, 9],
        val_correct,
        5,
    )["weighted_auc"]
    return {
        "run_name": run_dir.name,
        "subset": subset,
        "seed": int(config.get("seed", 0)),
        "validation_diagnostic": float(diagnostic),
        "validation_msp_aurc": aurc_from_scores(val_features[:, 0], val_correct),
        "validation_pmrc_aurc": aurc_from_scores(val_scores, val_correct),
        "test_msp_aurc": aurc_from_scores(test_features[:, 0], test_correct),
        "test_pmrc_aurc": aurc_from_scores(test_scores, test_correct),
    }


def relative_gain(msp: float, selected: float) -> float:
    return 100.0 * (msp - selected) / msp


def leave_one_subset_out(rows: list[dict]) -> list[dict]:
    results = []
    subsets = sorted({row["subset"] for row in rows})
    candidates = np.linspace(0.30, 0.80, 101)
    for held_out in subsets:
        source = [row for row in rows if row["subset"] != held_out]
        target = [row for row in rows if row["subset"] == held_out]
        choices = []
        for threshold in candidates:
            gains = [
                relative_gain(
                    row["validation_msp_aurc"],
                    row["validation_pmrc_aurc"]
                    if row["validation_diagnostic"] > threshold
                    else row["validation_msp_aurc"],
                )
                for row in source
            ]
            choices.append(
                (
                    float(np.mean(gains)),
                    -abs(float(threshold) - 0.5),
                    float(threshold),
                    sum(
                        row["validation_diagnostic"] > threshold
                        for row in source
                    ),
                )
            )
        source_gain, _, threshold, source_selected = max(choices)
        target_gains = []
        target_selected = 0
        for row in target:
            use_pmrc = row["validation_diagnostic"] > threshold
            target_selected += int(use_pmrc)
            selected_aurc = (
                row["test_pmrc_aurc"] if use_pmrc else row["test_msp_aurc"]
            )
            target_gains.append(
                relative_gain(row["test_msp_aurc"], selected_aurc)
            )
        results.append(
            {
                "held_out_subset": held_out,
                "source_subsets": sorted({row["subset"] for row in source}),
                "selected_threshold": float(threshold),
                "source_validation_gain_pct": float(source_gain),
                "source_pmrc_selected": int(source_selected),
                "source_runs": len(source),
                "heldout_pmrc_selected": int(target_selected),
                "heldout_runs": len(target),
                "heldout_mean_gain_pct": float(np.mean(target_gains)),
                "heldout_min_gain_pct": float(np.min(target_gains)),
                "heldout_max_gain_pct": float(np.max(target_gains)),
                "heldout_regressions": int(
                    np.sum(np.asarray(target_gains) < -1e-12)
                ),
            }
        )
    return results


def lower_bound_gate(
    runs_root: Path,
    bootstrap: int,
) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(20261026)
    rows = []
    for run_dir in sorted(runs_root.iterdir()):
        required = (
            "posthoc_val_features.npy",
            "posthoc_val_labels.npy",
            "posthoc_val_predictions.npy",
            "posthoc_test_features.npy",
            "posthoc_test_scores.npy",
            "posthoc_test_labels.npy",
            "posthoc_test_predictions.npy",
        )
        if not all((run_dir / name).is_file() for name in required):
            continue
        val_features = np.load(run_dir / "posthoc_val_features.npy")
        val_labels = np.load(run_dir / "posthoc_val_labels.npy")
        val_predictions = np.load(run_dir / "posthoc_val_predictions.npy")
        test_features = np.load(run_dir / "posthoc_test_features.npy")
        pmrc_scores = np.load(run_dir / "posthoc_test_scores.npy")
        test_labels = np.load(run_dir / "posthoc_test_labels.npy")
        test_predictions = np.load(run_dir / "posthoc_test_predictions.npy")
        val_correct = val_predictions == val_labels
        test_correct = test_predictions == test_labels
        point = stratified_auc(
            val_features[:, 0],
            val_features[:, 9],
            val_correct,
            5,
        )["weighted_auc"]
        bootstrap_values = []
        for _ in range(bootstrap):
            sample = rng.integers(0, len(val_correct), len(val_correct))
            value = stratified_auc(
                val_features[sample, 0],
                val_features[sample, 9],
                val_correct[sample],
                5,
            )["weighted_auc"]
            if value is not None:
                bootstrap_values.append(value)
        lower = float(np.quantile(bootstrap_values, 0.05))
        use_pmrc = lower > 0.5
        msp_aurc = aurc_from_scores(test_features[:, 0], test_correct)
        pmrc_aurc = aurc_from_scores(pmrc_scores, test_correct)
        selected = pmrc_aurc if use_pmrc else msp_aurc
        rows.append(
            {
                "run_name": run_dir.name,
                "validation_diagnostic": float(point),
                "validation_diagnostic_q05": lower,
                "selected_method": "pmrc" if use_pmrc else "msp",
                "msp_aurc": msp_aurc,
                "pmrc_aurc": pmrc_aurc,
                "gated_aurc": selected,
                "gated_relative_gain_pct": relative_gain(msp_aurc, selected),
            }
        )
    gains = np.asarray([row["gated_relative_gain_pct"] for row in rows])
    summary = {
        "n_runs": len(rows),
        "pmrc_selected": int(
            sum(row["selected_method"] == "pmrc" for row in rows)
        ),
        "positive_runs": int(np.sum(gains > 1e-12)),
        "unchanged_runs": int(np.sum(np.abs(gains) <= 1e-12)),
        "regressing_runs": int(np.sum(gains < -1e-12)),
        "mean_gain_pct": float(np.mean(gains)),
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    web_rows = []
    for run_dir in sorted(args.runs_root.glob("elr_resnet50_webfg_*")):
        if (
            (run_dir / "posthoc_reliability.json").is_file()
            and (run_dir / "posthoc_val_scores.npy").is_file()
        ):
            web_rows.append(load_web_run(run_dir))
    lowo = leave_one_subset_out(web_rows)
    lower_rows, lower_summary = lower_bound_gate(
        args.runs_root,
        args.bootstrap,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "gate_prospective_checks.json",
        {
            "leave_one_subset_out": lowo,
            "lower_bound_gate": {
                "summary": lower_summary,
                "runs": lower_rows,
            },
        },
    )
    lines = [
        "# Gate prospective checks",
        "",
        "## WebFG leave-one-subset-out",
        "",
        "The threshold is selected using source-subset validation gains only. "
        "The held-out subset contributes only its fixed test result.",
        "",
        "| Held out | Threshold | Source validation gain | Held-out selected | "
        "Held-out mean test gain | Range | Regressions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in lowo:
        lines.append(
            f"| {row['held_out_subset']} | {row['selected_threshold']:.2f} | "
            f"{row['source_validation_gain_pct']:+.2f}% | "
            f"{row['heldout_pmrc_selected']}/{row['heldout_runs']} | "
            f"{row['heldout_mean_gain_pct']:+.2f}% | "
            f"{row['heldout_min_gain_pct']:+.2f}% to "
            f"{row['heldout_max_gain_pct']:+.2f}% | "
            f"{row['heldout_regressions']} |"
        )
    lines.extend(
        [
            "",
            "## Lower-bound gate",
            "",
            "PMRC is selected only when the 5th percentile of the calibration "
            "bootstrap distribution of within-MSP prototype AUROC exceeds 0.5.",
            "",
            f"Across {lower_summary['n_runs']} runs, the lower-bound gate selects "
            f"PMRC in {lower_summary['pmrc_selected']} runs, improves "
            f"{lower_summary['positive_runs']}, leaves "
            f"{lower_summary['unchanged_runs']} unchanged, and produces "
            f"{lower_summary['regressing_runs']} regressions. Mean gated gain is "
            f"{lower_summary['mean_gain_pct']:+.2f}%.",
        ]
    )
    (args.output_dir / "gate_prospective_checks.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"leave_one_subset_out": lowo, "summary": lower_summary}, indent=2))


if __name__ == "__main__":
    main()
