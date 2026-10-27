from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prototype_independence import stratified_auc


RUNS_ROOT = PROJECT_ROOT / "runs"
OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "outputs"
GATE_THRESHOLD = 0.5


def read_optional(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_seed(seed: int) -> dict | None:
    run_dir = RUNS_ROOT / f"elr_resnet50_stanford_dogs_semantic40_seed{seed}"
    posthoc = read_optional(run_dir / "posthoc_reliability.json")
    independence = read_optional(run_dir / "prototype_independence.json")
    if posthoc is None:
        return None
    results = posthoc["test_results"]
    msp = results["msp"]
    pmrc = results["learned_reliability"]
    val_features = np.load(run_dir / "posthoc_val_features.npy")
    val_labels = np.load(run_dir / "posthoc_val_labels.npy")
    val_predictions = np.load(run_dir / "posthoc_val_predictions.npy")
    gate_auc = float(
        stratified_auc(
            val_features[:, 0],
            val_features[:, 9],
            val_predictions == val_labels,
            5,
        )["weighted_auc"]
    )
    test_within_msp_auc = (
        independence["stratified_by_msp"]["weighted_auc"]
        if independence is not None
        else None
    )
    row = {
        "seed": seed,
        "test_accuracy": float(posthoc["test_accuracy"]),
        "msp_aurc": float(msp["aurc"]),
        "pmrc_aurc": float(pmrc["aurc"]),
        "relative_gain_pct": float(
            100.0 * (msp["aurc"] - pmrc["aurc"]) / msp["aurc"]
        ),
        "delta_aurc_ci95_low": float(posthoc["learned_minus_msp_aurc_ci95"][0]),
        "delta_aurc_ci95_high": float(posthoc["learned_minus_msp_aurc_ci95"][1]),
        "msp_risk90": float(msp["risk_at_90"]),
        "pmrc_risk90": float(pmrc["risk_at_90"]),
        "validation_gate_auc": gate_auc,
        "test_within_msp_auc": test_within_msp_auc,
        "gate_selected_method": (
            "pmrc" if gate_auc is not None and gate_auc > GATE_THRESHOLD else "msp"
        ),
        "gated_aurc": (
            float(pmrc["aurc"])
            if gate_auc is not None and gate_auc > GATE_THRESHOLD
            else float(msp["aurc"])
        ),
        "temperature_aurc": None,
        "pnorm_aurc": None,
        "knn_aurc": None,
        "trust_aurc": None,
    }
    selective = read_optional(run_dir / "selective_baselines.json")
    if selective is not None:
        test = selective.get("test", {})
        row["temperature_aurc"] = test.get("temperature_scaling", {}).get("aurc")
        row["pnorm_aurc"] = test.get("maxlogit_pnorm", {}).get("aurc")
    embedding = read_optional(run_dir / "selective_embedding_baselines.json")
    if embedding is not None:
        baselines = embedding.get("baselines", {})
        row["knn_aurc"] = baselines.get("knn_label_agreement", {}).get("test_aurc")
        row["trust_aurc"] = baselines.get("trust_score", {}).get("test_aurc")
    return row


def optional_mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row[key] is not None]
    return float(np.mean(values)) if values else None


def main() -> None:
    rows = [row for seed in range(3) if (row := load_seed(seed)) is not None]
    if not rows:
        raise RuntimeError("No completed Stanford Dogs external-domain runs")
    gains = [row["relative_gain_pct"] for row in rows]
    baseline_keys = ("temperature_aurc", "pnorm_aurc", "knn_aurc", "trust_aurc")
    baseline_wins = sum(
        all(
            row[key] is not None and row["pmrc_aurc"] < float(row[key])
            for key in baseline_keys
        )
        for row in rows
    )
    summary = {
        "setting": "Stanford Dogs, 120 classes, 40% semantic-neighbor noise",
        "gate_threshold": GATE_THRESHOLD,
        "completed_seeds": [row["seed"] for row in rows],
        "mean_msp_aurc": optional_mean(rows, "msp_aurc"),
        "mean_pmrc_aurc": optional_mean(rows, "pmrc_aurc"),
        "mean_relative_gain_pct": float(np.mean(gains)),
        "seed_standard_deviation_pct": (
            float(np.std(gains, ddof=1)) if len(gains) > 1 else 0.0
        ),
        "positive_seeds": sum(gain > 0 for gain in gains),
        "pmrc_selected": sum(row["gate_selected_method"] == "pmrc" for row in rows),
        "all_baseline_wins": baseline_wins,
        "gate_direction_matches": sum(
            (row["gate_selected_method"] == "pmrc")
            == (row["relative_gain_pct"] > 0)
            for row in rows
        ),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "stanford_dogs_external_validation.json").write_text(
        json.dumps({"summary": summary, "runs": rows}, indent=2),
        encoding="utf-8",
    )
    with (OUTPUT_ROOT / "stanford_dogs_external_validation.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Stanford Dogs external-domain validation",
        "",
        "The official train/test split is fixed. Five official train images per "
        "class form the clean validation/calibration set, and the remaining "
        "training labels receive 40% semantic-neighbor noise. The official test "
        "set is untouched.",
        "",
        "| Seed | MSP AURC | PMRC AURC | Relative gain | 95% CI | "
        "Gate AUROC | Gate |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        gate = (
            "not available"
            if row["validation_gate_auc"] is None
            else f"{row['validation_gate_auc']:.4f}"
        )
        lines.append(
            f"| {row['seed']} | {row['msp_aurc']:.4f} | "
            f"{row['pmrc_aurc']:.4f} | {row['relative_gain_pct']:+.2f}% | "
            f"[{row['delta_aurc_ci95_low']:+.4f}, "
            f"{row['delta_aurc_ci95_high']:+.4f}] | {gate} | "
            f"{row['gate_selected_method'].upper()} |"
        )
    lines.extend(
        [
            "",
            f"Mean relative gain: {summary['mean_relative_gain_pct']:+.2f}%. "
            f"Positive seeds: {summary['positive_seeds']}/{len(rows)}. "
            f"Gate direction matches: {summary['gate_direction_matches']}/{len(rows)}. "
            f"PMRC beats all four baselines in "
            f"{summary['all_baseline_wins']}/{len(rows)}.",
            "",
            "| Baseline | Mean AURC |",
            "|---|---:|",
        ]
    )
    for key, label in (
        ("temperature_aurc", "Temperature"),
        ("pnorm_aurc", "MaxLogit-pNorm"),
        ("knn_aurc", "KNN agreement"),
        ("trust_aurc", "Trust Score"),
        ("pmrc_aurc", "PMRC"),
    ):
        value = optional_mean(rows, key)
        lines.append(
            f"| {label} | {'not run' if value is None else f'{value:.4f}'} |"
        )
    lines.extend(
        [
            "",
            "Stanford Dogs is derived from ImageNet and the backbone is "
            "ImageNet-pretrained. This experiment therefore measures downstream "
            "dataset and split transport under synthetic semantic noise, not "
            "transport to a pretraining-independent real-web domain.",
        ]
    )
    (OUTPUT_ROOT / "stanford_dogs_external_validation.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
