from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "outputs"
METHODS = ("ce", "elr", "gce")


def optional_mean(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return float(np.mean(present)) if present else None


def read_optional(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_run(method: str, seed: int) -> dict | None:
    run_dir = RUNS_ROOT / f"{method}_resnet50_cub_semantic40_seed{seed}"
    posthoc = read_optional(run_dir / "posthoc_reliability.json")
    if posthoc is None:
        return None
    results = posthoc["test_results"]
    msp = results["msp"]
    pmrc = results["learned_reliability"]
    row: dict[str, float | int | str | None] = {
        "method": method.upper(),
        "seed": seed,
        "test_accuracy": float(posthoc["test_accuracy"]),
        "msp_aurc": float(msp["aurc"]),
        "pmrc_aurc": float(pmrc["aurc"]),
        "relative_gain_pct": float(
            100.0 * (msp["aurc"] - pmrc["aurc"]) / msp["aurc"]
        ),
        "msp_risk90": float(msp["risk_at_90"]),
        "pmrc_risk90": float(pmrc["risk_at_90"]),
        "delta_aurc_ci95_low": float(posthoc["learned_minus_msp_aurc_ci95"][0]),
        "delta_aurc_ci95_high": float(posthoc["learned_minus_msp_aurc_ci95"][1]),
        "temperature_aurc": None,
        "pnorm_aurc": None,
        "knn_aurc": None,
        "trust_aurc": None,
        "within_msp_auc": None,
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

    independence = read_optional(run_dir / "prototype_independence.json")
    if independence is not None:
        row["within_msp_auc"] = independence["stratified_by_msp"]["weighted_auc"]
    return row


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for method in METHODS:
        selected = [
            row for row in rows if str(row["method"]).lower() == method
        ]
        if not selected:
            continue
        summaries.append(
            {
                "method": method,
                "n_seeds": len(selected),
                "msp_aurc_mean": optional_mean(
                    [row["msp_aurc"] for row in selected]
                ),
                "pmrc_aurc_mean": optional_mean(
                    [row["pmrc_aurc"] for row in selected]
                ),
                "relative_gain_pct_mean": optional_mean(
                    [row["relative_gain_pct"] for row in selected]
                ),
                "positive_seeds": sum(
                    row["relative_gain_pct"] > 0 for row in selected
                ),
                "risk90_improvement_mean": optional_mean(
                    [
                        row["msp_risk90"] - row["pmrc_risk90"]
                        for row in selected
                    ]
                ),
                "temperature_aurc_mean": optional_mean(
                    [row["temperature_aurc"] for row in selected]
                ),
                "pnorm_aurc_mean": optional_mean(
                    [row["pnorm_aurc"] for row in selected]
                ),
                "knn_aurc_mean": optional_mean(
                    [row["knn_aurc"] for row in selected]
                ),
                "trust_aurc_mean": optional_mean(
                    [row["trust_aurc"] for row in selected]
                ),
                "within_msp_auc_mean": optional_mean(
                    [row["within_msp_auc"] for row in selected]
                ),
            }
        )
    return summaries


def fmt(value: float | None, digits: int = 4) -> str:
    return "not run" if value is None else f"{value:.{digits}f}"


def main() -> None:
    rows = [
        row
        for method in METHODS
        for seed in range(3)
        if (row := load_run(method, seed)) is not None
    ]
    if not rows:
        raise RuntimeError("No completed CUB semantic-40 training-objective runs")
    summary = summarize(rows)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "setting": "CUB ResNet-50 semantic-40, seeds 0-2",
        "runs": rows,
        "summaries": summary,
    }
    (OUTPUT_ROOT / "training_objective_transfer.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    with (OUTPUT_ROOT / "training_objective_transfer.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# CUB semantic-40 training-objective transfer",
        "",
        "The classifier architecture, data split, prototype construction, and "
        "post-hoc calibration protocol are held fixed. Only the training "
        "objective changes.",
        "",
        "| Objective | Seeds | MSP AURC | PMRC AURC | Relative gain | "
        "Improving seeds | Risk@90 reduction | Within-MSP AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        risk_text = "not available" if row["risk90_improvement_mean"] is None else (
            f"{100 * row['risk90_improvement_mean']:+.2f} pp"
        )
        lines.append(
            f"| {row['method']} | {row['n_seeds']} | "
            f"{row['msp_aurc_mean']:.4f} | {row['pmrc_aurc_mean']:.4f} | "
            f"{row['relative_gain_pct_mean']:+.2f}% | "
            f"{row['positive_seeds']}/{row['n_seeds']} | "
            f"{risk_text} | {fmt(row['within_msp_auc_mean'])} |"
        )
    lines.extend(
        [
            "",
            "| Objective | Seed | MSP AURC | PMRC AURC | Relative gain | "
            "PMRC minus MSP 95\\% CI |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['seed']} | {row['msp_aurc']:.4f} | "
            f"{row['pmrc_aurc']:.4f} | {row['relative_gain_pct']:+.2f}% | "
            f"[{row['delta_aurc_ci95_low']:+.4f}, "
            f"{row['delta_aurc_ci95_high']:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "| Objective | Temperature | pNorm | KNN | Trust Score | PMRC |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['method']} | {fmt(row['temperature_aurc_mean'])} | "
            f"{fmt(row['pnorm_aurc_mean'])} | {fmt(row['knn_aurc_mean'])} | "
            f"{fmt(row['trust_aurc_mean'])} | "
            f"{row['pmrc_aurc_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "These results test transfer across three training objectives. "
            "They do not substitute for evaluations of DGCL or CYFLOD.",
        ]
    )
    (OUTPUT_ROOT / "training_objective_transfer.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
