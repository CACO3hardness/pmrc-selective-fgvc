from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "outputs"


def relative_improvement(row: dict, score: str) -> float:
    return 100.0 * (row["msp_aurc"] - row[f"{score}_aurc"]) / row["msp_aurc"]


def main() -> None:
    calibration = json.loads(
        (OUTPUT_ROOT / "calibration_robustness.json").read_text(encoding="utf-8")
    )
    heldout = json.loads(
        (OUTPUT_ROOT / "heldout_class_calibration.json").read_text(
            encoding="utf-8"
        )
    )
    risk_control = json.loads(
        (OUTPUT_ROOT / "risk_control_baseline.json").read_text(encoding="utf-8")
    )
    gate_sensitivity = json.loads(
        (OUTPUT_ROOT / "gate_threshold_sensitivity.json").read_text(
            encoding="utf-8"
        )
    )
    decision_cost = json.loads(
        (OUTPUT_ROOT / "decision_cost_analysis.json").read_text(
            encoding="utf-8"
        )
    )
    risk_coverage = json.loads(
        (OUTPUT_ROOT / "risk_constrained_coverage.json").read_text(
            encoding="utf-8"
        )
    )
    risk_paired = json.loads(
        (OUTPUT_ROOT / "risk_violation_paired.json").read_text(
            encoding="utf-8"
        )
    )
    gate_checks = json.loads(
        (OUTPUT_ROOT / "gate_prospective_checks.json").read_text(
            encoding="utf-8"
        )
    )
    contamination = json.loads(
        (OUTPUT_ROOT / "prototype_contamination.json").read_text(
            encoding="utf-8"
        )
    )
    prototype_association = json.loads(
        (OUTPUT_ROOT / "prototype_gain_association.json").read_text(
            encoding="utf-8"
        )
    )
    prototype_factor = json.loads(
        (OUTPUT_ROOT / "prototype_factor_partition.json").read_text(
            encoding="utf-8"
        )
    )
    prototype_purification = json.loads(
        (OUTPUT_ROOT / "prototype_purification_intervention.json").read_text(
            encoding="utf-8"
        )
    )
    training_transfer = json.loads(
        (OUTPUT_ROOT / "training_objective_transfer.json").read_text(
            encoding="utf-8"
        )
    )
    stanford_external = json.loads(
        (OUTPUT_ROOT / "stanford_dogs_external_validation.json").read_text(
            encoding="utf-8"
        )
    )
    openset_proxy = json.loads(
        (OUTPUT_ROOT / "openset_webfg_proxy.json").read_text(
            encoding="utf-8"
        )
    )
    strict_openset_payload = json.loads(
        (OUTPUT_ROOT / "openset_cub160.json").read_text(encoding="utf-8")
    )
    strict_openset = strict_openset_payload.get(
        "runs",
        [strict_openset_payload],
    )
    posthoc_cost = json.loads(
        (OUTPUT_ROOT / "posthoc_cost_benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    cv_stability = json.loads(
        (OUTPUT_ROOT / "cv_stability_summary.json").read_text(
            encoding="utf-8"
        )
    )

    lines = [
        "# Expanded validity evidence",
        "",
        "All analyses reuse frozen classifiers. No test labels are used to fit",
        "calibrators, tune gates, or select thresholds.",
        "",
        "## Calibration-label and class-coverage robustness",
        "",
        "| Perturbation | Rate | Evaluations | Mean direct PMRC gain | "
        "Mean gated gain | Non-regression rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    calibration_groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in calibration["rows"]:
        calibration_groups[
            (row["condition"], float(row["perturbation_rate"]))
        ].append(row)
    for (condition, rate), rows in sorted(calibration_groups.items()):
        direct = [relative_improvement(row, "pmrc") for row in rows]
        gated = [relative_improvement(row, "gated") for row in rows]
        label = {
            "clean": "Clean calibration",
            "label_noise": "Symmetric calibration-label noise",
            "missing_classes": "Missing calibration classes",
        }[condition]
        lines.append(
            f"| {label} | {100 * rate:.0f}% | {len(rows)} | "
            f"{np.mean(direct):+.1f}% | {np.mean(gated):+.1f}% | "
            f"{np.mean([value >= 0 for value in gated]):.1%} |"
        )

    lines.extend(
        [
            "",
            "The clean parameter estimate is held fixed within each run. Only",
            "the calibration labels or calibration-class coverage are perturbed.",
            "",
            "## Calibration and gate transfer to unseen classes",
            "",
            "| Held-out fraction | Evaluations | Mean direct PMRC gain | "
            "Mean gated gain | Direct non-regression | Gated non-regression |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    heldout_groups: dict[float, list[dict]] = defaultdict(list)
    for row in heldout["rows"]:
        heldout_groups[float(row["heldout_fraction"])].append(row)
    for fraction, rows in sorted(heldout_groups.items()):
        direct = [relative_improvement(row, "pmrc") for row in rows]
        gated = [relative_improvement(row, "gated") for row in rows]
        lines.append(
            f"| {100 * fraction:.0f}% | {len(rows)} | "
            f"{np.mean(direct):+.1f}% | {np.mean(gated):+.1f}% | "
            f"{np.mean([value >= 0 for value in direct]):.1%} | "
            f"{np.mean([value >= 0 for value in gated]):.1%} |"
        )

    lines.extend(
        [
            "",
            "For each split, calibration uses only the non-held-out classes and",
            "evaluation uses only held-out classes. Repeats use five independent",
            "class partitions.",
            "",
            "## Split LTT risk-control baseline",
            "",
            "Clopper-Pearson upper bounds use a 10% family-wise error budget",
            "after a union correction over all candidate thresholds.",
            "",
            "| Target risk | Score | Runs accepted | Mean test coverage | "
            "Mean test risk |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for target in risk_control["target_risks"]:
        for score in ("msp", "pmrc"):
            rows = [
                row
                for row in risk_control["rows"]
                if float(row["target_risk"]) == float(target)
                and row["score"] == score
            ]
            accepted = [row for row in rows if row["ltt_test_coverage"] > 0]
            risks = [
                row["ltt_test_risk"]
                for row in accepted
                if row["ltt_test_risk"] is not None
            ]
            lines.append(
                f"| {100 * target:.0f}% | {score.upper()} | "
                f"{len(accepted)}/{len(rows)} | "
                f"{np.mean([row['ltt_test_coverage'] for row in rows]):.1%} | "
                f"{(f'{np.mean(risks):.1%}' if risks else 'not available')} |"
            )

    lines.extend(
        [
            "",
            "At the 5% target, the family-wise finite-sample bound accepts no",
            "samples in most runs. The comparison is nonetheless useful because",
            "it quantifies the price of a formal risk guarantee relative to the",
            "empirical calibration used in the main experiments.",
            "",
            "## Empirical and finite-sample risk violations",
            "",
            "| Target risk | Empirical MSP violations | Empirical PMRC violations | "
            "LTT MSP violations | LTT PMRC violations |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for target in risk_coverage["target_risks"]:
        empirical_rows = [
            row
            for row in risk_coverage["rows"]
            if float(row["target_risk"]) == float(target)
        ]
        empirical_msp = sum(
            row["msp_test_risk"] > float(target) for row in empirical_rows
        )
        empirical_pmrc = sum(
            row["pmrc_test_risk"] > float(target) for row in empirical_rows
        )
        ltt_counts = {}
        for score in ("msp", "pmrc"):
            accepted = [
                row
                for row in risk_control["rows"]
                if float(row["target_risk"]) == float(target)
                and row["score"] == score
                and row["ltt_test_coverage"] > 0
                and row["ltt_test_risk"] is not None
            ]
            ltt_counts[score] = sum(
                row["ltt_test_risk"] > float(target) for row in accepted
            )
        lines.append(
            f"| {100 * target:.0f}% | {empirical_msp}/{len(empirical_rows)} | "
            f"{empirical_pmrc}/{len(empirical_rows)} | "
            f"{ltt_counts['msp']} | {ltt_counts['pmrc']} |"
        )

    lines.extend(
        [
            "",
            "Empirical violations are counted over all runs, including runs that "
            "accept no samples. LTT violations are counted among runs that accept "
            "at least one sample.",
            "",
            "Paired McNemar comparisons of empirical violations:",
            "",
            "| Target risk | PMRC-only | MSP-only | Exact p |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in risk_paired["rows"]:
        lines.append(
            f"| {100 * row['target_risk']:.0f}% | "
            f"{row['pmrc_only_violation']} | "
            f"{row['msp_only_violation']} | "
            f"{row['mcnemar_exact_p']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Gate-threshold sensitivity",
            "",
            "| Scope | Threshold | Mean AURC gain | Non-regression | Regressions |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in gate_sensitivity["results"]:
        lines.append(
            f"| {row['scope']} | {row['threshold']:.3f} | "
            f"{row['mean_gain_pct']:+.2f}% | "
            f"{100 * row['nonregression_rate']:.1f}% | {row['worse_count']} |"
        )
    lines.extend(
        [
            "",
            "## Gate transfer and lower-bound checks",
            "",
            "| Held-out WebFG subset | Selected threshold | Held-out PMRC selected | "
            "Mean test gain | Regressions |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in gate_checks["leave_one_subset_out"]:
        lines.append(
            f"| {row['held_out_subset']} | {row['selected_threshold']:.2f} | "
            f"{row['heldout_pmrc_selected']}/{row['heldout_runs']} | "
            f"{row['heldout_mean_gain_pct']:+.2f}% | "
            f"{row['heldout_regressions']} |"
        )
    lower = gate_checks["lower_bound_gate"]["summary"]
    lines.extend(
        [
            "",
            f"The lower-bound gate selects PMRC in {lower['pmrc_selected']}/"
            f"{lower['n_runs']} runs, improves {lower['positive_runs']}, leaves "
            f"{lower['unchanged_runs']} unchanged, and produces "
            f"{lower['regressing_runs']} regressions.",
            "",
            "## Prototype contamination and concentration",
            "",
            "| Run | Corruption | Noisy-clean prototype cosine | "
            "Noisy-margin corruption AUROC | Clean-margin corruption AUROC |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in contamination["runs"]:
        lines.append(
            f"| {row['run_name']} | {100 * row['corruption_rate']:.1f}% | "
            f"{row['mean_prototype_cosine']:.4f} | "
            f"{row['noisy_margin_corruption_auc']:.4f} | "
            f"{row['clean_margin_corruption_auc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Across these eight runs, prototype cosine versus PMRC gain has "
            f"Pearson r={prototype_association['pearson_r']:.2f} "
            f"(p={prototype_association['pearson_p']:.3f}) and Spearman "
            f"rho={prototype_association['spearman_rho']:.2f} "
            f"(p={prototype_association['spearman_p']:.3f}). This exploratory "
            "association is dataset-confounded and is not interpreted causally.",
            "",
            "### Class-level factor partition",
            "",
            "| Factor | Low-group gain | High-group gain | Low minus high | Runs favoring low |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in prototype_factor["summaries"]:
        lines.append(
            f"| {row['factor']} | {row['low_mean_gain_pct']:+.1f}% | "
            f"{row['high_mean_gain_pct']:+.1f}% | "
            f"{row['low_minus_high_pct']:+.1f} pp | "
            f"{row['runs_favoring_low']}/{row['n_runs']} |"
        )
    lines.extend(
        [
            "",
            "### Prototype purification intervention",
            "",
            "| Condition | Mean PMRC gain | Mean gate AUROC | PMRC selected |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in prototype_purification["summaries"]:
        lines.append(
            f"| {row['condition']} | {row['mean_relative_gain_pct']:+.2f}% | "
            f"{row['mean_validation_gate_auc']:.4f} | "
            f"{row['pmrc_selected']}/{row['n_runs']} |"
        )
    for row in prototype_purification["pairwise"]:
        lines.extend(
            [
                "",
                f"{row['improved']} versus {row['baseline']}: "
                f"{row['mean_difference_pct']:+.2f} points "
                f"(95% CI {row['ci95'][0]:+.2f} to {row['ci95'][1]:+.2f}; "
                f"{row['runs_improved']}/{row['n_runs']} runs).",
            ]
        )
    lines.extend(
        [
            "",
            "### WebFG cross-domain open-set proxy",
            "",
            "| Known subset | PMRC AURC gain | MSP unknown AUROC | PMRC unknown AUROC | Gate |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in openset_proxy["runs"]:
        lines.append(
            f"| {row['known_subset']} | {row['relative_gain_pct']:+.2f}% | "
            f"{row['msp_unknown_auroc']:.4f} | "
            f"{row['pmrc_unknown_auroc']:.4f} | "
            f"{row['gate_selected_method']} |"
        )
    lines.extend(
        [
            "",
            "PMRC is not suitable for open-set claims in the tested cross-domain proxy.",
            "",
            "### Strict CUB 160/40 open-set evaluation",
            "",
            "Each classifier was trained on 160 known classes only. The other 40 "
            "CUB classes were excluded from training, validation, prototype "
            "construction, and calibration.",
            "",
            "| Seed | AURC gain | MSP unknown AUROC | PMRC unknown AUROC | Gate |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(strict_openset, key=lambda item: int(item["seed"])):
        lines.append(
            f"| {row['seed']} | {row['relative_gain_pct']:+.2f}% | "
            f"{row['msp_unknown_auroc']:.4f} | "
            f"{row['pmrc_unknown_auroc']:.4f} | "
            f"{row['gate_selected_method']} |"
        )
    strict_gains = [float(row["relative_gain_pct"]) for row in strict_openset]
    lines.extend(
        [
            "",
            f"Across {len(strict_gains)} seeds, the mean strict open-set AURC "
            f"change was {np.mean(strict_gains):+.2f}%. This supports a "
            "closed-set scope rather than a dedicated open-set detector claim.",
        ]
    )
    lines.extend(
        [
            "",
            "## Training-objective transfer",
            "",
            "| Objective | Seeds | MSP AURC | PMRC AURC | Relative gain | "
            "Improving seeds |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in training_transfer["summaries"]:
        lines.append(
            f"| {row['method'].upper()} | {row['n_seeds']} | "
            f"{row['msp_aurc_mean']:.4f} | {row['pmrc_aurc_mean']:.4f} | "
            f"{row['relative_gain_pct_mean']:+.2f}% | "
            f"{row['positive_seeds']}/{row['n_seeds']} |"
        )
    lines.extend(
        [
            "",
            "## Stanford Dogs external-domain validation",
            "",
            "The protocol and gate threshold were frozen before training. "
            "Official test membership was preserved, and exact duplicate "
            "train-side images were removed before splitting.",
            "",
            "| Seed | MSP AURC | PMRC AURC | Relative gain | Gate AUROC | Gate |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in stanford_external["runs"]:
        lines.append(
            f"| {row['seed']} | {row['msp_aurc']:.4f} | "
            f"{row['pmrc_aurc']:.4f} | {row['relative_gain_pct']:+.2f}% | "
            f"{row['validation_gate_auc']:.4f} | "
            f"{row['gate_selected_method']} |"
        )
    stanford_summary = stanford_external["summary"]
    lines.extend(
        [
            "",
            f"Mean relative AURC gain is "
            f"{stanford_summary['mean_relative_gain_pct']:+.2f}%; "
            f"all {stanford_summary['positive_seeds']} completed seeds improve "
            f"and the gate direction matches in "
            f"{stanford_summary['gate_direction_matches']}/"
            f"{len(stanford_external['runs'])} runs.",
            "",
            "Stanford Dogs shares ImageNet lineage with the pretrained backbone. "
            "This validates downstream dataset and split transport, not a "
            "pretraining-independent real-web domain.",
        ]
    )
    lines.extend(
        [
            "",
            "### Cost and calibration stability",
            "",
            "| Run | Single view | Two view ratio | Total post-hoc |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in posthoc_cost["runs"]:
        lines.append(
            f"| {row['run_name']} | {row['test_single_view_seconds']:.1f}s | "
            f"{row['two_view_to_single_view_ratio']:.2f}x | "
            f"{row['full_posthoc_seconds']:.1f}s |"
        )
    lines.extend(
        [
            "",
            f"Across {cv_stability['n_runs']} runs, the median relative span "
            "between best and worst candidate CV AURC was "
            f"{100 * cv_stability['relative_cv_aurc_span']['median']:.1f}%.",
        ]
    )
    lines.extend(
        [
            "",
            "## Decision cost",
            "",
            "Error cost is normalized to one. Thresholds are selected on",
            "calibration data and evaluated on fixed test data.",
            "",
            "| Rejection cost | MSP cost | PMRC cost | PMRC lower | "
            "MSP coverage | PMRC coverage |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rejection_cost in decision_cost["rejection_costs"]:
        selected = [
            row
            for row in decision_cost["rows"]
            if row["rejection_cost"] == rejection_cost
        ]
        by_method = {
            method: {
                row["run_name"]: row
                for row in selected
                if row["score"] == method
            }
            for method in ("msp", "pmrc")
        }
        common = sorted(set(by_method["msp"]) & set(by_method["pmrc"]))
        msp_cost = float(
            np.mean([by_method["msp"][name]["test_cost"] for name in common])
        )
        pmrc_cost = float(
            np.mean([by_method["pmrc"][name]["test_cost"] for name in common])
        )
        lower = sum(
            by_method["pmrc"][name]["test_cost"]
            < by_method["msp"][name]["test_cost"]
            for name in common
        )
        msp_coverage = float(
            np.mean(
                [by_method["msp"][name]["test_coverage"] for name in common]
            )
        )
        pmrc_coverage = float(
            np.mean(
                [by_method["pmrc"][name]["test_coverage"] for name in common]
            )
        )
        lines.append(
            f"| {rejection_cost:.2f} | {msp_cost:.4f} | {pmrc_cost:.4f} | "
            f"{lower}/{len(common)} | {msp_coverage:.1%} | "
            f"{pmrc_coverage:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Artifact files",
            "",
            "- `calibration_robustness.{json,csv}`",
            "- `heldout_class_calibration.{json,csv}`",
            "- `risk_control_baseline.{json,csv}`",
            "- `risk_constrained_coverage.{json,csv}`",
            "- `gate_prospective_checks.{json,md}`",
            "- `gate_threshold_sensitivity.{json,csv}`",
            "- `prototype_contamination.{json,csv,md}`",
            "- `prototype_gain_association.{json,md}`",
            "- `prototype_factor_partition.{json,csv,md}`",
            "- `prototype_purification_intervention.{json,csv,md}`",
            "- `training_objective_transfer.{json,csv,md}`",
            "- `stanford_dogs_external_validation.{json,csv,md}`",
            "- `openset_webfg_proxy.{json,csv,md}`",
            "- `openset_cub160.{json,csv}`",
            "- `openset_cub160_summary.{json,csv,md}`",
            "- `posthoc_cost_benchmark.{json,csv,md}`",
            "- `cv_stability_summary.{json,md}`",
            "- `decision_cost_analysis.{json,csv}`",
        ]
    )
    (OUTPUT_ROOT / "expanded_validity_evidence.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("Wrote expanded_validity_evidence.md")


if __name__ == "__main__":
    main()
