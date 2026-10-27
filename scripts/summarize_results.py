from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgel.phase2_queue import dataset_settings, task_name
from rgel.utils import read_json, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT.parent.parent / "outputs"
PHASE2_CONFIG = PROJECT_ROOT / "configs" / "phase2_queue.yaml"
NOISE_PATTERN = re.compile(
    r"^(?P<dataset>.+)_(?P<kind>semantic|symmetric)_(?P<rate>\d+)_seed(?P<seed>\d+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize PMRC experiment results.")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--phase2-config",
        type=Path,
        default=PHASE2_CONFIG,
    )
    return parser.parse_args()


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return float(array.mean()), std


def mean_optional(values: list[float | None]) -> tuple[float | None, int]:
    present = [float(value) for value in values if value is not None]
    return (float(np.mean(present)) if present else None, len(present))


def infer_noise(config: dict) -> tuple[str, float, float | None]:
    dataset = str(config.get("dataset", "")).lower()
    root = str(config.get("root", "")).lower()
    if dataset.startswith("webfg_") or "webfg_496" in root:
        # WebFG supplies naturally web-collected noisy training labels. Its
        # corruption rate is uncontrolled and therefore has no requested rate.
        return "web", 0.0, None
    raw_path = config.get("noise_csv")
    if not raw_path:
        return "clean", 0.0, 0.0
    noise_path = Path(str(raw_path))
    if noise_path.stem.startswith("cifar100n_human"):
        metadata_path = PROJECT_ROOT / "data" / "CIFAR100N" / "metadata.json"
        empirical_rate = None
        if metadata_path.is_file():
            try:
                empirical_rate = float(
                    read_json(metadata_path)["empirical_noise_rate"]
                )
            except (KeyError, TypeError, ValueError):
                empirical_rate = None
        return "human", float(empirical_rate or 0.402), empirical_rate
    match = NOISE_PATTERN.match(noise_path.stem)
    if match is None:
        return "unknown", 0.0, None
    kind = match.group("kind")
    requested_rate = int(match.group("rate")) / 100.0
    metadata_path = noise_path.with_suffix(".json")
    empirical_rate = None
    if metadata_path.is_file():
        try:
            metadata = read_json(metadata_path)
            empirical_rate = float(
                metadata.get("rate_empirical", metadata.get("empirical_noise_rate"))
            )
        except (KeyError, TypeError, ValueError):
            empirical_rate = None
    return kind, requested_rate, empirical_rate


def load_run(run_dir: Path) -> dict | None:
    posthoc_path = run_dir / "posthoc_reliability.json"
    config_path = run_dir / "config.json"
    if not posthoc_path.is_file() or not config_path.is_file():
        return None
    config = read_json(config_path)
    posthoc = read_json(posthoc_path)
    kind, requested_rate, empirical_rate = infer_noise(config)
    test_results = posthoc["test_results"]
    msp = test_results["msp"]
    pmrc = test_results["learned_reliability"]
    dataset = str(config.get("dataset", "unknown")).lower()
    if dataset in {"cub", "cub200"}:
        dataset = "cub"
    elif dataset.startswith("aircraft"):
        dataset = "aircraft"
    elif dataset.startswith("flowers"):
        dataset = "flowers102"
    elif dataset == "unknown":
        lowered_name = run_dir.name.lower()
        if "cub" in lowered_name:
            dataset = "cub"
        elif "aircraft" in lowered_name:
            dataset = "aircraft"
        elif "flowers" in lowered_name:
            dataset = "flowers102"
    row = {
        "run_name": run_dir.name,
        "dataset": dataset,
        "method": str(config.get("method", "unknown")),
        "model": str(config.get("model", "unknown")),
        "noise_kind": kind,
        "requested_noise_rate": requested_rate,
        "empirical_noise_rate": empirical_rate,
        "seed": int(config.get("seed", 0)),
        "test_accuracy": float(posthoc["test_accuracy"]),
        "msp_aurc": float(msp["aurc"]),
        "pmrc_aurc": float(pmrc["aurc"]),
        "aurc_improvement": float(msp["aurc"] - pmrc["aurc"]),
        "relative_aurc_improvement_pct": float(
            100.0 * (msp["aurc"] - pmrc["aurc"]) / msp["aurc"]
        ),
        "msp_risk_at_90": float(msp["risk_at_90"]),
        "pmrc_risk_at_90": float(pmrc["risk_at_90"]),
        "risk90_improvement": float(msp["risk_at_90"] - pmrc["risk_at_90"]),
        "bootstrap_ci95_low": None,
        "bootstrap_ci95_high": None,
        "pnorm_aurc": None,
        "temperature_scaling_aurc": None,
        "knn_agreement_aurc": None,
        "embedding_trust_aurc": None,
        "prototype_within_msp_auc": None,
        "prototype_within_msp_auc_ci95_low": None,
        "prototype_within_msp_auc_ci95_high": None,
        "class_pair_delta_aurc": None,
        "class_pair_delta_ci95_low": None,
        "class_pair_delta_ci95_high": None,
    }
    ci = posthoc.get("learned_minus_msp_aurc_ci95")
    if isinstance(ci, list) and len(ci) == 2:
        row["bootstrap_ci95_low"] = float(ci[0])
        row["bootstrap_ci95_high"] = float(ci[1])

    baseline_path = run_dir / "selective_baselines.json"
    if baseline_path.is_file():
        baseline = read_json(baseline_path)
        pnorm = baseline.get("test", {}).get("maxlogit_pnorm", {})
        if "aurc" in pnorm:
            row["pnorm_aurc"] = float(pnorm["aurc"])
        temperature_scaling = baseline.get("test", {}).get(
            "temperature_scaling",
            {},
        )
        if "aurc" in temperature_scaling:
            row["temperature_scaling_aurc"] = float(temperature_scaling["aurc"])

    embedding_baseline_path = run_dir / "selective_embedding_baselines.json"
    if embedding_baseline_path.is_file():
        embedding_baseline = read_json(embedding_baseline_path)
        baselines = embedding_baseline.get("baselines", {})
        knn = baselines.get("knn_label_agreement", {})
        trust = baselines.get("trust_score", {})
        if "test_aurc" in knn:
            row["knn_agreement_aurc"] = float(knn["test_aurc"])
        if "test_aurc" in trust:
            row["embedding_trust_aurc"] = float(trust["test_aurc"])

    independence_path = run_dir / "prototype_independence.json"
    if independence_path.is_file():
        independence = read_json(independence_path)
        stratified = independence["stratified_by_msp"]
        row["prototype_within_msp_auc"] = float(stratified["weighted_auc"])
        ci = stratified.get("weighted_auc_ci95")
        if isinstance(ci, list) and len(ci) == 2:
            row["prototype_within_msp_auc_ci95_low"] = float(ci[0])
            row["prototype_within_msp_auc_ci95_high"] = float(ci[1])

    class_pair_path = run_dir / "class_pair_reliability.json"
    if class_pair_path.is_file():
        class_pair = read_json(class_pair_path)
        row["class_pair_delta_aurc"] = float(
            class_pair["all_plus_pair_minus_current_all_aurc"]
        )
        ci = class_pair.get("all_plus_pair_minus_current_all_aurc_ci95")
        if isinstance(ci, list) and len(ci) == 2:
            row["class_pair_delta_ci95_low"] = float(ci[0])
            row["class_pair_delta_ci95_high"] = float(ci[1])
    return row


def expected_phase2_runs(config_path: Path) -> list[dict]:
    if not config_path.is_file():
        return []
    with config_path.open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    expected = []
    for dataset_spec in spec["tasks"]:
        dataset = str(dataset_spec["dataset"])
        for seed in dataset_spec["seeds"]:
            for noise_spec in dataset_spec["noises"]:
                kind = str(noise_spec["kind"])
                rate = float(noise_spec["rate"])
                dataset_settings(dataset, int(seed))
                name = task_name(dataset, kind, rate, int(seed))
                expected.append(
                    {
                        "task": name,
                        "run_name": f"elr_resnet50_{name}",
                        "dataset": dataset,
                        "seed": int(seed),
                        "noise_kind": kind,
                        "requested_noise_rate": rate,
                    }
                )
    return expected


def cell_key(row: dict) -> tuple[str, str, str, str, float]:
    return (
        row["dataset"],
        row["method"],
        row["model"],
        row["noise_kind"],
        float(row["requested_noise_rate"]),
    )


def summarize_cells(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str, float], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[cell_key(row)].append(row)

    summaries = []
    for (dataset, method, model, kind, rate), cell_rows in sorted(grouped.items()):
        cell_rows.sort(key=lambda item: item["seed"])
        msp_mean, msp_std = mean_std([row["msp_aurc"] for row in cell_rows])
        pmrc_mean, pmrc_std = mean_std([row["pmrc_aurc"] for row in cell_rows])
        risk_msp_mean, risk_msp_std = mean_std(
            [row["msp_risk_at_90"] for row in cell_rows]
        )
        risk_pmrc_mean, risk_pmrc_std = mean_std(
            [row["pmrc_risk_at_90"] for row in cell_rows]
        )
        relative = [row["relative_aurc_improvement_pct"] for row in cell_rows]
        pnorm_values = [
            row["pnorm_aurc"] for row in cell_rows if row["pnorm_aurc"] is not None
        ]
        temperature_aurc, temperature_count = mean_optional(
            [row["temperature_scaling_aurc"] for row in cell_rows]
        )
        knn_aurc, knn_count = mean_optional(
            [row["knn_agreement_aurc"] for row in cell_rows]
        )
        trust_aurc, trust_count = mean_optional(
            [row["embedding_trust_aurc"] for row in cell_rows]
        )
        within_msp_auc, within_msp_count = mean_optional(
            [row["prototype_within_msp_auc"] for row in cell_rows]
        )
        summaries.append(
            {
                "dataset": dataset,
                "method": method,
                "model": model,
                "noise_kind": kind,
                "requested_noise_rate": rate,
                "mean_empirical_noise_rate": (
                    float(
                        np.mean(
                            [
                                row["empirical_noise_rate"]
                                for row in cell_rows
                                if row["empirical_noise_rate"] is not None
                            ]
                        )
                    )
                    if any(
                        row["empirical_noise_rate"] is not None for row in cell_rows
                    )
                    else None
                ),
                "seeds": [row["seed"] for row in cell_rows],
                "n_seeds": len(cell_rows),
                "msp_aurc_mean": msp_mean,
                "msp_aurc_std": msp_std,
                "pmrc_aurc_mean": pmrc_mean,
                "pmrc_aurc_std": pmrc_std,
                "relative_aurc_improvement_pct_mean": float(np.mean(relative)),
                "relative_aurc_improvement_pct_min": float(np.min(relative)),
                "relative_aurc_improvement_pct_max": float(np.max(relative)),
                "all_seeds_improved": bool(all(value > 0 for value in relative)),
                "msp_risk90_mean": risk_msp_mean,
                "msp_risk90_std": risk_msp_std,
                "pmrc_risk90_mean": risk_pmrc_mean,
                "pmrc_risk90_std": risk_pmrc_std,
                "pnorm_aurc_mean": (
                    float(np.mean(pnorm_values)) if pnorm_values else None
                ),
                "pnorm_seeds_available": len(pnorm_values),
                "temperature_aurc_mean": temperature_aurc,
                "temperature_seeds_available": temperature_count,
                "knn_agreement_aurc_mean": knn_aurc,
                "knn_seeds_available": knn_count,
                "embedding_trust_aurc_mean": trust_aurc,
                "trust_seeds_available": trust_count,
                "prototype_within_msp_auc_mean": within_msp_auc,
                "within_msp_seeds_available": within_msp_count,
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_rate(value: float | None, kind: str | None = None) -> str:
    if value is None:
        return "not measured"
    if kind == "web":
        return "uncontrolled"
    return f"{100.0 * value:.1f}%"


def write_markdown(
    path: Path,
    rows: list[dict],
    summaries: list[dict],
    missing_runs: list[dict],
) -> None:
    lines = [
        "# PMRC 结果自动汇总",
        "",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "指标均为测试集结果。PMRC 相对改善按每个种子计算后取均值；",
        "n=1 时标准差记为 0，不能据此声称统计显著性。",
        "",
        "## 按实验设置汇总",
        "",
            "| 数据集 | 方法 | Backbone | 噪声 | 请求率 | 实际率 | 种子数 | MSP AURC | PMRC AURC | 相对改善 | 90%覆盖风险 MSP → PMRC |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
                "| {dataset} | {method} | {model} | {kind} | {requested} | {empirical} | "
            "{n_seeds} | {msp:.4f} ± {msp_std:.4f} | "
            "{pmrc:.4f} ± {pmrc_std:.4f} | {relative:.1f}% "
            "({relative_min:.1f}–{relative_max:.1f}%) | "
            "{risk_msp:.4f} → {risk_pmrc:.4f} |".format(
                dataset=item["dataset"],
                method=item["method"],
                model=item["model"],
                kind=item["noise_kind"],
                requested=format_rate(
                    item["requested_noise_rate"],
                    item["noise_kind"],
                ),
                empirical=format_rate(item["mean_empirical_noise_rate"]),
                n_seeds=item["n_seeds"],
                msp=item["msp_aurc_mean"],
                msp_std=item["msp_aurc_std"],
                pmrc=item["pmrc_aurc_mean"],
                pmrc_std=item["pmrc_aurc_std"],
                relative=item["relative_aurc_improvement_pct_mean"],
                relative_min=item["relative_aurc_improvement_pct_min"],
                relative_max=item["relative_aurc_improvement_pct_max"],
                risk_msp=item["msp_risk90_mean"],
                risk_pmrc=item["pmrc_risk90_mean"],
            )
        )

    lines.extend(
        [
            "",
            "### 按设置的基线均值",
            "",
            "只对已有对应结果的种子取均值；样本数与主结果种子数不一定相同。",
            "",
            "| 数据集/方法/backbone/噪声 | 率 | PMRC | Temperature | pNorm | KNN | Trust | 分层内原型 AUROC |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summaries:
        def display_optional(value: float | None) -> str:
            return "—" if value is None else f"{value:.4f}"

        lines.append(
            f"| {item['dataset']}/{item['method']}/{item['model']}/{item['noise_kind']} | "
            f"{format_rate(item['requested_noise_rate'], item['noise_kind'])} | "
            f"{item['pmrc_aurc_mean']:.4f} | "
            f"{display_optional(item['temperature_aurc_mean'])} | "
            f"{display_optional(item['pnorm_aurc_mean'])} | "
            f"{display_optional(item['knn_agreement_aurc_mean'])} | "
            f"{display_optional(item['embedding_trust_aurc_mean'])} | "
            f"{display_optional(item['prototype_within_msp_auc_mean'])} |"
        )

    lines.extend(
        [
            "",
            "## 逐种子结果",
            "",
            "| Run | Seed | Acc | MSP AURC | PMRC AURC | 改善 | PMRC−MSP 的 bootstrap 95% CI |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda item: item["run_name"]):
        ci = "—"
        if row["bootstrap_ci95_low"] is not None:
            ci = (
                f"[{row['bootstrap_ci95_low']:.4f}, "
                f"{row['bootstrap_ci95_high']:.4f}]"
            )
        lines.append(
            f"| {row['run_name']} | {row['seed']} | "
            f"{100.0 * row['test_accuracy']:.2f}% | {row['msp_aurc']:.4f} | "
            f"{row['pmrc_aurc']:.4f} | "
            f"{row['relative_aurc_improvement_pct']:.1f}% | {ci} |"
        )

    analysis_rows = [
        row
        for row in rows
        if any(
            row[field] is not None
            for field in (
                "pnorm_aurc",
                "temperature_scaling_aurc",
                "knn_agreement_aurc",
                "embedding_trust_aurc",
                "prototype_within_msp_auc",
            )
        )
    ]
    lines.extend(
        [
            "",
            "## 强基线与独立性分析",
            "",
            "| Run | PMRC AURC | Temperature AURC | pNorm AURC | KNN 一致性 AURC | Trust Score AURC | MSP 分层内的原型 AUROC |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(analysis_rows, key=lambda item: item["run_name"]):
        def display(value: float | None) -> str:
            return "—" if value is None else f"{value:.4f}"

        within_auc = "—"
        if row["prototype_within_msp_auc"] is not None:
            within_auc = f"{row['prototype_within_msp_auc']:.4f}"
            if row["prototype_within_msp_auc_ci95_low"] is not None:
                within_auc += (
                    f" [{row['prototype_within_msp_auc_ci95_low']:.4f}, "
                    f"{row['prototype_within_msp_auc_ci95_high']:.4f}]"
                )
        lines.append(
            f"| {row['run_name']} | {row['pmrc_aurc']:.4f} | "
            f"{display(row['temperature_scaling_aurc'])} | "
            f"{display(row['pnorm_aurc'])} | "
            f"{display(row['knn_agreement_aurc'])} | "
            f"{display(row['embedding_trust_aurc'])} | {within_auc} |"
        )

    class_pair_rows = [
        row for row in rows if row["class_pair_delta_aurc"] is not None
    ]
    lines.extend(
        [
            "",
            "## 类对级增量复核",
            "",
            "ΔAURC = all+pair − current_all；负值表示加入类对特征更好。",
            "",
            "| Run | ΔAURC | 配对 bootstrap 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for row in sorted(class_pair_rows, key=lambda item: item["run_name"]):
        lines.append(
            f"| {row['run_name']} | {row['class_pair_delta_aurc']:+.4f} | "
            f"[{row['class_pair_delta_ci95_low']:+.4f}, "
            f"{row['class_pair_delta_ci95_high']:+.4f}] |"
        )

    lines.extend(["", "## Phase 2 尚未完成的运行", ""])
    if missing_runs:
        for item in missing_runs:
            lines.append(
                f"- `{item['run_name']}`：{item['dataset']}，"
                f"{item['noise_kind']} {format_rate(item['requested_noise_rate'])}，"
                f"seed {item['seed']}"
            )
    else:
        lines.append("- 无；Phase 2 固定矩阵已全部产生 PMRC 结果。")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    rows = []
    for posthoc_path in sorted(args.runs_root.glob("*/posthoc_reliability.json")):
        row = load_run(posthoc_path.parent)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda item: item["run_name"])
    summaries = summarize_cells(rows)

    available_runs = {row["run_name"] for row in rows}
    expected = expected_phase2_runs(args.phase2_config)
    missing_runs = [
        item for item in expected if item["run_name"] not in available_runs
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_completed_runs": len(rows),
        "runs": rows,
        "summaries": summaries,
        "phase2_expected_runs": len(expected),
        "phase2_completed_runs": len(expected) - len(missing_runs),
        "phase2_missing_runs": missing_runs,
    }
    write_json(args.output_dir / "results_summary_latest.json", output)
    write_csv(args.output_dir / "results_summary_latest.csv", rows)
    write_csv(args.output_dir / "results_by_setting_latest.csv", summaries)
    write_markdown(
        args.output_dir / "results_summary_latest.md",
        rows,
        summaries,
        missing_runs,
    )
    print(
        f"Wrote {len(rows)} completed runs; "
        f"Phase 2 {len(expected) - len(missing_runs)}/{len(expected)} complete."
    )


if __name__ == "__main__":
    main()
