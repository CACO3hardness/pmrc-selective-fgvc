from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgel.data import (
    CUBManifestDataset,
    CUBTwoViewDataset,
    build_transforms,
    load_manifest,
    load_noise_map,
)
from rgel.models import build_model
from rgel.utils import read_json, write_json
from scripts.posthoc_reliability import (
    aurc_from_scores,
    cross_validate_logistic,
    extract_features,
    fit_logistic,
    reliability_features,
    risk_at_coverage,
    score_logistic,
)
from scripts.prototype_contamination_diagnostic import DEFAULT_RUNS
from scripts.prototype_independence import stratified_auc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Intervene on prototype labels by loss filtering or clean labels."
    )
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--filter-fraction", type=float, default=0.40)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def extract_train_data(
    config: dict,
    run_dir: Path,
    train_rows,
    num_classes: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    noise_map = load_noise_map(config.get("noise_csv"))
    dataset = CUBManifestDataset(
        config["root"],
        train_rows,
        build_transforms(int(config.get("image_size", 224)), train=False),
        noise_map=noise_map,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=False,
        num_workers=0,
    )
    model = build_model(
        config["model"],
        num_classes,
        pretrained=False,
        freeze_stages=int(config.get("freeze_stages", 0)),
    ).to(device)
    checkpoint = torch.load(
        run_dir / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    feature_batches = []
    noisy_batches = []
    clean_batches = []
    loss_batches = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        noisy = batch["target"].to(device, non_blocking=True)
        clean = batch["clean_target"].to(device, non_blocking=True)
        features, logits = extract_features(model, images)
        losses = F.cross_entropy(logits, noisy, reduction="none")
        feature_batches.append(F.normalize(features.float(), dim=1).cpu().numpy())
        noisy_batches.append(noisy.cpu().numpy())
        clean_batches.append(clean.cpu().numpy())
        loss_batches.append(losses.float().cpu().numpy())
    return (
        np.concatenate(feature_batches),
        np.concatenate(noisy_batches),
        np.concatenate(clean_batches),
        np.concatenate(loss_batches),
    )


def prototypes_from_labels(
    features: np.ndarray,
    labels: np.ndarray,
    selected: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    sums = np.zeros((num_classes, features.shape[1]), dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.int64)
    np.add.at(sums, labels[selected], features[selected])
    np.add.at(counts, labels[selected], 1)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Prototype intervention has empty classes: {missing[:10]}")
    prototypes = sums / counts[:, None]
    prototypes /= np.clip(
        np.linalg.norm(prototypes, axis=1, keepdims=True),
        1e-12,
        None,
    )
    return prototypes.astype(np.float32)


def loss_filter_mask(
    noisy_labels: np.ndarray,
    losses: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    keep = np.ones(len(noisy_labels), dtype=bool)
    excluded = np.zeros(len(noisy_labels), dtype=bool)
    for class_id in np.unique(noisy_labels):
        class_indices = np.flatnonzero(noisy_labels == class_id)
        remove_count = int(round(fraction * len(class_indices)))
        if remove_count >= len(class_indices):
            remove_count = len(class_indices) - 1
        order = class_indices[np.argsort(-losses[class_indices])]
        excluded[order[:remove_count]] = True
        keep[order[:remove_count]] = False
    return keep, excluded


def score_condition(
    config: dict,
    run_dir: Path,
    val_rows,
    test_rows,
    prototypes: np.ndarray,
    device: torch.device,
    weight_decay: float,
    folds_count: int = 5,
) -> dict[str, float | str]:
    model = build_model(
        config["model"],
        len(prototypes),
        pretrained=False,
        freeze_stages=int(config.get("freeze_stages", 0)),
    ).to(device)
    checkpoint = torch.load(
        run_dir / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    image_size = int(config.get("image_size", 224))
    batch_size = int(config.get("batch_size", 32))
    val_loader = DataLoader(
        CUBTwoViewDataset(
            config["root"],
            val_rows,
            build_transforms(image_size, False),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        CUBTwoViewDataset(
            config["root"],
            test_rows,
            build_transforms(image_size, False),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    val_features, val_labels, val_predictions = reliability_features(
        model,
        val_loader,
        prototypes,
        device,
    )
    test_features, test_labels, test_predictions = reliability_features(
        model,
        test_loader,
        prototypes,
        device,
    )
    val_correct = val_predictions == val_labels
    test_correct = test_predictions == test_labels
    calibrator, mean, std = fit_logistic(
        val_features,
        val_correct,
        weight_decay,
    )
    val_scores = score_logistic(calibrator, val_features, mean, std)
    test_scores = score_logistic(calibrator, test_features, mean, std)
    validation_gate = stratified_auc(
        val_features[:, 0],
        val_features[:, 9],
        val_correct,
        5,
    )["weighted_auc"]
    use_pmrc = bool(validation_gate is not None and validation_gate > 0.5)
    deployed_scores = test_scores if use_pmrc else test_features[:, 0]
    msp_aurc = aurc_from_scores(test_features[:, 0], test_correct)
    pmrc_aurc = aurc_from_scores(test_scores, test_correct)
    gated_aurc = aurc_from_scores(deployed_scores, test_correct)
    return {
        "msp_aurc": float(msp_aurc),
        "pmrc_aurc": float(pmrc_aurc),
        "gated_aurc": float(gated_aurc),
        "relative_gain_pct": float(100.0 * (msp_aurc - pmrc_aurc) / msp_aurc),
        "validation_gate_auc": float(validation_gate),
        "selected_method": "pmrc" if use_pmrc else "msp",
        "risk90": float(risk_at_coverage(deployed_scores, test_correct, 0.9)),
    }


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    for condition in ("all_noisy", "loss_filtered", "clean_oracle"):
        selected = [row for row in rows if row["condition"] == condition]
        output.append(
            {
                "condition": condition,
                "n_runs": len(selected),
                "mean_relative_gain_pct": float(
                    np.mean([row["relative_gain_pct"] for row in selected])
                ),
                "mean_gated_gain_pct": float(
                    np.mean(
                        [
                            100.0
                            * (row["msp_aurc"] - row["gated_aurc"])
                            / row["msp_aurc"]
                            for row in selected
                        ]
                    )
                ),
                "pmrc_selected": int(
                    sum(row["selected_method"] == "pmrc" for row in selected)
                ),
                "mean_validation_gate_auc": float(
                    np.mean([row["validation_gate_auc"] for row in selected])
                ),
            }
        )
    return output


def pairwise_summaries(rows: list[dict]) -> list[dict]:
    by_run: dict[str, dict[str, float]] = {}
    for row in rows:
        by_run.setdefault(row["run_name"], {})[row["condition"]] = float(
            row["relative_gain_pct"]
        )
    output = []
    rng = np.random.default_rng(17)
    for improved, baseline in (
        ("loss_filtered", "all_noisy"),
        ("clean_oracle", "loss_filtered"),
        ("clean_oracle", "all_noisy"),
    ):
        differences = np.asarray(
            [
                values[improved] - values[baseline]
                for values in by_run.values()
                if improved in values and baseline in values
            ]
        )
        bootstrap = np.asarray(
            [
                differences[
                    rng.integers(0, len(differences), len(differences))
                ].mean()
                for _ in range(5000)
            ]
        )
        output.append(
            {
                "improved": improved,
                "baseline": baseline,
                "n_runs": int(len(differences)),
                "mean_difference_pct": float(differences.mean()),
                "ci95": [
                    float(np.quantile(bootstrap, 0.025)),
                    float(np.quantile(bootstrap, 0.975)),
                ],
                "runs_improved": int(np.sum(differences > 0)),
            }
        )
    return output


def write_outputs(output_dir: Path, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = summarize(rows)
    pairwise = pairwise_summaries(rows)
    write_json(
        output_dir / "prototype_purification_intervention.json",
        {"summaries": summaries, "pairwise": pairwise, "rows": rows},
    )
    with (output_dir / "prototype_purification_intervention.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Prototype purification intervention",
        "",
        "This intervention changes only prototype labels or the samples used to "
        "construct prototypes. The classifier, calibration split, descriptors, "
        "and evaluation test set remain fixed.",
        "",
        "| Condition | Runs | Mean PMRC gain | Mean gated gain | PMRC selected | Mean gate AUROC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['condition']} | {row['n_runs']} | "
            f"{row['mean_relative_gain_pct']:+.2f}% | "
            f"{row['mean_gated_gain_pct']:+.2f}% | "
            f"{row['pmrc_selected']}/{row['n_runs']} | "
            f"{row['mean_validation_gate_auc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "| Comparison | Mean difference | 95% CI | Runs improved |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in pairwise:
        lines.append(
            f"| {row['improved']} vs {row['baseline']} | "
            f"{row['mean_difference_pct']:+.2f} pp | "
            f"[{row['ci95'][0]:+.2f}, {row['ci95'][1]:+.2f}] | "
            f"{row['runs_improved']}/{row['n_runs']} |"
        )
    (output_dir / "prototype_purification_intervention.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = args.output_dir / "prototype_purification_intervention.json"
    output_rows = []
    if (args.append or args.summarize_only) and output_path.is_file():
        output_rows = json.loads(
            output_path.read_text(encoding="utf-8")
        )["rows"]
    if args.summarize_only:
        write_outputs(args.output_dir, output_rows)
        print(json.dumps(summarize(output_rows), indent=2))
        return
    for run_name in args.runs:
        run_dir = args.runs_root / run_name
        if not (run_dir / "best.pt").is_file():
            continue
        config = read_json(run_dir / "config.json")
        rows = load_manifest(config["manifest"])
        train_rows = [row for row in rows if row.split == "train"]
        val_rows = [row for row in rows if row.split == "val"]
        test_rows = [row for row in rows if row.split == "test"]
        num_classes = max(row.class_id for row in rows) + 1
        features, noisy_labels, clean_labels, losses = extract_train_data(
            config,
            run_dir,
            train_rows,
            num_classes,
            device,
        )
        keep, excluded = loss_filter_mask(
            noisy_labels,
            losses,
            args.filter_fraction,
        )
        posthoc = read_json(run_dir / "posthoc_reliability.json")
        weight_decay = float(posthoc["selected_weight_decay"])
        conditions = {
            "all_noisy": prototypes_from_labels(
                features,
                noisy_labels,
                np.ones(len(features), dtype=bool),
                num_classes,
            ),
            "loss_filtered": prototypes_from_labels(
                features,
                noisy_labels,
                keep,
                num_classes,
            ),
            "clean_oracle": prototypes_from_labels(
                features,
                clean_labels,
                np.ones(len(features), dtype=bool),
                num_classes,
            ),
        }
        run_rows = []
        for condition, prototypes in conditions.items():
            metrics = score_condition(
                config,
                run_dir,
                val_rows,
                test_rows,
                prototypes,
                device,
                weight_decay,
            )
            run_rows.append(
                {
                    "run_name": run_name,
                    "dataset": str(config.get("dataset", "unknown")),
                    "condition": condition,
                    "filtered_samples": int(excluded.sum()),
                    "filtered_corruption_fraction": (
                        float((noisy_labels[excluded] != clean_labels[excluded]).mean())
                        if excluded.any()
                        else 0.0
                    ),
                    **metrics,
                }
            )
        output_rows = [
            row for row in output_rows if row["run_name"] != run_name
        ] + run_rows
        write_outputs(args.output_dir, output_rows)
        print(run_name, "complete", flush=True)

    summaries = summarize(output_rows)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
