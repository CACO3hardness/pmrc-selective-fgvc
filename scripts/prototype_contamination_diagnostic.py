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
    build_transforms,
    load_manifest,
    load_noise_map,
)
from rgel.models import build_model
from rgel.utils import read_json, write_json
from scripts.posthoc_reliability import extract_features
from scripts.prototype_independence import roc_auc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "outputs"
DEFAULT_RUNS = [
    "elr_resnet50_cub_semantic40_seed0",
    "elr_resnet50_cub_semantic40_seed1",
    "elr_resnet50_cub_semantic40_seed2",
    "elr_resnet50_aircraft_semantic40_seed0",
    "elr_resnet50_flowers102_semantic375_seed0",
    "elr_resnet50_flowers102_semantic375_seed1",
    "elr_resnet50_flowers102_semantic375_seed2",
    "elr_resnet18_cifar100n_human_seed0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distinguish prototype contamination from feature concentration."
    )
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    return parser.parse_args()


def build_prototypes(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((num_classes, features.shape[1]), dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.int64)
    np.add.at(sums, labels, features)
    np.add.at(counts, labels, 1)
    if np.any(counts == 0):
        raise ValueError("Prototype diagnostic requires every class in training")
    prototypes = sums / counts[:, None]
    prototypes /= np.clip(
        np.linalg.norm(prototypes, axis=1, keepdims=True),
        1e-12,
        None,
    )
    return prototypes.astype(np.float32), counts


def prototype_margin(
    features: np.ndarray,
    prototypes: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    similarity = features @ prototypes.T
    observed = similarity[np.arange(len(labels)), labels].copy()
    similarity[np.arange(len(labels)), labels] = -np.inf
    return observed - similarity.max(axis=1)


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2:
        return None
    left = left[valid]
    right = right[valid]
    if left.std() < 1e-12 or right.std() < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


@torch.inference_mode()
def extract_train_views(
    config: dict,
    run_dir: Path,
    train_rows,
    num_classes: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        features, _ = extract_features(model, images)
        feature_batches.append(
            F.normalize(features.float(), dim=1).cpu().numpy()
        )
        noisy_batches.append(batch["target"].numpy())
        clean_batches.append(batch["clean_target"].numpy())
    return (
        np.concatenate(feature_batches),
        np.concatenate(noisy_batches),
        np.concatenate(clean_batches),
    )


def run_diagnostic(
    run_dir: Path,
    device: torch.device,
) -> dict[str, float | int | str | None]:
    config = read_json(run_dir / "config.json")
    rows = load_manifest(config["manifest"])
    train_rows = [row for row in rows if row.split == "train"]
    num_classes = max(row.class_id for row in rows) + 1

    features, noisy_labels, clean_labels = extract_train_views(
        config,
        run_dir,
        train_rows,
        num_classes,
        device,
    )
    noisy_prototypes, _ = build_prototypes(
        features,
        noisy_labels,
        num_classes,
    )
    clean_prototypes, clean_counts = build_prototypes(
        features,
        clean_labels,
        num_classes,
    )

    prototype_cosine = np.sum(noisy_prototypes * clean_prototypes, axis=1)
    observed_similarity = np.sum(features * noisy_prototypes[noisy_labels], axis=1)
    true_similarity = np.sum(features * clean_prototypes[clean_labels], axis=1)
    noisy_margin = prototype_margin(features, noisy_prototypes, noisy_labels)
    clean_margin = prototype_margin(features, clean_prototypes, clean_labels)
    corrupted = noisy_labels != clean_labels

    class_corruption_rate = np.array(
        [
            float(corrupted[clean_labels == class_id].mean())
            if np.any(clean_labels == class_id)
            else np.nan
            for class_id in range(num_classes)
        ]
    )
    concentration = np.array(
        [
            true_similarity[clean_labels == class_id].mean()
            for class_id in range(num_classes)
        ]
    )
    prototype_corruption_correlation = safe_correlation(
        prototype_cosine,
        class_corruption_rate,
    )
    concentration_corruption_correlation = safe_correlation(
        concentration,
        class_corruption_rate,
    )

    return {
        "run_name": run_dir.name,
        "dataset": str(config.get("dataset", "unknown")),
        "model": str(config.get("model", "unknown")),
        "seed": int(config.get("seed", 0)),
        "train_images": int(len(features)),
        "corruption_rate": float(corrupted.mean()),
        "mean_prototype_cosine": float(prototype_cosine.mean()),
        "min_prototype_cosine": float(prototype_cosine.min()),
        "mean_similarity_noisy_prototype": float(observed_similarity.mean()),
        "mean_similarity_clean_prototype": float(true_similarity.mean()),
        "noisy_margin_clean": float(noisy_margin[~corrupted].mean()),
        "noisy_margin_corrupt": float(noisy_margin[corrupted].mean()),
        "clean_margin_clean": float(clean_margin[~corrupted].mean()),
        "clean_margin_corrupt": float(clean_margin[corrupted].mean()),
        "noisy_margin_corruption_auc": float(roc_auc(-noisy_margin, corrupted)),
        "clean_margin_corruption_auc": float(roc_auc(-clean_margin, corrupted)),
        "prototype_cosine_vs_class_corruption_corr": prototype_corruption_correlation,
        "within_class_concentration_vs_corruption_corr": (
            concentration_corruption_correlation
        ),
        "clean_prototype_class_count": int(np.sum(clean_counts > 0)),
    }


def write_outputs(output_dir: Path, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "prototype_contamination.json", {"runs": rows})
    with (output_dir / "prototype_contamination.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Prototype contamination versus concentration",
        "",
        "Clean prototypes are a diagnostic upper bound and are never available "
        "to the deployed method.",
        "",
        "| Run | Corruption | Prototype cosine | Noisy margin clean/corrupt | "
        "Noisy-margin corruption AUROC | Clean-margin corruption AUROC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run_name']} | {100 * row['corruption_rate']:.1f}% | "
            f"{row['mean_prototype_cosine']:.4f} | "
            f"{row['noisy_margin_clean']:.4f} / "
            f"{row['noisy_margin_corrupt']:.4f} | "
            f"{row['noisy_margin_corruption_auc']:.4f} | "
            f"{row['clean_margin_corruption_auc']:.4f} |"
        )
    (output_dir / "prototype_contamination.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for run_name in args.runs:
        run_dir = args.runs_root / run_name
        if not (run_dir / "best.pt").is_file():
            print(f"Skipping missing run: {run_name}", flush=True)
            continue
        row = run_diagnostic(run_dir, device)
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
    write_outputs(args.output_dir, rows)


if __name__ == "__main__":
    main()
