from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgel.data import CUBManifestDataset, CUBTwoViewDataset, build_transforms, load_manifest, load_noise_map
from rgel.models import build_model
from rgel.utils import read_json, write_json
from scripts.posthoc_reliability import (
    class_prototypes,
    cross_validate_logistic,
    fit_logistic,
    reliability_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "outputs"
DEFAULT_RUNS = [
    "elr_resnet50_cub_semantic40_seed0",
    "elr_resnet50_aircraft_semantic40_seed0",
    "elr_resnet50_flowers102_semantic375_seed0",
    "elr_resnet18_cifar100n_human_seed0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PMRC post-hoc cost.")
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def time_single_view(model, loader, device: torch.device) -> float:
    for batch in loader:
        model(batch["image"].to(device, non_blocking=True))
        break
    synchronize(device)
    started = time.perf_counter()
    for batch in loader:
        model(batch["image"].to(device, non_blocking=True))
    synchronize(device)
    return time.perf_counter() - started


def benchmark_run(run_dir: Path, device: torch.device) -> dict:
    config = read_json(run_dir / "config.json")
    rows = load_manifest(config["manifest"])
    train_rows = [row for row in rows if row.split == "train"]
    val_rows = [row for row in rows if row.split == "val"]
    test_rows = [row for row in rows if row.split == "test"]
    num_classes = max(row.class_id for row in rows) + 1
    model = build_model(
        config["model"],
        num_classes,
        pretrained=False,
        freeze_stages=int(config.get("freeze_stages", 0)),
    ).to(device)
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image_size = int(config.get("image_size", 224))
    batch_size = int(config.get("batch_size", 32))
    noise_map = load_noise_map(config.get("noise_csv"))
    train_loader = torch.utils.data.DataLoader(
        CUBManifestDataset(
            config["root"],
            train_rows,
            build_transforms(image_size, False),
            noise_map=noise_map,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        CUBTwoViewDataset(
            config["root"],
            val_rows,
            build_transforms(image_size, False),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        CUBTwoViewDataset(
            config["root"],
            test_rows,
            build_transforms(image_size, False),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    single_view_test_loader = torch.utils.data.DataLoader(
        CUBManifestDataset(
            config["root"],
            test_rows,
            build_transforms(image_size, False),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    synchronize(device)
    started = time.perf_counter()
    prototypes = class_prototypes(
        model,
        train_loader,
        num_classes,
        device,
    )
    synchronize(device)
    prototype_seconds = time.perf_counter() - started

    started = time.perf_counter()
    reliability_features(model, val_loader, prototypes, device)
    synchronize(device)
    validation_seconds = time.perf_counter() - started

    started = time.perf_counter()
    reliability_features(model, test_loader, prototypes, device)
    synchronize(device)
    test_two_view_seconds = time.perf_counter() - started
    single_view_seconds = time_single_view(model, single_view_test_loader, device)

    val_features = np.load(run_dir / "posthoc_val_features.npy")
    val_labels = np.load(run_dir / "posthoc_val_labels.npy")
    val_predictions = np.load(run_dir / "posthoc_val_predictions.npy")
    val_correct = val_predictions == val_labels
    rng = np.random.default_rng(7)
    indices = np.arange(len(val_rows))
    rng.shuffle(indices)
    folds = np.array_split(indices, 5)
    started = time.perf_counter()
    for weight_decay in (1e-4, 1e-3, 1e-2, 1e-1):
        cross_validate_logistic(
            val_features,
            val_correct,
            folds,
            weight_decay,
        )
    calibration_seconds = time.perf_counter() - started
    return {
        "run_name": run_dir.name,
        "dataset": str(config.get("dataset", "unknown")),
        "model": str(config.get("model", "unknown")),
        "train_images": len(train_rows),
        "validation_images": len(val_rows),
        "test_images": len(test_rows),
        "prototype_feature_seconds": prototype_seconds,
        "validation_two_view_seconds": validation_seconds,
        "test_single_view_seconds": single_view_seconds,
        "test_two_view_seconds": test_two_view_seconds,
        "two_view_to_single_view_ratio": test_two_view_seconds / single_view_seconds,
        "five_fold_cv_all_decays_seconds": calibration_seconds,
        "full_posthoc_seconds": (
            prototype_seconds
            + validation_seconds
            + test_two_view_seconds
            + calibration_seconds
        ),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for run_name in args.runs:
        run_dir = args.runs_root / run_name
        if not (run_dir / "best.pt").is_file():
            continue
        row = benchmark_run(run_dir, device)
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "posthoc_cost_benchmark.json", {"runs": rows})
    with (args.output_dir / "posthoc_cost_benchmark.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# PMRC post-hoc cost benchmark",
        "",
        "Timing includes frozen-model feature extraction and calibration on the "
        "RTX 5060 Laptop GPU. The two-view ratio compares descriptor extraction "
        "on test images with one ordinary classifier forward pass.",
        "",
        "| Run | Single view | Two view | Ratio | Prototype pass | Five-fold CV | Total |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run_name']} | {row['test_single_view_seconds']:.2f}s | "
            f"{row['test_two_view_seconds']:.2f}s | "
            f"{row['two_view_to_single_view_ratio']:.2f}x | "
            f"{row['prototype_feature_seconds']:.2f}s | "
            f"{row['five_fold_cv_all_decays_seconds']:.2f}s | "
            f"{row['full_posthoc_seconds']:.2f}s |"
        )
    (args.output_dir / "posthoc_cost_benchmark.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
