from __future__ import annotations

import argparse
import base64
import csv
import json
import random
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import CIFAR100

from rgel.data import ManifestRow, validate_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_URL = (
    "https://api.github.com/repos/UCSC-REAL/cifar-10-100n/"
    "contents/data/CIFAR-100_human.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare CIFAR-100N real human label noise."
    )
    parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--torch-cache", type=Path, default=PROJECT_ROOT / ".torch_cache")
    parser.add_argument("--manifest-dir", type=Path, default=PROJECT_ROOT / "data" / "manifests")
    parser.add_argument("--noise-dir", type=Path, default=PROJECT_ROOT / "data" / "noise")
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    return parser.parse_args()


def download_labels(path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        request = urllib.request.Request(
            LABEL_URL,
            headers={"User-Agent": "Codex"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.load(response)
        path.write_bytes(base64.b64decode(payload["content"]))
    return torch.load(path, map_location="cpu", weights_only=False)


def export_images(dataset: CIFAR100, directory: Path, prefix: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, (image, _) in enumerate(dataset):
        path = directory / f"{prefix}_{index:05d}.png"
        if not path.is_file():
            Image.fromarray(np.asarray(image)).save(path, optimize=True)


def write_manifest(
    path: Path,
    rows: list[ManifestRow],
    seed: int,
    val_fraction: float,
) -> None:
    validate_manifest(rows)
    payload = {
        "dataset": "cifar100n",
        "dataset_root": str(path.parent.parent.resolve()),
        "val_fraction": val_fraction,
        "seed": seed,
        "records": [row.__dict__ for row in rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    dataset_root = args.out_root / "CIFAR100N"
    images_root = dataset_root / "images"
    labels_path = dataset_root / "CIFAR-100_human.pt"
    labels = download_labels(labels_path)
    clean_labels = np.asarray(labels["clean_label"], dtype=np.int64)
    noisy_labels = np.asarray(labels["noisy_label"], dtype=np.int64)

    train_dataset = CIFAR100(
        root=args.torch_cache,
        train=True,
        download=True,
    )
    test_dataset = CIFAR100(
        root=args.torch_cache,
        train=False,
        download=True,
    )
    if len(train_dataset) != len(clean_labels):
        raise ValueError("CIFAR-100 image and human-label counts differ")
    export_images(train_dataset, images_root / "train", "train")
    export_images(test_dataset, images_root / "test", "test")

    class_names = train_dataset.classes
    indices_by_class: dict[int, list[int]] = {
        class_id: [] for class_id in range(len(class_names))
    }
    for index, class_id in enumerate(clean_labels):
        indices_by_class[int(class_id)].append(index)

    metadata = {
        "dataset": "cifar100n",
        "source": "UCSC-REAL/cifar-10-100n",
        "train_images": len(train_dataset),
        "test_images": len(test_dataset),
        "classes": len(class_names),
        "empirical_noise_rate": float((clean_labels != noisy_labels).mean()),
        "val_fraction": args.val_fraction,
        "seeds": args.seeds,
    }
    for seed in args.seeds:
        rng = random.Random(seed)
        train_indices: list[int] = []
        val_indices: list[int] = []
        for class_id in range(len(class_names)):
            class_indices = list(indices_by_class[class_id])
            rng.shuffle(class_indices)
            n_val = max(1, int(round(len(class_indices) * args.val_fraction)))
            n_val = min(n_val, len(class_indices) - 1)
            val_indices.extend(class_indices[:n_val])
            train_indices.extend(class_indices[n_val:])
        rng.shuffle(train_indices)

        rows = []
        for split, split_indices in (
            ("train", train_indices),
            ("val", val_indices),
            ("test", list(range(len(test_dataset)))),
        ):
            for index in sorted(split_indices):
                if split == "test":
                    image_id = len(train_dataset) + index + 1
                    relative_path = f"CIFAR100N/images/test/test_{index:05d}.png"
                    class_id = int(test_dataset.targets[index])
                else:
                    image_id = index + 1
                    relative_path = f"CIFAR100N/images/train/train_{index:05d}.png"
                    class_id = int(clean_labels[index])
                rows.append(
                    ManifestRow(
                        image_id=image_id,
                        relative_path=relative_path,
                        class_id=class_id,
                        class_name=str(class_names[class_id]),
                        split=split,
                    )
                )

        manifest_path = args.manifest_dir / f"cifar100n_seed{seed}.json"
        write_manifest(manifest_path, rows, seed, args.val_fraction)

        noise_path = args.noise_dir / f"cifar100n_human_seed{seed}.csv"
        noise_path.parent.mkdir(parents=True, exist_ok=True)
        with noise_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["image_id", "clean_label", "noisy_label", "is_noisy"],
            )
            writer.writeheader()
            for index in train_indices:
                clean_label = int(clean_labels[index])
                noisy_label = int(noisy_labels[index])
                writer.writerow(
                    {
                        "image_id": index + 1,
                        "clean_label": clean_label,
                        "noisy_label": noisy_label,
                        "is_noisy": int(clean_label != noisy_label),
                    }
                )

        split_counts = {
            split: sum(row.split == split for row in rows)
            for split in ("train", "val", "test")
        }
        split_noise_rate = float(
            (
                clean_labels[train_indices]
                != noisy_labels[train_indices]
            ).mean()
        )
        metadata[f"seed{seed}"] = {
            **split_counts,
            "train_noise_rate": split_noise_rate,
            "manifest": str(manifest_path),
            "noise_csv": str(noise_path),
        }
        print({"seed": seed, **split_counts, "noise": split_noise_rate})

    with (dataset_root / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
