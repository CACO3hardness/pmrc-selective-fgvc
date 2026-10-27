from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from rgel.data import CUBManifestDataset, build_transforms, load_manifest
from rgel.noise import empirical_noise_rate, sample_class_noise, write_noise_csv
from rgel.utils import set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a fixed symmetric or semantic-neighbor noise file."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metadata-out", type=Path, required=True)
    parser.add_argument("--kind", choices=("semantic", "symmetric"), required=True)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--model", choices=("resnet18", "resnet50"), default="resnet50")
    return parser.parse_args()


def _feature_model(name: str, device: torch.device) -> nn.Module:
    from torchvision import models

    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Identity()
    return model.eval().to(device)


@torch.inference_mode()
def extract_class_centroids(
    args: argparse.Namespace,
    train_rows: list[Any],
    num_classes: int,
    device: torch.device,
) -> np.ndarray:
    dataset = CUBManifestDataset(
        args.root,
        train_rows,
        build_transforms(args.image_size, train=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = _feature_model(args.model, device)
    dimension = 512 if args.model == "resnet18" else 2048
    sums = np.zeros((num_classes, dimension), dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.int64)

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["clean_target"].cpu().numpy()
        features = model(images).float().cpu().numpy()
        features /= np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-12, None)
        np.add.at(sums, labels, features)
        np.add.at(counts, labels, 1)
        if batch_index == 1 or batch_index % 25 == 0 or batch_index == len(loader):
            print(f"class centroids: batch {batch_index}/{len(loader)}", flush=True)

    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Classes without training examples: {missing}")
    centroids = sums / counts[:, None]
    centroids /= np.clip(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12, None)
    return centroids.astype(np.float32)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.rate <= 1.0:
        raise ValueError("--rate must be in [0, 1]")
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_manifest(args.manifest)
    train_rows = [row for row in rows if row.split == "train"]
    classes = sorted({row.class_id for row in rows})
    num_classes = len(classes)
    if classes != list(range(num_classes)):
        raise ValueError("Class IDs must be contiguous from zero")

    if args.kind == "semantic":
        if args.top_k >= num_classes:
            raise ValueError("--top-k must be smaller than the number of classes")
        centroids = extract_class_centroids(args, train_rows, num_classes, device)
        similarity = centroids @ centroids.T
        np.fill_diagonal(similarity, -np.inf)
        top_indices = np.argpartition(-similarity, kth=args.top_k - 1, axis=1)[:, : args.top_k]
        neighbor_map: dict[int, list[int]] = {}
        similar_map: dict[int, list[float]] | None = {}
        for class_id in classes:
            order = np.argsort(-similarity[class_id, top_indices[class_id]])
            chosen = top_indices[class_id, order]
            neighbor_map[class_id] = [int(value) for value in chosen]
            assert similar_map is not None
            similar_map[class_id] = [float(similarity[class_id, value]) for value in chosen]
        centroid_path = args.metadata_out.with_suffix(".centroids.npy")
        centroid_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(centroid_path, centroids)
    else:
        neighbor_map = {
            class_id: [other for other in classes if other != class_id]
            for class_id in classes
        }
        similar_map = None
        centroid_path = None

    entries = sample_class_noise(
        train_rows,
        neighbor_map=neighbor_map,
        similar_map=similar_map,
        rate=args.rate,
        seed=args.seed,
    )
    write_noise_csv(args.out, entries)
    changed_counts = Counter(
        entry.clean_label for entry in entries if entry.is_noisy
    )
    metadata = {
        "kind": args.kind,
        "rate_requested": args.rate,
        "rate_empirical": empirical_noise_rate(entries),
        "seed": args.seed,
        "top_k": args.top_k if args.kind == "semantic" else None,
        "model": args.model if args.kind == "semantic" else None,
        "device": str(device),
        "train_images": len(train_rows),
        "changed_images": sum(entry.is_noisy for entry in entries),
        "changed_by_clean_class": {str(key): value for key, value in sorted(changed_counts.items())},
        "noise_csv": str(args.out.resolve()),
        "centroids_npy": str(centroid_path.resolve()) if centroid_path else None,
        "neighbor_map": {str(key): value for key, value in neighbor_map.items()},
    }
    write_json(args.metadata_out, metadata)
    print(
        f"Wrote {args.out} with empirical noise rate {metadata['rate_empirical']:.4f} "
        f"to {len(train_rows)} training images."
    )


if __name__ == "__main__":
    main()
