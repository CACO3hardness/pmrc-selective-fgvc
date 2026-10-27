from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader

from rgel.data import CUBManifestDataset, build_transforms, load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile image-pipeline throughput.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 4, 8])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--steps", type=int, default=40)
    return parser.parse_args()


def measure(root: Path, manifest: Path, workers: int, batch_size: int, image_size: int, steps: int) -> None:
    rows = [row for row in load_manifest(manifest) if row.split == "train"]
    dataset = CUBManifestDataset(root, rows, build_transforms(image_size, train=True))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    images = 0
    started = time.perf_counter()
    for index, batch in enumerate(loader, start=1):
        images += len(batch["target"])
        if index >= steps:
            break
    elapsed = time.perf_counter() - started
    print(
        {
            "workers": workers,
            "images": images,
            "seconds": elapsed,
            "images_per_second": images / elapsed,
        }
    )


def main() -> None:
    args = parse_args()
    for workers in args.workers:
        measure(
            args.root,
            args.manifest,
            workers,
            args.batch_size,
            args.image_size,
            args.steps,
        )


if __name__ == "__main__":
    main()

