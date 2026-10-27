from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader

from rgel.data import CUBManifestDataset, build_transforms, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 4, 8])
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=20)
    args = parser.parse_args()
    rows = [
        row
        for row in load_manifest("data/manifests/cifar100n_seed0.json")
        if row.split == "train"
    ]
    dataset = CUBManifestDataset(
        "data",
        rows,
        build_transforms(args.image_size, train=True),
    )
    for workers in args.workers:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=workers,
            persistent_workers=False,
            pin_memory=True,
        )
        started = time.perf_counter()
        seen = 0
        for _ in loader:
            seen += 1
            if seen == args.batches:
                break
        elapsed = time.perf_counter() - started
        print(
            {
                "workers": workers,
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "batches": seen,
                "seconds": elapsed,
                "images_per_second": seen * args.batch_size / elapsed,
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
