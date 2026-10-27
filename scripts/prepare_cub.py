from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgel.data import ManifestRow, read_metadata, stratified_train_val_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a fixed CUB manifest.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, stats = build_manifest(args.root, args.val_fraction, args.seed)
    write_manifest(args.out, rows, args.root, args.val_fraction, args.seed)
    print(stats)


def build_manifest(root: Path, val_fraction: float, seed: int) -> tuple[list[ManifestRow], dict]:
    rows, metadata = read_metadata(root)
    train, rest = stratified_train_val_split(rows, val_fraction, seed)
    rows = train + rest
    split_counts = {split: sum(row.split == split for row in rows) for split in ("train", "val", "test")}
    class_counts = {split: len({row.class_id for row in rows if row.split == split}) for split in split_counts}
    stats = {
        **metadata,
        "split_counts": split_counts,
        "class_counts": class_counts,
        "val_fraction": val_fraction,
        "seed": seed,
    }
    return rows, stats


def write_manifest(
    path: Path,
    rows: list[ManifestRow],
    root: Path,
    val_fraction: float,
    seed: int,
) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "CUB_200_2011",
        "dataset_root": str(root.resolve()),
        "val_fraction": val_fraction,
        "seed": seed,
        "records": [row.__dict__ for row in rows],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
