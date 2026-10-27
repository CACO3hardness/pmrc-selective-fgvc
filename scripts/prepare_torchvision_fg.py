from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgel.data import ManifestRow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an ImageFolder-compatible manifest.")
    parser.add_argument("--dataset", choices=("aircraft", "flowers102"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def dataset_splits(name: str, root: Path, download: bool) -> tuple[object, object]:
    if name == "aircraft":
        from torchvision.datasets import FGVCAircraft

        return (
            FGVCAircraft(root, split="trainval", download=download),
            FGVCAircraft(root, split="test", download=download),
        )
    if name == "flowers102":
        from torchvision.datasets import Flowers102

        return (
            Flowers102(root, split="train", download=download),
            Flowers102(root, split="test", download=download),
        )
    raise ValueError(name)


def records_from_dataset(dataset, root: Path, split: str) -> list[ManifestRow]:
    classes = getattr(dataset, "classes", None)
    records: list[ManifestRow] = []
    for image_id, (image_file, class_id) in enumerate(
        zip(dataset._image_files, dataset._labels),
        start=1,
    ):
        relative_path = Path(image_file).resolve().relative_to(root.resolve()).as_posix()
        class_name = str(classes[class_id]) if classes else str(class_id)
        records.append(
            ManifestRow(
                image_id=image_id,
                relative_path=relative_path,
                class_id=int(class_id),
                class_name=class_name,
                split=split,
            )
        )
    return records


def main() -> None:
    args = parse_args()
    train_dataset, test_dataset = dataset_splits(args.dataset, args.root, args.download)
    train_rows = records_from_dataset(train_dataset, args.root, "train")
    test_rows = records_from_dataset(test_dataset, args.root, "test")

    random.Random(args.seed).shuffle(train_rows)
    by_class: dict[int, list[ManifestRow]] = {}
    for row in train_rows:
        by_class.setdefault(row.class_id, []).append(row)
    final_train: list[ManifestRow] = []
    final_val: list[ManifestRow] = []
    for class_id in sorted(by_class):
        class_rows = by_class[class_id]
        n_val = max(1, int(round(len(class_rows) * args.val_fraction)))
        n_val = min(n_val, len(class_rows) - 1)
        final_val.extend(
            ManifestRow(
                image_id=row.image_id,
                relative_path=row.relative_path,
                class_id=row.class_id,
                class_name=row.class_name,
                split="val",
            )
            for row in class_rows[:n_val]
        )
        final_train.extend(
            ManifestRow(
                image_id=row.image_id,
                relative_path=row.relative_path,
                class_id=row.class_id,
                class_name=row.class_name,
                split="train",
            )
            for row in class_rows[n_val:]
        )

    rows = final_train + final_val + test_rows
    if len({row.image_id for row in rows}) != len(rows):
        # Dataset-specific image IDs are only for CUB; use a global ordinal here.
        offset = 0
        reset: list[ManifestRow] = []
        for row in rows:
            offset += 1
            reset.append(
                ManifestRow(
                    image_id=offset,
                    relative_path=row.relative_path,
                    class_id=row.class_id,
                    class_name=row.class_name,
                    split=row.split,
                )
            )
        rows = reset

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": args.dataset,
        "dataset_root": str(args.root.resolve()),
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "records": [row.__dict__ for row in rows],
    }
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(
        {
            "dataset": args.dataset,
            "train": len(final_train),
            "val": len(final_val),
            "test": len(test_rows),
            "classes": len(by_class),
        }
    )


if __name__ == "__main__":
    main()

