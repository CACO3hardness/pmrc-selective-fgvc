from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgel.data import ManifestRow, validate_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBSETS = ("web-bird", "web-aircraft", "web-car")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare WebFG-496 manifests from official extracted archives."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "data" / "WebFG_496" / "extracted",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "WebFG_496",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    return parser.parse_args()


def read_label_list(path: Path) -> list[tuple[str, int]]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        relative_path, raw_label = line.rsplit(":", 1)
        entries.append((relative_path.strip().replace("\\", "/"), int(raw_label)))
    return entries


def class_names(entries: list[tuple[str, int]]) -> dict[int, str]:
    result: dict[int, str] = {}
    for relative_path, class_id in entries:
        name = Path(relative_path).parent.name
        previous = result.setdefault(class_id, name)
        if previous != name:
            raise ValueError(
                f"Class {class_id} has multiple folder names: {previous}, {name}"
            )
    return result


def make_manifest(
    root: Path,
    subset: str,
    manifest_dir: Path,
    val_fraction: float,
    split_seed: int,
) -> dict:
    subset_root = root / subset
    train_entries = read_label_list(subset_root / "train-list.txt")
    clean_entries = read_label_list(subset_root / "val-list.txt")
    names = class_names(train_entries)
    clean_names = class_names(clean_entries)
    if names != clean_names:
        missing = sorted(set(names) - set(clean_names))
        extra = sorted(set(clean_names) - set(names))
        raise ValueError(f"{subset}: train/clean class mismatch {missing=} {extra=}")

    class_ids = sorted(names)
    if class_ids != list(range(len(class_ids))):
        raise ValueError(f"{subset}: class IDs are not contiguous from zero")

    by_class: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for entry in clean_entries:
        by_class[entry[1]].append(entry)
    rng = random.Random(split_seed)
    val_paths: set[str] = set()
    for class_id in class_ids:
        class_entries = list(by_class[class_id])
        rng.shuffle(class_entries)
        n_val = max(1, int(round(len(class_entries) * val_fraction)))
        n_val = min(n_val, len(class_entries) - 1)
        val_paths.update(path for path, _ in class_entries[:n_val])

    rows: list[ManifestRow] = []
    image_id = 0
    for relative_path, class_id in train_entries:
        image_id += 1
        rows.append(
            ManifestRow(
                image_id=image_id,
                relative_path=f"{subset}/{relative_path}",
                class_id=class_id,
                class_name=names[class_id],
                split="train",
            )
        )
    for relative_path, class_id in clean_entries:
        image_id += 1
        split = "val" if relative_path in val_paths else "test"
        rows.append(
            ManifestRow(
                image_id=image_id,
                relative_path=f"{subset}/{relative_path}",
                class_id=class_id,
                class_name=names[class_id],
                split=split,
            )
        )
    validate_manifest(rows)
    missing = [
        row.relative_path
        for row in rows
        if not (root / row.relative_path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{subset}: {len(missing)} image files missing, e.g. {missing[:3]}"
        )
    payload = {
        "dataset": f"webfg_{subset.removeprefix('web-')}",
        "dataset_root": str(root.resolve()),
        "val_fraction": val_fraction,
        "split_seed": split_seed,
        "labels_are_noisy": True,
        "records": [row.__dict__ for row in rows],
    }
    manifest_path = manifest_dir / f"webfg_{subset.removeprefix('web-')}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    counts = {
        split: sum(row.split == split for row in rows)
        for split in ("train", "val", "test")
    }
    return {
        "subset": subset,
        "classes": len(class_ids),
        **counts,
        "manifest": str(manifest_path),
    }


def main() -> None:
    args = parse_args()
    metadata = {
        "dataset": "WebFG-496",
        "root": str(args.root.resolve()),
        "train_labels": "web-collected noisy labels supplied by the official folders",
        "clean_test_source": "official val directories and val-list files",
        "val_fraction": args.val_fraction,
        "split_seed": args.split_seed,
        "subsets": [],
    }
    for subset in SUBSETS:
        result = make_manifest(
            args.root,
            subset,
            args.manifest_dir,
            args.val_fraction,
            args.split_seed,
        )
        metadata["subsets"].append(result)
        print(result)
    args.metadata_dir.mkdir(parents=True, exist_ok=True)
    with (args.metadata_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
