from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Webly supervised datasets often contain partially truncated JPEGs. Pillow can
# decode their available scan data when this flag is enabled; without it the
# DataLoader workers abort the entire training run.
ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass(frozen=True)
class ManifestRow:
    image_id: int
    relative_path: str
    class_id: int
    class_name: str
    split: str


def _resolve_cub_root(root: str | Path) -> Path:
    root = Path(root)
    if (root / "images.txt").is_file():
        return root
    nested = root / "CUB_200_2011"
    if (nested / "images.txt").is_file():
        return nested
    raise FileNotFoundError(
        f"Cannot find CUB metadata under {root}. Expected images.txt directly in the root "
        "or in CUB_200_2011/."
    )


def _read_pairs(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip().split(maxsplit=1) for line in handle if line.strip()]


def read_metadata(root: str | Path) -> tuple[list[ManifestRow], dict[str, object]]:
    base = _resolve_cub_root(root)
    image_paths = {int(key): value for key, value in _read_pairs(base / "images.txt")}
    class_labels = {
        int(key): int(value) - 1
        for key, value in _read_pairs(base / "image_class_labels.txt")
    }
    train_flags = {
        int(key): bool(int(value))
        for key, value in _read_pairs(base / "train_test_split.txt")
    }
    class_names = {
        int(key) - 1: value
        for key, value in _read_pairs(base / "classes.txt")
    }

    ids = set(image_paths)
    if ids != set(class_labels) or ids != set(train_flags):
        raise ValueError("CUB metadata files do not contain identical image ID sets")

    rows = [
        ManifestRow(
            image_id=image_id,
            relative_path=(
                image_paths[image_id]
                if image_paths[image_id].startswith("images/")
                else f"images/{image_paths[image_id].lstrip('/')}"
            ),
            class_id=class_labels[image_id],
            class_name=class_names[class_labels[image_id]],
            split="train" if train_flags[image_id] else "test",
        )
        for image_id in sorted(ids)
    ]
    stats: dict[str, object] = {
        "dataset_root": str(base.resolve()),
        "images": len(rows),
        "classes": len(class_names),
        "official_train": sum(row.split == "train" for row in rows),
        "official_test": sum(row.split == "test" for row in rows),
    }
    return rows, stats


def load_manifest(path: str | Path) -> list[ManifestRow]:
    import json

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    rows = payload.get("records")
    if not isinstance(rows, list):
        raise ValueError(f"Manifest has no records list: {path}")

    parsed = [
        ManifestRow(
            image_id=int(row["image_id"]),
            relative_path=str(row["relative_path"]),
            class_id=int(row["class_id"]),
            class_name=str(row["class_name"]),
            split=str(row["split"]),
        )
        for row in rows
    ]
    validate_manifest(parsed)
    return parsed


def validate_manifest(rows: Iterable[ManifestRow]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("Manifest is empty")

    image_ids = [row.image_id for row in rows]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Manifest contains duplicate image IDs")

    valid_splits = {"train", "val", "test"}
    bad_splits = sorted({row.split for row in rows} - valid_splits)
    if bad_splits:
        raise ValueError(f"Unknown splits: {bad_splits}")

    train_ids = {row.class_id for row in rows if row.split == "train"}
    val_ids = {row.class_id for row in rows if row.split == "val"}
    test_ids = {row.class_id for row in rows if row.split == "test"}
    if not train_ids:
        raise ValueError("Manifest has no training rows")
    if not val_ids:
        raise ValueError("Manifest has no validation rows")
    if not test_ids:
        raise ValueError("Manifest has no test rows")
    if train_ids != val_ids or train_ids != test_ids:
        raise ValueError("Train, validation and test must contain the same class IDs")


def load_noise_map(path: str | Path | None) -> dict[int, int]:
    if path is None:
        return {}

    result: dict[int, int] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "noisy_label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Noise CSV must contain {sorted(required)}")
        for row in reader:
            image_id = int(row["image_id"])
            noisy_label = int(row["noisy_label"])
            if image_id in result:
                raise ValueError(f"Duplicate noise entry for image {image_id}")
            result[image_id] = noisy_label
    return result


class CUBManifestDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        rows: list[ManifestRow],
        transform: transforms.Compose,
        noise_map: dict[int, int] | None = None,
    ) -> None:
        self.root = Path(root)
        self.rows = rows
        self.transform = transform
        self.noise_map = noise_map or {}

        missing = [
            str(self.root / row.relative_path)
            for row in rows
            if not (self.root / row.relative_path).is_file()
        ]
        if missing:
            preview = "\n".join(missing[:5])
            raise FileNotFoundError(f"{len(missing)} image files are missing:\n{preview}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        image = Image.open(self.root / row.relative_path).convert("RGB")
        image = self.transform(image)
        target = self.noise_map.get(row.image_id, row.class_id)
        return {
            "image": image,
            "target": target,
            "clean_target": row.class_id,
            "image_id": row.image_id,
            "index": index,
        }


class CUBTwoViewDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        rows: list[ManifestRow],
        transform: transforms.Compose,
        noise_map: dict[int, int] | None = None,
    ) -> None:
        self.root = Path(root)
        self.rows = rows
        self.transform = transform
        self.noise_map = noise_map or {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        image = Image.open(self.root / row.relative_path).convert("RGB")
        view_a = self.transform(image)
        view_b = self.transform(image)
        target = self.noise_map.get(row.image_id, row.class_id)
        return {
            "image_a": view_a,
            "image_b": view_b,
            "target": target,
            "clean_target": row.class_id,
            "image_id": row.image_id,
            "index": index,
        }


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    resize_size = int(round(image_size * 1.14))
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def rows_for_split(rows: list[ManifestRow], split: str) -> list[ManifestRow]:
    return [row for row in rows if row.split == split]


def stratified_train_val_split(
    rows: list[ManifestRow],
    val_fraction: float,
    seed: int,
) -> tuple[list[ManifestRow], list[ManifestRow]]:
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be between 0 and 0.5")

    train_source = rows_for_split(rows, "train")
    test_rows = rows_for_split(rows, "test")
    by_class: dict[int, list[ManifestRow]] = {}
    for row in train_source:
        by_class.setdefault(row.class_id, []).append(row)

    rng = random.Random(seed)
    train_rows: list[ManifestRow] = []
    val_rows: list[ManifestRow] = []
    for class_id in sorted(by_class):
        class_rows = list(by_class[class_id])
        rng.shuffle(class_rows)
        n_val = max(1, int(round(len(class_rows) * val_fraction)))
        if n_val >= len(class_rows):
            raise ValueError(f"Class {class_id} has too few training images")
        val_rows.extend(
            ManifestRow(
                image_id=row.image_id,
                relative_path=row.relative_path,
                class_id=row.class_id,
                class_name=row.class_name,
                split="val",
            )
            for row in class_rows[:n_val]
        )
        train_rows.extend(
            ManifestRow(
                image_id=row.image_id,
                relative_path=row.relative_path,
                class_id=row.class_id,
                class_name=row.class_name,
                split="train",
            )
            for row in class_rows[n_val:]
        )

    # Keep test rows unchanged.
    return train_rows, val_rows + test_rows
