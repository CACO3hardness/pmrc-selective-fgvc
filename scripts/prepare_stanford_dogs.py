from __future__ import annotations

import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgel.data import ManifestRow, validate_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "StanfordDogs"
PARQUET_ROOT = DATA_ROOT / "parquet"
IMAGE_ROOT = DATA_ROOT / "images"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "stanford_dogs_external.json"
VAL_PER_CLASS = 5


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_image(value: object) -> tuple[bytes, str]:
    if not isinstance(value, dict):
        raise TypeError(f"Unexpected image payload: {type(value)}")
    raw = value.get("bytes")
    source_path = str(value.get("path") or "")
    if not isinstance(raw, bytes) or not raw:
        raise ValueError(f"Missing image bytes for source {source_path!r}")
    with Image.open(io.BytesIO(raw)) as image:
        image.verify()
    with Image.open(io.BytesIO(raw)) as image:
        image_format = (image.format or "JPEG").lower()
    suffix = ".png" if image_format == "png" else ".jpg"
    return raw, suffix


def iter_parquet_rows(path: Path):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=64, columns=["label", "pixel_values"]):
        yield from batch.to_pylist()


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_files = sorted(PARQUET_ROOT.glob("train-*.parquet"))
    test_files = sorted(PARQUET_ROOT.glob("test-*.parquet"))
    if not train_files or not test_files:
        raise FileNotFoundError("Stanford Dogs parquet files are incomplete")
    schema_metadata = pq.ParquetFile(train_files[0]).schema_arrow.metadata
    huggingface_metadata = json.loads(schema_metadata[b"huggingface"])
    class_names = huggingface_metadata["info"]["features"]["label"]["names"]
    if len(class_names) != 120:
        raise ValueError(f"Expected 120 class names, found {len(class_names)}")

    records: list[ManifestRow] = []
    source_metadata: list[dict[str, object]] = []
    raw_hashes: dict[int, str] = {}
    seen_source_keys: set[str] = set()
    image_id = 0
    for split, files in (("train_source", train_files), ("test", test_files)):
        for parquet_file in files:
            for row in iter_parquet_rows(parquet_file):
                class_id = int(row["label"])
                raw, suffix = normalize_image(row["pixel_values"])
                source_path = str(row["pixel_values"].get("path") or "")
                source_key = f"{class_id}:{source_path}"
                if source_key in seen_source_keys:
                    raise ValueError(f"Duplicate source image key: {source_key}")
                seen_source_keys.add(source_key)
                raw_hash = hashlib.sha256(raw).hexdigest()
                raw_hashes[image_id] = raw_hash
                filename = f"{digest(source_key)[:20]}{suffix}"
                relative_path = Path("images") / f"{class_id:03d}" / filename
                output_path = DATA_ROOT / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(raw)
                records.append(
                    ManifestRow(
                        image_id=image_id,
                        relative_path=relative_path.as_posix(),
                        class_id=class_id,
                        class_name=str(class_names[class_id]),
                        split="train" if split == "train_source" else "test",
                    )
                )
                source_metadata.append(
                    {
                        "image_id": image_id,
                        "official_split": "train" if split == "train_source" else "test",
                        "source_path": source_path,
                        "parquet_file": parquet_file.name,
                        "raw_sha256": raw_hash,
                    }
                )
                image_id += 1

    test_hashes = {
        raw_hashes[row.image_id] for row in records if row.split == "test"
    }
    test_hash_counts = Counter(
        raw_hashes[row.image_id] for row in records if row.split == "test"
    )
    train_hash_counts = Counter(
        raw_hashes[row.image_id] for row in records if row.split == "train"
    )
    removed_train = [
        row
        for row in records
        if row.split == "train"
        and (
            raw_hashes[row.image_id] in test_hashes
            or train_hash_counts[raw_hashes[row.image_id]] > 1
        )
    ]
    removed_ids = {row.image_id for row in removed_train}
    records = [
        row for row in records if row.split == "test" or row.image_id not in removed_ids
    ]
    source_metadata = [
        row for row in source_metadata if int(row["image_id"]) not in removed_ids
    ]

    by_class_train: dict[int, list[ManifestRow]] = defaultdict(list)
    for row in records:
        if row.split == "train":
            by_class_train[row.class_id].append(row)
    val_ids: set[int] = set()
    for class_id, class_rows in sorted(by_class_train.items()):
        ordered = sorted(
            class_rows,
            key=lambda row: digest(f"{row.image_id}:{row.relative_path}"),
        )
        if len(ordered) <= VAL_PER_CLASS:
            raise ValueError(f"Class {class_id} has too few train images")
        val_ids.update(row.image_id for row in ordered[:VAL_PER_CLASS])

    records = [
        ManifestRow(
            image_id=row.image_id,
            relative_path=row.relative_path,
            class_id=row.class_id,
            class_name=row.class_name,
            split=(
                "val"
                if row.split == "train" and row.image_id in val_ids
                else row.split
            ),
        )
        for row in records
    ]
    validate_manifest(records)
    split_counts = {
        split: sum(row.split == split for row in records)
        for split in ("train", "val", "test")
    }
    class_counts = {
        split: len({row.class_id for row in records if row.split == split})
        for split in ("train", "val", "test")
    }
    if split_counts["val"] != 600 or split_counts["test"] != 6174:
        raise ValueError(f"Unexpected validation/test counts: {split_counts}")
    if split_counts["train"] < 13000:
        raise ValueError(f"Too few training images after deduplication: {split_counts}")
    if class_counts != {"train": 120, "val": 120, "test": 120}:
        raise ValueError(f"Unexpected class coverage: {class_counts}")

    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "dataset": "StanfordDogs",
                "source": "https://huggingface.co/datasets/amaye15/stanford-dogs",
                "upstream": "Khosla et al., FGVC Workshop at CVPR 2011",
                "split_rule": (
                    "Official train/test membership. Exact pixel duplicates in "
                    "the train side are removed when the same hash occurs in the "
                    "official test set or more than once in train. Exactly five "
                    "retained train images per class are assigned to "
                    "val/calibration by sorted SHA-256 of the saved relative path."
                ),
                "deduplication": {
                    "removed_train_images": len(removed_train),
                    "test_images_retained": split_counts["test"],
                    "test_duplicate_groups_retained": sum(
                        count > 1 for count in test_hash_counts.values()
                    ),
                },
                "split_counts": split_counts,
                "class_counts": class_counts,
                "records": [row.__dict__ for row in records],
                "source_metadata": source_metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(MANIFEST_PATH), "splits": split_counts}, indent=2))


if __name__ == "__main__":
    main()
