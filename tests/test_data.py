from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from rgel.data import (
    CUBManifestDataset,
    ManifestRow,
    build_transforms,
    read_metadata,
    stratified_train_val_split,
    validate_manifest,
)


class DataTests(unittest.TestCase):
    def test_cub_metadata_and_stratified_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "CUB_200_2011"
            (root / "images").mkdir(parents=True)
            image_lines: list[str] = []
            label_lines: list[str] = []
            split_lines: list[str] = []
            class_lines = ["1 alpha", "2 beta"]

            image_id = 1
            for class_id in (1, 2):
                for index in range(8):
                    relative = Path("images") / f"{class_id:03d}" / f"{index}.jpg"
                    destination = root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (32, 32), color=(class_id * 40, index * 20, 50)).save(destination)
                    image_lines.append(f"{image_id} {relative.as_posix()}")
                    label_lines.append(f"{image_id} {class_id}")
                    split_lines.append(f"{image_id} {int(index < 6)}")
                    image_id += 1

            (root / "images.txt").write_text("\n".join(image_lines), encoding="utf-8")
            (root / "image_class_labels.txt").write_text("\n".join(label_lines), encoding="utf-8")
            (root / "train_test_split.txt").write_text("\n".join(split_lines), encoding="utf-8")
            (root / "classes.txt").write_text("\n".join(class_lines), encoding="utf-8")

            source, stats = read_metadata(root)
            self.assertEqual(stats["images"], 16)
            train, val_test = stratified_train_val_split(source, 0.2, seed=3)
            rows = train + val_test
            validate_manifest(rows)
            self.assertEqual(sum(row.split == "train" for row in rows), 10)
            self.assertEqual(sum(row.split == "val" for row in rows), 2)
            self.assertEqual(sum(row.split == "test" for row in rows), 4)

            dataset = CUBManifestDataset(
                root,
                train[:2],
                build_transforms(32, train=False),
                noise_map={train[0].image_id: 1 if train[0].class_id == 0 else 0},
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["image"].shape), (3, 32, 32))
            self.assertNotEqual(sample["target"], sample["clean_target"])


if __name__ == "__main__":
    unittest.main()

