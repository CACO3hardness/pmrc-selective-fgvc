from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from rgel.data import read_metadata, stratified_train_val_split
from rgel.engine import train
from scripts.prepare_cub import write_manifest


def create_synthetic_cub(root: Path) -> None:
    (root / "images").mkdir(parents=True)
    image_lines: list[str] = []
    label_lines: list[str] = []
    split_lines: list[str] = []
    image_id = 1
    for class_id in range(1, 4):
        for index in range(6):
            relative = Path("images") / f"{class_id:03d}" / f"{index}.jpg"
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            color = (class_id * 60, index * 30, (class_id + index) * 20)
            Image.new("RGB", (64, 64), color=color).save(destination)
            image_lines.append(f"{image_id} {relative.as_posix()}")
            label_lines.append(f"{image_id} {class_id}")
            split_lines.append(f"{image_id} {int(index < 4)}")
            image_id += 1

    (root / "images.txt").write_text("\n".join(image_lines), encoding="utf-8")
    (root / "image_class_labels.txt").write_text("\n".join(label_lines), encoding="utf-8")
    (root / "train_test_split.txt").write_text("\n".join(split_lines), encoding="utf-8")
    (root / "classes.txt").write_text("1 alpha\n2 beta\n3 gamma", encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        root = work / "CUB_200_2011"
        create_synthetic_cub(root)
        source, _ = read_metadata(root)
        train_rows, rest = stratified_train_val_split(source, 0.25, seed=0)
        rows = train_rows + rest
        manifest = work / "manifest.json"
        write_manifest(manifest, rows, root, 0.25, 0)
        config = {
            "root": str(root),
            "manifest": str(manifest),
            "noise_csv": None,
            "output_dir": "runs",
            "run_name": "smoke_resnet18",
            "model": "resnet18",
            "pretrained": False,
            "freeze_stages": 0,
            "image_size": 64,
            "batch_size": 4,
            "num_workers": 0,
            "epochs": 2,
            "lr": 0.001,
            "weight_decay": 0.0,
            "label_smoothing": 0.0,
            "grad_clip": 1.0,
            "amp": True,
            "amp_dtype": "bf16",
            "seed": 0,
        }
        result = train(config)
        print(result["test"])


if __name__ == "__main__":
    main()
