from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgel.data import read_metadata, stratified_train_val_split
from rgel.engine import train
from rgel.utils import read_json
from scripts.prepare_cub import write_manifest
from scripts.smoke_train import create_synthetic_cub


def main() -> None:
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        root = work / "CUB_200_2011"
        create_synthetic_cub(root)
        source, _ = read_metadata(root)
        train_rows, rest = stratified_train_val_split(source, 0.25, seed=0)
        manifest = work / "manifest.json"
        write_manifest(manifest, train_rows + rest, root, 0.25, 0)
        config = {
            "root": str(root),
            "manifest": str(manifest),
            "noise_csv": None,
            "output_dir": str(work / "runs"),
            "run_name": "smoke_resume",
            "method": "ce",
            "model": "resnet18",
            "pretrained": False,
            "freeze_stages": 0,
            "image_size": 64,
            "batch_size": 4,
            "num_workers": 0,
            "epochs": 1,
            "lr": 0.001,
            "weight_decay": 0.0,
            "label_smoothing": 0.0,
            "grad_clip": 1.0,
            "amp": True,
            "amp_dtype": "bf16",
            "seed": 0,
            "resume": True,
        }
        train(config)
        last_path = work / "runs" / "smoke_resume" / "last.pt"
        if not last_path.is_file():
            raise RuntimeError("First stage did not write last.pt")

        config["epochs"] = 2
        result = train(config)
        history = read_json(work / "runs" / "smoke_resume" / "history.json")
        if len(history) != 2:
            raise RuntimeError(f"Resume produced {len(history)} history rows, expected 2")
        print(
            {
                "resume_ok": True,
                "epochs": len(history),
                "test_accuracy": result["test"]["accuracy"],
            }
        )


if __name__ == "__main__":
    main()

