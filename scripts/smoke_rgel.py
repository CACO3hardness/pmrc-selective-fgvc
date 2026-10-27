from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgel.data import read_metadata, stratified_train_val_split
from rgel.rgel_train import train_rgel
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
            "run_name": "smoke_rgel",
            "model": "resnet18",
            "pretrained": False,
            "freeze_stages": 0,
            "image_size": 64,
            "batch_size": 4,
            "num_workers": 0,
            "epochs": 2,
            "lr": 0.001,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "amp": False,
            "amp_dtype": "bf16",
            "seed": 0,
            "resume": True,
            "vacuity_weight": 0.1,
            "consistency_weight": 0.5,
            "gate_warmup_epochs": 0,
            "prototype_momentum": 0.9,
            "prototype_threshold": 0.65,
            "reliability_agreement_weight": 0.5,
            "reliability_prototype_weight": 0.35,
            "reliability_confidence_weight": 0.15,
        }
        result = train_rgel(config)
        history = read_json(work / "runs" / "smoke_rgel" / "history.json")
        print(
            {
                "rgel_smoke_ok": len(history) == 2,
                "test_accuracy": result["test"]["accuracy"],
            }
        )


if __name__ == "__main__":
    main()
