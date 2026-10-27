from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgel.data import CUBManifestDataset, CUBTwoViewDataset, build_transforms, load_manifest
from rgel.models import build_model
from rgel.utils import read_json, write_json
from scripts.openset_webfg_proxy import evaluate_scores
from scripts.posthoc_reliability import (
    class_prototypes,
    fit_logistic,
    reliability_features,
    score_logistic,
)
from scripts.prototype_independence import stratified_auc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PMRC on a classifier trained only on known CUB classes."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=RUNS_ROOT / "elr_resnet50_cub_openset160_semantic40_seed0",
    )
    parser.add_argument(
        "--unknown-manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "cub_openset160_unknown.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / "openset_cub160.json",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append or replace the matching seed in an existing output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.run_dir / "config.json")
    manifest = load_manifest(config["manifest"])
    train_rows = [row for row in manifest if row.split == "train"]
    val_rows = [row for row in manifest if row.split == "val"]
    known_test_rows = [row for row in manifest if row.split == "test"]
    unknown_payload = json.loads(
        args.unknown_manifest.read_text(encoding="utf-8")
    )
    unknown_rows = [
        type(known_test_rows[0])(**row)
        for row in unknown_payload["records"]
    ]
    num_classes = len({row.class_id for row in manifest})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        config["model"],
        num_classes,
        pretrained=False,
        freeze_stages=int(config.get("freeze_stages", 0)),
    ).to(device)
    checkpoint = torch.load(
        args.run_dir / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image_size = int(config.get("image_size", 224))
    batch_size = int(config.get("batch_size", 32))
    train_loader = DataLoader(
        CUBManifestDataset(
            config["root"],
            train_rows,
            build_transforms(image_size, False),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    prototypes = class_prototypes(
        model,
        train_loader,
        num_classes,
        device,
    )
    unknown_loader = DataLoader(
        CUBTwoViewDataset(
            config["root"],
            unknown_rows,
            build_transforms(image_size, False),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    unknown_features, _, _ = reliability_features(
        model,
        unknown_loader,
        prototypes,
        device,
    )
    val_features = np.load(args.run_dir / "posthoc_val_features.npy")
    val_labels = np.load(args.run_dir / "posthoc_val_labels.npy")
    val_predictions = np.load(args.run_dir / "posthoc_val_predictions.npy")
    val_correct = val_predictions == val_labels
    posthoc = read_json(args.run_dir / "posthoc_reliability.json")
    calibrator, mean, std = fit_logistic(
        val_features,
        val_correct,
        float(posthoc["selected_weight_decay"]),
    )
    unknown_scores = score_logistic(
        calibrator,
        unknown_features,
        mean,
        std,
    )
    known_features = np.load(args.run_dir / "posthoc_test_features.npy")
    known_scores = np.load(args.run_dir / "posthoc_test_scores.npy")
    msp_scores = np.concatenate((known_features[:, 0], unknown_features[:, 0]))
    pmrc_scores = np.concatenate((known_scores, unknown_scores))
    known_flag = np.concatenate(
        (
            np.ones(len(known_features), dtype=bool),
            np.zeros(len(unknown_features), dtype=bool),
        )
    )
    gate_auc = stratified_auc(
        val_features[:, 0],
        val_features[:, 9],
        val_correct,
        5,
    )["weighted_auc"]
    config_seed = int(config.get("seed", 0))
    result = {
        "run_dir": str(args.run_dir),
        "seed": config_seed,
        "known_test_count": len(known_test_rows),
        "unknown_test_count": len(unknown_rows),
        "validation_gate_auc": float(gate_auc),
        "gate_selected_method": "pmrc" if gate_auc > 0.5 else "msp",
        **evaluate_scores(msp_scores, pmrc_scores, known_flag),
    }
    if args.append and args.output.is_file():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        rows = [
            row
            for row in payload.get("runs", [])
            if int(row.get("seed", -1)) != config_seed
        ]
        rows.append(result)
        write_json(args.output, {"runs": rows})
    else:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
