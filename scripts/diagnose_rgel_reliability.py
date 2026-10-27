from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from rgel.data import CUBTwoViewDataset, build_transforms, load_manifest
from rgel.evidential import EvidentialResNet
from rgel.metrics import metrics_from_probabilities
from rgel.rgel_train import _reliability
from rgel.utils import read_json, set_seed, write_json
from scripts.diagnose_rgel import selective_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RGEL reliability on two test views.")
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.run_dir / "config.json")
    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(config.get("seed", 0)) + 10000)
    rows = load_manifest(config["manifest"])
    test_rows = [row for row in rows if row.split == "test"]
    dataset = CUBTwoViewDataset(
        config["root"],
        test_rows,
        build_transforms(int(config.get("image_size", 224)), train=True),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 16)),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = EvidentialResNet(
        config["model"],
        num_classes=len({row.class_id for row in rows}),
        pretrained=False,
        freeze_stages=int(config.get("freeze_stages", 0)),
        prototype_momentum=float(config.get("prototype_momentum", 0.9)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    prediction_probabilities: list[np.ndarray] = []
    reliabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            image_a = batch["image_a"].to(device, non_blocking=True)
            image_b = batch["image_b"].to(device, non_blocking=True)
            targets = batch["clean_target"].to(device, non_blocking=True)
            features_a, logits_a, alpha_a = model(image_a)
            features_b, logits_b, alpha_b = model(image_b)
            reliability = _reliability(
                model,
                features_a,
                features_b,
                logits_a,
                logits_b,
                alpha_a,
                alpha_b,
                targets,
                config,
            )
            probability = (
                torch.softmax(logits_a, dim=1) + torch.softmax(logits_b, dim=1)
            ) * 0.5
            prediction_probabilities.append(probability.cpu().numpy())
            reliabilities.append(reliability.cpu().numpy())
            labels.append(targets.cpu().numpy())

    probabilities = np.concatenate(prediction_probabilities)
    reliability = np.concatenate(reliabilities)
    labels_np = np.concatenate(labels)
    probability_metrics = metrics_from_probabilities(probabilities, labels_np)
    correct = probabilities.argmax(axis=1) == labels_np
    top_probability = probabilities.max(axis=1)
    result = {
        "two_view_probability": probability_metrics,
        "reliability_only": selective_metrics(reliability, correct),
        "probability_only": selective_metrics(top_probability, correct),
        "probability_times_reliability": selective_metrics(
            top_probability * reliability,
            correct,
        ),
        "mean_reliability": float(reliability.mean()),
        "reliability_std": float(reliability.std()),
    }
    np.save(args.run_dir / "test_reliability.npy", reliability)
    np.save(args.run_dir / "test_two_view_probabilities.npy", probabilities)
    np.save(args.run_dir / "test_eval_labels.npy", labels_np)
    write_json(args.run_dir / "reliability_diagnostics.json", result)
    print(result)


if __name__ == "__main__":
    main()
