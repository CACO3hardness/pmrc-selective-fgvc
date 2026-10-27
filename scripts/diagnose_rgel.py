from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from rgel.data import CUBManifestDataset, build_transforms, load_manifest
from rgel.evidential import EvidentialResNet
from rgel.metrics import metrics_from_probabilities, softmax
from rgel.utils import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare RGEL confidence heads.")
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def selective_metrics(scores: np.ndarray, correct: np.ndarray) -> dict[str, float]:
    order = np.argsort(-scores, kind="stable")
    ordered_correct = correct[order]
    risks = np.cumsum(~ordered_correct) / np.arange(1, len(correct) + 1)
    coverage = np.arange(1, len(correct) + 1) / len(correct)
    aurc = float(
        np.trapezoid(
            np.concatenate(([0.0], risks)),
            np.concatenate(([0.0], coverage)),
        )
    )
    result = {"aurc": aurc}
    for level in (0.7, 0.8, 0.9):
        index = max(0, int(np.ceil(level * len(correct))) - 1)
        result[f"risk_at_{int(level * 100)}"] = float(risks[index])
    return result


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    config = read_json(args.run_dir / "config.json")
    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=False)
    rows = load_manifest(config["manifest"])
    test_rows = [row for row in rows if row.split == "test"]
    loader = DataLoader(
        CUBManifestDataset(
            config["root"],
            test_rows,
            build_transforms(int(config.get("image_size", 224)), train=False),
        ),
        batch_size=int(config.get("batch_size", 16)),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EvidentialResNet(
        config["model"],
        num_classes=len({row.class_id for row in rows}),
        pretrained=False,
        freeze_stages=int(config.get("freeze_stages", 0)),
        prototype_momentum=float(config.get("prototype_momentum", 0.9)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    logits_list: list[np.ndarray] = []
    alpha_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["clean_target"].to(device, non_blocking=True)
        _, logits, alpha = model(images)
        logits_list.append(logits.float().cpu().numpy())
        alpha_list.append(alpha.float().cpu().numpy())
        labels_list.append(labels.cpu().numpy())

    logits = np.concatenate(logits_list)
    alpha = np.concatenate(alpha_list)
    labels = np.concatenate(labels_list)
    alpha_probability = alpha / alpha.sum(axis=1, keepdims=True)
    logit_probability = softmax(logits)
    prediction = alpha_probability.argmax(axis=1)
    correct = prediction == labels
    top_logit_probability = logit_probability.max(axis=1)
    vacuity = alpha.shape[1] / alpha.sum(axis=1)

    diagnostics = {
        "alpha_probability": metrics_from_probabilities(alpha_probability, labels),
        "logit_probability": metrics_from_probabilities(logit_probability, labels),
        "logit_confidence": selective_metrics(top_logit_probability, correct),
        "evidence_strength": selective_metrics(1.0 - vacuity, correct),
        "combined_confidence": selective_metrics(
            top_logit_probability * (1.0 - vacuity),
            correct,
        ),
        "mean_vacuity": float(vacuity.mean()),
        "mean_total_evidence": float(alpha.sum(axis=1).mean()),
        "mean_top_alpha_probability": float(alpha_probability.max(axis=1).mean()),
        "mean_top_logit_probability": float(top_logit_probability.mean()),
    }
    np.save(args.run_dir / "test_logits.npy", logits)
    np.save(args.run_dir / "test_alpha.npy", alpha)
    np.save(args.run_dir / "test_labels.npy", labels)
    write_json(args.run_dir / "diagnostic_metrics.json", diagnostics)
    print(diagnostics)


if __name__ == "__main__":
    main()

