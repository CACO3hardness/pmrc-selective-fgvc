from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from rgel.data import CUBManifestDataset, build_transforms, load_manifest
from rgel.models import build_model
from rgel.utils import read_json, write_json
from scripts.posthoc_reliability import aurc_from_scores, risk_at_coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate post-hoc selective baselines.")
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def collect_logits(config: dict, model: torch.nn.Module, rows, device: torch.device):
    loader = DataLoader(
        CUBManifestDataset(
            config["root"],
            rows,
            build_transforms(int(config.get("image_size", 224)), train=False),
        ),
        batch_size=int(config.get("batch_size", 16)),
        shuffle=False,
        num_workers=0,
    )
    logits = []
    labels = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["clean_target"].to(device, non_blocking=True)
        logits.append(model(images).float().cpu().numpy())
        labels.append(targets.cpu().numpy())
    return np.concatenate(logits), np.concatenate(labels)


def maxlogit_pnorm(logits: np.ndarray, p: float) -> np.ndarray:
    centered = logits - logits.mean(axis=1, keepdims=True)
    norm = np.power(np.power(np.abs(centered), p).sum(axis=1), 1.0 / p)
    norm = np.clip(norm, 1e-12, None)
    return centered.max(axis=1) / norm


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    logits_tensor = torch.from_numpy(logits.astype(np.float32))
    labels_tensor = torch.from_numpy(labels.astype(np.int64))
    log_temperature = nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.Adam([log_temperature], lr=0.05)
    loss_function = nn.CrossEntropyLoss()
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = loss_function(logits_tensor / temperature, labels_tensor)
        loss.backward()
        optimizer.step()
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


def msp_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    probability = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    return probability.max(axis=1)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    config = read_json(args.run_dir / "config.json")
    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=False)
    rows = load_manifest(config["manifest"])
    num_classes = len({row.class_id for row in rows})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        config["model"],
        num_classes,
        pretrained=False,
        freeze_stages=int(config.get("freeze_stages", 0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    val_rows = [row for row in rows if row.split == "val"]
    test_rows = [row for row in rows if row.split == "test"]
    val_logits, val_labels = collect_logits(config, model, val_rows, device)
    np.save(args.run_dir / "selective_val_logits.npy", val_logits)
    np.save(args.run_dir / "selective_val_labels.npy", val_labels)
    cached_test_logits = args.run_dir / "test_logits.npy"
    cached_test_labels = args.run_dir / "posthoc_test_labels.npy"
    if cached_test_logits.is_file() and cached_test_labels.is_file():
        test_logits = np.load(cached_test_logits)
        test_labels = np.load(cached_test_labels)
    else:
        test_logits, test_labels = collect_logits(config, model, test_rows, device)
    val_predictions = val_logits.argmax(axis=1)
    test_predictions = test_logits.argmax(axis=1)
    val_correct = val_predictions == val_labels
    test_correct = test_predictions == test_labels

    msp_test = msp_from_logits(test_logits)
    candidates = {}
    for p in range(1, 11):
        candidates[p] = aurc_from_scores(maxlogit_pnorm(val_logits, p), val_correct)
    best_p = min(candidates, key=candidates.get)
    pnorm_test = maxlogit_pnorm(test_logits, best_p)
    maxlogit_test = test_logits.max(axis=1)
    selected_temperature = fit_temperature(val_logits, val_labels)
    temperature_scaled_test = msp_from_logits(
        test_logits,
        selected_temperature,
    )

    result = {
        "run_dir": str(args.run_dir),
        "validation_size": len(val_rows),
        "test_size": len(test_rows),
        "pnorm_validation_aurc_by_p": candidates,
        "selected_p": best_p,
        "test": {
            "msp": {
                "aurc": aurc_from_scores(msp_test, test_correct),
                "risk_at_90": risk_at_coverage(msp_test, test_correct, 0.9),
            },
            "maxlogit": {
                "aurc": aurc_from_scores(maxlogit_test, test_correct),
                "risk_at_90": risk_at_coverage(maxlogit_test, test_correct, 0.9),
            },
            "maxlogit_pnorm": {
                "aurc": aurc_from_scores(pnorm_test, test_correct),
                "risk_at_90": risk_at_coverage(pnorm_test, test_correct, 0.9),
            },
            "temperature_scaling": {
                "temperature": selected_temperature,
                "aurc": aurc_from_scores(temperature_scaled_test, test_correct),
                "risk_at_90": risk_at_coverage(
                    temperature_scaled_test,
                    test_correct,
                    0.9,
                ),
            },
        },
    }
    write_json(args.run_dir / "selective_baselines.json", result)
    print(result)


if __name__ == "__main__":
    main()
