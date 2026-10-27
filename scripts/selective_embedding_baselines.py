from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from rgel.data import (
    CUBManifestDataset,
    build_transforms,
    load_manifest,
    load_noise_map,
)
from rgel.models import build_model
from rgel.utils import read_json, set_seed, write_json
from scripts.posthoc_reliability import (
    aurc_from_scores,
    extract_features,
    risk_at_coverage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate KNN and trust-score selective baselines."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 5, 10, 20])
    return parser.parse_args()


@torch.inference_mode()
def collect_embeddings(
    config: dict,
    model: nn.Module,
    rows,
    device: torch.device,
    noise_map: dict[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        CUBManifestDataset(
            config["root"],
            rows,
            build_transforms(int(config.get("image_size", 224)), train=False),
            noise_map=noise_map,
        ),
        batch_size=int(config.get("batch_size", 16)),
        shuffle=False,
        num_workers=0,
    )
    embeddings = []
    logits_rows = []
    targets = []
    clean_targets = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        features, logits = extract_features(model, images)
        embeddings.append(F.normalize(features.float(), dim=1).cpu())
        logits_rows.append(logits.float().cpu())
        targets.append(batch["target"].cpu())
        clean_targets.append(batch["clean_target"].cpu())
    return (
        torch.cat(embeddings),
        torch.cat(logits_rows),
        torch.cat(targets),
        torch.cat(clean_targets),
    )


@torch.inference_mode()
def score_queries(
    query_embeddings: torch.Tensor,
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    predictions: torch.Tensor,
    k_values: list[int],
    device: torch.device,
    batch_size: int = 512,
) -> dict[str, dict[int, np.ndarray]]:
    max_k = max(k_values)
    train_embeddings = train_embeddings.to(device)
    train_labels = train_labels.to(device)
    predictions = predictions.to(device)
    outputs: dict[str, dict[int, list[np.ndarray]]] = {
        "knn_label_agreement": {k: [] for k in k_values},
        "knn_mean_similarity": {k: [] for k in k_values},
        "trust_score": {k: [] for k in k_values},
    }
    for start in range(0, len(query_embeddings), batch_size):
        query = query_embeddings[start : start + batch_size].to(device)
        prediction = predictions[start : start + batch_size]
        similarity = query @ train_embeddings.T
        top_similarity, top_indices = similarity.topk(max_k, dim=1)
        top_labels = train_labels[top_indices]

        same_class = train_labels[None, :] == prediction[:, None]
        same_similarity = similarity.masked_fill(~same_class, -torch.inf)
        other_similarity = similarity.masked_fill(same_class, -torch.inf)
        for k in k_values:
            agreement = (top_labels[:, :k] == prediction[:, None]).float().mean(
                dim=1
            )
            mean_similarity = top_similarity[:, :k].mean(dim=1)
            same_top = same_similarity.topk(min(k, len(train_labels)), dim=1).values
            same_distance = 1.0 - same_top.mean(dim=1)
            other_distance = 1.0 - other_similarity.max(dim=1).values
            trust_score = other_distance / same_distance.clamp_min(1e-6)
            outputs["knn_label_agreement"][k].append(agreement.cpu().numpy())
            outputs["knn_mean_similarity"][k].append(mean_similarity.cpu().numpy())
            outputs["trust_score"][k].append(trust_score.cpu().numpy())
    return {
        method: {
            k: np.concatenate(values).astype(np.float32)
            for k, values in method_values.items()
        }
        for method, method_values in outputs.items()
    }


def select_and_evaluate(
    val_scores: dict[int, np.ndarray],
    test_scores: dict[int, np.ndarray],
    val_correct: np.ndarray,
    test_correct: np.ndarray,
) -> dict:
    validation_aurc = {
        str(k): aurc_from_scores(scores, val_correct)
        for k, scores in val_scores.items()
    }
    selected_k = int(min(validation_aurc, key=validation_aurc.get))
    scores = test_scores[selected_k]
    return {
        "validation_aurc_by_k": validation_aurc,
        "selected_k": selected_k,
        "test_aurc": aurc_from_scores(scores, test_correct),
        "risk_at_70": risk_at_coverage(scores, test_correct, 0.7),
        "risk_at_80": risk_at_coverage(scores, test_correct, 0.8),
        "risk_at_90": risk_at_coverage(scores, test_correct, 0.9),
    }


def main() -> None:
    args = parse_args()
    set_seed(7)
    torch.set_num_threads(1)
    config = read_json(args.run_dir / "config.json")
    checkpoint = torch.load(
        args.run_dir / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
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

    train_rows = [row for row in rows if row.split == "train"]
    val_rows = [row for row in rows if row.split == "val"]
    test_rows = [row for row in rows if row.split == "test"]
    noise_map = load_noise_map(config.get("noise_csv"))
    train_embeddings, _, train_labels, _ = collect_embeddings(
        config,
        model,
        train_rows,
        device,
        noise_map=noise_map,
    )
    val_embeddings, val_logits, _, val_labels = collect_embeddings(
        config,
        model,
        val_rows,
        device,
    )
    test_embeddings, test_logits, _, test_labels = collect_embeddings(
        config,
        model,
        test_rows,
        device,
    )
    val_predictions = val_logits.argmax(dim=1)
    test_predictions = test_logits.argmax(dim=1)
    val_correct = (val_predictions == val_labels).numpy()
    test_correct = (test_predictions == test_labels).numpy()

    val_scores = score_queries(
        val_embeddings,
        train_embeddings,
        train_labels,
        val_predictions,
        args.k_values,
        device,
    )
    test_scores = score_queries(
        test_embeddings,
        train_embeddings,
        train_labels,
        test_predictions,
        args.k_values,
        device,
    )
    results = {
        method: select_and_evaluate(
            val_scores[method],
            test_scores[method],
            val_correct,
            test_correct,
        )
        for method in val_scores
    }

    val_probability = torch.softmax(val_logits, dim=1)
    test_probability = torch.softmax(test_logits, dim=1)
    val_msp = val_probability.max(dim=1).values.numpy()
    test_msp = test_probability.max(dim=1).values.numpy()
    output = {
        "run_dir": str(args.run_dir),
        "validation_size": len(val_rows),
        "test_size": len(test_rows),
        "test_accuracy": float(test_correct.mean()),
        "train_neighbor_labels": "noisy training labels",
        "baselines": results,
        "msp": {
            "aurc": aurc_from_scores(test_msp, test_correct),
            "risk_at_90": risk_at_coverage(test_msp, test_correct, 0.9),
        },
    }
    write_json(args.run_dir / "selective_embedding_baselines.json", output)
    print(output)


if __name__ == "__main__":
    main()
