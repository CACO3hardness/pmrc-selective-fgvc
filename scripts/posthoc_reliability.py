from __future__ import annotations

import argparse
import math
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
    CUBTwoViewDataset,
    ManifestRow,
    build_transforms,
    load_manifest,
)
from rgel.models import build_model, forward_features
from rgel.utils import read_json, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test post-hoc selective reliability.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def loader_for(
    root: str,
    rows: list[ManifestRow],
    image_size: int,
    batch_size: int,
    two_view: bool,
    noise_map: dict[int, int] | None = None,
) -> DataLoader:
    dataset = (
        CUBTwoViewDataset(root, rows, build_transforms(image_size, train=False))
        if two_view
        else CUBManifestDataset(
            root,
            rows,
            build_transforms(image_size, train=False),
            noise_map=noise_map,
        )
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def extract_features(model: nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return forward_features(model, images)


@torch.inference_mode()
def class_prototypes(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> np.ndarray:
    feature_dim = None
    sums = None
    counts = np.zeros(num_classes, dtype=np.int64)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["target"].cpu().numpy()
        features, _ = extract_features(model, images)
        features = F.normalize(features.float(), dim=1).cpu().numpy()
        if sums is None:
            feature_dim = features.shape[1]
            sums = np.zeros((num_classes, feature_dim), dtype=np.float64)
        np.add.at(sums, labels, features)
        np.add.at(counts, labels, 1)
    assert sums is not None
    missing_classes = np.flatnonzero(counts == 0)
    if missing_classes.size:
        raise ValueError(
            "Cannot construct prototypes for observed-label classes without "
            f"training samples: {missing_classes[:10].tolist()}"
        )
    prototypes = sums / counts[:, None]
    prototypes /= np.clip(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12, None)
    return prototypes.astype(np.float32)


@torch.inference_mode()
def reliability_features(
    model: nn.Module,
    loader: DataLoader,
    prototypes: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    labels: list[int] = []
    predictions: list[int] = []
    prototype_tensor = torch.from_numpy(prototypes).to(device)
    for batch in loader:
        image_a = batch["image_a"].to(device, non_blocking=True)
        image_b = torch.flip(batch["image_b"].to(device, non_blocking=True), dims=[3])
        targets = batch["clean_target"].numpy()
        features_a, logits_a = extract_features(model, image_a)
        features_b, logits_b = extract_features(model, image_b)
        probability_a = torch.softmax(logits_a, dim=1)
        probability_b = torch.softmax(logits_b, dim=1)

        sorted_probability = probability_a.sort(dim=1, descending=True).values
        top_probability = sorted_probability[:, 0]
        margin = sorted_probability[:, 0] - sorted_probability[:, 1]
        entropy = -(probability_a * torch.log(probability_a.clamp_min(1e-8))).sum(dim=1)
        energy = torch.logsumexp(logits_a, dim=1)
        feature_norm = torch.linalg.vector_norm(features_a, dim=1)
        flip_agreement = torch.sqrt(probability_a * probability_b).sum(dim=1)
        flip_feature_similarity = F.cosine_similarity(features_a, features_b, dim=1)
        flip_prediction_agreement = (
            probability_a.argmax(dim=1) == probability_b.argmax(dim=1)
        ).float()
        logit_view_distance = torch.linalg.vector_norm(logits_a - logits_b, dim=1)

        normalized_features = F.normalize(features_a, dim=1)
        similarity = normalized_features @ prototype_tensor.T
        predicted = probability_a.argmax(dim=1)
        observed_similarity = similarity.gather(1, predicted[:, None]).squeeze(1)
        masked = similarity.clone()
        masked.scatter_(1, predicted[:, None], -float("inf"))
        prototype_margin = observed_similarity - masked.max(dim=1).values

        values = torch.stack(
            (
                top_probability,
                margin,
                -entropy,
                energy,
                feature_norm,
                flip_agreement,
                flip_feature_similarity,
                flip_prediction_agreement,
                logit_view_distance,
                prototype_margin,
            ),
            dim=1,
        )
        rows.extend(values.float().cpu().tolist())
        labels.extend(targets.tolist())
        predictions.extend(predicted.cpu().tolist())
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(predictions, dtype=np.int64),
    )


def aurc_from_scores(scores: np.ndarray, correct: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    ordered = correct[order]
    risks = np.cumsum(~ordered) / np.arange(1, len(correct) + 1)
    coverage = np.arange(1, len(correct) + 1) / len(correct)
    return float(
        np.trapezoid(
            np.concatenate(([0.0], risks)),
            np.concatenate(([0.0], coverage)),
        )
    )


def risk_at_coverage(scores: np.ndarray, correct: np.ndarray, level: float) -> float:
    order = np.argsort(-scores, kind="stable")
    risks = np.cumsum(~correct[order]) / np.arange(1, len(correct) + 1)
    index = max(0, int(np.ceil(level * len(correct))) - 1)
    return float(risks[index])


def fit_logistic(
    features: np.ndarray,
    targets: np.ndarray,
    weight_decay: float,
    steps: int = 500,
) -> tuple[nn.Linear, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-6] = 1.0
    normalized = (features - mean) / std
    x = torch.from_numpy(normalized.astype(np.float32))
    y = torch.from_numpy(targets.astype(np.float32))
    model = nn.Linear(x.shape[1], 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=weight_decay)
    loss_function = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(x).squeeze(1), y)
        loss.backward()
        optimizer.step()
    return model, mean, std


def score_logistic(
    model: nn.Linear,
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    normalized = (features - mean) / std
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(normalized.astype(np.float32)))).squeeze(1).numpy()


def cross_validate_logistic(
    features: np.ndarray,
    targets: np.ndarray,
    folds: list[np.ndarray],
    weight_decay: float,
) -> tuple[np.ndarray, float]:
    out_of_fold = np.zeros(len(targets), dtype=np.float32)
    for fold_index in range(len(folds)):
        validation_index = folds[fold_index]
        training_index = np.concatenate(
            [folds[index] for index in range(len(folds)) if index != fold_index]
        )
        fit, mean, std = fit_logistic(
            features[training_index],
            targets[training_index],
            weight_decay,
        )
        out_of_fold[validation_index] = score_logistic(
            fit,
            features[validation_index],
            mean,
            std,
        )
    return out_of_fold, aurc_from_scores(out_of_fold, targets)


def main() -> None:
    args = parse_args()
    set_seed(7)
    # The calibrator operates on small tabular batches. Allowing PyTorch to use
    # every CPU thread creates heavy oversubscription and can make the bootstrap
    # and cross-validation phase much slower than the GPU feature extraction.
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

    train_rows = [row for row in rows if row.split == "train"]
    val_rows = [row for row in rows if row.split == "val"]
    test_rows = [row for row in rows if row.split == "test"]
    image_size = int(config.get("image_size", 224))
    batch_size = int(config.get("batch_size", 16))
    from rgel.data import load_noise_map

    noise_map = load_noise_map(config.get("noise_csv"))
    prototypes = class_prototypes(
        model,
        loader_for(
            config["root"],
            train_rows,
            image_size,
            batch_size,
            two_view=False,
            noise_map=noise_map,
        ),
        num_classes,
        device,
    )
    val_features, val_labels, val_predictions = reliability_features(
        model,
        loader_for(config["root"], val_rows, image_size, batch_size, two_view=True),
        prototypes,
        device,
    )
    test_features, test_labels, test_predictions = reliability_features(
        model,
        loader_for(config["root"], test_rows, image_size, batch_size, two_view=True),
        prototypes,
        device,
    )
    val_correct = val_predictions == val_labels
    test_correct = test_predictions == test_labels

    rng = np.random.default_rng(7)
    indices = np.arange(len(val_rows))
    rng.shuffle(indices)
    folds = np.array_split(indices, args.folds)
    candidates = [1e-4, 1e-3, 1e-2, 1e-1]
    cv_results: dict[str, float] = {}
    for weight_decay in candidates:
        _, cv_results[str(weight_decay)] = cross_validate_logistic(
            val_features,
            val_correct,
            folds,
            weight_decay,
        )
    selected_decay = float(min(cv_results, key=cv_results.get))
    calibrator, mean, std = fit_logistic(val_features, val_correct, selected_decay)
    val_scores = score_logistic(calibrator, val_features, mean, std)
    test_scores = score_logistic(calibrator, test_features, mean, std)

    named_scores = {
        "msp": test_features[:, 0],
        "margin": test_features[:, 1],
        "negative_entropy": test_features[:, 2],
        "energy": test_features[:, 3],
        "learned_reliability": test_scores,
    }
    results = {
        name: {
            "aurc": aurc_from_scores(scores, test_correct),
            "risk_at_70": risk_at_coverage(scores, test_correct, 0.7),
            "risk_at_80": risk_at_coverage(scores, test_correct, 0.8),
            "risk_at_90": risk_at_coverage(scores, test_correct, 0.9),
        }
        for name, scores in named_scores.items()
    }
    feature_groups = {
        "confidence": [0, 1, 2, 3],
        "confidence_plus_view": [0, 1, 2, 3, 5, 6, 7, 8],
        "confidence_plus_prototype": [0, 1, 2, 3, 4, 9],
        "all": list(range(test_features.shape[1])),
    }
    group_results: dict[str, dict[str, Any]] = {}
    for group_name, selected_features in feature_groups.items():
        selected_features = list(selected_features)
        group_cv = {}
        for weight_decay in (1e-3, 1e-2):
            _, value = cross_validate_logistic(
                val_features[:, selected_features],
                val_correct,
                folds,
                weight_decay,
            )
            group_cv[str(weight_decay)] = value
        selected_group_decay = float(min(group_cv, key=group_cv.get))
        group_model, group_mean, group_std = fit_logistic(
            val_features[:, selected_features],
            val_correct,
            selected_group_decay,
        )
        group_test_scores = score_logistic(
            group_model,
            test_features[:, selected_features],
            group_mean,
            group_std,
        )
        group_results[group_name] = {
            "features": selected_features,
            "cv_aurc": group_cv,
            "selected_weight_decay": selected_group_decay,
            "test_aurc": aurc_from_scores(group_test_scores, test_correct),
            "risk_at_70": risk_at_coverage(group_test_scores, test_correct, 0.7),
            "risk_at_80": risk_at_coverage(group_test_scores, test_correct, 0.8),
            "risk_at_90": risk_at_coverage(group_test_scores, test_correct, 0.9),
        }
    point_difference = results["learned_reliability"]["aurc"] - results["msp"]["aurc"]
    bootstrap_differences = []
    for _ in range(args.bootstrap):
        sample = rng.integers(0, len(test_rows), len(test_rows))
        bootstrap_differences.append(
            aurc_from_scores(test_scores[sample], test_correct[sample])
            - aurc_from_scores(test_features[sample, 0], test_correct[sample])
        )

    output = {
        "run_dir": str(args.run_dir),
        "validation_accuracy": float(val_correct.mean()),
        "test_accuracy": float(test_correct.mean()),
        "feature_names": [
            "msp",
            "margin",
            "negative_entropy",
            "energy",
            "feature_norm",
            "flip_agreement",
            "flip_feature_similarity",
            "flip_prediction_agreement",
            "logit_view_distance",
            "prototype_margin",
        ],
        "cv_aurc": cv_results,
        "selected_weight_decay": selected_decay,
        "validation_learned_aurc": aurc_from_scores(val_scores, val_correct),
        "test_results": results,
        "feature_group_results": group_results,
        "calibrator_weights": calibrator.weight.detach().squeeze(0).tolist(),
        "calibrator_bias": float(calibrator.bias.detach()),
        "reliability_mean_correct": float(test_scores[test_correct].mean()),
        "reliability_mean_wrong": float(test_scores[~test_correct].mean()),
        "learned_minus_msp_aurc": float(point_difference),
        "learned_minus_msp_aurc_ci95": [
            float(np.quantile(bootstrap_differences, 0.025)),
            float(np.quantile(bootstrap_differences, 0.975)),
        ],
    }
    write_json(args.run_dir / "posthoc_reliability.json", output)
    np.save(args.run_dir / "posthoc_test_features.npy", test_features)
    np.save(args.run_dir / "posthoc_test_scores.npy", test_scores)
    np.save(args.run_dir / "posthoc_val_features.npy", val_features)
    np.save(args.run_dir / "posthoc_val_scores.npy", val_scores)
    np.save(args.run_dir / "posthoc_val_labels.npy", val_labels)
    np.save(args.run_dir / "posthoc_val_predictions.npy", val_predictions)
    np.save(args.run_dir / "posthoc_test_labels.npy", test_labels)
    np.save(args.run_dir / "posthoc_test_predictions.npy", test_predictions)
    print(output)


if __name__ == "__main__":
    main()
