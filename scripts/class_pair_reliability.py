from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rgel.data import load_manifest, load_noise_map
from rgel.models import build_model
from rgel.utils import read_json, set_seed, write_json
from scripts.posthoc_reliability import (
    aurc_from_scores,
    class_prototypes,
    cross_validate_logistic,
    extract_features,
    fit_logistic,
    loader_for,
    risk_at_coverage,
    score_logistic,
)


FEATURE_NAMES = [
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
]
PAIR_FEATURE_NAMES = [
    "pair_margin_mean",
    "pair_margin_min",
    "pair_margin_std",
    "pair_similarity_rank_agreement",
    "pair_logit_rank_agreement",
]
ALL_FEATURE_NAMES = FEATURE_NAMES + PAIR_FEATURE_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate class-pair reliability features without retraining."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def nearest_class_neighbors(prototypes: np.ndarray, top_k: int) -> np.ndarray:
    prototypes = prototypes.astype(np.float64, copy=False)
    similarity = prototypes @ prototypes.T
    np.fill_diagonal(similarity, -np.inf)
    top_k = min(top_k, len(prototypes) - 1)
    return np.argsort(-similarity, axis=1, kind="stable")[:, :top_k].astype(
        np.int64
    )


def pair_rank_agreement(values_a: torch.Tensor, values_b: torch.Tensor) -> torch.Tensor:
    pair_i, pair_j = torch.triu_indices(
        values_a.shape[1],
        values_a.shape[1],
        offset=1,
        device=values_a.device,
    )
    difference_a = values_a[:, pair_i] - values_a[:, pair_j]
    difference_b = values_b[:, pair_i] - values_b[:, pair_j]
    non_tied = (difference_a != 0) & (difference_b != 0)
    agreements = (difference_a.sign() == difference_b.sign()) & non_tied
    denominator = non_tied.sum(dim=1).clamp_min(1)
    return agreements.sum(dim=1).float() / denominator


@torch.inference_mode()
def collect_features(
    model: nn.Module,
    loader,
    prototypes: np.ndarray,
    neighbors: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_rows: list[list[float]] = []
    pair_rows: list[list[float]] = []
    labels: list[int] = []
    predictions: list[int] = []
    prototype_tensor = torch.from_numpy(prototypes).to(device)
    neighbor_tensor = torch.from_numpy(neighbors).to(device)

    for batch in loader:
        image_a = batch["image_a"].to(device, non_blocking=True)
        image_b = torch.flip(
            batch["image_b"].to(device, non_blocking=True),
            dims=[3],
        )
        targets = batch["clean_target"].numpy()
        features_a, logits_a = extract_features(model, image_a)
        features_b, logits_b = extract_features(model, image_b)
        probability_a = torch.softmax(logits_a, dim=1)
        probability_b = torch.softmax(logits_b, dim=1)

        sorted_probability = probability_a.sort(dim=1, descending=True).values
        top_probability = sorted_probability[:, 0]
        margin = sorted_probability[:, 0] - sorted_probability[:, 1]
        entropy = -(probability_a * torch.log(probability_a.clamp_min(1e-8))).sum(
            dim=1
        )
        energy = torch.logsumexp(logits_a, dim=1)
        feature_norm = torch.linalg.vector_norm(features_a, dim=1)
        flip_agreement = torch.sqrt(probability_a * probability_b).sum(dim=1)
        flip_feature_similarity = F.cosine_similarity(features_a, features_b, dim=1)
        flip_prediction_agreement = (
            probability_a.argmax(dim=1) == probability_b.argmax(dim=1)
        ).float()
        logit_view_distance = torch.linalg.vector_norm(logits_a - logits_b, dim=1)

        normalized_features_a = F.normalize(features_a, dim=1)
        normalized_features_b = F.normalize(features_b, dim=1)
        similarity_a = normalized_features_a @ prototype_tensor.T
        similarity_b = normalized_features_b @ prototype_tensor.T
        predicted = probability_a.argmax(dim=1)
        observed_similarity = similarity_a.gather(1, predicted[:, None]).squeeze(1)
        masked = similarity_a.clone()
        masked.scatter_(1, predicted[:, None], -float("inf"))
        prototype_margin = observed_similarity - masked.max(dim=1).values

        base_values = torch.stack(
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

        batch_indices = torch.arange(len(predicted), device=device)
        selected_neighbors = neighbor_tensor[predicted]
        anchor_similarity_a = similarity_a[batch_indices, predicted]
        anchor_similarity_b = similarity_b[batch_indices, predicted]
        neighbor_similarity_a = similarity_a[batch_indices[:, None], selected_neighbors]
        neighbor_similarity_b = similarity_b[batch_indices[:, None], selected_neighbors]
        pair_margins = anchor_similarity_a[:, None] - neighbor_similarity_a

        similarity_vectors_a = torch.cat(
            (anchor_similarity_a[:, None], neighbor_similarity_a),
            dim=1,
        )
        similarity_vectors_b = torch.cat(
            (anchor_similarity_b[:, None], neighbor_similarity_b),
            dim=1,
        )
        logit_vectors_a = torch.cat(
            (
                logits_a.gather(1, predicted[:, None]),
                logits_a.gather(1, selected_neighbors),
            ),
            dim=1,
        )
        logit_vectors_b = torch.cat(
            (
                logits_b.gather(1, predicted[:, None]),
                logits_b.gather(1, selected_neighbors),
            ),
            dim=1,
        )
        pair_values = torch.stack(
            (
                pair_margins.mean(dim=1),
                pair_margins.min(dim=1).values,
                pair_margins.std(dim=1, unbiased=False),
                pair_rank_agreement(similarity_vectors_a, similarity_vectors_b),
                pair_rank_agreement(logit_vectors_a, logit_vectors_b),
            ),
            dim=1,
        )
        base_rows.extend(base_values.float().cpu().tolist())
        pair_rows.extend(pair_values.float().cpu().tolist())
        labels.extend(targets.tolist())
        predictions.extend(predicted.cpu().tolist())

    return (
        np.asarray(base_rows, dtype=np.float32),
        np.asarray(pair_rows, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(predictions, dtype=np.int64),
    )


def select_and_evaluate(
    val_features: np.ndarray,
    test_features: np.ndarray,
    val_correct: np.ndarray,
    test_correct: np.ndarray,
    folds: list[np.ndarray],
) -> tuple[dict, np.ndarray]:
    candidates = (1e-4, 1e-3, 1e-2, 1e-1)
    cv_results = {}
    for weight_decay in candidates:
        _, cv_aurc = cross_validate_logistic(
            val_features,
            val_correct,
            folds,
            weight_decay,
        )
        cv_results[str(weight_decay)] = cv_aurc
    selected_weight_decay = float(min(cv_results, key=cv_results.get))
    calibrator, mean, std = fit_logistic(
        val_features,
        val_correct,
        selected_weight_decay,
    )
    test_scores = score_logistic(
        calibrator,
        test_features,
        mean,
        std,
    )
    result = {
        "cv_aurc": cv_results,
        "selected_weight_decay": selected_weight_decay,
        "test_aurc": aurc_from_scores(test_scores, test_correct),
        "risk_at_70": risk_at_coverage(test_scores, test_correct, 0.7),
        "risk_at_80": risk_at_coverage(test_scores, test_correct, 0.8),
        "risk_at_90": risk_at_coverage(test_scores, test_correct, 0.9),
        "calibrator_weights": calibrator.weight.detach().squeeze(0).tolist(),
        "calibrator_bias": float(calibrator.bias.detach()),
    }
    return result, test_scores


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
    image_size = int(config.get("image_size", 224))
    batch_size = int(config.get("batch_size", 16))
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
    neighbors = nearest_class_neighbors(prototypes, args.top_k)
    val_base, val_pair, val_labels, val_predictions = collect_features(
        model,
        loader_for(
            config["root"],
            val_rows,
            image_size,
            batch_size,
            two_view=True,
        ),
        prototypes,
        neighbors,
        device,
    )
    test_base, test_pair, test_labels, test_predictions = collect_features(
        model,
        loader_for(
            config["root"],
            test_rows,
            image_size,
            batch_size,
            two_view=True,
        ),
        prototypes,
        neighbors,
        device,
    )
    val_correct = val_predictions == val_labels
    test_correct = test_predictions == test_labels

    rng = np.random.default_rng(7)
    val_indices = np.arange(len(val_correct))
    rng.shuffle(val_indices)
    folds = np.array_split(val_indices, args.folds)

    group_indices = {
        "confidence": list(range(4)),
        "current_confidence_plus_prototype": [0, 1, 2, 3, 4, 9],
        "current_all": list(range(len(FEATURE_NAMES))),
        "pair_only": list(range(len(FEATURE_NAMES), len(FEATURE_NAMES) + len(PAIR_FEATURE_NAMES))),
        "confidence_plus_pair": list(range(4))
        + list(
            range(
                len(FEATURE_NAMES),
                len(FEATURE_NAMES) + len(PAIR_FEATURE_NAMES),
            )
        ),
        "all_plus_pair": list(
            range(len(FEATURE_NAMES) + len(PAIR_FEATURE_NAMES))
        ),
    }
    val_combined = np.concatenate((val_base, val_pair), axis=1)
    test_combined = np.concatenate((test_base, test_pair), axis=1)
    group_results = {}
    test_scores = {}
    for name, selected in group_indices.items():
        result, scores = select_and_evaluate(
            val_combined[:, selected],
            test_combined[:, selected],
            val_correct,
            test_correct,
            folds,
        )
        result["feature_names"] = ALL_FEATURE_NAMES
        result["selected_indices"] = selected
        result["selected_feature_names"] = [
            ALL_FEATURE_NAMES[index] for index in selected
        ]
        group_results[name] = result
        test_scores[name] = scores

    bootstrap_differences = []
    pair_scores = test_scores["all_plus_pair"]
    current_scores = test_scores["current_all"]
    for _ in range(args.bootstrap):
        sample = rng.integers(0, len(test_correct), len(test_correct))
        bootstrap_differences.append(
            aurc_from_scores(pair_scores[sample], test_correct[sample])
            - aurc_from_scores(current_scores[sample], test_correct[sample])
        )

    output = {
        "run_dir": str(args.run_dir),
        "top_k": int(neighbors.shape[1]),
        "prototype_source": "current model trained features grouped by noisy training labels",
        "validation_accuracy": float(val_correct.mean()),
        "test_accuracy": float(test_correct.mean()),
        "feature_names": ALL_FEATURE_NAMES,
        "pair_feature_names": PAIR_FEATURE_NAMES,
        "group_results": group_results,
        "all_plus_pair_minus_current_all_aurc": float(
            group_results["all_plus_pair"]["test_aurc"]
            - group_results["current_all"]["test_aurc"]
        ),
        "all_plus_pair_minus_current_all_aurc_ci95": [
            float(np.quantile(bootstrap_differences, 0.025)),
            float(np.quantile(bootstrap_differences, 0.975)),
        ],
        "pair_margins_mean_by_prediction_correctness": {
            "correct": float(test_pair[test_correct, 0].mean()),
            "wrong": float(test_pair[~test_correct, 0].mean()),
        },
        "pair_similarity_rank_agreement_mean_by_prediction_correctness": {
            "correct": float(test_pair[test_correct, 3].mean()),
            "wrong": float(test_pair[~test_correct, 3].mean()),
        },
        "pair_logit_rank_agreement_mean_by_prediction_correctness": {
            "correct": float(test_pair[test_correct, 4].mean()),
            "wrong": float(test_pair[~test_correct, 4].mean()),
        },
    }
    write_json(args.run_dir / "class_pair_reliability.json", output)
    np.save(args.run_dir / "class_pair_val_base_features.npy", val_base)
    np.save(args.run_dir / "class_pair_test_base_features.npy", test_base)
    np.save(args.run_dir / "class_pair_val_features.npy", val_pair)
    np.save(args.run_dir / "class_pair_test_features.npy", test_pair)
    np.save(args.run_dir / "class_pair_val_labels.npy", val_labels)
    np.save(args.run_dir / "class_pair_val_predictions.npy", val_predictions)
    np.save(args.run_dir / "class_pair_test_labels.npy", test_labels)
    np.save(args.run_dir / "class_pair_test_predictions.npy", test_predictions)
    np.save(args.run_dir / "class_pair_neighbors.npy", neighbors)
    print(output)


if __name__ == "__main__":
    main()
