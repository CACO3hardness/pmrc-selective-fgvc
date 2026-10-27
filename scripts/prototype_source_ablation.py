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
    cross_validate_logistic,
    extract_features,
    fit_logistic,
    risk_at_coverage,
    score_logistic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare noisy, clean, pretrained, and shuffled prototypes."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


@torch.inference_mode()
def collect_features(
    config: dict,
    model: nn.Module,
    rows,
    device: torch.device,
    noise_map: dict[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    features = []
    logits_rows = []
    labels = []
    noisy_labels = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        batch_features, batch_logits = extract_features(model, images)
        features.append(F.normalize(batch_features.float(), dim=1).cpu())
        logits_rows.append(batch_logits.float().cpu())
        labels.append(batch["clean_target"].cpu())
        noisy_labels.append(batch["target"].cpu())
    return (
        torch.cat(features),
        torch.cat(logits_rows),
        torch.cat(labels),
        torch.cat(noisy_labels),
    )


def build_prototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> np.ndarray:
    features_np = features.numpy()
    labels_np = labels.numpy()
    sums = np.zeros((num_classes, features_np.shape[1]), dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.int64)
    np.add.at(sums, labels_np, features_np)
    np.add.at(counts, labels_np, 1)
    missing = np.flatnonzero(counts == 0)
    if len(missing):
        raise ValueError(f"Missing prototype classes: {missing[:10].tolist()}")
    prototypes = sums / counts[:, None]
    prototypes /= np.clip(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12, None)
    return prototypes.astype(np.float32)


def prototype_margin(
    features: torch.Tensor,
    logits: torch.Tensor,
    prototypes: np.ndarray,
) -> np.ndarray:
    similarity = features @ torch.from_numpy(prototypes).T
    predicted = logits.argmax(dim=1)
    batch_indices = torch.arange(len(features))
    observed = similarity[batch_indices, predicted]
    masked = similarity.clone()
    masked[batch_indices, predicted] = -torch.inf
    return (observed - masked.max(dim=1).values).numpy().astype(np.float32)


def binary_auroc(scores: np.ndarray, correct: np.ndarray) -> float:
    correct = np.asarray(correct, dtype=bool)
    n_positive = int(correct.sum())
    n_negative = int(len(correct) - n_positive)
    if n_positive == 0 or n_negative == 0:
        raise ValueError("AUROC is undefined without both correct and incorrect samples")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    positive_rank_sum = ranks[correct].sum()
    return float(
        (
            positive_rank_sum
            - n_positive * (n_positive + 1) / 2.0
        )
        / (n_positive * n_negative)
    )


def paired_bootstrap_aurc_difference(
    reference_scores: np.ndarray,
    comparison_scores: np.ndarray,
    correct: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, object]:
    point_difference = aurc_from_scores(
        reference_scores, correct
    ) - aurc_from_scores(comparison_scores, correct)
    differences = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sample = rng.integers(0, len(correct), len(correct))
        sample_correct = correct[sample]
        if sample_correct.all() or not sample_correct.any():
            differences[index] = np.nan
            continue
        differences[index] = aurc_from_scores(
            reference_scores[sample], sample_correct
        ) - aurc_from_scores(comparison_scores[sample], sample_correct)
    finite = differences[np.isfinite(differences)]
    if len(finite) == 0:
        raise ValueError("No valid bootstrap samples contained both outcomes")
    low, high = np.quantile(finite, [0.025, 0.975])
    return {
        "difference_reference_minus_comparison": float(point_difference),
        "ci95": [float(low), float(high)],
        "n_valid_bootstrap": int(len(finite)),
        "reference_better_in_valid_bootstrap": float(np.mean(finite < 0.0)),
    }


def evaluate_features(
    val_features: np.ndarray,
    val_correct: np.ndarray,
    test_features: np.ndarray,
    test_correct: np.ndarray,
    folds: list[np.ndarray],
) -> tuple[dict, np.ndarray]:
    cv_results = {}
    for weight_decay in (1e-4, 1e-3, 1e-2, 1e-1):
        _, value = cross_validate_logistic(
            val_features,
            val_correct,
            folds,
            weight_decay,
        )
        cv_results[str(weight_decay)] = value
    selected_weight_decay = float(min(cv_results, key=cv_results.get))
    calibrator, mean, std = fit_logistic(
        val_features,
        val_correct,
        selected_weight_decay,
    )
    scores = score_logistic(calibrator, test_features, mean, std)
    return (
        {
            "cv_aurc": cv_results,
            "selected_weight_decay": selected_weight_decay,
            "test_aurc": aurc_from_scores(scores, test_correct),
            "risk_at_90": risk_at_coverage(scores, test_correct, 0.9),
        },
        scores,
    )


def main() -> None:
    args = parse_args()
    set_seed(7)
    torch.set_num_threads(1)
    config = read_json(args.run_dir / "config.json")
    rows = load_manifest(config["manifest"])
    train_rows = [row for row in rows if row.split == "train"]
    val_rows = [row for row in rows if row.split == "val"]
    test_rows = [row for row in rows if row.split == "test"]
    num_classes = len({row.class_id for row in rows})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise_map = load_noise_map(config.get("noise_csv"))

    checkpoint = torch.load(
        args.run_dir / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    trained_model = build_model(
        config["model"],
        num_classes,
        pretrained=False,
        freeze_stages=int(config.get("freeze_stages", 0)),
    ).to(device)
    trained_model.load_state_dict(checkpoint["model_state"])
    trained_model.eval()
    pretrained_model = build_model(
        config["model"],
        num_classes,
        pretrained=True,
        freeze_stages=int(config.get("freeze_stages", 0)),
    ).to(device)
    pretrained_model.eval()

    train_features, _, clean_train_labels, noisy_train_labels = collect_features(
        config,
        trained_model,
        train_rows,
        device,
        noise_map=noise_map,
    )
    if noise_map and np.array_equal(
        clean_train_labels.numpy(),
        noisy_train_labels.numpy(),
    ):
        raise RuntimeError(
            "Noisy and clean training labels are identical despite a noise map; "
            "refusing to run the prototype-source ablation."
        )
    val_features_model, val_logits, val_labels, _ = collect_features(
        config,
        trained_model,
        val_rows,
        device,
    )
    test_features_model, test_logits, test_labels, _ = collect_features(
        config,
        trained_model,
        test_rows,
        device,
    )
    pretrained_train_features, _, _, _ = collect_features(
        config,
        pretrained_model,
        train_rows,
        device,
    )
    pretrained_val_features, pretrained_val_logits, _, _ = collect_features(
        config,
        pretrained_model,
        val_rows,
        device,
    )
    pretrained_test_features, pretrained_test_logits, _, _ = collect_features(
        config,
        pretrained_model,
        test_rows,
        device,
    )

    noisy_prototypes = build_prototypes(
        train_features,
        noisy_train_labels,
        num_classes,
    )
    clean_prototypes = build_prototypes(
        train_features,
        clean_train_labels,
        num_classes,
    )
    pretrained_prototypes = build_prototypes(
        pretrained_train_features,
        noisy_train_labels,
        num_classes,
    )
    shuffled_prototypes = noisy_prototypes[
        np.random.default_rng(19).permutation(num_classes)
    ]

    margins = {
        "noisy": prototype_margin(
            test_features_model,
            test_logits,
            noisy_prototypes,
        ),
        "clean_oracle": prototype_margin(
            test_features_model,
            test_logits,
            clean_prototypes,
        ),
        "pretrained": prototype_margin(
            pretrained_test_features,
            test_logits,
            pretrained_prototypes,
        ),
        "shuffled": prototype_margin(
            test_features_model,
            test_logits,
            shuffled_prototypes,
        ),
    }
    val_margins = {
        "noisy": prototype_margin(
            val_features_model,
            val_logits,
            noisy_prototypes,
        ),
        "clean_oracle": prototype_margin(
            val_features_model,
            val_logits,
            clean_prototypes,
        ),
        "pretrained": prototype_margin(
            pretrained_val_features,
            val_logits,
            pretrained_prototypes,
        ),
        "shuffled": prototype_margin(
            val_features_model,
            val_logits,
            shuffled_prototypes,
        ),
    }

    val_base = np.load(args.run_dir / "posthoc_val_features.npy")
    test_base = np.load(args.run_dir / "posthoc_test_features.npy")
    if val_labels.numpy().tolist() != np.load(args.run_dir / "posthoc_val_labels.npy").tolist():
        raise ValueError("Validation ordering differs from posthoc features")
    if test_labels.numpy().tolist() != np.load(args.run_dir / "posthoc_test_labels.npy").tolist():
        raise ValueError("Test ordering differs from posthoc features")
    val_predictions = np.load(args.run_dir / "posthoc_val_predictions.npy")
    test_predictions = np.load(args.run_dir / "posthoc_test_predictions.npy")
    val_correct = val_predictions == val_labels.numpy()
    test_correct = test_predictions == test_labels.numpy()

    rng = np.random.default_rng(7)
    val_indices = np.arange(len(val_correct))
    rng.shuffle(val_indices)
    folds = np.array_split(val_indices, 5)
    results = {}
    score_arrays = {}
    results["without_prototype"], score_arrays["without_prototype"] = evaluate_features(
        val_base[:, :9],
        val_correct,
        test_base[:, :9],
        test_correct,
        folds,
    )
    for name in ("noisy", "clean_oracle", "pretrained", "shuffled"):
        val_with_margin = val_base.copy()
        test_with_margin = test_base.copy()
        val_with_margin[:, 9] = val_margins[name]
        test_with_margin[:, 9] = margins[name]
        results[name], score_arrays[name] = evaluate_features(
            val_with_margin,
            val_correct,
            test_with_margin,
            test_correct,
            folds,
        )
        results[name]["margin_auroc"] = binary_auroc(
            margins[name],
            test_correct,
        )

    paired_tests = {}
    for comparison in ("without_prototype", "clean_oracle", "pretrained", "shuffled"):
        paired_tests[f"noisy_minus_{comparison}"] = (
            paired_bootstrap_aurc_difference(
                score_arrays["noisy"],
                score_arrays[comparison],
                test_correct,
                rng,
                args.bootstrap,
            )
        )

    output = {
        "run_dir": str(args.run_dir),
        "prototype_variants": results,
        "paired_bootstrap": paired_tests,
        "interpretation": {
            "positive_aurc_difference": "noisy prototypes improve over comparison",
            "clean_oracle": "diagnostic upper bound only; clean train labels are unavailable",
        },
    }
    np.savez_compressed(
        args.run_dir / "prototype_source_scores.npz",
        correct=test_correct,
        **{
            f"{name}_val": scores
            for name, scores in score_arrays.items()
        },
        **{
            f"{name}_test": scores
            for name, scores in score_arrays.items()
        },
    )
    write_json(args.run_dir / "prototype_source_ablation.json", output)
    print(output)


if __name__ == "__main__":
    main()
