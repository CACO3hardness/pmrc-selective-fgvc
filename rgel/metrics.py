from __future__ import annotations

from typing import Any

import numpy as np


def macro_f1_score(predictions: np.ndarray, labels: np.ndarray) -> float:
    predictions = np.asarray(predictions, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    classes = np.unique(np.concatenate((predictions, labels)))
    scores: list[float] = []
    for class_id in classes:
        true_positive = np.sum((predictions == class_id) & (labels == class_id))
        false_positive = np.sum((predictions == class_id) & (labels != class_id))
        false_negative = np.sum((predictions != class_id) & (labels == class_id))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores)) if scores else 0.0


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be [N, C]")
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities have different lengths")

    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1.0:
            in_bin = (confidence >= lower) & (confidence <= upper)
        else:
            in_bin = (confidence >= lower) & (confidence < upper)
        fraction = in_bin.mean()
        if fraction == 0.0:
            continue
        error += fraction * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


def risk_coverage_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    order = np.argsort(-confidence, kind="stable")
    ordered_correct = correct[order]
    risks = np.cumsum(~ordered_correct) / np.arange(1, len(labels) + 1)
    coverage = np.arange(1, len(labels) + 1) / len(labels)
    return coverage, risks


def aurc(probabilities: np.ndarray, labels: np.ndarray) -> float:
    coverage, risks = risk_coverage_curve(probabilities, labels)
    coverage = np.concatenate(([0.0], coverage))
    risks = np.concatenate(([0.0], risks))
    return float(np.trapezoid(risks, coverage))


def evaluation_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    coverage_levels: tuple[float, ...] = (0.7, 0.8, 0.9),
) -> dict[str, Any]:
    logits = np.asarray(logits, dtype=np.float64)
    probabilities = softmax(logits)
    return metrics_from_probabilities(probabilities, labels, coverage_levels)


def metrics_from_probabilities(
    probabilities: np.ndarray,
    labels: np.ndarray,
    coverage_levels: tuple[float, ...] = (0.7, 0.8, 0.9),
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    selected_probability = probabilities[np.arange(len(labels)), labels]
    nll = float(-np.log(np.clip(selected_probability, 1e-12, 1.0)).mean())
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    brier = float(((probabilities - one_hot) ** 2).sum(axis=1).mean())

    coverage, risks = risk_coverage_curve(probabilities, labels)
    selective: dict[str, dict[str, float]] = {}
    for level in coverage_levels:
        index = max(0, int(np.ceil(level * len(labels))) - 1)
        selective[str(level)] = {
            "risk": float(risks[index]),
            "accuracy": float(1.0 - risks[index]),
        }

    return {
        "n": int(len(labels)),
        "accuracy": float((predictions == labels).mean()),
        "macro_f1": macro_f1_score(predictions, labels),
        "nll": nll,
        "brier": brier,
        "ece_15": expected_calibration_error(probabilities, labels, n_bins=15),
        "aurc": aurc(probabilities, labels),
        "high_confidence_error_rate": float(
            ((predictions != labels) & (confidence >= 0.8)).mean()
        ),
        "selective": selective,
    }
