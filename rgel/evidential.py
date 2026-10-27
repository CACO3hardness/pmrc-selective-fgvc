from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models

from rgel.metrics import metrics_from_probabilities, softmax


def _freeze_stages(model: nn.Module, freeze_stages: int) -> None:
    if freeze_stages < 0 or freeze_stages > 4:
        raise ValueError("freeze_stages must be in [0, 4]")
    if freeze_stages >= 1:
        for module in (model.conv1, model.bn1, model.layer1):
            for parameter in module.parameters():
                parameter.requires_grad = False
    if freeze_stages >= 2:
        for module in model.layer2:
            for parameter in module.parameters():
                parameter.requires_grad = False
    if freeze_stages >= 3:
        for module in model.layer3:
            for parameter in module.parameters():
                parameter.requires_grad = False
    if freeze_stages >= 4:
        for module in model.layer4:
            for parameter in module.parameters():
                parameter.requires_grad = False


class EvidentialResNet(nn.Module):
    def __init__(
        self,
        name: str,
        num_classes: int,
        pretrained: bool,
        freeze_stages: int,
        prototype_momentum: float,
    ) -> None:
        super().__init__()
        if name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            feature_dim = 512
        elif name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported model: {name}")
        _freeze_stages(model, freeze_stages)
        model.fc = nn.Identity()
        self.encoder = model
        self.head = nn.Linear(feature_dim, num_classes)
        self.register_buffer("prototypes", torch.zeros(num_classes, feature_dim))
        self.register_buffer("prototype_initialized", torch.zeros(num_classes, dtype=torch.bool))
        self.prototype_momentum = prototype_momentum

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_features = self.encoder(images)
        features = F.normalize(raw_features, dim=1)
        logits = self.head(raw_features)
        alpha = torch.exp(logits.clamp(min=-20.0, max=20.0))
        return features, logits, alpha

    @torch.no_grad()
    def prototype_confidence(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        normalized_prototypes = F.normalize(self.prototypes, dim=1)
        similarity = features @ normalized_prototypes.T
        observed = similarity.gather(1, labels[:, None]).squeeze(1)
        masked = similarity.clone()
        masked.scatter_(1, labels[:, None], -float("inf"))
        uninitialized = ~self.prototype_initialized
        if uninitialized.any():
            masked[:, uninitialized] = -float("inf")
        nearest_other = masked.max(dim=1).values
        valid = self.prototype_initialized[labels] & torch.isfinite(nearest_other)
        confidence = torch.sigmoid(observed - nearest_other)
        return torch.where(valid, confidence, torch.full_like(confidence, 0.5))

    @torch.no_grad()
    def update_prototypes(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        reliable: torch.Tensor,
    ) -> None:
        for class_id in labels.unique():
            class_mask = (labels == class_id) & reliable
            if not class_mask.any():
                continue
            mean_feature = F.normalize(features[class_mask].mean(dim=0), dim=0)
            if self.prototype_initialized[class_id]:
                prototype = (
                    self.prototype_momentum * self.prototypes[class_id]
                    + (1.0 - self.prototype_momentum) * mean_feature
                )
                self.prototypes[class_id] = F.normalize(prototype, dim=0)
            else:
                self.prototypes[class_id] = mean_feature
                self.prototype_initialized[class_id] = True


def rgel_losses(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    alpha_a: torch.Tensor,
    alpha_b: torch.Tensor,
    labels: torch.Tensor,
    reliability: torch.Tensor,
    vacuity_weight: float,
    consistency_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    probability_a = F.softmax(logits_a, dim=1)
    probability_b = F.softmax(logits_b, dim=1)
    cross_entropy_a = F.cross_entropy(logits_a, labels, reduction="none")
    cross_entropy_b = F.cross_entropy(logits_b, labels, reduction="none")
    supervised = (0.5 * (cross_entropy_a + cross_entropy_b) * reliability).mean()

    consistency = 0.5 * (
        F.kl_div(torch.log(probability_a.clamp_min(1e-8)), probability_b, reduction="batchmean")
        + F.kl_div(torch.log(probability_b.clamp_min(1e-8)), probability_a, reduction="batchmean")
    )
    num_classes = alpha_a.shape[1]
    vacuity_a = torch.logsumexp(logits_a, dim=1) - math.log(num_classes)
    vacuity_b = torch.logsumexp(logits_b, dim=1) - math.log(num_classes)
    vacuity = (0.5 * (vacuity_a + vacuity_b) * (1.0 - reliability)).mean()
    total = supervised + consistency_weight * consistency + vacuity_weight * vacuity
    return total, {
        "supervised": float(supervised.detach()),
        "consistency": float(consistency.detach()),
        "vacuity": float(vacuity.detach()),
        "reliability_mean": float(reliability.mean().detach()),
    }


@torch.inference_mode()
def evaluate_evidential(
    model: EvidentialResNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    logits_values: list[np.ndarray] = []
    alpha_values: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["clean_target"].to(device, non_blocking=True)
        _, logits, alpha = model(images)
        logits_np = logits.float().cpu().numpy()
        probabilities.append(softmax(logits_np))
        logits_values.append(logits_np)
        alpha_values.append(alpha.float().cpu().numpy())
        labels.append(targets.cpu().numpy())
    probability_np = np.concatenate(probabilities)
    label_np = np.concatenate(labels)
    metrics = metrics_from_probabilities(probability_np, label_np)
    nll = -np.log(np.clip(probability_np[np.arange(len(label_np)), label_np], 1e-12, 1.0)).mean()
    metrics["nll"] = float(nll)
    metrics["loss"] = float(nll)
    return (
        metrics,
        probability_np,
        label_np,
        np.concatenate(logits_values),
        np.concatenate(alpha_values),
    )
