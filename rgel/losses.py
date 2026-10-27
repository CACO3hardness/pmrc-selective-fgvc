from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ELRLoss(nn.Module):
    """Early-learning regularization loss.

    This is a compact implementation of:
    Liu et al., "Early-Learning Regularization Prevents Memorization of Noisy
    Labels", NeurIPS 2020. The training loader must provide a stable sample
    index in [0, num_examples).
    """

    def __init__(
        self,
        num_examples: int,
        num_classes: int,
        beta: float = 0.9,
        regularization: float = 3.0,
        label_smoothing: float = 0.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if num_examples < 1:
            raise ValueError("num_examples must be positive")
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1)")
        self.beta = beta
        self.regularization = regularization
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        target_device = device or torch.device("cpu")
        self.register_buffer(
            "target",
            torch.zeros(num_examples, num_classes, device=target_device),
        )

    def forward(
        self,
        indices: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = F.softmax(logits, dim=1)
        detached = probabilities.detach()
        normalized = detached / detached.sum(dim=1, keepdim=True)
        with torch.no_grad():
            self.target[indices] = (
                self.beta * self.target[indices]
                + (1.0 - self.beta) * normalized
            )

        cross_entropy = self.cross_entropy(logits, labels)
        agreement = (self.target[indices] * probabilities).sum(dim=1)
        regularizer = torch.log(torch.clamp(1.0 - agreement, min=1e-4)).mean()
        return cross_entropy + self.regularization * regularizer


class GCELoss(nn.Module):
    """Generalized cross entropy for noisy-label classification."""

    def __init__(self, q: float = 0.7) -> None:
        super().__init__()
        if not 0.0 < q <= 1.0:
            raise ValueError("q must be in (0, 1]")
        self.q = q

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probabilities = F.softmax(logits, dim=1)
        selected = probabilities.gather(1, labels[:, None]).squeeze(1)
        return ((1.0 - selected.clamp_min(1e-8).pow(self.q)) / self.q).mean()
