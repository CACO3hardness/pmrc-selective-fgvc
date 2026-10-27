from __future__ import annotations

import unittest

import torch

from rgel.evidential import EvidentialResNet, rgel_losses
from rgel.models import build_model, forward_features


class RGELTests(unittest.TestCase):
    def test_rgel_loss_is_finite(self) -> None:
        alpha_a = torch.tensor([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0]])
        alpha_b = torch.tensor([[1.8, 1.2, 1.0], [1.1, 1.9, 1.0]])
        labels = torch.tensor([0, 1])
        reliability = torch.tensor([0.9, 0.2])
        loss, components = rgel_losses(
            torch.tensor([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]]),
            torch.tensor([[1.8, 0.2, -1.0], [0.1, 1.9, -1.0]]),
            alpha_a,
            alpha_b,
            labels,
            reliability,
            vacuity_weight=0.1,
            consistency_weight=0.5,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("reliability_mean", components)

    def test_prototype_confidence_shape(self) -> None:
        model = EvidentialResNet(
            "resnet18",
            num_classes=3,
            pretrained=False,
            freeze_stages=0,
            prototype_momentum=0.9,
        )
        features = torch.nn.functional.normalize(torch.randn(2, 512), dim=1)
        labels = torch.tensor([0, 2])
        confidence = model.prototype_confidence(features, labels)
        self.assertEqual(tuple(confidence.shape), (2,))
        self.assertTrue(torch.isfinite(confidence).all())

    def test_convnext_feature_adapter(self) -> None:
        model = build_model(
            "convnext_tiny",
            num_classes=3,
            pretrained=False,
        )
        model.eval()
        with torch.inference_mode():
            features, logits = forward_features(
                model,
                torch.randn(1, 3, 64, 64),
            )
        self.assertEqual(tuple(features.shape), (1, 768))
        self.assertEqual(tuple(logits.shape), (1, 3))


if __name__ == "__main__":
    unittest.main()
