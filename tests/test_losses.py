from __future__ import annotations

import unittest

import torch

from rgel.losses import ELRLoss, GCELoss


class LossTests(unittest.TestCase):
    def test_elr_loss_is_finite_and_updates_targets(self) -> None:
        criterion = ELRLoss(
            num_examples=4,
            num_classes=3,
            beta=0.9,
            regularization=3.0,
        )
        indices = torch.tensor([0, 1])
        logits = torch.tensor(
            [[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]],
            requires_grad=True,
        )
        labels = torch.tensor([0, 1])
        loss = criterion(indices, logits, labels)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(criterion.target[:2].sum()), 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_gce_loss_is_finite_and_differentiable(self) -> None:
        criterion = GCELoss(q=0.7)
        logits = torch.tensor(
            [[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]],
            requires_grad=True,
        )
        labels = torch.tensor([0, 1])
        loss = criterion(logits, labels)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()
