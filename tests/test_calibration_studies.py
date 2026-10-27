from __future__ import annotations

import unittest

import numpy as np

from scripts.calibration_robustness import corrupt_labels
from scripts.decision_cost_analysis import choose_cost_threshold
from scripts.risk_control_baseline import ltt_threshold


class CalibrationStudyTests(unittest.TestCase):
    def test_corrupt_labels_flips_exact_count_without_identity(self) -> None:
        labels = np.arange(100) % 5
        corrupted = corrupt_labels(
            labels,
            rate=0.1,
            num_classes=5,
            rng=np.random.default_rng(3),
        )
        changed = corrupted != labels
        self.assertEqual(int(changed.sum()), 10)
        self.assertTrue(np.all(corrupted[changed] != labels[changed]))

    def test_ltt_threshold_can_certify_separable_scores(self) -> None:
        scores = np.linspace(1.0, 0.0, 400)
        correct = np.arange(400) < 200
        result = ltt_threshold(scores, correct, 0.1, 0.1)
        self.assertTrue(np.isfinite(result["threshold"]))
        self.assertLessEqual(result["calibration_upper_bound"], 0.1)

    def test_decision_cost_prefers_selective_rule_when_rejection_is_cheap(self) -> None:
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        correct = np.array([True, True, False, False, False])
        threshold, coverage, risk, cost = choose_cost_threshold(
            scores,
            correct,
            rejection_cost=0.1,
        )
        self.assertGreaterEqual(threshold, 0.8)
        self.assertEqual(coverage, 0.4)
        self.assertEqual(risk, 0.0)
        self.assertEqual(cost, 0.06)


if __name__ == "__main__":
    unittest.main()
