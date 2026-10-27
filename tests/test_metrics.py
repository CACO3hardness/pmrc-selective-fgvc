from __future__ import annotations

import unittest

import numpy as np

from rgel.metrics import aurc, evaluation_metrics, expected_calibration_error


class MetricTests(unittest.TestCase):
    def test_perfect_predictions_have_zero_risk(self) -> None:
        logits = np.array([[10.0, 0.0], [0.0, 10.0]])
        labels = np.array([0, 1])
        metrics = evaluation_metrics(logits, labels)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["aurc"], 0.0)
        self.assertLess(metrics["ece_15"], 1e-3)

    def test_ece_is_bounded(self) -> None:
        probabilities = np.array([[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]])
        labels = np.array([0, 0, 1])
        score = expected_calibration_error(probabilities, labels)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_aurc_penalizes_confident_errors(self) -> None:
        confident_wrong = np.array([[0.9, 0.1], [0.8, 0.2]])
        less_confident_wrong = np.array([[0.55, 0.45], [0.6, 0.4]])
        labels = np.array([1, 0])
        self.assertGreater(
            aurc(confident_wrong, labels),
            aurc(less_confident_wrong, labels),
        )


if __name__ == "__main__":
    unittest.main()
