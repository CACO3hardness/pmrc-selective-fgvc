from __future__ import annotations

import unittest

import numpy as np

from scripts.prototype_source_ablation import (
    binary_auroc,
    paired_bootstrap_aurc_difference,
)


class PrototypeSourceAblationTests(unittest.TestCase):
    def test_binary_auroc_is_bounded(self) -> None:
        scores = np.array([4.0, 3.0, 2.0, 1.0])
        correct = np.array([True, True, False, False])
        self.assertEqual(binary_auroc(scores, correct), 1.0)

    def test_bootstrap_reports_reference_better_direction(self) -> None:
        correct = np.array([True] * 200 + [False] * 200)
        reference = np.where(correct, 1.0, 0.0)
        comparison = -reference
        result = paired_bootstrap_aurc_difference(
            reference,
            comparison,
            correct,
            np.random.default_rng(3),
            200,
        )
        self.assertLess(result["difference_reference_minus_comparison"], 0)
        self.assertGreater(result["reference_better_in_valid_bootstrap"], 0.9)


if __name__ == "__main__":
    unittest.main()
