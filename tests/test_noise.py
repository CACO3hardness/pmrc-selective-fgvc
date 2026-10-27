from __future__ import annotations

import unittest

from rgel.data import ManifestRow
from rgel.noise import empirical_noise_rate, sample_class_noise


def rows() -> list[ManifestRow]:
    result = []
    image_id = 1
    for class_id in range(3):
        for _ in range(10):
            result.append(
                ManifestRow(
                    image_id=image_id,
                    relative_path=f"images/{image_id}.jpg",
                    class_id=class_id,
                    class_name=f"class-{class_id}",
                    split="train",
                )
            )
            image_id += 1
    return result


class NoiseTests(unittest.TestCase):
    def test_noise_rate_and_labels(self) -> None:
        neighbor_map = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
        entries = sample_class_noise(rows(), neighbor_map, None, rate=0.4, seed=7)
        self.assertEqual(len(entries), 30)
        self.assertAlmostEqual(empirical_noise_rate(entries), 0.4)
        for entry in entries:
            if entry.is_noisy:
                self.assertNotEqual(entry.noisy_label, entry.clean_label)
            else:
                self.assertEqual(entry.noisy_label, entry.clean_label)

    def test_generation_is_deterministic(self) -> None:
        neighbor_map = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
        first = sample_class_noise(rows(), neighbor_map, None, rate=0.3, seed=11)
        second = sample_class_noise(rows(), neighbor_map, None, rate=0.3, seed=11)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

