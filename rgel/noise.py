from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from rgel.data import ManifestRow


@dataclass(frozen=True)
class NoiseEntry:
    image_id: int
    clean_label: int
    noisy_label: int
    is_noisy: bool


def sample_class_noise(
    rows: Sequence[ManifestRow],
    neighbor_map: Mapping[int, Sequence[int]],
    similar_map: Mapping[int, Sequence[float]] | None,
    rate: float,
    seed: int,
) -> list[NoiseEntry]:
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be in [0, 1]")

    by_class: dict[int, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        by_class[row.class_id].append(row)

    rng = random.Random(seed)
    entries: list[NoiseEntry] = []
    for class_id in sorted(by_class):
        class_rows = list(by_class[class_id])
        n_noise = int(round(rate * len(class_rows)))
        if n_noise >= len(class_rows):
            n_noise = max(0, len(class_rows) - 1)

        chosen = set(rng.sample(range(len(class_rows)), n_noise))
        neighbors = [int(value) for value in neighbor_map.get(class_id, []) if int(value) != class_id]
        if n_noise and not neighbors:
            raise ValueError(f"No valid neighbors for class {class_id}")

        if similar_map and neighbors:
            raw_weights = [max(float(similar_map[class_id][i]), 1e-6) for i in range(len(neighbors))]
        else:
            raw_weights = [1.0] * len(neighbors)

        for index, row in enumerate(class_rows):
            noisy_label = row.class_id
            if index in chosen:
                noisy_label = rng.choices(neighbors, weights=raw_weights, k=1)[0]
            entries.append(
                NoiseEntry(
                    image_id=row.image_id,
                    clean_label=row.class_id,
                    noisy_label=int(noisy_label),
                    is_noisy=index in chosen,
                )
            )

    entries.sort(key=lambda item: item.image_id)
    return entries


def write_noise_csv(path: str | Path, entries: Sequence[NoiseEntry]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "clean_label", "noisy_label", "is_noisy"],
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "image_id": entry.image_id,
                    "clean_label": entry.clean_label,
                    "noisy_label": entry.noisy_label,
                    "is_noisy": int(entry.is_noisy),
                }
            )


def empirical_noise_rate(entries: Sequence[NoiseEntry]) -> float:
    if not entries:
        return 0.0
    return sum(entry.is_noisy for entry in entries) / len(entries)

