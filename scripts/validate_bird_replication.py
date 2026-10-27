from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgel.utils import read_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
CONFIG_ROOT = RUNS_ROOT / "webfg496_configs"
OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "outputs"
SEEDS = tuple(range(6))
IGNORED_CONFIG_FIELDS = {"seed", "run_name"}


def normalized_config(config: dict) -> dict:
    return {
        key: value
        for key, value in config.items()
        if key not in IGNORED_CONFIG_FIELDS
    }


def main() -> None:
    records = []
    reference_config = None
    reference_labels = None
    failures = []
    for seed in SEEDS:
        config_path = CONFIG_ROOT / f"webfg_bird_seed{seed}.yaml"
        run_dir = RUNS_ROOT / f"elr_resnet50_webfg_bird_seed{seed}"
        if not config_path.is_file():
            failures.append(f"missing config for seed {seed}")
            continue
        import yaml

        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        normalized = normalized_config(config)
        if reference_config is None:
            reference_config = normalized
            config_equal = True
        else:
            config_equal = normalized == reference_config
            if not config_equal:
                failures.append(f"config mismatch for seed {seed}")

        labels_equal = None
        label_path = run_dir / "test_labels.npy"
        if label_path.is_file():
            labels = np.load(label_path)
            if reference_labels is None:
                reference_labels = labels
                labels_equal = True
            else:
                labels_equal = bool(np.array_equal(labels, reference_labels))
                if not labels_equal:
                    failures.append(f"test-label mismatch for seed {seed}")
        records.append(
            {
                "seed": seed,
                "config_exists": True,
                "config_equal_except_seed_and_run_name": config_equal,
                "test_labels_available": labels_equal is not None,
                "test_labels_equal": labels_equal,
            }
        )

    report = {
        "config_fields_compared": sorted(reference_config or {}),
        "ignored_fields": sorted(IGNORED_CONFIG_FIELDS),
        "runs": records,
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }
    output_path = OUTPUT_ROOT / "webfg_bird_replication_integrity.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
