from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgel.data import load_manifest, validate_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic known/unknown CUB class split."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "cub_seed0.json",
    )
    parser.add_argument("--known-classes", type=int, default=160)
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "cub_openset160_seed0.json",
    )
    parser.add_argument(
        "--unknown-out",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "cub_openset160_unknown.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_manifest(args.source)
    class_ids = sorted({row.class_id for row in rows})
    if args.known_classes >= len(class_ids):
        raise ValueError("known-classes must be smaller than total classes")
    known_ids = set(class_ids[: args.known_classes])
    known_rows = [row for row in rows if row.class_id in known_ids]
    unknown_test_rows = [
        row
        for row in rows
        if row.class_id not in known_ids and row.split == "test"
    ]
    validate_manifest(known_rows)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(
        json.dumps(
            {
                "dataset": "CUB_200_2011_openset_known",
                "source_manifest": str(args.source.resolve()),
                "known_classes": args.known_classes,
                "unknown_classes": len(class_ids) - args.known_classes,
                "records": [row.__dict__ for row in known_rows],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    args.unknown_out.write_text(
        json.dumps(
            {
                "dataset": "CUB_200_2011_openset_unknown_test",
                "source_manifest": str(args.source.resolve()),
                "records": [row.__dict__ for row in unknown_test_rows],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"known classes={args.known_classes}, known rows={len(known_rows)}, "
        f"unknown test rows={len(unknown_test_rows)}"
    )


if __name__ == "__main__":
    main()
