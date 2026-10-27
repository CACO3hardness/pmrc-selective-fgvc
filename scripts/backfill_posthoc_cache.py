from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CACHE = (
    "posthoc_val_features.npy",
    "posthoc_val_labels.npy",
    "posthoc_val_predictions.npy",
    "posthoc_val_scores.npy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute post-hoc outputs for runs with incomplete caches."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "runs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = []
    for posthoc_path in sorted(args.runs_root.glob("*/posthoc_reliability.json")):
        run_dir = posthoc_path.parent
        if not (run_dir / "config.json").is_file():
            continue
        if not (run_dir / "best.pt").is_file():
            continue
        if not all((run_dir / name).is_file() for name in REQUIRED_CACHE):
            candidates.append(run_dir)

    print(f"Backfilling post-hoc caches for {len(candidates)} runs.", flush=True)
    failures = []
    for run_dir in candidates:
        log_path = args.runs_root / "e0_logs" / f"posthoc_backfill_{run_dir.name}.log"
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {run_dir.name}\n")
            log.flush()
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "posthoc_reliability.py"),
                    "--run-dir",
                    str(run_dir),
                ],
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            failures.append(run_dir.name)
    print(f"Backfill complete. Failures: {failures}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
