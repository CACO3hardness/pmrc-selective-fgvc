from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgel.utils import read_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate temperature-scaling and pNorm baselines for completed runs."
    )
    parser.add_argument("--runs-root", type=Path, default=PROJECT_ROOT / "runs")
    return parser.parse_args()


def needs_baseline(run_dir: Path) -> bool:
    result_path = run_dir / "selective_baselines.json"
    if not result_path.is_file():
        return True
    try:
        result = read_json(result_path)
        return result["test"]["temperature_scaling"]["aurc"] is None
    except (KeyError, TypeError, ValueError):
        return True


def main() -> None:
    args = parse_args()
    log_root = args.runs_root / "e0_logs"
    candidates = [
        path
        for path in sorted(args.runs_root.iterdir())
        if path.is_dir()
        and (path / "posthoc_reliability.json").is_file()
        and needs_baseline(path)
    ]
    print(f"Populating baselines for {len(candidates)} runs.", flush=True)
    failures = []
    for run_dir in candidates:
        log_path = log_root / f"all_baselines_{run_dir.name}.log"
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {run_dir.name}\n")
            log.flush()
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "selective_baselines.py"),
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
    print(f"Complete. Failures: {failures}", flush=True)


if __name__ == "__main__":
    main()
