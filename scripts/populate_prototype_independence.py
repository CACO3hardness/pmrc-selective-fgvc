from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute prototype-independence analysis for completed runs."
    )
    parser.add_argument("--runs-root", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_root = args.runs_root / "e0_logs"
    candidates = []
    for run_dir in sorted(args.runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "posthoc_reliability.json").is_file():
            continue
        result_path = run_dir / "prototype_independence.json"
        if args.overwrite or not result_path.is_file():
            candidates.append(run_dir)

    print(f"Computing prototype independence for {len(candidates)} runs.", flush=True)
    failures = []
    for run_dir in candidates:
        log_path = log_root / f"prototype_independence_{run_dir.name}.log"
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {run_dir.name}\n")
            log.flush()
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "prototype_independence.py"),
                    "--run-dir",
                    str(run_dir),
                    "--bins",
                    "5",
                    "--bootstrap",
                    str(args.bootstrap),
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
