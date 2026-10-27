from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = (
    "elr_resnet50_cub_semantic40_seed0",
    "elr_resnet50_cub_semantic40_seed1",
    "elr_resnet50_cub_semantic40_seed2",
    "elr_resnet50_aircraft_semantic40_seed0",
    "elr_resnet50_flowers102_semantic375_seed0",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run prototype-source controls on representative noisy runs."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=list(DEFAULT_RUNS),
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = PROJECT_ROOT / "runs"
    log_root = runs_root / "e0_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    failures = []
    completed = []

    for run_name in args.runs:
        run_dir = runs_root / run_name
        output_path = run_dir / "prototype_source_ablation.json"
        if output_path.is_file() and not args.overwrite:
            print(f"skip {run_name}: result exists", flush=True)
            completed.append(run_name)
            continue
        if not (run_dir / "best.pt").is_file():
            print(f"skip {run_name}: best.pt missing", flush=True)
            failures.append(run_name)
            continue

        log_path = log_root / f"prototype_source_{run_name}.log"
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {run_name}\n")
            log.flush()
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "prototype_source_ablation.py"),
                    "--run-dir",
                    str(run_dir),
                    "--bootstrap",
                    str(args.bootstrap),
                ],
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            failures.append(run_name)
        else:
            completed.append(run_name)

    summary = []
    for run_name in completed:
        path = runs_root / run_name / "prototype_source_ablation.json"
        if not path.is_file():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        summary.append(
            {
                "run_name": run_name,
                "variants": result["prototype_variants"],
                "paired_bootstrap": result["paired_bootstrap"],
            }
        )
    output_path = PROJECT_ROOT.parent.parent / "outputs" / "prototype_source_summary.json"
    output_path.write_text(
        json.dumps(
            {
                "requested_runs": list(args.runs),
                "completed_runs": completed,
                "failures": failures,
                "results": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Completed {len(completed)} runs; failures={failures}", flush=True)
    print(f"Summary: {output_path}", flush=True)


if __name__ == "__main__":
    main()
