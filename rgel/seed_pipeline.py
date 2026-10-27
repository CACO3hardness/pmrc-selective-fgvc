from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import yaml

from rgel.engine import train
from rgel.suite import SuiteLock
from rgel.utils import read_json, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one ELR seed and PMRC analysis.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()
    config_path = args.config.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    run_dir = RUNS_ROOT / str(config["run_name"])
    tag = args.tag or f"seed{args.seed}"
    status_path = RUNS_ROOT / f"{tag}_pipeline_status.json"
    log_path = RUNS_ROOT / "e0_logs" / f"{tag}_pipeline.log"
    lock = SuiteLock(RUNS_ROOT / f"{tag}.lock")
    if not lock.acquire():
        print(f"Seed {args.seed} pipeline is already running.")
        return 0

    status = {
        "seed": args.seed,
        "status": "running",
        "pid": os.getpid(),
        "run_name": config["run_name"],
        "started_at": now(),
    }
    write_json(status_path, status)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not (run_dir / "test_metrics.json").is_file():
            status["stage"] = "training"
            write_json(status_path, status)
            with log_path.open("a", encoding="utf-8", buffering=1) as log:
                log.write(f"\n===== {now()} train seed {args.seed} =====\n")
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    train(config)

        if not (run_dir / "posthoc_reliability.json").is_file():
            status["stage"] = "posthoc_reliability"
            write_json(status_path, status)
            with log_path.open("a", encoding="utf-8", buffering=1) as log:
                log.write(f"\n===== {now()} PMRC seed {args.seed} =====\n")
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
                raise RuntimeError(f"PMRC analysis failed with code {result.returncode}")

        posthoc = read_json(run_dir / "posthoc_reliability.json")
        status.update(
            {
                "status": "complete",
                "stage": "complete",
                "finished_at": now(),
                "msp_aurc": posthoc["test_results"]["msp"]["aurc"],
                "pmrc_aurc": posthoc["test_results"]["learned_reliability"]["aurc"],
            }
        )
        write_json(status_path, status)
        return 0
    except Exception:
        status["status"] = "failed"
        status["finished_at"] = now()
        write_json(status_path, status)
        with (RUNS_ROOT / "e0_logs" / "seed_errors.log").open("a", encoding="utf-8") as handle:
            handle.write(f"\n===== {now()} seed {args.seed} =====\n")
            traceback.print_exc(file=handle)
        return 1
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
