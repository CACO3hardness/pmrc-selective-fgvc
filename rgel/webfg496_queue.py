from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import yaml

from rgel.phase2_queue import run_logged
from rgel.suite import SuiteLock
from rgel.utils import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
CONFIG_ROOT = RUNS_ROOT / "webfg496_configs"
LOG_ROOT = RUNS_ROOT / "e0_logs"
SUBSETS = ("bird", "aircraft", "car")


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WebFG-496 real-noise queue.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--subsets",
        nargs="+",
        choices=SUBSETS,
        default=list(SUBSETS),
    )
    parser.add_argument(
        "--status-name",
        default="webfg496_status",
        help="Status JSON stem under runs/.",
    )
    parser.add_argument("--epochs", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock = SuiteLock(RUNS_ROOT / "webfg496.lock")
    if not lock.acquire():
        print("WebFG-496 queue is already running.")
        return
    status_path = RUNS_ROOT / f"{args.status_name}.json"
    status = {
        "status": "running",
        "pid": os.getpid(),
        "started_at": now(),
        "tasks": {},
    }
    write_json(status_path, status)
    try:
        for seed in args.seeds:
            for subset in args.subsets:
                name = f"webfg_{subset}_seed{seed}"
                run_name = f"elr_resnet50_{name}"
                run_dir = RUNS_ROOT / run_name
                status["tasks"][name] = {
                    "status": (
                        "complete"
                        if (run_dir / "posthoc_reliability.json").is_file()
                        else "pending"
                    ),
                    "run_name": run_name,
                }
                status["current"] = name
                write_json(status_path, status)
                if (run_dir / "posthoc_reliability.json").is_file():
                    continue

                CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
                config_path = CONFIG_ROOT / f"{name}.yaml"
                config = {
                    "dataset": f"webfg_{subset}",
                    "root": str(PROJECT_ROOT / "data" / "WebFG_496" / "extracted"),
                    "manifest": str(
                        PROJECT_ROOT / "data" / "manifests" / f"webfg_{subset}.json"
                    ),
                    "noise_csv": None,
                    "output_dir": str(RUNS_ROOT),
                    "run_name": run_name,
                    "method": "elr",
                    "elr_beta": 0.9,
                    "elr_lambda": 3.0,
                    "model": "resnet50",
                    "pretrained": True,
                    "freeze_stages": 1,
                    "image_size": 224,
                    "batch_size": 64,
                    "num_workers": 4,
                    "epochs": args.epochs,
                    "lr": 0.0001,
                    "weight_decay": 0.0001,
                    "label_smoothing": 0.0,
                    "grad_clip": 1.0,
                    "amp": True,
                    "amp_dtype": "bf16",
                    "seed": int(seed),
                    "resume": True,
                }
                with config_path.open("w", encoding="utf-8") as handle:
                    yaml.safe_dump(config, handle, sort_keys=False)

                status["tasks"][name]["status"] = "running"
                write_json(status_path, status)
                run_logged(
                    [
                        sys.executable,
                        "-m",
                        "rgel.seed_pipeline",
                        "--config",
                        str(config_path),
                        "--seed",
                        str(seed),
                        "--tag",
                        name,
                    ],
                    LOG_ROOT / f"{name}_queue.log",
                    # The machine may be suspended while this queue is active.
                    # Wall-clock silence then does not distinguish a sleeping
                    # laptop from a stalled worker, so WebFG uses status/log
                    # inspection rather than an automatic kill threshold.
                    stall_seconds=None,
                    activity_paths=[LOG_ROOT / f"{name}_pipeline.log"],
                )
                status["tasks"][name]["status"] = "complete"
                write_json(status_path, status)
        status["status"] = "complete"
        status["finished_at"] = now()
        write_json(status_path, status)
    except Exception as error:
        status["status"] = "failed"
        status["error"] = repr(error)
        status["finished_at"] = now()
        write_json(status_path, status)
        raise
    finally:
        lock.release()


if __name__ == "__main__":
    main()
