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
CONFIG_ROOT = RUNS_ROOT / "cifar100n_configs"
STATUS_PATH = RUNS_ROOT / "cifar100n_status.json"
LOG_ROOT = RUNS_ROOT / "e0_logs"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CIFAR-100N real-noise queue.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock = SuiteLock(RUNS_ROOT / "cifar100n.lock")
    if not lock.acquire():
        print("CIFAR-100N queue is already running.")
        return
    status = {
        "status": "running",
        "pid": os.getpid(),
        "started_at": now(),
        "tasks": {},
    }
    write_json(STATUS_PATH, status)
    try:
        for seed in args.seeds:
            name = f"cifar100n_human_seed{seed}"
            run_name = f"elr_resnet18_{name}"
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
            write_json(STATUS_PATH, status)
            if (run_dir / "posthoc_reliability.json").is_file():
                continue

            CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
            config_path = CONFIG_ROOT / f"{name}.yaml"
            config = {
                "dataset": "cifar100n",
                "root": str(PROJECT_ROOT / "data"),
                "manifest": str(
                    PROJECT_ROOT / "data" / "manifests" / f"cifar100n_seed{seed}.json"
                ),
                "noise_csv": str(
                    PROJECT_ROOT
                    / "data"
                    / "noise"
                    / f"cifar100n_human_seed{seed}.csv"
                ),
                "output_dir": str(RUNS_ROOT),
                "run_name": run_name,
                "method": "elr",
                "elr_beta": 0.9,
                "elr_lambda": 3.0,
                "model": "resnet18",
                "pretrained": True,
                "freeze_stages": 1,
                "image_size": 64,
                "batch_size": 256,
                "num_workers": 4,
                "epochs": args.epochs,
                "lr": 0.0002,
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
            write_json(STATUS_PATH, status)
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
                activity_paths=[LOG_ROOT / f"{name}_pipeline.log"],
            )
            status["tasks"][name]["status"] = "complete"
            write_json(STATUS_PATH, status)
        status["status"] = "complete"
        status["finished_at"] = now()
        write_json(STATUS_PATH, status)
    except Exception as error:
        status["status"] = "failed"
        status["error"] = repr(error)
        status["finished_at"] = now()
        write_json(STATUS_PATH, status)
        raise
    finally:
        lock.release()


if __name__ == "__main__":
    main()
