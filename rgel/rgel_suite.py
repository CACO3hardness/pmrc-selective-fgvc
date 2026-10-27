from __future__ import annotations

import contextlib
import json
import os
import time
import traceback
from pathlib import Path

import yaml

from rgel.rgel_train import train_rgel
from rgel.suite import SuiteLock, load_status
from rgel.utils import read_json, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
E0_LOCK_PATH = RUNS_ROOT / "e0_suite.lock"
RGEL_STATUS_PATH = RUNS_ROOT / "rgel_pilot_status.json"
RGEL_LOG = RUNS_ROOT / "e0_logs" / "rgel_semantic40_pilot_v5.log"
RGEL_CONFIG = PROJECT_ROOT / "configs" / "cub_rgel_semantic40_pilot.yaml"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def update_status(value: dict) -> None:
    value["updated_at"] = now()
    write_json(RGEL_STATUS_PATH, value)


def main() -> int:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    with RGEL_CONFIG.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    result_path = RUNS_ROOT / str(config["run_name"]) / "test_metrics.json"
    status = {
        "suite": "rgel_pilot",
        "status": "waiting_for_e0_controls",
        "pid": os.getpid(),
        "run_name": config["run_name"],
        "waiting_since": now(),
    }
    update_status(status)

    if result_path.is_file():
        status["status"] = "complete"
        save_result = read_json(result_path)
        status["test_accuracy"] = save_result["test"]["accuracy"]
        status["test_aurc"] = save_result["test"]["aurc"]
        update_status(status)
        return 0

    lock = SuiteLock(E0_LOCK_PATH)
    while not lock.acquire():
        e0_status_path = RUNS_ROOT / "e0_suite_status.json"
        if e0_status_path.is_file():
            e0_status = read_json(e0_status_path)
            status["e0_status"] = e0_status.get("status")
            if e0_status.get("status") == "failed":
                status["status"] = "blocked_by_e0_failure"
                update_status(status)
                return 1
        update_status(status)
        time.sleep(20)

    try:
        e0_status = load_status()
        if e0_status.get("status") != "complete":
            status["status"] = "blocked_by_incomplete_e0"
            status["e0_status"] = e0_status.get("status")
            update_status(status)
            return 1

        status["status"] = "running"
        status["started_at"] = now()
        update_status(status)
        RGEL_LOG.parent.mkdir(parents=True, exist_ok=True)
        with RGEL_LOG.open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"\n===== {now()} start RGEL pilot =====\n")
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                result = train_rgel(config)
        status["status"] = "complete"
        status["finished_at"] = now()
        status["test_accuracy"] = result["test"]["accuracy"]
        status["test_aurc"] = result["test"]["aurc"]
        status["test_ece_15"] = result["test"]["ece_15"]
        update_status(status)
        return 0
    except Exception:
        status["status"] = "failed"
        status["finished_at"] = now()
        update_status(status)
        with (RUNS_ROOT / "e0_logs" / "rgel_errors.log").open("a", encoding="utf-8") as handle:
            traceback.print_exc(file=handle)
        return 1
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
