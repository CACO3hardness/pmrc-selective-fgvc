from __future__ import annotations

import contextlib
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, TextIO

import yaml

from rgel.engine import train
from rgel.utils import read_json, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
STATUS_PATH = RUNS_ROOT / "e0_suite_status.json"
LOCK_PATH = RUNS_ROOT / "e0_suite.lock"
LOG_ROOT = RUNS_ROOT / "e0_logs"

TASKS = [
    ("ce_clean", PROJECT_ROOT / "configs" / "cub_e0_clean.yaml"),
    ("ce_semantic40", PROJECT_ROOT / "configs" / "cub_e0_semantic40.yaml"),
    ("elr_semantic40", PROJECT_ROOT / "configs" / "cub_elr_semantic40.yaml"),
    ("ce_symmetric40", PROJECT_ROOT / "configs" / "cub_e0_symmetric40.yaml"),
    ("elr_clean", PROJECT_ROOT / "configs" / "cub_elr_clean.yaml"),
]


class SuiteLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: TextIO | None = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            if os.name == "nt":
                import msvcrt

                self.handle.seek(0)
                self.handle.write("0")
                self.handle.flush()
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            self.handle = None
            return False
        return True

    def release(self) -> None:
        if self.handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_status() -> dict[str, Any]:
    if STATUS_PATH.is_file():
        try:
            return read_json(STATUS_PATH)
        except (OSError, ValueError):
            pass
    return {
        "suite": "e0",
        "status": "pending",
        "tasks": {},
        "history": [],
    }


def save_status(status: dict[str, Any]) -> None:
    status["updated_at"] = now()
    write_json(STATUS_PATH, status)


def task_result_path(config: dict[str, Any]) -> Path:
    return RUNS_ROOT / str(config["run_name"]) / "test_metrics.json"


def task_checkpoint_path(config: dict[str, Any]) -> Path:
    return RUNS_ROOT / str(config["run_name"]) / "last.pt"


def run_task(
    task_name: str,
    config_path: Path,
    status: dict[str, Any],
) -> None:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    run_name = str(config["run_name"])
    task = status["tasks"].setdefault(task_name, {})
    task.update(
        {
            "config": str(config_path.relative_to(PROJECT_ROOT)),
            "run_name": run_name,
            "last_started_at": now(),
        }
    )

    if task_result_path(config).is_file():
        task["status"] = "complete"
        task["last_finished_at"] = now()
        save_status(status)
        return

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{run_name}.log"
    task["status"] = "resuming" if task_checkpoint_path(config).is_file() else "running"
    task["log"] = str(log_path.relative_to(PROJECT_ROOT))
    save_status(status)
    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"\n===== {now()} start {run_name} =====\n")
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                result = train(config)
        task["status"] = "complete"
        task["last_finished_at"] = now()
        task["test_accuracy"] = result["test"]["accuracy"]
        task["test_aurc"] = result["test"]["aurc"]
        task["test_ece_15"] = result["test"]["ece_15"]
        status["history"].append(
            {
                "task": task_name,
                "finished_at": now(),
                "test_accuracy": result["test"]["accuracy"],
                "test_aurc": result["test"]["aurc"],
                "test_ece_15": result["test"]["ece_15"],
            }
        )
        save_status(status)
    except Exception:
        task["status"] = "failed"
        task["last_finished_at"] = now()
        with (LOG_ROOT / "suite_errors.log").open("a", encoding="utf-8") as error_log:
            error_log.write(f"\n===== {now()} {run_name} =====\n")
            traceback.print_exc(file=error_log)
        save_status(status)
        raise


def main() -> int:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    lock = SuiteLock(LOCK_PATH)
    if not lock.acquire():
        print("E0 suite is already running.")
        return 0

    status = load_status()
    status.update(
        {
            "status": "running",
            "pid": os.getpid(),
            "started_at": now(),
            "project_root": str(PROJECT_ROOT),
        }
    )
    save_status(status)
    try:
        for task_name, config_path in TASKS:
            run_task(task_name, config_path, status)
        status["status"] = "complete"
        status["finished_at"] = now()
        save_status(status)
        print("E0 suite complete.")
        return 0
    except Exception:
        status["status"] = "failed"
        status["finished_at"] = now()
        save_status(status)
        return 1
    finally:
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
