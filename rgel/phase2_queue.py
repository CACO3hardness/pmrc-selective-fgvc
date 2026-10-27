from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from rgel.suite import SuiteLock
from rgel.utils import read_json, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
GENERATED_CONFIGS = RUNS_ROOT / "phase2_configs"
QUEUE_CONFIG = PROJECT_ROOT / "configs" / "phase2_queue.yaml"
STATUS_PATH = RUNS_ROOT / "phase2_status.json"
LOG_ROOT = RUNS_ROOT / "e0_logs"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def run_logged(
    command: list[str],
    log_path: Path,
    stall_seconds: int | None = 900,
    max_attempts: int = 3,
    activity_paths: list[Path] | None = None,
    poll_seconds: float = 5.0,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    activity_paths = activity_paths or []
    for attempt in range(1, max_attempts + 1):
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n===== {now()} attempt {attempt}: {' '.join(command)} =====\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            stalled = False
            while process.poll() is None:
                time.sleep(poll_seconds)
                last_write = max(
                    path.stat().st_mtime
                    for path in [log_path, *activity_paths]
                    if path.is_file()
                )
                if (
                    stall_seconds is not None
                    and time.time() - last_write > stall_seconds
                ):
                    stalled = True
                    log.write(
                        f"\n===== {now()} watchdog: no output for "
                        f"{stall_seconds}s, terminating pid {process.pid} =====\n"
                    )
                    log.flush()
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=20)
                    break
            return_code = process.wait()
        if return_code == 0:
            return
        if attempt < max_attempts:
            with log_path.open("a", encoding="utf-8") as log:
                reason = "watchdog stall" if stalled else f"exit code {return_code}"
                log.write(f"\n===== {now()} retrying after {reason} =====\n")
            time.sleep(10)
            continue
        raise RuntimeError(
            f"Command failed after {max_attempts} attempts "
            f"(last code {return_code}, stalled={stalled}): {command}"
        )


def dataset_settings(dataset: str, seed: int) -> dict:
    if dataset == "cub":
        manifest = PROJECT_ROOT / "data" / "manifests" / f"cub_seed{seed}.json"
        root = PROJECT_ROOT / "data" / "CUB_200_2011_256"
        val_fraction = 0.1
        epochs = 20
        prepare_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "prepare_cub.py"),
            "--root",
            str(PROJECT_ROOT / "data" / "CUB_200_2011"),
            "--out",
            str(manifest),
            "--val-fraction",
            str(val_fraction),
            "--seed",
            str(seed),
        ]
    elif dataset == "aircraft":
        manifest = PROJECT_ROOT / "data" / "manifests" / f"aircraft_seed{seed}.json"
        root = PROJECT_ROOT / "data" / "FGVC_Aircraft_256"
        val_fraction = 0.1
        epochs = 20
        prepare_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "prepare_torchvision_fg.py"),
            "--dataset",
            "aircraft",
            "--root",
            str(PROJECT_ROOT / "data"),
            "--out",
            str(manifest),
            "--val-fraction",
            str(val_fraction),
            "--seed",
            str(seed),
        ]
    elif dataset == "flowers102":
        manifest = PROJECT_ROOT / "data" / "manifests" / f"flowers102_seed{seed}.json"
        root = PROJECT_ROOT / "data" / "Flowers102_256"
        val_fraction = 0.2
        epochs = 40
        prepare_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "prepare_torchvision_fg.py"),
            "--dataset",
            "flowers102",
            "--root",
            str(PROJECT_ROOT / "data"),
            "--out",
            str(manifest),
            "--val-fraction",
            str(val_fraction),
            "--seed",
            str(seed),
        ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return {
        "manifest": manifest,
        "root": root,
        "epochs": epochs,
        "prepare_command": prepare_command,
    }


def task_name(dataset: str, kind: str, rate: float, seed: int) -> str:
    if dataset == "flowers102":
        rate_tag = "25" if abs(rate - 0.2) < 1e-9 else "375"
    else:
        rate_tag = str(int(round(rate * 100)))
    return f"{dataset}_{kind}{rate_tag}_seed{seed}"


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        still_active = 259
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(
                handle,
                ctypes.byref(exit_code),
            ):
                return False
            return exit_code.value == still_active
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def wait_for_existing_pipelines(status: dict) -> None:
    while True:
        active = []
        for path in RUNS_ROOT.glob("*_pipeline_status.json"):
            if path == STATUS_PATH:
                continue
            try:
                value = read_json(path)
            except (OSError, ValueError):
                continue
            pid = int(value.get("pid", -1))
            if value.get("status") == "running" and process_is_alive(pid):
                active.append(path.stem)
        if not active:
            return
        status["status"] = "waiting_for_existing_pipeline"
        status["waiting_for"] = active
        write_json(STATUS_PATH, status)
        time.sleep(20)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the PMRC paper experiment queue.")
    parser.add_argument("--config", type=Path, default=QUEUE_CONFIG)
    args = parser.parse_args()
    lock = SuiteLock(RUNS_ROOT / "phase2.lock")
    if not lock.acquire():
        print("Phase 2 queue is already running.")
        return 0

    with args.config.open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    status = {
        "status": "waiting_for_existing_pipeline",
        "pid": os.getpid(),
        "started_at": now(),
        "tasks": {},
    }
    write_json(STATUS_PATH, status)
    try:
        wait_for_existing_pipelines(status)
        status["status"] = "running"
        status.pop("waiting_for", None)
        write_json(STATUS_PATH, status)
        for dataset_spec in spec["tasks"]:
            dataset = str(dataset_spec["dataset"])
            for seed in dataset_spec["seeds"]:
                settings = dataset_settings(dataset, int(seed))
                if not settings["manifest"].is_file():
                    run_logged(
                        settings["prepare_command"],
                        LOG_ROOT / "phase2_prepare.log",
                    )
                for noise_spec in dataset_spec["noises"]:
                    kind = str(noise_spec["kind"])
                    rate = float(noise_spec["rate"])
                    name = task_name(dataset, kind, rate, int(seed))
                    run_name = f"elr_resnet50_{name}"
                    run_dir = RUNS_ROOT / run_name
                    status["tasks"][name] = {
                        "status": "complete" if (run_dir / "test_metrics.json").is_file() else "pending",
                        "run_name": run_name,
                    }
                    write_json(STATUS_PATH, status)
                    if (run_dir / "test_metrics.json").is_file():
                        continue

                    noise_path = (
                        PROJECT_ROOT
                        / "data"
                        / "noise"
                        / f"{dataset}_{kind}_{int(round(rate * 100))}_seed{seed}.csv"
                    )
                    metadata_path = noise_path.with_suffix(".json")
                    if not noise_path.is_file():
                        train_root = (
                            str(settings["root"])
                            if dataset != "cub"
                            else str(PROJECT_ROOT / "data" / "CUB_200_2011_256")
                        )
                        run_logged(
                            [
                                sys.executable,
                                str(PROJECT_ROOT / "scripts" / "make_noise.py"),
                                "--root",
                                train_root,
                                "--manifest",
                                str(settings["manifest"]),
                                "--out",
                                str(noise_path),
                                "--metadata-out",
                                str(metadata_path),
                                "--kind",
                                kind,
                                "--top-k",
                                "5",
                                "--rate",
                                str(rate),
                                "--seed",
                                str(seed),
                                "--batch-size",
                                "64",
                                "--num-workers",
                                "0",
                            ],
                            LOG_ROOT / "phase2_noise.log",
                        )

                    GENERATED_CONFIGS.mkdir(parents=True, exist_ok=True)
                    config_path = GENERATED_CONFIGS / f"{name}.yaml"
                    config = {
                        "dataset": dataset,
                        "root": str(settings["root"]),
                        "manifest": str(settings["manifest"]),
                        "noise_csv": str(noise_path),
                        "output_dir": str(RUNS_ROOT),
                        "run_name": run_name,
                        "method": "elr",
                        "elr_beta": 0.9,
                        "elr_lambda": 3.0,
                        "model": "resnet50",
                        "pretrained": True,
                        "freeze_stages": 0,
                        "image_size": 224,
                        "batch_size": 32,
                        "num_workers": 0,
                        "epochs": settings["epochs"],
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
                    status["current"] = name
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
        return 0
    except Exception as error:
        status["status"] = "failed"
        status["error"] = repr(error)
        status["finished_at"] = now()
        write_json(STATUS_PATH, status)
        return 1
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
