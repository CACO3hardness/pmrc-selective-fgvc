from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgel.utils import read_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"


def main() -> None:
    status_path = RUNS_ROOT / "e0_suite_status.json"
    if not status_path.is_file():
        print("E0 has not started.")
        return

    status = read_json(status_path)
    print(f"Suite: {status.get('status')}  updated: {status.get('updated_at')}")
    print(f"PID: {status.get('pid')}")
    for task_name, task in status.get("tasks", {}).items():
        print(f"\n[{task_name}] {task.get('status')}")
        run_name = task.get("run_name")
        if not run_name:
            continue
        run_dir = RUNS_ROOT / run_name
        run_state_path = run_dir / "run_state.json"
        if run_state_path.is_file():
            state = read_json(run_state_path)
            print(
                f"  epochs: {state.get('completed_epochs')}/{state.get('total_epochs')}  "
                f"best val: {state.get('best_val_accuracy')}"
            )
            if state.get("last_seconds") is not None:
                print(f"  last epoch seconds: {state['last_seconds']:.1f}")
            if state.get("test_accuracy") is not None:
                print(
                    f"  test acc: {state['test_accuracy']:.4f}  "
                    f"AURC: {state['test_aurc']:.4f}"
                )
        log_path = PROJECT_ROOT / str(task.get("log", ""))
        if log_path.is_file():
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-3:]
            for line in tail:
                print(f"  log: {line}")

    rgel_status_path = RUNS_ROOT / "rgel_pilot_status.json"
    if rgel_status_path.is_file():
        rgel_status = read_json(rgel_status_path)
        print(f"\n[RGEL pilot] {rgel_status.get('status')}")
        rgel_run_dir = RUNS_ROOT / str(
            rgel_status.get("run_name", "rgel_resnet50_cub_semantic40_seed0_pilot")
        )
        rgel_state_path = rgel_run_dir / "run_state.json"
        if rgel_state_path.is_file():
            state = read_json(rgel_state_path)
            print(
                f"  epochs: {state.get('completed_epochs')}/{state.get('total_epochs')}  "
                f"best val: {state.get('best_val_accuracy')}"
            )
            if state.get("test_accuracy") is not None:
                print(
                    f"  test acc: {state['test_accuracy']:.4f}  "
                    f"AURC: {state['test_aurc']:.4f}"
                )

    for pipeline_path in sorted(RUNS_ROOT.glob("*_pipeline_status.json")):
        pipeline_status = read_json(pipeline_path)
        print(f"\n[{pipeline_path.stem}] {pipeline_status.get('status')}")
        if pipeline_status.get("status") == "aborted":
            print(f"  reason: {pipeline_status.get('reason', 'unspecified')}")
            continue
        pipeline_run_dir = RUNS_ROOT / str(pipeline_status.get("run_name", ""))
        pipeline_state_path = pipeline_run_dir / "run_state.json"
        if pipeline_state_path.is_file():
            state = read_json(pipeline_state_path)
            print(
                f"  epochs: {state.get('completed_epochs')}/{state.get('total_epochs')}  "
                f"best val: {state.get('best_val_accuracy')}"
            )
        if pipeline_status.get("msp_aurc") is not None:
            print(
                f"  MSP AURC: {pipeline_status['msp_aurc']:.4f}  "
                f"PMRC AURC: {pipeline_status['pmrc_aurc']:.4f}"
            )

    phase2_path = RUNS_ROOT / "phase2_status.json"
    if phase2_path.is_file():
        phase2 = read_json(phase2_path)
        print(f"\n[Phase 2 queue] {phase2.get('status')}")
        if phase2.get("current"):
            print(f"  current: {phase2['current']}")
        if phase2.get("waiting_for"):
            print(f"  waiting for: {', '.join(phase2['waiting_for'])}")
        tasks = phase2.get("tasks", {})
        completed = sum(value.get("status") == "complete" for value in tasks.values())
        print(f"  tasks complete: {completed}/{len(tasks)}")


if __name__ == "__main__":
    main()
