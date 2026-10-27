from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgel.utils import read_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
def main() -> None:
    parser = argparse.ArgumentParser(description="Show WebFG queue status.")
    parser.add_argument(
        "--status-name",
        default="webfg496_status",
        help="Status JSON stem under runs/.",
    )
    args = parser.parse_args()
    status_path = RUNS_ROOT / f"{args.status_name}.json"
    if not status_path.is_file():
        print("WebFG queue has not started.")
        return
    status = read_json(status_path)
    print(
        f"WebFG: {status.get('status')}  current={status.get('current', '-')}  "
        f"pid={status.get('pid')}"
    )
    for name, task in status.get("tasks", {}).items():
        run_name = task.get("run_name", "")
        run_dir = RUNS_ROOT / run_name
        print(f"\n[{name}] {task.get('status')}")
        state_path = run_dir / "run_state.json"
        if state_path.is_file():
            state = read_json(state_path)
            print(
                f"  epochs: {state.get('completed_epochs')}/"
                f"{state.get('total_epochs')}  "
                f"best val: {state.get('best_val_accuracy')}"
            )
            if state.get("last_seconds") is not None:
                print(f"  last epoch seconds: {state['last_seconds']:.1f}")
        posthoc_path = run_dir / "posthoc_reliability.json"
        if posthoc_path.is_file():
            result = read_json(posthoc_path)
            msp = result["test_results"]["msp"]["aurc"]
            pmrc = result["test_results"]["learned_reliability"]["aurc"]
            print(
                f"  test acc: {result['test_accuracy']:.4f}  "
                f"MSP AURC: {msp:.4f}  PMRC AURC: {pmrc:.4f}"
            )
        log_path = RUNS_ROOT / "e0_logs" / f"{name}_pipeline.log"
        if log_path.is_file():
            lines = log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            for line in lines[-2:]:
                print(f"  log: {line[:180]}")


if __name__ == "__main__":
    main()
