from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from rgel.phase2_queue import run_logged


class Phase2QueueTests(unittest.TestCase):
    def test_activity_file_prevents_false_stall(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            queue_log = root / "queue.log"
            activity_log = root / "pipeline.log"
            code = (
                "import pathlib, time\n"
                f"path = pathlib.Path(r'{activity_log}')\n"
                "for index in range(12):\n"
                "    path.write_text(str(index), encoding='utf-8')\n"
                "    time.sleep(0.05)\n"
            )
            run_logged(
                [sys.executable, "-c", code],
                queue_log,
                stall_seconds=0.25,
                max_attempts=1,
                activity_paths=[activity_log],
                poll_seconds=0.05,
            )
            self.assertTrue(activity_log.is_file())
            self.assertNotIn("watchdog", queue_log.read_text(encoding="utf-8"))

    def test_disabled_watchdog_allows_silent_process(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            queue_log = root / "queue.log"
            code = "import time; time.sleep(0.3)"
            run_logged(
                [sys.executable, "-c", code],
                queue_log,
                stall_seconds=None,
                max_attempts=1,
                poll_seconds=0.05,
            )
            self.assertNotIn("watchdog", queue_log.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
