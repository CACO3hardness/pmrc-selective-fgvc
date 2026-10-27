from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_ROOT = PROJECT_ROOT / "data" / "manifests"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "WebFG_496" / "extracted"
SUBSETS = ("bird", "aircraft", "car")


def check_image(path_text: str) -> tuple[str, str | None]:
    path = Path(path_text)
    try:
        with Image.open(path) as image:
            image.load()
            image.convert("RGB")
    except Exception as error:
        return path_text, f"{type(error).__name__}: {error}"
    return path_text, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that every WebFG manifest image can be decoded."
    )
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report: dict[str, object] = {"data_root": str(args.data_root), "subsets": {}}
    total_failures = 0

    for subset in SUBSETS:
        manifest_path = args.manifest_root / f"webfg_{subset}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        relative_paths = [str(row["relative_path"]) for row in manifest["records"]]
        paths = [str(args.data_root / relative_path) for relative_path in relative_paths]
        failures: list[dict[str, str]] = []

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(check_image, path) for path in paths]
            for future in as_completed(futures):
                path, error = future.result()
                if error is not None:
                    failures.append({"path": path, "error": error})

        total_failures += len(failures)
        report["subsets"][subset] = {
            "checked": len(paths),
            "failures": len(failures),
            "failure_details": sorted(failures, key=lambda item: item["path"]),
        }
        print(f"{subset}: checked={len(paths)} failures={len(failures)}", flush=True)

    report["total_failures"] = total_failures
    report_path = args.manifest_root / "webfg_image_validation.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"report: {report_path}", flush=True)
    if total_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
