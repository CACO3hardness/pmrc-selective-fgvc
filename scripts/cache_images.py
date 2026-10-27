from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from rgel.data import load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize all manifest images once into a training cache."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--shorter-side", type=int, default=256)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def resize_one(source: Path, destination: Path, shorter_side: int, quality: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = image.convert("RGB")
        width, height = image.size
        scale = shorter_side / min(width, height)
        if scale < 1.0:
            new_size = (max(shorter_side, round(width * scale)), max(shorter_side, round(height * scale)))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        image.save(destination, format="JPEG", quality=quality, optimize=True)


def main() -> None:
    args = parse_args()
    if args.shorter_side < 224:
        raise ValueError("--shorter-side must be at least 224")
    rows = load_manifest(args.manifest)
    jobs = [
        (args.root / row.relative_path, args.out_root / row.relative_path)
        for row in rows
    ]
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                resize_one,
                source,
                destination,
                args.shorter_side,
                args.quality,
            )
            for source, destination in jobs
        ]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed == 1 or completed % 500 == 0 or completed == len(jobs):
                print(f"cached {completed}/{len(jobs)}", flush=True)
    print(f"Resized image cache written to {args.out_root.resolve()}")


if __name__ == "__main__":
    main()

