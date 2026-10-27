from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests
import urllib3


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "StanfordDogs"
PARQUET_ROOT = DATA_ROOT / "parquet"
REPOSITORY = "amaye15/stanford-dogs"
FILES = (
    "data/train-00000-of-00002.parquet",
    "data/train-00001-of-00002.parquet",
    "data/test-00000-of-00001.parquet",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    PARQUET_ROOT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "Codex StanfordDogs downloader"
    metadata = {}
    for relative in FILES:
        output = PARQUET_ROOT / Path(relative).name
        url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/main/{relative}"
        head = session.head(url, timeout=30, verify=False, allow_redirects=True)
        head.raise_for_status()
        expected_size = int(head.headers["content-length"])
        if output.is_file() and output.stat().st_size == expected_size:
            print(f"Already complete: {output.name}", flush=True)
        else:
            temporary = output.with_suffix(output.suffix + ".part")
            with session.get(
                url,
                stream=True,
                timeout=(30, 300),
                verify=False,
                allow_redirects=True,
            ) as response:
                response.raise_for_status()
                downloaded = 0
                last_report = 0
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if downloaded - last_report >= 64 * 1024 * 1024:
                            print(
                                f"{output.name}: "
                                f"{downloaded / (1024**2):.1f}/"
                                f"{expected_size / (1024**2):.1f} MiB",
                                flush=True,
                            )
                            last_report = downloaded
            if temporary.stat().st_size != expected_size:
                raise RuntimeError(
                    f"Incomplete download for {output.name}: "
                    f"{temporary.stat().st_size} != {expected_size}"
                )
            temporary.replace(output)
        metadata[output.name] = {
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
            "source": url,
        }
        print(f"Verified: {output.name} ({output.stat().st_size} bytes)", flush=True)
    (DATA_ROOT / "download_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(DATA_ROOT / "download_metadata.json")


if __name__ == "__main__":
    main()
