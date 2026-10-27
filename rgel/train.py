from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from rgel.engine import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an E0 baseline.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.run_name is not None:
        config["run_name"] = args.run_name
    train(config)


if __name__ == "__main__":
    main()

