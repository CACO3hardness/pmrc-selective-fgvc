from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from rgel.rgel_train import train_rgel


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RGEL pilot.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.run_name:
        config["run_name"] = args.run_name
    if args.output_dir:
        config["output_dir"] = str(args.output_dir)
    train_rgel(config)


if __name__ == "__main__":
    main()

