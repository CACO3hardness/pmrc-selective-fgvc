from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgel.data import (
    CUBManifestDataset,
    CUBTwoViewDataset,
    build_transforms,
    load_manifest,
)
from rgel.models import build_model
from rgel.utils import read_json, set_seed, write_json
from scripts.posthoc_reliability import (
    aurc_from_scores,
    class_prototypes,
    fit_logistic,
    reliability_features,
    risk_at_coverage,
    score_logistic,
)
from scripts.prototype_independence import roc_auc, stratified_auc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-domain open-set proxy using WebFG subset models."
    )
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--unknown-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--known-subsets",
        nargs="+",
        choices=("bird", "aircraft", "car"),
        default=["bird", "aircraft", "car"],
    )
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


def evaluate_scores(
    msp_scores: np.ndarray,
    pmrc_scores: np.ndarray,
    known_flag: np.ndarray,
) -> dict[str, float]:
    correct = known_flag.astype(bool)
    msp_aurc = aurc_from_scores(msp_scores, correct)
    pmrc_aurc = aurc_from_scores(pmrc_scores, correct)
    coverage = np.arange(1, len(correct) + 1) / len(correct)
    order = np.argsort(-pmrc_scores, kind="stable")
    index = max(0, int(np.ceil(0.9 * len(correct))) - 1)
    accepted = order[: index + 1]
    return {
        "n": int(len(correct)),
        "known_count": int(known_flag.sum()),
        "unknown_count": int((~known_flag).sum()),
        "unknown_rate": float(1.0 - known_flag.mean()),
        "msp_aurc": float(msp_aurc),
        "pmrc_aurc": float(pmrc_aurc),
        "relative_gain_pct": float(100.0 * (msp_aurc - pmrc_aurc) / msp_aurc),
        "msp_unknown_auroc": float(roc_auc(msp_scores, known_flag)),
        "pmrc_unknown_auroc": float(roc_auc(pmrc_scores, known_flag)),
        "msp_risk90": float(risk_at_coverage(msp_scores, correct, 0.9)),
        "pmrc_risk90": float(risk_at_coverage(pmrc_scores, correct, 0.9)),
        "pmrc_unknown_acceptance90": float(1.0 - known_flag[accepted].mean()),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = args.output_dir / "openset_webfg_proxy.json"
    rows = []
    if args.append and output_path.is_file():
        rows = json.loads(output_path.read_text(encoding="utf-8"))["runs"]
    for known_subset in args.known_subsets:
        run_name = f"elr_resnet50_webfg_{known_subset}_seed0"
        run_dir = args.runs_root / run_name
        config = read_json(run_dir / "config.json")
        manifest = load_manifest(config["manifest"])
        val_rows = [row for row in manifest if row.split == "val"]
        known_test_rows = [row for row in manifest if row.split == "test"]
        num_classes = len({row.class_id for row in manifest})
        model = build_model(
            config["model"],
            num_classes,
            pretrained=False,
            freeze_stages=int(config.get("freeze_stages", 0)),
        ).to(device)
        checkpoint = torch.load(
            run_dir / "best.pt",
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        train_rows = [row for row in manifest if row.split == "train"]
        train_loader = DataLoader(
            CUBManifestDataset(
                config["root"],
                train_rows,
                build_transforms(int(config.get("image_size", 224)), False),
            ),
            batch_size=int(config.get("batch_size", 64)),
            shuffle=False,
            num_workers=0,
        )
        prototypes = class_prototypes(
            model,
            train_loader,
            num_classes,
            device,
        )

        known_features = np.load(run_dir / "posthoc_test_features.npy")
        known_scores = np.load(run_dir / "posthoc_test_scores.npy")
        val_features = np.load(run_dir / "posthoc_val_features.npy")
        val_labels = np.load(run_dir / "posthoc_val_labels.npy")
        val_predictions = np.load(run_dir / "posthoc_val_predictions.npy")
        val_correct = val_predictions == val_labels
        posthoc = read_json(run_dir / "posthoc_reliability.json")
        calibrator, mean, std = fit_logistic(
            val_features,
            val_correct,
            float(posthoc["selected_weight_decay"]),
        )

        unknown_pool = []
        for other_subset in ("bird", "aircraft", "car"):
            if other_subset == known_subset:
                continue
            other_manifest = load_manifest(
                PROJECT_ROOT / "data" / "manifests" / f"webfg_{other_subset}.json"
            )
            unknown_pool.extend(
                row for row in other_manifest if row.split == "test"
            )
        target_unknown = int(
            round(
                args.unknown_fraction
                / (1.0 - args.unknown_fraction)
                * len(known_test_rows)
            )
        )
        target_unknown = min(target_unknown, len(unknown_pool))
        chosen = rng.choice(len(unknown_pool), target_unknown, replace=False)
        unknown_rows = [unknown_pool[int(index)] for index in sorted(chosen)]
        unknown_loader = DataLoader(
            CUBTwoViewDataset(
                config["root"],
                unknown_rows,
                build_transforms(int(config.get("image_size", 224)), False),
            ),
            batch_size=int(config.get("batch_size", 64)),
            shuffle=False,
            num_workers=0,
        )
        unknown_features, _, _ = reliability_features(
            model,
            unknown_loader,
            prototypes,
            device,
        )
        unknown_scores = score_logistic(
            calibrator,
            unknown_features,
            mean,
            std,
        )
        msp_scores = np.concatenate((known_features[:, 0], unknown_features[:, 0]))
        pmrc_scores = np.concatenate((known_scores, unknown_scores))
        known_flag = np.concatenate(
            (
                np.ones(len(known_features), dtype=bool),
                np.zeros(len(unknown_features), dtype=bool),
            )
        )
        gate_auc = stratified_auc(
            val_features[:, 0],
            val_features[:, 9],
            val_correct,
            5,
        )["weighted_auc"]
        metrics = evaluate_scores(msp_scores, pmrc_scores, known_flag)
        new_row = (
            {
                "known_subset": known_subset,
                "unknown_subsets": [
                    value
                    for value in ("bird", "aircraft", "car")
                    if value != known_subset
                ],
                "validation_gate_auc": float(gate_auc),
                "gate_selected_method": "pmrc" if gate_auc > 0.5 else "msp",
                **metrics,
            }
        )
        rows = [row for row in rows if row["known_subset"] != known_subset]
        rows.append(new_row)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_path, {"runs": rows})
        print(json.dumps(new_row), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "openset_webfg_proxy.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# WebFG cross-domain open-set proxy",
        "",
        "Known-domain calibration and test images come from the named subset. "
        "Unknown images are sampled from the other two WebFG subsets at a fixed "
        "20% prevalence. Unknown samples count as errors for selective risk.",
        "",
        "| Known subset | Unknown | PMRC AURC gain | MSP unknown AUROC | PMRC unknown AUROC | PMRC risk@90 | Gate |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['known_subset']} | {', '.join(row['unknown_subsets'])} | "
            f"{row['relative_gain_pct']:+.2f}% | "
            f"{row['msp_unknown_auroc']:.4f} | "
            f"{row['pmrc_unknown_auroc']:.4f} | "
            f"{row['pmrc_risk90']:.4f} | "
            f"{row['gate_selected_method']} |"
        )
    lines.extend(
        [
            "",
            "This is a cross-domain open-set proxy, not a classifier retrained "
            "on a restricted known-class set.",
        ]
    )
    (args.output_dir / "openset_webfg_proxy.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
