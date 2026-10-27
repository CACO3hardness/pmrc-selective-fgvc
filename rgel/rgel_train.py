from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from rgel.data import (
    CUBManifestDataset,
    CUBTwoViewDataset,
    ManifestRow,
    build_transforms,
    load_manifest,
    load_noise_map,
)
from rgel.evidential import EvidentialResNet, evaluate_evidential, rgel_losses
from rgel.utils import set_seed, write_json


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _eval_loader(
    root: Path,
    rows: list[ManifestRow],
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        CUBManifestDataset(root, rows, build_transforms(image_size, train=False)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _reliability(
    model: EvidentialResNet,
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    alpha_a: torch.Tensor,
    alpha_b: torch.Tensor,
    targets: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    probability_a = torch.softmax(logits_a, dim=1)
    probability_b = torch.softmax(logits_b, dim=1)
    agreement = torch.sqrt(probability_a * probability_b).sum(dim=1)
    prototype_confidence = 0.5 * (
        model.prototype_confidence(features_a, targets)
        + model.prototype_confidence(features_b, targets)
    )
    entropy_a = -(probability_a * torch.log(probability_a.clamp_min(1e-8))).sum(dim=1)
    entropy_b = -(probability_b * torch.log(probability_b.clamp_min(1e-8))).sum(dim=1)
    normalized_entropy = 0.5 * (entropy_a + entropy_b) / math.log(alpha_a.shape[1])
    confidence = 1.0 - normalized_entropy

    agreement_weight = float(config.get("reliability_agreement_weight", 0.5))
    prototype_weight = float(config.get("reliability_prototype_weight", 0.35))
    confidence_weight = float(config.get("reliability_confidence_weight", 0.15))
    total_weight = agreement_weight + prototype_weight + confidence_weight
    reliability = (
        agreement_weight * agreement
        + prototype_weight * prototype_confidence
        + confidence_weight * confidence
    ) / total_weight
    return reliability.clamp(0.0, 1.0)


def train_rgel(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 0))
    set_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    root = Path(config["root"]).resolve()
    rows = load_manifest(config["manifest"])
    train_rows = [row for row in rows if row.split == "train"]
    val_rows = [row for row in rows if row.split == "val"]
    test_rows = [row for row in rows if row.split == "test"]
    num_classes = len({row.class_id for row in rows})
    noise_map = load_noise_map(config.get("noise_csv"))

    image_size = int(config.get("image_size", 224))
    batch_size = int(config.get("batch_size", 16))
    num_workers = int(config.get("num_workers", 0))
    val_loader = _eval_loader(root, val_rows, image_size, batch_size, num_workers)
    test_loader = _eval_loader(root, test_rows, image_size, batch_size, num_workers)

    device = _device()
    model = EvidentialResNet(
        str(config.get("model", "resnet50")),
        num_classes,
        pretrained=bool(config.get("pretrained", True)),
        freeze_stages=int(config.get("freeze_stages", 0)),
        prototype_momentum=float(config.get("prototype_momentum", 0.9)),
    ).to(device)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config.get("lr", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    epochs = int(config.get("epochs", 10))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    use_amp = bool(config.get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if config.get("amp_dtype", "bf16") == "bf16" else torch.float16
    grad_clip = float(config.get("grad_clip", 1.0))
    vacuity_weight = float(config.get("vacuity_weight", 0.1))
    consistency_weight = float(config.get("consistency_weight", 0.5))
    prototype_threshold = float(config.get("prototype_threshold", 0.65))
    gate_warmup_epochs = int(config.get("gate_warmup_epochs", 3))

    output_dir = Path(config.get("output_dir", "runs")) / str(config["run_name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config)
    checkpoint_path = output_dir / "best.pt"
    last_checkpoint_path = output_dir / "last.pt"
    history: list[dict[str, Any]] = []
    best_val_accuracy = -1.0
    best_epoch = -1
    start_epoch = 1

    if bool(config.get("resume", True)) and last_checkpoint_path.is_file():
        checkpoint = torch.load(last_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        history = list(checkpoint["history"])
        best_val_accuracy = float(checkpoint["best_val_accuracy"])
        best_epoch = int(checkpoint["best_epoch"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"resuming RGEL at epoch {start_epoch}/{epochs}")

    for epoch in range(start_epoch, epochs + 1):
        set_seed(seed + epoch)
        dataset = CUBTwoViewDataset(
            root,
            train_rows,
            build_transforms(image_size, train=True),
            noise_map=noise_map,
        )
        generator = torch.Generator().manual_seed(seed + epoch)
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
            generator=generator,
        )
        model.train()
        started = time.perf_counter()
        totals = {"loss": 0.0, "supervised": 0.0, "consistency": 0.0, "vacuity": 0.0, "reliability": 0.0}
        running_correct = 0
        running_total = 0

        for batch_index, batch in enumerate(train_loader, start=1):
            image_a = batch["image_a"].to(device, non_blocking=True)
            image_b = batch["image_b"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                features_a, logits_a, alpha_a = model(image_a)
                features_b, logits_b, alpha_b = model(image_b)
                reliability = _reliability(
                    model,
                    features_a,
                    features_b,
                    logits_a,
                    logits_b,
                    alpha_a,
                    alpha_b,
                    targets,
                    config,
                )
                if epoch <= gate_warmup_epochs:
                    reliability = torch.ones_like(reliability)
                loss, components = rgel_losses(
                    logits_a,
                    logits_b,
                    alpha_a,
                    alpha_b,
                    targets,
                    reliability,
                    vacuity_weight,
                    consistency_weight,
                )

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if epoch > gate_warmup_epochs:
                with torch.no_grad():
                    model.update_prototypes(
                        features_a.float(),
                        targets,
                        reliability >= prototype_threshold,
                    )

            probability = (
                torch.softmax(logits_a.detach(), dim=1)
                + torch.softmax(logits_b.detach(), dim=1)
            ) * 0.5
            running_correct += int((probability.argmax(dim=1) == targets).sum())
            running_total += len(targets)
            totals["loss"] += float(loss.detach()) * len(targets)
            for key in ("supervised", "consistency", "vacuity"):
                totals[key] += components[key] * len(targets)
            totals["reliability"] += components["reliability_mean"] * len(targets)
            if batch_index == 1 or batch_index % 25 == 0 or batch_index == len(train_loader):
                print(
                    f"epoch={epoch} batch={batch_index}/{len(train_loader)} "
                    f"loss={totals['loss'] / running_total:.4f} "
                    f"reliability={totals['reliability'] / running_total:.3f}"
                )

        scheduler.step()
        val_metrics, *_ = evaluate_evidential(model, val_loader, device)
        elapsed = time.perf_counter() - started
        record = {
            "epoch": epoch,
            "train_loss": totals["loss"] / max(running_total, 1),
            "train_accuracy_noisy_target": running_correct / max(running_total, 1),
            "train_supervised": totals["supervised"] / max(running_total, 1),
            "train_consistency": totals["consistency"] / max(running_total, 1),
            "train_vacuity": totals["vacuity"] / max(running_total, 1),
            "train_reliability": totals["reliability"] / max(running_total, 1),
            "val": val_metrics,
            "seconds": elapsed,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        if len(history) != epoch:
            raise RuntimeError("RGEL resume history is inconsistent")
        write_json(output_dir / "history.json", history)
        print(json.dumps(record, ensure_ascii=False))

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = float(val_metrics["accuracy"])
            best_epoch = epoch
            _save_checkpoint(
                checkpoint_path,
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "best_epoch": best_epoch,
                    "best_val_accuracy": best_val_accuracy,
                },
            )
        _save_checkpoint(
            last_checkpoint_path,
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "config": config,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_val_accuracy": best_val_accuracy,
                "history": history,
            },
        )
        write_json(
            output_dir / "run_state.json",
            {
                "run_name": config["run_name"],
                "status": "running" if epoch < epochs else "trained",
                "completed_epochs": epoch,
                "total_epochs": epochs,
                "best_epoch": best_epoch,
                "best_val_accuracy": best_val_accuracy,
                "last_val_accuracy": val_metrics["accuracy"],
                "last_seconds": elapsed,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics, test_probabilities, test_labels, test_logits, test_alpha = (
        evaluate_evidential(model, test_loader, device)
    )
    np.save(output_dir / "test_probabilities.npy", test_probabilities)
    np.save(output_dir / "test_logits.npy", test_logits)
    np.save(output_dir / "test_alpha.npy", test_alpha)
    np.save(output_dir / "test_labels.npy", test_labels)
    result = {
        "run_name": config["run_name"],
        "method": "rgel",
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "history": history,
        "test": test_metrics,
    }
    write_json(output_dir / "test_metrics.json", result)
    write_json(
        output_dir / "run_state.json",
        {
            "run_name": config["run_name"],
            "status": "complete",
            "completed_epochs": epochs,
            "total_epochs": epochs,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_accuracy,
            "test_accuracy": test_metrics["accuracy"],
            "test_aurc": test_metrics["aurc"],
            "test_ece_15": test_metrics["ece_15"],
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    print(json.dumps({"test": test_metrics, "best_epoch": best_epoch}, ensure_ascii=False))
    return result
