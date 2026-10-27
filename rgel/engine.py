from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from rgel.data import CUBManifestDataset, ManifestRow, build_transforms, load_noise_map
from rgel.losses import ELRLoss, GCELoss
from rgel.metrics import evaluation_metrics
from rgel.models import build_model, trainable_parameter_count
from rgel.utils import set_seed, write_json


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_loader(
    root: Path,
    rows: list[ManifestRow],
    transform: Any,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    noise_map: dict[int, int] | None = None,
    generator_seed: int | None = None,
) -> DataLoader:
    dataset = CUBManifestDataset(root, rows, transform, noise_map=noise_map)
    generator = None
    if generator_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(generator_seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total_loss = 0.0
    total = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["clean_target"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += float(loss) * len(labels)
        total += len(labels)
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels)
    metrics = evaluation_metrics(logits_np, labels_np)
    metrics["loss"] = total_loss / max(total, 1)
    return metrics, logits_np, labels_np


def train(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 0))
    set_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    from rgel.data import load_manifest

    root = Path(config["root"]).resolve()
    rows = load_manifest(config["manifest"])
    train_rows = [row for row in rows if row.split == "train"]
    val_rows = [row for row in rows if row.split == "val"]
    test_rows = [row for row in rows if row.split == "test"]
    num_classes = len({row.class_id for row in rows})

    noise_map = load_noise_map(config.get("noise_csv"))
    unknown_noise_ids = set(noise_map) - {row.image_id for row in train_rows}
    if unknown_noise_ids:
        raise ValueError(f"Noise CSV includes non-training IDs: {sorted(unknown_noise_ids)[:5]}")
    invalid_noise_labels = {
        label for label in noise_map.values() if label < 0 or label >= num_classes
    }
    if invalid_noise_labels:
        raise ValueError(
            "Noise CSV includes labels outside the class range: "
            f"{sorted(invalid_noise_labels)[:10]}"
        )

    image_size = int(config.get("image_size", 224))
    batch_size = int(config.get("batch_size", 16))
    num_workers = int(config.get("num_workers", 4))
    val_loader = _build_loader(
        root,
        val_rows,
        build_transforms(image_size, train=False),
        batch_size,
        num_workers,
        shuffle=False,
    )
    test_loader = _build_loader(
        root,
        test_rows,
        build_transforms(image_size, train=False),
        batch_size,
        num_workers,
        shuffle=False,
    )

    device = _device()
    model = build_model(
        str(config.get("model", "resnet50")),
        num_classes,
        pretrained=bool(config.get("pretrained", True)),
        freeze_stages=int(config.get("freeze_stages", 0)),
    ).to(device)
    eval_criterion = nn.CrossEntropyLoss()
    method = str(config.get("method", "ce")).lower()
    if method == "ce":
        train_criterion: nn.Module = nn.CrossEntropyLoss(
            label_smoothing=float(config.get("label_smoothing", 0.0))
        )
    elif method == "elr":
        train_criterion = ELRLoss(
            num_examples=len(train_rows),
            num_classes=num_classes,
            beta=float(config.get("elr_beta", 0.9)),
            regularization=float(config.get("elr_lambda", 3.0)),
            label_smoothing=float(config.get("label_smoothing", 0.0)),
            device=device,
        )
    elif method == "gce":
        train_criterion = GCELoss(
            q=float(config.get("gce_q", 0.7)),
        )
    else:
        raise ValueError(f"Unsupported method: {method}")
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config.get("lr", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    epochs = int(config.get("epochs", 20))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    use_amp = bool(config.get("amp", True)) and device.type == "cuda"
    amp_dtype_name = str(config.get("amp_dtype", "bf16")).lower()
    if amp_dtype_name == "bf16":
        amp_dtype = torch.bfloat16
    elif amp_dtype_name == "fp16":
        amp_dtype = torch.float16
    else:
        raise ValueError("amp_dtype must be 'bf16' or 'fp16'")
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if use_amp and amp_dtype == torch.float16
        else None
    )
    grad_clip = float(config.get("grad_clip", 1.0))

    output_dir = Path(config.get("output_dir", "runs")) / str(config["run_name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history: list[dict[str, Any]] = []
    best_val_accuracy = -1.0
    best_epoch = -1
    start_epoch = 1
    checkpoint_path = output_dir / "best.pt"
    last_checkpoint_path = output_dir / "last.pt"
    resume = bool(config.get("resume", True))

    if resume and last_checkpoint_path.is_file():
        checkpoint = torch.load(last_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "criterion_state" in checkpoint:
            train_criterion.load_state_dict(checkpoint["criterion_state"])
        best_val_accuracy = float(checkpoint["best_val_accuracy"])
        best_epoch = int(checkpoint["best_epoch"])
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint["epoch"]) + 1
        print(
            f"resuming {config['run_name']} at epoch {start_epoch}/{epochs} "
            f"(best validation accuracy {best_val_accuracy:.4f})"
        )

    for epoch in range(start_epoch, epochs + 1):
        set_seed(seed + epoch)
        train_loader = _build_loader(
            root,
            train_rows,
            build_transforms(image_size, train=True),
            batch_size,
            num_workers,
            shuffle=True,
            noise_map=noise_map,
            generator_seed=seed + epoch,
        )
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        optimizer_steps = 0
        started = time.perf_counter()

        for batch_index, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            indices = batch["index"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                logits = model(images)
                if method == "elr":
                    assert isinstance(train_criterion, ELRLoss)
                    loss = train_criterion(indices, logits, targets)
                else:
                    loss = train_criterion(logits, targets)

            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                step_result = scaler.step(optimizer)
                scaler.update()
                if step_result is not None:
                    optimizer_steps += 1
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer_steps += 1

            running_loss += float(loss.detach()) * len(targets)
            running_correct += int((logits.argmax(dim=1) == targets).sum())
            running_total += len(targets)
            if batch_index == 1 or batch_index % 25 == 0 or batch_index == len(train_loader):
                print(
                    f"epoch={epoch} batch={batch_index}/{len(train_loader)} "
                    f"loss={running_loss / running_total:.4f}"
                )

        if optimizer_steps:
            scheduler.step()
        val_metrics, _, _ = evaluate(model, val_loader, eval_criterion, device)
        elapsed = time.perf_counter() - started
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(running_total, 1),
            "train_accuracy_noisy_target": running_correct / max(running_total, 1),
            "val": val_metrics,
            "seconds": elapsed,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        if len(history) != epoch:
            raise RuntimeError("Resume history is inconsistent with the current epoch")
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
                    "num_classes": num_classes,
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
                "criterion_state": train_criterion.state_dict(),
                "config": config,
                "num_classes": num_classes,
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
    test_metrics, test_logits, test_labels = evaluate(
        model,
        test_loader,
        eval_criterion,
        device,
    )
    np.save(output_dir / "test_logits.npy", test_logits)
    np.save(output_dir / "test_labels.npy", test_labels)

    result = {
        "run_name": config["run_name"],
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "num_classes": num_classes,
        "trainable_parameters": trainable_parameter_count(model),
        "records": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        ),
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
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_hours": sum(record["seconds"] for record in history) / 3600.0,
        },
    )
    print(json.dumps({"test": test_metrics, "best_epoch": best_epoch}, ensure_ascii=False))
    return result
