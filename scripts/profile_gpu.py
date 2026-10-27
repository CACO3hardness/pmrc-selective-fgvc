from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from rgel.models import build_model, trainable_parameter_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure ResNet training memory and step time.")
    parser.add_argument("--model", choices=("resnet18", "resnet50"), default="resnet50")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=200)
    parser.add_argument("--freeze-stages", type=int, default=0)
    parser.add_argument("--batches", type=int, nargs="+", default=[4, 8, 16, 24, 32])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    return parser.parse_args()


def profile(
    model_name: str,
    image_size: int,
    num_classes: int,
    freeze_stages: int,
    batch_size: int,
    warmup: int,
    steps: int,
) -> dict[str, float | int | str]:
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = build_model(model_name, num_classes, True, freeze_stages).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    def one_step() -> None:
        images = torch.randn(batch_size, 3, image_size, image_size, device=device)
        targets = torch.randint(num_classes, (batch_size,), device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            loss = criterion(model(images), targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(steps):
        one_step()
    torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - started) / steps

    result = {
        "model": model_name,
        "freeze_stages": freeze_stages,
        "image_size": image_size,
        "batch_size": batch_size,
        "trainable_parameters": trainable_parameter_count(model),
        "seconds_per_step": elapsed,
        "images_per_second": batch_size / elapsed,
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
    }
    del model, optimizer, criterion, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    print(
        f"GPU: {torch.cuda.get_device_name(0)}, "
        f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GiB"
    )
    for batch_size in args.batches:
        try:
            result = profile(
                args.model,
                args.image_size,
                args.num_classes,
                args.freeze_stages,
                batch_size,
                args.warmup,
                args.steps,
            )
            print(result, flush=True)
        except torch.OutOfMemoryError:
            print(f"batch_size={batch_size}: CUDA out of memory", flush=True)
            torch.cuda.empty_cache()
            break


if __name__ == "__main__":
    main()
