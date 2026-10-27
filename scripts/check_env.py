from __future__ import annotations

import platform
import sys


def main() -> None:
    print("python", sys.version.split()[0], platform.platform())
    try:
        import torch
        import torchvision
    except ImportError as error:
        raise SystemExit(f"PyTorch or torchvision is unavailable: {error}") from error

    print("torch", torch.__version__)
    print("torchvision", torchvision.__version__)
    print("cuda_available", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Install a CUDA-enabled PyTorch build.")

    device = torch.device("cuda")
    print("gpu", torch.cuda.get_device_name(device))
    properties = torch.cuda.get_device_properties(device)
    print("memory_gb", round(properties.total_memory / (1024**3), 2))
    print("capability", torch.cuda.get_device_capability(device))
    tensor = torch.randn((1024, 1024), device=device)
    print("matmul_ok", torch.isfinite(tensor @ tensor).all().item())


if __name__ == "__main__":
    main()

