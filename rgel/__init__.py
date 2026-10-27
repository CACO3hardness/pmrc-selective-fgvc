"""E0 utilities for fine-grained recognition under semantic label noise."""

import os
from pathlib import Path


# The managed desktop sandbox cannot write to the user's default ~/.cache/torch.
os.environ.setdefault(
    "TORCH_HOME",
    str(Path(__file__).resolve().parents[1] / ".torch_cache"),
)

__version__ = "0.1.0"
