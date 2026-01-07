from __future__ import annotations

import os
from typing import Any


def save_checkpoint(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    import torch

    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path: str) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu")

