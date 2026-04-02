"""Project configuration.

Centralizes all configurable settings. Uses environment variables
where appropriate for paths and sensitive settings.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch


def _get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_num_workers() -> int:
    """Get reasonable number of dataloader workers."""
    return min(4, os.cpu_count() or 2)


IMAGENET_PATH = os.environ.get("IMAGENET_PATH", "/run/media/samiel/K_USB_256/imagenet/")

DATA_DIR = Path(os.environ.get("VIT_DATA_DIR", "data"))
DETECTION_DIR = DATA_DIR / "detection"
LOGITS_DIR = DATA_DIR / "logits"


@dataclass
class ModelConfig:
    """Configuration for model and data loading."""

    data_root: str = field(default_factory=lambda: IMAGENET_PATH)
    batch_size: int = 100
    max_batches: int | None = 1
    num_workers: int = field(default_factory=_get_num_workers)
    device: torch.device = field(default_factory=_get_device)
    use_train: bool = False


SUPPORTED_MODELS = {
    "vit_tiny": "vit_tiny_patch16_224",
    "vit_small": "vit_small_patch16_224",
    "vit_base": "vit_base_patch16_224",
    "vit_large": "vit_large_patch16_224",
    "deit_tiny": "deit_tiny_patch16_224",
    "deit_small": "deit_small_patch16_224",
    "deit_base": "deit_base_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "swin_small": "swin_small_patch4_window7_224",
    "swin_base": "swin_base_patch4_window7_224",
    "beit_base": "beit_base_patch16_224",
}
