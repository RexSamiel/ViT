# settings.py
# Configuration definitions for ViT fault injection experiments

import os
import torch
from dataclasses import dataclass


@dataclass
class Config:
    """
    Configuration for Vision Transformer fault injection evaluation.
    """

    root_dir: str = "/run/media/samiel/K_USB_256/imagenet/"
    model_name: str = "vit_base_patch16_224"
    model_key: str = "vit_base"
    batch_size: int = 100
    num_workers: int = min(4, os.cpu_count() or 2)
    use_amp: bool = True
    max_batches: int | None = 1  # Limit number of batches (optional)

    @property
    def device(self) -> torch.device:
        """Automatically select CUDA if available, otherwise CPU."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
