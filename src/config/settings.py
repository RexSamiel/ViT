import os
import torch
from dataclasses import dataclass, field


@dataclass
class Config:
    root_dir: str = "/run/media/samiel/K_USB_256/imagenet/"
    model_name: str = "vit_base_patch16_224"
    model_key: str = "vit_base"
    batch_size: int = 100
    num_workers: int = field(default_factory=lambda: min(4, os.cpu_count() or 2))
    use_amp: bool = True
    max_batches: int | None = 1
    _device: torch.device = field(default=None, init=False, repr=False)

    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device
