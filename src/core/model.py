import functools
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timm
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
from typing import Optional

from src.config.settings import Config
from src.core.library.imagenet_loader import ImageNetValDataset
from src.core.library.logits import FaultFreeLogits


class ModelRunner:
    """Handles model loading, data loading, and inference caching."""

    def __init__(self, config: Config, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.device = config.device

        self.model = self._load_model()
        self.dataloader = self._create_dataloader()
        self.ff_logits = FaultFreeLogits(config.model_key)

    def _load_model(self) -> nn.Module:
        """Load pretrained model and set to eval mode."""
        if self.verbose:
            print(f"Loading model: {self.config.model_name}")

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        model = timm.create_model(self.config.model_name, pretrained=True).to(
            self.config.device
        )
        model.eval()

        if self.config.device.type == "cuda":
            dummy = torch.randn(1, 3, 224, 224, device=self.config.device)
            with torch.inference_mode():
                _ = model(dummy)
            del dummy
            torch.cuda.empty_cache()

        if self.verbose:

    def _get_transform(self):
        """Get appropriate transform for the model."""
        try:
            data_cfg = resolve_data_config(self.model.pretrained_cfg)
            return create_transform(is_training=False, **data_cfg)
        except Exception:
            return transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def _create_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        transform = self._get_transform()
        dataset = ImageNetValDataset(self.config.root_dir, "val", transform)

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
        )

    @functools.lru_cache(maxsize=None)
    def cached_batches(
        self,
        batch_size: int,
        device: torch.device,
        max_batches: int | None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Load and cache data batches in memory for repeated evaluation."""
        dataloader = DataLoader(
            self.dataloader.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.dataloader.pin_memory,
        )

        batches = []
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            batches.append((images, labels))

            if max_batches is not None and (i + 1) >= max_batches:
                break

        return tuple(batches)

    def clear_cache(self) -> None:
        """Clear the batch cache."""
        self.cached_batches.cache_clear()

    def get_batches(self) -> tuple:
        """Get cached batches for evaluation."""
        return self.cached_batches(
            self.config.batch_size,
            self.device,
            self.config.max_batches,
        )

    def inference(self, images: torch.Tensor, use_amp: bool = True) -> torch.Tensor:
        """Run inference on a batch of images."""
        with torch.inference_mode():
            with torch.autocast(
                device_type=self.config.device.type,
                enabled=use_amp and self.config.device.type == "cuda",
            ):
                return self.model(images)
