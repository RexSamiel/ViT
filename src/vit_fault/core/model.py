"""Model loading and data management."""

import os
import functools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
from dataclasses import dataclass, field
from pathlib import Path

from vit_fault.core.data import ImageNetDataset


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


@dataclass
class ModelConfig:
    """Configuration for model and data loading."""

    data_root: str = "/run/media/samiel/K_USB_256/imagenet/"
    batch_size: int = 100
    max_batches: int | None = 1
    num_workers: int = field(default_factory=lambda: min(4, os.cpu_count() or 2))
    device: torch.device = field(default_factory=lambda: torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ))


class Model:
    """ViT model wrapper with data loading."""

    def __init__(
        self,
        name: str,
        config: ModelConfig | None = None,
        verbose: bool = True,
    ):
        """Load a pretrained ViT model.

        Args:
            name: Model key (e.g., "vit_tiny") or full timm name
            config: Optional configuration, uses defaults if None
            verbose: Print loading info
        """
        self.name = name
        self.config = config or ModelConfig()
        self.verbose = verbose

        # Resolve model name
        self.model_name = SUPPORTED_MODELS.get(name, name)

        # Load model
        self.net = self._load_model()
        self.dataloader = self._create_dataloader()
        self._logits_cache = LogitsCache(name)

    def _load_model(self) -> nn.Module:
        """Load pretrained model."""
        if self.verbose:
            print(f"Loading model: {self.model_name}")

        model = timm.create_model(self.model_name, pretrained=True)
        model = model.to(self.config.device)
        model.eval()

        # Warmup for CUDA
        if self.config.device.type == "cuda":
            with torch.inference_mode():
                dummy = torch.randn(1, 3, 224, 224, device=self.config.device)
                _ = model(dummy)
            torch.cuda.empty_cache()

        return model

    def _create_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        try:
            data_cfg = resolve_data_config(self.net.pretrained_cfg)
            transform = create_transform(is_training=False, **data_cfg)
        except Exception:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        dataset = ImageNetDataset(self.config.data_root, transform=transform)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    @functools.lru_cache(maxsize=1)
    def get_batches(self) -> tuple:
        """Get cached data batches."""
        batches = []
        for i, (images, labels) in enumerate(self.dataloader):
            images = images.to(self.config.device, non_blocking=True)
            labels = labels.to(self.config.device, non_blocking=True)
            batches.append((images, labels))

            if self.config.max_batches and (i + 1) >= self.config.max_batches:
                break

        return tuple(batches)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run inference."""
        with torch.inference_mode():
            return self.net(images)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward(images)

    @property
    def ff_logits(self) -> "LogitsCache":
        """Access fault-free logits cache."""
        return self._logits_cache

    def save_baseline(self) -> None:
        """Run model on all batches and save fault-free logits for SDC comparison.

        This must be run once before fault injection experiments to establish
        the baseline logits that SDC metrics are computed against.
        """
        print(f"Computing fault-free logits for {self.name}...")
        logits_list = []
        labels_list = []

        batches = self.get_batches()
        for i, (images, labels) in enumerate(batches):
            print(f"  Batch {i + 1}/{len(batches)}", end="\r")
            with torch.inference_mode():
                outputs = self.net(images)
            logits_list.append(outputs.cpu())
            labels_list.append(labels.cpu())

        self._logits_cache.save(logits_list, labels_list)
        print(f"\nBaseline saved: {len(batches)} batches, {sum(l.shape[0] for l in logits_list)} samples")


class LogitsCache:
    """Cache for fault-free logits (baseline for SDC computation)."""

    def __init__(self, model_key: str):
        self.path = Path("data/logits") / f"ff_logits_{model_key}.pt"
        self.data = None
        self._load()

    def _load(self):
        if self.path.exists():
            self.data = torch.load(self.path, weights_only=False)
            print(f"✓ Fault-free logits loaded from {self.path}")

    def save(self, logits: list, labels: list):
        """Save fault-free logits."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "logits": torch.cat(logits),
            "labels": torch.cat(labels),
        }, self.path)
        print(f"✓ Saved fault-free logits to {self.path}")

    def get_batch(self, batch_idx: int, batch_size: int, device: torch.device):
        """Get fault-free logits for a batch."""
        if self.data is None:
            raise RuntimeError("Fault-free logits not available. Run baseline first.")

        start = batch_idx * batch_size
        end = start + batch_size
        return self.data["logits"][start:end].to(device)

    @property
    def available(self) -> bool:
        return self.data is not None
