"""Model loading and data management."""

import functools

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
from pathlib import Path

from core.data import ImageNetDataset
from core.config import ModelConfig, SUPPORTED_MODELS, logits_path


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

        self.model_name = SUPPORTED_MODELS.get(name, name)

        self.net = self._load_model()
        self.dataloader = self._create_dataloader()
        n_samples = config.batch_size * config.max_batches if config and config.max_batches else None
        split = "train" if (config and config.use_train) else "val"
        self._logits_cache = LogitsCache(name, n_samples, split=split)

    def _load_model(self) -> nn.Module:
        """Load pretrained model."""
        if self.verbose:
            print(f"Loading model: {self.model_name}")

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        model = timm.create_model(self.model_name, pretrained=True)
        model = model.to(self.config.device)
        model.eval()

        if self.config.device.type == "cuda":
            with torch.inference_mode():
                dummy = torch.randn(1, 3, 224, 224, device=self.config.device)
                _ = model(dummy)
            torch.cuda.empty_cache()

        return model

    def _create_dataloader(self) -> DataLoader:
        """Create dataloader (validation or training based on config)."""
        split = "train" if self.config.use_train else "val"
        is_training = self.config.use_train

        try:
            data_cfg = resolve_data_config(self.net.pretrained_cfg)
            transform = create_transform(is_training=is_training, **data_cfg)
        except Exception:
            from torchvision import transforms

            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        dataset = ImageNetDataset(
            self.config.data_root,
            transform=transform,
            split=split,
        )

        if self.verbose:
            print(f"Using {split} data: {len(dataset)} samples")

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=is_training,
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

        n_samples = sum(l.shape[0] for l in logits_list)
        self._logits_cache.save(logits_list, labels_list, n_samples)
        print(f"\nBaseline saved: {len(batches)} batches, {n_samples} samples")


class LogitsCache:
    """Cache for fault-free logits (baseline for SDC computation)."""

    def __init__(self, model_key: str, n_samples: int | None, split: str = "val"):
        self.model_key = model_key
        self._n_samples = n_samples
        self._split = split
        self.data = None
        self._loaded = False

    def _check_meta(self, data: dict, path) -> bool:
        """Return False and print a rejection message if metadata mismatches."""
        meta = data.get("meta")
        if meta is None:
            print(f"✗ Logits file {path} has no metadata — re-save with current code.")
            return False
        if meta.get("split") != self._split:
            print(
                f"✗ Logits split mismatch: file='{meta.get('split')}' "
                f"expected='{self._split}'. Re-save with correct split."
            )
            return False
        if self._n_samples is not None and meta.get("n_samples") != self._n_samples:
            print(
                f"✗ Logits sample count mismatch: file={meta.get('n_samples')} "
                f"expected={self._n_samples}. Re-save with correct batch_size/max_batches."
            )
            return False
        return True

    def _load(self):
        """Lazy-load logits on first access."""
        if self._loaded:
            return
        self._loaded = True
        if self._n_samples is not None:
            path = logits_path(self.model_key, self._n_samples)
            if path.exists():
                data = torch.load(path, weights_only=False)
                if self._check_meta(data, path):
                    self.data = data
                    print(f"✓ Fault-free logits loaded from {path}")
        else:
            import re
            candidates = sorted(
                (logits_path(self.model_key, 1).parent).glob("*_samples.pt"),
                key=lambda p: int(m.group(1)) if (m := re.search(r"(\d+)_samples", p.name)) else 0,
                reverse=True,
            )
            for candidate in candidates:
                data = torch.load(candidate, weights_only=False)
                if self._check_meta(data, candidate):
                    self.data = data
                    print(f"✓ Fault-free logits loaded from {candidate}")
                    break

    def save(self, logits: list, labels: list, n_samples: int):
        """Save fault-free logits to data/{model}/logits/{n_samples}_samples.pt."""
        path = logits_path(self.model_key, n_samples)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "logits": torch.cat(logits),
            "labels": torch.cat(labels),
            "meta": {"n_samples": n_samples, "split": self._split},
        }, path)
        print(f"✓ Saved fault-free logits to {path}")

    def preload_to_device(self, device: torch.device):
        """Move all logits to GPU once — avoids per-batch CPU→GPU transfers."""
        self._load()
        if self.data is not None and self.data["logits"].device != device:
            self.data["logits"] = self.data["logits"].to(device)

    def get_batch(self, batch_idx: int, batch_size: int, device: torch.device):
        """Get fault-free logits for a batch."""
        self._load()
        if self.data is None:
            raise RuntimeError("Fault-free logits not available. Run baseline first.")

        start = batch_idx * batch_size
        end = start + batch_size
        t = self.data["logits"][start:end]
        return t if t.device == device else t.to(device)

    @property
    def available(self) -> bool:
        self._load()
        return self.data is not None
