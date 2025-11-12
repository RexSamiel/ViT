# runner.py
# Reusable Vision Transformer runner + standalone script mode

import torch
import functools
import timm
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataclasses import asdict

from src.config.settings import Config
from src.data.imagenet_loader import ImageNetValDataset
from src.utils.logits import FaultFreeLogits
from src.utils.helper import SUPPORTED_MODELS


class MetricsTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total_loss: float = 0.0
        self.total_samples: int = 0
        self.top1_correct: float = 0.0
        self.top5_correct: float = 0.0
        self.sdc_rates: list[torch.Tensor] = []
        self.sdc_magnitudes: list[torch.Tensor] = []
        self.pred_sdc_count: int = 0
        self.pred_top5_sdc_count: int = 0

    def update_accuracy(
        self, outputs: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor
    ) -> None:
        batch_size: int = labels.size(0)
        predictions: torch.Tensor = outputs.argmax(dim=1)
        self.top1_correct += (predictions == labels).sum().item()

        top5_predictions: torch.Tensor = outputs.topk(5, dim=1)[1]
        top5_matches: torch.Tensor = (labels.unsqueeze(1) == top5_predictions).any(
            dim=1
        )
        self.top5_correct += top5_matches.sum().item()

        self.total_loss += loss.item() * batch_size
        self.total_samples += batch_size

    def calculate_sdc(
        self, faulty_logits: torch.Tensor, ff_logits: torch.Tensor
    ) -> None:
        """
        Strict logit SDC and MSDC calculation (no tolerance)
        """
        diff: torch.Tensor = ff_logits - faulty_logits
        sdc_rate: torch.Tensor = (diff != 0).float().mean(dim=1)
        sdc_magnitude: torch.Tensor = diff.abs().mean(dim=1)
        self.sdc_rates.append(sdc_rate.cpu())
        self.sdc_magnitudes.append(sdc_magnitude.cpu())

    def calculate_top(
        self, faulty_logits: torch.Tensor, ff_logits: torch.Tensor
    ) -> None:
        """
        Calculate top-1 and top-5 prediction changes
        """
        # Top-1
        faulty_top1 = faulty_logits.argmax(dim=1)
        ff_top1 = ff_logits.argmax(dim=1)
        self.pred_sdc_count += (faulty_top1 != ff_top1).sum().item()

        # Top-5
        faulty_top5 = faulty_logits.topk(5, dim=1)[1]
        ff_top5 = ff_logits.topk(5, dim=1)[1]
        # Count samples where top-5 differs
        pred_top5_changed = ~(
            (ff_top5.unsqueeze(1) == faulty_top5.unsqueeze(0)).any(dim=1)
        ).any(dim=1)
        self.pred_top5_sdc_count += pred_top5_changed.sum().item()

    def get_results(self) -> dict[str, float | int] | None:
        if self.total_samples == 0:
            return None

        results: dict[str, float | int] = {
            "samples": self.total_samples,
            "top1_acc": 100 * self.top1_correct / self.total_samples,
            "top5_acc": 100 * self.top5_correct / self.total_samples,
            "avg_loss": self.total_loss / self.total_samples,
            "logit_sdc_rate": 100 * torch.cat(self.sdc_rates).mean().item()
            if self.sdc_rates
            else 0.0,
            "msdc_avg": torch.cat(self.sdc_magnitudes).mean().item()
            if self.sdc_magnitudes
            else 0.0,
            "pred_sdc_rate": 100 * self.pred_sdc_count / self.total_samples,
            "pred_top5_sdc_rate": 100 * self.pred_top5_sdc_count / self.total_samples,
        }

        return results


class ModelEvaluator:
    def __init__(self, config: Config, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.device = config.device
        self.model = self._load_model()
        self.dataloader = self._create_dataloader()
        self.criterion = nn.CrossEntropyLoss()
        self.ff_logits = FaultFreeLogits(config.model_key)

    def _load_model(self):
        if self.verbose:
            print(f"Loading model: {self.config.model_name}")
        model = timm.create_model(self.config.model_name, pretrained=True).to(
            self.device
        )
        model.eval()
        if self.verbose:
            print(f"✓ Model loaded successfully on {self.device}")
        return model

    def _create_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        dataset = ImageNetValDataset(self.config.root_dir, "val", transform)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    @functools.lru_cache(maxsize=None)
    def cached_batches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        max_batches: int | None,
    ):
        loader = DataLoader(
            self.dataloader.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.dataloader.pin_memory,
        )
        batches = []
        for i, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device=device, dtype=dtype, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            batches.append((imgs, labels))
            if max_batches is not None and (i + 1) >= max_batches:
                break
        return tuple(batches)

    def clear_cache(self):
        self.cached_batches.cache_clear()


class Runner:
    """Reusable class for evaluating models."""

    def __init__(self, config: Config | None = None, verbose: bool = False):
        self.config = config or Config()
        self.verbose = verbose
        self.evaluator = ModelEvaluator(self.config, verbose=verbose)

    def run(
        self, compute_metrics: bool = True, save_logits: bool = False
    ) -> dict | None:
        tracker = MetricsTracker()
        logits_buffer, labels_buffer = [], []

        dtype = (
            torch.float16
            if (self.config.use_amp and self.config.device.type == "cuda")
            else torch.float32
        )
        batches = self.evaluator.cached_batches(
            self.config.batch_size, dtype, self.config.device, self.config.max_batches
        )

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(batches):
                with torch.autocast(
                    device_type="cuda",
                    enabled=(self.config.use_amp and self.config.device.type == "cuda"),
                ):
                    outputs = self.evaluator.model(images)
                    loss = self.evaluator.criterion(outputs, labels)

                if save_logits:
                    logits_buffer.append(outputs.cpu())
                    labels_buffer.append(labels.cpu())

                if compute_metrics:
                    tracker.update_accuracy(outputs, labels, loss)

                # Always compute SDC and prediction changes
                ff_logits = self.evaluator.ff_logits.get_batch(
                    batch_idx,
                    self.config.batch_size,
                    labels.size(0),
                    self.config.device,
                )
                if ff_logits is not None:
                    tracker.calculate_sdc(outputs, ff_logits)
                    tracker.calculate_top(outputs, ff_logits)

        if save_logits and logits_buffer:
            self.evaluator.ff_logits.save(logits_buffer, labels_buffer)

        results = tracker.get_results()
        if results:
            self._print_results(results)
        return results

    def _print_results(self, results: dict[str, float]) -> None:
        if not self.verbose:
            return
        print("\n" + "=" * 50)
        print(f"RESULTS for {self.config.model_key} ({self.config.model_name})")
        print("=" * 50)
        print(f"Samples:        {results['samples']}")
        print(f"Top-1 Accuracy: {results['top1_acc']:.2f}%")
        print(f"Top-5 Accuracy: {results['top5_acc']:.2f}%")
        print(f"Average Loss:   {results['avg_loss']:.4f}")
        print(f"Logit SDC Rate:  {results['logit_sdc_rate']:.2f}%")
        print(f"MSDC Average:    {results['msdc_avg']:.6f}")
        print(f"Top-1 Prediction SDC: {results['pred_sdc_rate']:.2f}%")
        print(f"Top-5 Prediction SDC: {results['pred_top5_sdc_rate']:.2f}%")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    print("Running ViT Runner in standalone mode (fault-free).")

    config = Config()
    config.model_key = "vit_base"
    config.model_name = SUPPORTED_MODELS[config.model_key]

    print(f"Using default config:\n{asdict(config)}\n")

    runner = Runner(config)
    runner.run(compute_metrics=True, save_logits=False)

