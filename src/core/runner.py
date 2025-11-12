import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataclasses import dataclass
import timm
import functools
import typing
import os
import time

from src.utils.logits import FaultFreeLogits
from src.data.imagenet_loader import ImageNetValDataset


class MetricsTracker:
    """Tracks accuracy, loss, and SDC metrics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total_loss: float = 0.0
        self.total_samples: int = 0
        self.top1_correct: float = 0.0
        self.top5_correct: float = 0.0
        self.sdc_rates: list[torch.Tensor] = []
        self.sdc_magnitudes: list[torch.Tensor] = []
        self.pred_sdc_rates: list[float] = []
        self.pred_top5_sdc_rates: list[float] = []

    def update_accuracy(
        self, outputs: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor
    ) -> None:
        batch_size = labels.size(0)
        predictions = outputs.argmax(dim=1)
        self.top1_correct += (predictions == labels).sum().item()

        top5_predictions = outputs.topk(5, dim=1)[1]
        top5_matches = (labels.unsqueeze(1) == top5_predictions).any(dim=1)
        self.top5_correct += top5_matches.sum().item()

        self.total_loss += loss.item() * batch_size
        self.total_samples += batch_size

    def update_sdc(
        self,
        faulty_logits: torch.Tensor,
        ff_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        # Logit-level SDC
        diff = ff_logits - faulty_logits
        threshold = 1e-6
        sdc_rate = (diff.abs() > threshold).float().mean(dim=1)
        sdc_magnitude = diff.abs().mean(dim=1)
        self.sdc_rates.append(sdc_rate.cpu())
        self.sdc_magnitudes.append(sdc_magnitude.cpu())

        # Get predictions
        pred_faulty = faulty_logits.argmax(dim=1)
        pred_ff = ff_logits.argmax(dim=1)

        # Top-1 SDC: predictions differ
        pred_changed = pred_faulty != pred_ff
        pred_sdc = pred_changed.float().mean()
        self.pred_sdc_rates.append(pred_sdc.item())

        # Top-5 SDC: Check if the top-5 sets differ
        top5_faulty = faulty_logits.topk(5, dim=1)[1]
        top5_ff = ff_logits.topk(5, dim=1)[1]

        # For each sample, check if top-5 sets are different
        top5_changed = torch.zeros(
            len(top5_faulty), dtype=torch.bool, device=faulty_logits.device
        )
        for i in range(len(top5_faulty)):
            # Compare sets (order doesn't matter)
            set_ff = set(top5_ff[i].tolist())
            set_faulty = set(top5_faulty[i].tolist())
            top5_changed[i] = set_ff != set_faulty

        pred_top5_sdc = top5_changed.float().mean()
        self.pred_top5_sdc_rates.append(pred_top5_sdc.item())

    def get_results(self) -> dict[str, float] | None:
        if self.total_samples == 0:
            return None

        results = {
            "samples": self.total_samples,
            "top1_acc": 100 * self.top1_correct / self.total_samples,
            "top5_acc": 100 * self.top5_correct / self.total_samples,
            "avg_loss": self.total_loss / self.total_samples,
        }

        if self.sdc_rates:
            results["logit_sdc_rate"] = 100 * torch.cat(self.sdc_rates).mean().item()
            results["msdc_avg"] = torch.cat(self.sdc_magnitudes).mean().item()
            results["pred_sdc_rate"] = (
                100 * sum(self.pred_sdc_rates) / len(self.pred_sdc_rates)
            )
            results["pred_top5_sdc_rate"] = (
                100 * sum(self.pred_top5_sdc_rates) / len(self.pred_top5_sdc_rates)
            )
        else:
            results["logit_sdc_rate"] = 0.0
            results["msdc_avg"] = 0.0
            results["pred_sdc_rate"] = 0.0
            results["pred_top5_sdc_rate"] = 0.0

        return results


@dataclass
class RunnerConfig:
    root_dir: str = "/home/samiel/Documents/thesis/ViT/data/imagenet"
    model_name: str = "vit_base_patch16_224"
    model_key: str = "vit_base"
    batch_size: int = 128
    num_workers: int = min(4, os.cpu_count() or 2)
    use_amp: bool = True
    max_batches: int | None = 50

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelEvaluator:
    """Handles model loading and data preparation."""

    def __init__(self, config: RunnerConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.device = config.device
        self.model = self._load_model()
        self.dataloader = self._create_dataloader()
        self.criterion = nn.CrossEntropyLoss()
        self.ff_logits = FaultFreeLogits(config.model_key)

    def _load_model(self) -> nn.Module:
        if self.verbose:
            print(f"Loading model: {self.config.model_name}")
        model = timm.create_model(self.config.model_name, pretrained=True).to(
            self.config.device
        )
        model.eval()
        if self.verbose:
            print(f"✓ Model loaded successfully on {self.config.device}")
        return model

    def _create_dataloader(self) -> DataLoader:
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
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        dataloader = DataLoader(
            self.dataloader.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.dataloader.pin_memory,
        )
        batches = []
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device=device, dtype=dtype, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            batches.append((images, labels))
            if max_batches is not None and (i + 1) >= max_batches:
                break
        return tuple(batches)

    def clear_cache(self):
        self.cached_batches.cache_clear()


class Runner:
    def __init__(self, config: RunnerConfig, verbose=True):
        self.config = config
        self.verbose = verbose
        self.evaluator = ModelEvaluator(config, verbose)

    def run(
        self,
        compute_metrics: bool = True,
        save_logits: bool = False,
        verbose: bool = True,
        compute_sdc: bool = False,
    ) -> dict | None:
        tracker = MetricsTracker()
        logits_buffer, labels_buffer = [], []

        dtype = (
            torch.float16
            if self.config.use_amp and self.config.device.type == "cuda"
            else torch.float32
        )
        batches = self.evaluator.cached_batches(
            self.config.batch_size, dtype, self.config.device, self.config.max_batches
        )

        start_time = time.perf_counter()
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

                    # Compute SDC if requested and fault-free logits available
                    if compute_sdc and self.evaluator.ff_logits.available:
                        ff_batch = self.evaluator.ff_logits.get_batch(
                            batch_idx,
                            self.config.batch_size,
                            outputs.size(0),
                            self.config.device,
                        )
                        tracker.update_sdc(outputs, ff_batch, labels)

        if save_logits and logits_buffer:
            self.evaluator.ff_logits.save(logits_buffer, labels_buffer)

        results = tracker.get_results()
        if verbose and results:
            self._print_results(results)
            print(f"⏱️ Total runtime: {time.perf_counter() - start_time:.2f} seconds\n")

        return results

    def _print_results(self, results: dict[str, float]) -> None:
        print("\n" + "=" * 50)
        print(f"RESULTS for {self.config.model_key} ({self.config.model_name})")
        print("=" * 50)
        print(f"Samples:        {results['samples']}")
        print(f"Top-1 Accuracy: {results['top1_acc']:.2f}%")
        print(f"Top-5 Accuracy: {results['top5_acc']:.2f}%")
        print(f"Average Loss:   {results['avg_loss']:.4f}")

        # Only print SDC metrics if they exist and are non-zero
        if results.get("logit_sdc_rate", 0.0) > 0:
            print(f"Logit SDC Rate:       {results['logit_sdc_rate']:.2f}%")
            print(f"MSDC Average:         {results['msdc_avg']:.6f}")
            print(f"Top-1 Prediction SDC: {results['pred_sdc_rate']:.2f}%")
            print(f"Top-5 Prediction SDC: {results['pred_top5_sdc_rate']:.2f}%")

        print("=" * 50 + "\n")
