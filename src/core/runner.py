import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataclasses import dataclass
import timm
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
import functools
import time

from src.utils.logits import FaultFreeLogits
from src.data.imagenet_loader import ImageNetValDataset
from src.config.settings import Config


class MetricsTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total_samples: int = 0
        self.top1_correct: float = 0.0
        self.top5_correct: float = 0.0

        self.sdc_rates: list[torch.Tensor] = []
        self.sdc_1_levels = []
        self.sdc_5_levels = []
        self.sdc_10_levels = []
        self.sdc_15_levels = []
        self.sdc_25_levels = []
        self.sdc_50_levels = []
        self.sdc_75_levels = []
        self.rel_changes = []

        self.sdc_magnitudes: list[torch.Tensor] = []
        self.pred_sdc_rates: list[float] = []
        self.pred_top5_sdc_rates: list[float] = []

    def update_accuracy(self, outputs: torch.Tensor, labels: torch.Tensor) -> None:
        batch_size = labels.size(0)
        predictions = outputs.argmax(dim=1)
        self.top1_correct += (predictions == labels).sum().item()

        top5_predictions = outputs.topk(5, dim=1)[1]
        top5_matches = (labels.unsqueeze(1) == top5_predictions).any(dim=1)
        self.top5_correct += top5_matches.sum().item()

        self.total_samples += batch_size

    def update_sdc(
        self,
        faulty_logits: torch.Tensor,
        ff_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        diff = ff_logits - faulty_logits

        sdc_rate = (diff != 0).float().mean(dim=1)
        self.sdc_rates.append(sdc_rate.cpu())

        sdc_magnitude = diff.abs().mean(dim=1)
        self.sdc_magnitudes.append(sdc_magnitude.cpu())

        print(sdc_rate)
        relative_change = diff.abs().mean(dim=1) / ff_logits.abs().mean(dim=1)

        sdc_1 = (relative_change >= 0.01).float()
        sdc_5 = (relative_change >= 0.05).float()
        sdc_10 = (relative_change >= 0.10).float()
        sdc_15 = (relative_change >= 0.15).float()
        sdc_25 = (relative_change >= 0.25).float()
        sdc_50 = (relative_change >= 0.50).float()
        sdc_75 = (relative_change >= 0.75).float()

        self.sdc_1_levels.append(sdc_1.cpu())
        self.sdc_5_levels.append(sdc_5.cpu())
        self.sdc_10_levels.append(sdc_10.cpu())
        self.sdc_15_levels.append(sdc_15.cpu())
        self.sdc_25_levels.append(sdc_25.cpu())
        self.sdc_50_levels.append(sdc_50.cpu())
        self.sdc_75_levels.append(sdc_75.cpu())

        pred_faulty = faulty_logits.argmax(dim=1)
        pred_ff = ff_logits.argmax(dim=1)
        pred_changed = pred_faulty != pred_ff
        pred_sdc = pred_changed.float().mean()
        self.pred_sdc_rates.append(pred_sdc.item())

        top5_faulty = faulty_logits.topk(5, dim=1)[1]
        top5_ff = ff_logits.topk(5, dim=1)[1]

        top5_changed = torch.zeros(
            len(top5_faulty), dtype=torch.bool, device=faulty_logits.device
        )
        for i in range(len(top5_faulty)):
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
        }

        if len(self.sdc_rates) == 0:
            results["logit_sdc_rate"] = 0.0
            results["msdc_avg"] = 0.0
            results["pred_sdc_rate"] = 0.0
            results["pred_top5_sdc_rate"] = 0.0
            results["sdc_1pct"] = 0.0
            results["sdc_5pct"] = 0.0
            results["sdc_10pct"] = 0.0
            results["sdc_15pct"] = 0.0
            results["sdc_25pct"] = 0.0
            results["sdc_50pct"] = 0.0
            results["sdc_75pct"] = 0.0
            return results

        results["logit_sdc_rate"] = 100 * torch.cat(self.sdc_rates).mean().item()
        results["msdc_avg"] = torch.cat(self.sdc_magnitudes).mean().item()
        results["pred_sdc_rate"] = (
            100 * sum(self.pred_sdc_rates) / len(self.pred_sdc_rates)
        )
        results["pred_top5_sdc_rate"] = (
            100 * sum(self.pred_top5_sdc_rates) / len(self.pred_top5_sdc_rates)
        )

        # Add threshold-based SDC percentages
        results["sdc_1pct"] = 100 * torch.cat(self.sdc_1_levels).mean().item()
        results["sdc_5pct"] = 100 * torch.cat(self.sdc_5_levels).mean().item()
        results["sdc_10pct"] = 100 * torch.cat(self.sdc_10_levels).mean().item()
        results["sdc_15pct"] = 100 * torch.cat(self.sdc_15_levels).mean().item()
        results["sdc_25pct"] = 100 * torch.cat(self.sdc_25_levels).mean().item()
        results["sdc_50pct"] = 100 * torch.cat(self.sdc_50_levels).mean().item()
        results["sdc_75pct"] = 100 * torch.cat(self.sdc_75_levels).mean().item()

        return results


class ModelEvaluator:
    def __init__(self, config: Config, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.device = config.device
        self.model = self._load_model()
        self.dataloader = self._create_dataloader()
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
        try:
            data_cfg = resolve_data_config(self.model.pretrained_cfg)
            transform = create_transform(is_training=False, **data_cfg)
        except Exception:
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
    def __init__(self, config: Config, verbose=True):
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

        unstable_fp16_models = ["beit"]
        use_amp_flag = (
            self.config.use_amp
            and self.config.device.type == "cuda"
            and not any(
                name in self.config.model_name.lower() for name in unstable_fp16_models
            )
        )

        dtype = torch.float16 if use_amp_flag else torch.float32

        batches = self.evaluator.cached_batches(
            self.config.batch_size, dtype, self.config.device, self.config.max_batches
        )

        start_time = time.perf_counter()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(batches):
                with torch.autocast(
                    device_type=self.config.device.type, enabled=use_amp_flag
                ):
                    if not use_amp_flag and images.dtype != torch.float32:
                        images = images.float()

                    labels_clone = labels.clone()

                    outputs = self.evaluator.model(images)
                    nan_logits_mask = torch.isnan(outputs).all(dim=1)
                    num_all_nan = nan_logits_mask.sum().item()
                    if num_all_nan and verbose:
                        print(
                            f"{num_all_nan}/{outputs.size(0)} samples had all-NaN logits in this batch"
                        )
                    labels_clone[nan_logits_mask] = 1001

                if save_logits:
                    logits_buffer.append(outputs.cpu())
                    labels_buffer.append(labels.cpu())

                if compute_metrics:
                    tracker.update_accuracy(outputs, labels_clone)

                    if compute_sdc and self.evaluator.ff_logits.available:
                        ff_batch = self.evaluator.ff_logits.get_batch(
                            batch_idx,
                            self.config.batch_size,
                            outputs.size(0),
                            self.config.device,
                        )
                        tracker.update_sdc(outputs, ff_batch, labels_clone)

        if save_logits and logits_buffer:
            self.evaluator.ff_logits.save(logits_buffer, labels_buffer)

        results = tracker.get_results()
        if verbose and results:
            self._print_results(results)
            print(f"Total runtime: {time.perf_counter() - start_time:.2f} seconds\n")

        return results

    def _print_results(self, results: dict[str, float]) -> None:
        print("\n" + "=" * 50)
        print(f"RESULTS for {self.config.model_key} ({self.config.model_name})")
        print("=" * 50)
        print(f"Samples:        {results['samples']}")
        print(f"Top-1 Accuracy: {results['top1_acc']:.2f}%")
        print(f"Top-5 Accuracy: {results['top5_acc']:.2f}%")

        if results.get("logit_sdc_rate", 0.0) > 0:
            print(f"Logit SDC Rate:       {results['logit_sdc_rate']:.2f}%")
            print(f"MSDC Average:         {results['msdc_avg']:.6f}")
            print(f"SDC ≥1%:              {results['sdc_1pct']:.2f}%")
            print(f"SDC ≥5%:              {results['sdc_5pct']:.2f}%")
            print(f"SDC ≥10%:             {results['sdc_10pct']:.2f}%")
            print(f"SDC ≥15%:             {results['sdc_15pct']:.2f}%")
            print(f"SDC ≥25%:             {results['sdc_25pct']:.2f}%")
            print(f"SDC ≥50%:             {results['sdc_50pct']:.2f}%")
            print(f"SDC ≥75%:             {results['sdc_75pct']:.2f}%")
            print(f"Top-1 Prediction SDC: {results['pred_sdc_rate']:.2f}%")
            print(f"Top-5 Prediction SDC: {results['pred_top5_sdc_rate']:.2f}%")

        print("=" * 50 + "\n")
