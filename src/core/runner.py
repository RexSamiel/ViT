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
from src.utils.formatting import print_run_results


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
        self.sdc_20_levels = []
        self.sdc_25_levels = []
        self.sdc_50_levels = []

        self.sdc_magnitudes: list[torch.Tensor] = []

        self.critical_top1_sdc_rates: list[float] = []
        self.critical_top5_sdc_rates: list[float] = []

    def update_accuracy(self, outputs: torch.Tensor, labels: torch.Tensor) -> None:
        batch_size = labels.size(0)
        predictions = outputs.argmax(dim=1)
        self.top1_correct += (predictions == labels).sum().item()

        top5_predictions = outputs.topk(5, dim=1)[1]
        top5_matches = (labels.unsqueeze(1) == top5_predictions).any(dim=1)
        self.top5_correct += top5_matches.sum().item()

        self.total_samples += batch_size

    def _logit_sdc_rate(self, faulty: torch.Tensor, faultfree: torch.Tensor) -> None:
        """Percentage of logits that changed. NaN counts as changed."""
        diff = faultfree - faulty
        sdc_rate = (diff != 0).float().mean(dim=1)
        self.sdc_rates.append(sdc_rate.detach().cpu())

    def _sdc_magnitude(self, faulty: torch.Tensor, faultfree: torch.Tensor) -> None:
        """Mean absolute difference between faulty and faultfree logits per sample."""
        diff = faultfree - faulty
        magnitude = diff.abs().mean(dim=1)
        self.sdc_magnitudes.append(magnitude.detach().cpu())

    def _relative_sdc(self, faulty: torch.Tensor, faultfree: torch.Tensor) -> None:
        """
        Percentage of samples where relative change >= threshold.
        relative_change = mean(|diff|) / mean(|faultfree|).
        Samples with zero mean faultfree logits excluded (division by zero).
        """
        diff = faultfree - faulty
        abs_diff_mean = diff.abs().mean(dim=1)
        abs_ff_mean = faultfree.abs().mean(dim=1)

        nonzero_mask = abs_ff_mean != 0
        if not nonzero_mask.any():
            return

        relative_change = abs_diff_mean[nonzero_mask] / abs_ff_mean[nonzero_mask]

        sdc_levels = (
            torch.stack(
                [
                    (relative_change >= 0.01).float(),
                    (relative_change >= 0.05).float(),
                    (relative_change >= 0.10).float(),
                    (relative_change >= 0.15).float(),
                    (relative_change >= 0.20).float(),
                    (relative_change >= 0.25).float(),
                    (relative_change >= 0.50).float(),
                ],
                dim=0,
            )
            .detach()
            .cpu()
        )

        self.sdc_1_levels.append(sdc_levels[0])
        self.sdc_5_levels.append(sdc_levels[1])
        self.sdc_10_levels.append(sdc_levels[2])
        self.sdc_15_levels.append(sdc_levels[3])
        self.sdc_20_levels.append(sdc_levels[4])
        self.sdc_25_levels.append(sdc_levels[5])
        self.sdc_50_levels.append(sdc_levels[6])

    def _critical_top1_sdc(
        self, faulty: torch.Tensor, faultfree: torch.Tensor
    ) -> float:
        """Samples where faultfree top-1 logit changed AND prediction changed."""
        pred_faulty = faulty.argmax(dim=1)
        pred_ff = faultfree.argmax(dim=1)
        pred_changed = pred_faulty != pred_ff

        batch_idx = torch.arange(faulty.size(0), device=faulty.device)
        ff_top1_in_faulty = faulty[batch_idx, pred_ff]
        ff_top1_in_faultfree = faultfree[batch_idx, pred_ff]
        logit_changed = ff_top1_in_faulty != ff_top1_in_faultfree

        return (logit_changed & pred_changed).float().mean().item()

    def _critical_top5_sdc(
        self, faulty: torch.Tensor, faultfree: torch.Tensor
    ) -> float:
        """Samples where any faultfree top-5 logit changed AND top-5 set changed."""
        top5_faulty = faulty.topk(5, dim=1)[1]
        top5_ff = faultfree.topk(5, dim=1)[1]

        top5_ff_sorted, _ = top5_ff.sort(dim=1)
        top5_faulty_sorted, _ = top5_faulty.sort(dim=1)
        set_changed = (top5_ff_sorted != top5_faulty_sorted).any(dim=1)

        ff_top5_in_faultfree = torch.gather(faultfree, 1, top5_ff)
        ff_top5_in_faulty = torch.gather(faulty, 1, top5_ff)
        logits_changed = (ff_top5_in_faultfree != ff_top5_in_faulty).any(dim=1)

        return (logits_changed & set_changed).float().mean().item()

    def _nan_adjustment(self, nan_mask: torch.Tensor, has_valid: bool) -> None:
        """
        NaN outputs = catastrophic failure = 100% critical SDC.
        Adjusts critical rates: weighted average of valid samples + NaN samples (100%).
        """
        num_nan = nan_mask.sum().item()
        total = nan_mask.size(0)

        if has_valid:
            num_valid = total - num_nan
            self.critical_top1_sdc_rates[-1] = (
                self.critical_top1_sdc_rates[-1] * num_valid + num_nan
            ) / total
            self.critical_top5_sdc_rates[-1] = (
                self.critical_top5_sdc_rates[-1] * num_valid + num_nan
            ) / total
        else:
            self.critical_top1_sdc_rates.append(1.0)
            self.critical_top5_sdc_rates.append(1.0)

    def update_sdc(
        self,
        faulty_logits: torch.Tensor,
        ff_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Update all SDC metrics comparing faulty vs fault-free logits."""
        # Align batch sizes
        batch_size = min(faulty_logits.size(0), ff_logits.size(0))
        faulty_all = faulty_logits[:batch_size]
        faultfree_all = ff_logits[:batch_size]

        # Separate valid (non-NaN) samples
        nan_mask = torch.isnan(faulty_all).all(dim=1)
        has_valid = (~nan_mask).any()

        # Logit SDC - uses all samples (NaN = 100% changed)
        self._logit_sdc_rate(faulty_all, faultfree_all)

        # Other metrics - only valid samples
        if has_valid:
            faulty = faulty_all[~nan_mask]
            faultfree = faultfree_all[~nan_mask]

            self._sdc_magnitude(faulty, faultfree)
            self._relative_sdc(faulty, faultfree)
            self.critical_top1_sdc_rates.append(
                self._critical_top1_sdc(faulty, faultfree)
            )
            self.critical_top5_sdc_rates.append(
                self._critical_top5_sdc(faulty, faultfree)
            )

        # NaN = catastrophic failure = 100% critical
        if nan_mask.any():
            self._nan_adjustment(nan_mask, has_valid)

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
            results["critical_top1_sdc_rate"] = 0.0
            results["critical_top5_sdc_rate"] = 0.0
            results["sdc_1pct"] = 0.0
            results["sdc_5pct"] = 0.0
            results["sdc_10pct"] = 0.0
            results["sdc_15pct"] = 0.0
            results["sdc_20pct"] = 0.0
            results["sdc_25pct"] = 0.0
            results["sdc_50pct"] = 0.0
            return results

        results["logit_sdc_rate"] = 100 * torch.cat(self.sdc_rates).mean().item()

        if len(self.sdc_magnitudes) > 0:
            results["msdc_avg"] = torch.cat(self.sdc_magnitudes).mean().item()
        else:
            results["msdc_avg"] = float("nan")

        if len(self.critical_top1_sdc_rates) > 0:
            results["critical_top1_sdc_rate"] = (
                100
                * sum(self.critical_top1_sdc_rates)
                / len(self.critical_top1_sdc_rates)
            )
            results["critical_top5_sdc_rate"] = (
                100
                * sum(self.critical_top5_sdc_rates)
                / len(self.critical_top5_sdc_rates)
            )
        else:
            results["critical_top1_sdc_rate"] = 0.0
            results["critical_top5_sdc_rate"] = 0.0

        if len(self.sdc_1_levels) > 0:
            results["sdc_1pct"] = 100 * torch.cat(self.sdc_1_levels).mean().item()
            results["sdc_5pct"] = 100 * torch.cat(self.sdc_5_levels).mean().item()
            results["sdc_10pct"] = 100 * torch.cat(self.sdc_10_levels).mean().item()
            results["sdc_15pct"] = 100 * torch.cat(self.sdc_15_levels).mean().item()
            results["sdc_20pct"] = 100 * torch.cat(self.sdc_20_levels).mean().item()
            results["sdc_25pct"] = 100 * torch.cat(self.sdc_25_levels).mean().item()
            results["sdc_50pct"] = 100 * torch.cat(self.sdc_50_levels).mean().item()
        else:
            results["sdc_1pct"] = float("nan")
            results["sdc_5pct"] = float("nan")
            results["sdc_10pct"] = float("nan")
            results["sdc_15pct"] = float("nan")
            results["sdc_20pct"] = float("nan")
            results["sdc_25pct"] = float("nan")
            results["sdc_50pct"] = float("nan")

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
            persistent_workers=self.config.num_workers > 0,
        )

    @functools.lru_cache(maxsize=None)
    def cached_batches(
        self,
        batch_size: int,
        device: torch.device,
        max_batches: int | None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """
        Load and cache data batches in memory for repeated evaluation.
        All batches are stored as float32 for consistency.
        """
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

        batches = self.evaluator.cached_batches(
            self.config.batch_size, self.config.device, self.config.max_batches
        )

        start_time = time.perf_counter()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(batches):
                with torch.autocast(
                    device_type=self.config.device.type, enabled=use_amp_flag
                ):
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
        print_run_results(results, self.config.model_key, self.config.model_name)
