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
        self.sdc_20_levels = []
        self.sdc_25_levels = []
        self.sdc_50_levels = []

        self.sdc_magnitudes: list[torch.Tensor] = []

        # Critical SDC: fault-free prediction's logit changed AND this caused prediction to change
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

    def update_sdc(
        self,
        faulty_logits: torch.Tensor,
        ff_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        # Ensure both tensors have the same batch size (handle truncated batches)
        batch_size = min(faulty_logits.size(0), ff_logits.size(0))
        faulty_logits = faulty_logits[:batch_size]
        ff_logits = ff_logits[:batch_size]
        labels = labels[:batch_size]

        # Detect samples with all-NaN faulty logits
        nan_mask = torch.isnan(faulty_logits).all(dim=1)

        # Always compute logit SDC rate (NaN != valid number)
        diff = ff_logits - faulty_logits
        sdc_rate = (diff != 0).float().mean(dim=1)
        self.sdc_rates.append(sdc_rate.cpu())

        # Only compute MSDC and relative change metrics for non-NaN samples
        if (~nan_mask).any():
            valid_faulty = faulty_logits[~nan_mask]
            valid_ff = ff_logits[~nan_mask]

            valid_diff = valid_ff - valid_faulty
            sdc_magnitude = valid_diff.abs().mean(dim=1)
            self.sdc_magnitudes.append(sdc_magnitude.cpu())

            relative_change = valid_diff.abs().mean(dim=1) / valid_ff.abs().mean(dim=1)

            sdc_1 = (relative_change >= 0.01).float()
            sdc_5 = (relative_change >= 0.05).float()
            sdc_10 = (relative_change >= 0.10).float()
            sdc_15 = (relative_change >= 0.15).float()
            sdc_20 = (relative_change >= 0.20).float()
            sdc_25 = (relative_change >= 0.25).float()
            sdc_50 = (relative_change >= 0.50).float()

            self.sdc_1_levels.append(sdc_1.cpu())
            self.sdc_5_levels.append(sdc_5.cpu())
            self.sdc_10_levels.append(sdc_10.cpu())
            self.sdc_15_levels.append(sdc_15.cpu())
            self.sdc_20_levels.append(sdc_20.cpu())
            self.sdc_25_levels.append(sdc_25.cpu())
            self.sdc_50_levels.append(sdc_50.cpu())

        # Critical SDC: Check if fault-free prediction's logit changed AND label changed
        # Only compute for non-NaN samples
        if (~nan_mask).any():
            # Work with valid samples only
            valid_faulty = faulty_logits[~nan_mask]
            valid_ff = ff_logits[~nan_mask]
            valid_diff = diff[~nan_mask]

            # For top-1: Did the FF top-1 class's logit change AND did top-1 prediction change?
            pred_faulty = valid_faulty.argmax(dim=1)
            pred_ff = valid_ff.argmax(dim=1)
            pred_changed = pred_faulty != pred_ff

            # Get the fault-free top-1 prediction for each sample
            # Check if that specific class's logit changed
            batch_indices = torch.arange(
                valid_faulty.size(0), device=valid_faulty.device
            )
            ff_top1_logit_faulty = valid_faulty[batch_indices, pred_ff]
            ff_top1_logit_ff = valid_ff[batch_indices, pred_ff]
            ff_top1_logit_changed = ff_top1_logit_faulty != ff_top1_logit_ff

            # Critical top-1: FF top-1 class's logit changed AND prediction changed
            critical_top1 = (ff_top1_logit_changed & pred_changed).float().mean()
            self.critical_top1_sdc_rates.append(critical_top1.item())

            # For top-5: Did ANY of the FF top-5 classes' logits change AND did top-5 set change?
            top5_faulty = valid_faulty.topk(5, dim=1)[1]
            top5_ff = valid_ff.topk(5, dim=1)[1]

            # Check if top-5 set changed
            top5_changed = torch.zeros(
                len(top5_faulty), dtype=torch.bool, device=valid_faulty.device
            )
            for i in range(len(top5_faulty)):
                set_ff = set(top5_ff[i].tolist())
                set_faulty = set(top5_faulty[i].tolist())
                top5_changed[i] = set_ff != set_faulty

            # Check if ANY of the FF top-5 classes' logits changed
            ff_top5_logits_changed = torch.zeros(
                valid_faulty.size(0), dtype=torch.bool, device=valid_faulty.device
            )
            for i in range(valid_faulty.size(0)):
                for class_idx in top5_ff[i]:
                    if valid_faulty[i, class_idx] != valid_ff[i, class_idx]:
                        ff_top5_logits_changed[i] = True
                        break

            # Critical top-5: At least one FF top-5 class's logit changed AND top-5 set changed
            critical_top5 = (ff_top5_logits_changed & top5_changed).float().mean()
            self.critical_top5_sdc_rates.append(critical_top5.item())

        # Handle samples with all-NaN logits separately
        if nan_mask.any():
            num_nan_samples = nan_mask.sum().item()
            total_samples = nan_mask.size(0)

            # If we had some valid samples, we need to account for NaN samples
            # NaN logits = catastrophic failure = 100% critical corruption for those samples
            if (~nan_mask).any():
                # We already computed critical SDC for valid samples
                # Now we need to blend in the NaN samples (which are 100% critical)
                num_valid = (~nan_mask).sum().item()

                # Last appended values were for valid samples only
                # Adjust them to account for NaN samples being 100% critical
                last_top1 = self.critical_top1_sdc_rates[-1]
                last_top5 = self.critical_top5_sdc_rates[-1]

                # Weighted average: (valid_rate * num_valid + 1.0 * num_nan) / total
                adjusted_top1 = (
                    last_top1 * num_valid + 1.0 * num_nan_samples
                ) / total_samples
                adjusted_top5 = (
                    last_top5 * num_valid + 1.0 * num_nan_samples
                ) / total_samples

                # Replace the last values with adjusted ones
                self.critical_top1_sdc_rates[-1] = adjusted_top1
                self.critical_top5_sdc_rates[-1] = adjusted_top5
            else:
                # All samples have NaN logits - catastrophic failure - 100% critical SDC
                self.critical_top1_sdc_rates.append(1.0)
                self.critical_top5_sdc_rates.append(1.0)

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

        # Handle MSDC separately - might be empty even when sdc_rates is not
        if len(self.sdc_magnitudes) > 0:
            results["msdc_avg"] = torch.cat(self.sdc_magnitudes).mean().item()
        else:
            results["msdc_avg"] = float("nan")

        # Critical SDC rates
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

        # Handle threshold-based SDC percentages - might also be empty
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
        )

    @functools.lru_cache(maxsize=None)
    def cached_batches(
        self,
        batch_size: int,
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
            # Load to device as float32, we'll handle dtype conversion in the run loop
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

        # Get cached batches (always stored as float32)
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
            print(f"SDC ≥20%:             {results['sdc_20pct']:.2f}%")
            print(f"SDC ≥25%:             {results['sdc_25pct']:.2f}%")
            print(f"SDC ≥50%:             {results['sdc_50pct']:.2f}%")
            print(f"Critical Top-1 SDC:   {results['critical_top1_sdc_rate']:.2f}%")
            print(f"Critical Top-5 SDC:   {results['critical_top5_sdc_rate']:.2f}%")

        print("=" * 50 + "\n")
