import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import argparse
import os
from dataclasses import dataclass
import timm
import functools
import typing
import time

from logits import FaultFreeLogits
from dataset_loader import ImageNetValDataset
from fault_injection import inject_fault


SUPPORTED_MODELS: dict[str, str] = {
    # ViT models
    "vit_tiny": "vit_tiny_patch16_224",
    "vit_small": "vit_small_patch16_224",
    "vit_base": "vit_base_patch16_224",
    "vit_large": "vit_large_patch16_224",
    "vit_huge": "vit_huge_patch14_224",
    # Swin models
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "swin_small": "swin_small_patch4_window7_224",
    "swin_base": "swin_base_patch4_window7_224",
    "swin_large": "swin_large_patch4_window7_224",
    # BEiT models
    "beit_base": "beit_base_patch16_224",
    "beit_large": "beit_large_patch16_224",
}


def print_supported_models() -> None:
    print("\n" + "=" * 60)
    print("SUPPORTED MODELS")
    print("=" * 60)

    print("\nVision Transformer (ViT):")
    print("  - vit_tiny")
    print("  - vit_small")
    print("  - vit_base")
    print("  - vit_large")
    print("  - vit_huge")

    print("\nSwin Transformer:")
    print("  - swin_tiny")
    print("  - swin_small")
    print("  - swin_base")
    print("  - swin_large")

    print("\nBEiT:")
    print("  - beit_base")
    print("  - beit_large")

    print("\n" + "=" * 60)
    print("Usage: python script.py --model <model_name> [options]")
    print("Example: python script.py --model vit_base --faultfree --metrics")
    print("=" * 60 + "\n")


@dataclass
class Config:
    # /gpfs/mariana/home/svloor/Documents/vit/data/imagenet"
    root_dir: str = "/home/samiel/Documents/thesis/ViT/data/imagenet"
    model_name: str = "vit_base_patch16_224"
    model_key: str = "vit_base"
    batch_size: int = 128  # Default is 128
    num_workers: int = min(4, os.cpu_count() or 2)
    use_amp: bool = True
    max_batches: int | None = 50  # Default is None

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def update_accuracy(
        self, outputs: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor
    ) -> None:
        """Update accuracy metrics for a batch."""
        batch_size: int = labels.size(0)

        # Top-1 accuracy
        predictions: torch.Tensor = outputs.argmax(dim=1)
        self.top1_correct += (predictions == labels).sum().item()

        # Top-5 accuracy
        top5_predictions: torch.Tensor = outputs.topk(5, dim=1)[1]
        top5_matches: torch.Tensor = (labels.unsqueeze(1) == top5_predictions).any(
            dim=1
        )
        self.top5_correct += top5_matches.sum().item()

        # Loss
        self.total_loss += loss.item() * batch_size
        self.total_samples += batch_size

    def update_sdc(self, faulty_logits: torch.Tensor, ff_logits: torch.Tensor) -> None:
        diff: torch.Tensor = ff_logits - faulty_logits
        sdc_rate: torch.Tensor = (diff != 0).float().mean(dim=1)
        sdc_magnitude: torch.Tensor = diff.abs().mean(dim=1)

        self.sdc_rates.append(sdc_rate.cpu())
        self.sdc_magnitudes.append(sdc_magnitude.cpu())

    def get_results(self) -> dict[str, float | int] | None:
        if self.total_samples == 0:
            return None

        results: dict[str, float | int] = {
            "samples": self.total_samples,
            "top1_acc": 100 * self.top1_correct / self.total_samples,
            "top5_acc": 100 * self.top5_correct / self.total_samples,
            "avg_loss": self.total_loss / self.total_samples,
        }

        if self.sdc_rates:
            sdc_tensor: torch.Tensor = torch.cat(self.sdc_rates)
            msdc_tensor: torch.Tensor = torch.cat(self.sdc_magnitudes)

            results["sdc_rate"] = 100 * sdc_tensor.mean().item()
            results["msdc_avg"] = msdc_tensor.mean().item()

        return results


class ModelEvaluator:
    def __init__(self, config: Config) -> None:
        self.config: Config = config
        self.model: nn.Module = self._load_model()
        self.dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = (
            self._create_dataloader()
        )
        self.criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.ff_logits: FaultFreeLogits = FaultFreeLogits(config.model_key)

    def _load_model(self) -> nn.Module:
        print(f"Loading model: {self.config.model_name}")
        model: nn.Module = timm.create_model(
            self.config.model_name, pretrained=True
        ).to(self.config.device)
        _ = model.eval()
        print(f"✓ Model loaded successfully on {self.config.device}")
        return model

    def _create_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        transform: transforms.Compose = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        dataset: ImageNetValDataset = ImageNetValDataset(
            self.config.root_dir, "val", transform
        )

        return DataLoader[tuple[torch.Tensor, torch.Tensor]](
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    @functools.lru_cache(maxsize=None)
    def _cached_batches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        max_batches: int | None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """
        Build and cache the list of batches already moved to (device, dtype).
        Returns a tuple of (images_tensor, labels_tensor).
        Note: labels are NOT cast to the float dtype (they remain integer type).
        """
        dataset: Dataset[tuple[torch.Tensor, torch.Tensor]] = self.dataloader.dataset
        dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader[
            tuple[torch.Tensor, torch.Tensor]
        ](
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.dataloader.pin_memory,
        )

        batches: list[tuple[torch.Tensor, torch.Tensor]] = []
        batch: tuple[torch.Tensor, torch.Tensor]
        for i, batch in enumerate(dataloader):
            images: torch.Tensor
            labels: torch.Tensor
            images, labels = batch

            images = images.to(device=device, dtype=dtype, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)

            batches.append((images, labels))

            if max_batches is not None and (i + 1) >= max_batches:
                break

        return tuple(batches)

    def clear_prefetch_cache(self) -> None:
        """Clear the cached prefetched batches (frees memory)."""
        try:
            self._cached_batches.cache_clear()
            print("✓ Prefetch cache cleared.")
        except Exception:
            # In case cache doesn't exist or clearing fails
            pass

    def run(
        self,
        mode: str = "faultfree",
        compute_metrics: bool = False,
        save_logits: bool = False,
    ) -> None:
        if mode == "faulty":
            inject_fault(self.model, component_type="attention", verbose=True)
            print("✓ Fault injection applied to attention components")

        metrics: MetricsTracker = MetricsTracker()
        logits_buffer: list[torch.Tensor] = []
        labels_buffer: list[torch.Tensor] = []

        # choose an input dtype for prefetching images: use float16 on CUDA with AMP
        input_dtype: torch.dtype
        if self.config.use_amp and self.config.device.type == "cuda":
            input_dtype = torch.float16
        else:
            input_dtype = torch.float32

        batches: tuple[tuple[torch.Tensor, torch.Tensor], ...] = self._cached_batches(
            self.config.batch_size,
            input_dtype,
            self.config.device,
            self.config.max_batches,
        )

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(batches):
                if self.config.max_batches and batch_idx >= self.config.max_batches:
                    break

                outputs: torch.Tensor
                loss: torch.Tensor
                with torch.autocast(
                    device_type="cuda",
                    enabled=(self.config.use_amp and self.config.device.type == "cuda"),
                ):
                    outputs = typing.cast(torch.Tensor, self.model(images))
                    loss = typing.cast(torch.Tensor, self.criterion(outputs, labels))

                if save_logits and mode == "faultfree":
                    logits_buffer.append(outputs.cpu())
                    labels_buffer.append(labels.cpu())

                if compute_metrics:
                    metrics.update_accuracy(outputs, labels, loss)

                    if mode == "faulty" and self.ff_logits.available:
                        ff_batch: torch.Tensor = typing.cast(
                            torch.Tensor,
                            self.ff_logits.get_batch(
                                batch_idx,
                                self.config.batch_size,
                                outputs.size(0),
                                self.config.device,
                            ),
                        )
                        metrics.update_sdc(outputs, ff_batch)

        if save_logits and logits_buffer:
            typing.cast(typing.Any, self.ff_logits.save)(logits_buffer, labels_buffer)

        if compute_metrics:
            self._print_results(mode, metrics.get_results())

    def _print_results(self, mode: str, results: dict[str, float | int] | None) -> None:
        if results is None:
            print("No samples evaluated")
            return

        print("\n" + "=" * 50)
        print(f"RESULTS - {mode.upper()} MODE")
        print(f"Model: {self.config.model_key} ({self.config.model_name})")
        print("=" * 50)
        print(f"Samples:        {results['samples']}")
        print(f"Top-1 Accuracy: {results['top1_acc']:.2f}%")
        print(f"Top-5 Accuracy: {results['top5_acc']:.2f}%")
        print(f"Average Loss:   {results['avg_loss']:.4f}")

        if "sdc_rate" in results:
            print("\nSDC METRICS")
            print("-" * 50)
            print(f"SDC Rate:            {results['sdc_rate']:.2f}%")
            print(f"SDC Magnitude (avg): {results['msdc_avg']:.4f}")

        print("=" * 50 + "\n")

        output_file: str = f"results_{self.config.model_key}_{mode}.txt"
        with open(output_file, "w") as f:
            f.write(f"Model: {self.config.model_key} ({self.config.model_name})\n")
            f.write(f"Mode: {mode.upper()}\n")
            f.write(f"Samples: {results['samples']}\n")
            f.write(f"Top-1 Accuracy: {results['top1_acc']:.2f}%\n")
            f.write(f"Top-5 Accuracy: {results['top5_acc']:.2f}%\n")
            f.write(f"Average Loss: {results['avg_loss']:.4f}\n")

            if "sdc_rate" in results:
                f.write(f"\nSDC Rate: {results['sdc_rate']:.2f}%\n")
                f.write(f"SDC Magnitude (avg): {results['msdc_avg']:.4f}\n")

        print(f"✓ Results saved to {output_file}")


def main() -> None:
    start_time: float = time.perf_counter()
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Vision Transformer fault injection evaluation"
    )
    _ = parser.add_argument(
        "--model",
        type=str,
        help="Model to evaluate (e.g., vit_base, swin_small, beit_large)",
    )
    _ = parser.add_argument(
        "--faultfree", action="store_true", help="Run fault-free evaluation"
    )
    _ = parser.add_argument(
        "--faulty",
        action="store_true",
        help="Run evaluation with fault injection",
    )
    _ = parser.add_argument(
        "--metrics",
        action="store_true",
        help="Compute accuracy and SDC metrics",
    )
    _ = parser.add_argument(
        "--logits",
        action="store_true",
        help="Save fault-free logits for later comparison",
    )

    args: argparse.Namespace = parser.parse_args()

    model_arg: str | None = typing.cast(str | None, args.model)
    if not model_arg:
        print_supported_models()
        return

    if model_arg not in SUPPORTED_MODELS:
        print(f"\n Error: '{model_arg}' is not a supported model.")
        print_supported_models()
        return

    faultfree_arg: bool = typing.cast(bool, args.faultfree)
    faulty_arg: bool = typing.cast(bool, args.faulty)
    if not (faultfree_arg or faulty_arg):
        print("Please specify --faultfree or --faulty")
        return

    config: Config = Config()
    config.model_key = model_arg
    config.model_name = SUPPORTED_MODELS[model_arg]

    evaluator: ModelEvaluator = ModelEvaluator(config)

    mode: str = "faultfree" if faultfree_arg else "faulty"
    metrics_arg: bool = typing.cast(bool, args.metrics)
    logits_arg: bool = typing.cast(bool, args.logits)
    evaluator.run(mode=mode, compute_metrics=metrics_arg, save_logits=logits_arg)

    end_time: float = time.perf_counter()

    print(f"Total runtime = {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main()
