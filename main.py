import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import os
from pathlib import Path
from dataclasses import dataclass
import timm

from dataset_loader import ImageNetValDataset
from fault_injection import inject_fault


SUPPORTED_MODELS = {
    # Vision Transformer models
    "vit_tiny": "vit_tiny_patch16_224",
    "vit_small": "vit_small_patch16_224",
    "vit_base": "vit_base_patch16_224",
    "vit_large": "vit_large_patch16_224",
    "vit_huge": "vit_huge_patch14_224",
    # Swin Transformer models
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "swin_small": "swin_small_patch4_window7_224",
    "swin_base": "swin_base_patch4_window7_224",
    "swin_large": "swin_large_patch4_window7_224",
    # BEiT models
    "beit_base": "beit_base_patch16_224",
    "beit_large": "beit_large_patch16_224",
}


def print_supported_models():
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
    root_dir: str = "/gpfs/mariana/home/svloor/Documents/vit/data/imagenet"
    model_name: str = "vit_base_patch16_224"
    model_key: str = "vit_base"
    batch_size: int = 50
    num_workers: int = min(4, os.cpu_count() or 2)
    use_amp: bool = True
    max_batches: int | None = 10

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.total_samples = 0
        self.top1_correct = 0
        self.top5_correct = 0
        self.sdc_rates = []
        self.sdc_magnitudes = []

    def update_accuracy(self, outputs, labels, loss):
        """Update accuracy metrics for a batch."""
        batch_size = labels.size(0)

        # Top-1 accuracy
        predictions = outputs.argmax(dim=1)
        self.top1_correct += (predictions == labels).sum().item()

        # Top-5 accuracy
        top5_predictions = outputs.topk(5, dim=1)[1]
        top5_matches = (labels.unsqueeze(1) == top5_predictions).any(dim=1)
        self.top5_correct += top5_matches.sum().item()

        # Loss
        self.total_loss += loss.item() * batch_size
        self.total_samples += batch_size

    def update_sdc(self, faulty_outputs, ff_logits):
        diff = ff_logits - faulty_outputs
        sdc_rate = (diff != 0).float().mean(dim=1)
        sdc_magnitude = diff.abs().mean(dim=1)

        self.sdc_rates.append(sdc_rate.cpu())
        self.sdc_magnitudes.append(sdc_magnitude.cpu())

    def get_results(self):
        """Get final computed metrics."""
        if self.total_samples == 0:
            return None

        results = {
            "samples": self.total_samples,
            "top1_acc": 100 * self.top1_correct / self.total_samples,
            "top5_acc": 100 * self.top5_correct / self.total_samples,
            "avg_loss": self.total_loss / self.total_samples,
        }

        if self.sdc_rates:
            sdc_tensor = torch.cat(self.sdc_rates)
            msdc_tensor = torch.cat(self.sdc_magnitudes)

            results["sdc_rate"] = 100 * sdc_tensor.mean().item()
            results["msdc_avg"] = msdc_tensor.mean().item()
            results["msdc_min"] = msdc_tensor.min().item()
            results["msdc_max"] = msdc_tensor.max().item()

        return results


class FaultFreeLogits:
    def __init__(self, model_key):
        self.filename = f"ff_logits_{model_key}.pt"
        self.data = None
        self.load()

    def load(self):
        if Path(self.filename).exists():
            self.data = torch.load(self.filename, weights_only=False)
            print(f"✓ Fault-free logits loaded from {self.filename}")
        else:
            print(
                f"✗ Fault-free logits not found. Run with --faultfree --logits first."
            )

    def save(self, logits, labels):
        torch.save(
            {"logits": torch.cat(logits), "labels": torch.cat(labels)},
            self.filename,
        )
        print(f"✓ Fault-free logits saved to {self.filename}")

    def get_batch(self, batch_idx, batch_size, actual_size, device):
        if self.data is None:
            raise RuntimeError(
                "Fault-free logits required for SDC computation. "
                "Run: python script.py --model <model> --faultfree --logits"
            )

        start = batch_idx * batch_size
        end = start + actual_size
        return self.data["logits"][start:end].to(device)

    @property
    def available(self):
        return self.data is not None


class ModelEvaluator:
    def __init__(self, config):
        self.config = config
        self.model = self._load_model()
        self.dataloader = self._create_dataloader()
        self.criterion = nn.CrossEntropyLoss()
        self.ff_logits = FaultFreeLogits(config.model_key)

    def _load_model(self):
        print(f"Loading model: {self.config.model_name}")
        model = timm.create_model(self.config.model_name, pretrained=True).to(
            self.config.device
        )
        model.eval()
        print(f"✓ Model loaded successfully on {self.config.device}")
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

    def run(self, mode="faultfree", compute_metrics=False, save_logits=False):
        if mode == "faulty":
            inject_fault(self.model, component_type="attention", verbose=True)
            print("✓ Fault injection applied to attention components")

        metrics = MetricsTracker()
        logits_buffer = []
        labels_buffer = []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.dataloader):
                if self.config.max_batches and batch_idx >= self.config.max_batches:
                    break

                images = images.to(self.config.device, non_blocking=True)
                labels = labels.to(self.config.device, non_blocking=True)

                with torch.autocast(device_type="cuda", enabled=self.config.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                if save_logits and mode == "faultfree":
                    logits_buffer.append(outputs.cpu())
                    labels_buffer.append(labels.cpu())

                if compute_metrics:
                    metrics.update_accuracy(outputs, labels, loss)

                    if mode == "faulty" and self.ff_logits.available:
                        ff_batch = self.ff_logits.get_batch(
                            batch_idx,
                            self.config.batch_size,
                            outputs.size(0),
                            self.config.device,
                        )
                        metrics.update_sdc(outputs, ff_batch)

        if save_logits and logits_buffer:
            self.ff_logits.save(logits_buffer, labels_buffer)

        if compute_metrics:
            self._print_results(mode, metrics.get_results())

    def _print_results(self, mode, results):
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
            print(f"SDC Magnitude (min): {results['msdc_min']:.4f}")
            print(f"SDC Magnitude (max): {results['msdc_max']:.4f}")

        print("=" * 50 + "\n")

        output_file = f"results_{self.config.model_key}_{mode}.txt"
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
                f.write(f"SDC Magnitude (min): {results['msdc_min']:.4f}\n")
                f.write(f"SDC Magnitude (max): {results['msdc_max']:.4f}\n")

        print(f"✓ Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Vision Transformer fault injection evaluation"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to evaluate (e.g., vit_base, swin_small, beit_large)",
    )
    parser.add_argument(
        "--faultfree", action="store_true", help="Run fault-free evaluation"
    )
    parser.add_argument(
        "--faulty",
        action="store_true",
        help="Run evaluation with fault injection",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Compute accuracy and SDC metrics",
    )
    parser.add_argument(
        "--logits",
        action="store_true",
        help="Save fault-free logits for later comparison",
    )

    args = parser.parse_args()

    if not args.model:
        print_supported_models()
        return

    if args.model not in SUPPORTED_MODELS:
        print(f"\n❌ Error: '{args.model}' is not a supported model.")
        print_supported_models()
        return

    if not (args.faultfree or args.faulty):
        print("Please specify --faultfree or --faulty")
        return

    config = Config()
    config.model_key = args.model
    config.model_name = SUPPORTED_MODELS[args.model]

    evaluator = ModelEvaluator(config)

    mode = "faultfree" if args.faultfree else "faulty"
    evaluator.run(mode=mode, compute_metrics=args.metrics, save_logits=args.logits)


if __name__ == "__main__":
    main()
