import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataset_loader import ImageNetValDataset
from fault_injection import inject_fault


@dataclass
class Config:
    """Configuration for model evaluation."""

    root_dir: str = "/gpfs/mariana/home/svloor/Documents/vit/data/imagenet"
    model_name: str = "vit_base_patch16_224"
    batch_size: int = 128
    num_workers: int = min(4, os.cpu_count() or 2)
    use_amp: bool = True
    max_batches: int | None = None

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MetricsTracker:
    """Track and compute evaluation metrics."""

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
        """Update SDC metrics for a batch."""
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
    """Manage fault-free logits storage and loading."""

    FILENAME = "ff_logits.pt"

    def __init__(self):
        self.data = None
        self.load()

    def load(self):
        """Load fault-free logits if available."""
        if Path(self.FILENAME).exists():
            self.data = torch.load(self.FILENAME, weights_only=False)
            print("✓ Fault-free logits loaded")
        else:
            print("✗ Fault-free logits not found. Run with --faultfree --logits first.")

    def save(self, logits, labels):
        """Save fault-free logits and labels."""
        torch.save(
            {"logits": torch.cat(logits), "labels": torch.cat(labels)},
            self.FILENAME,
        )
        print(f"✓ Fault-free logits saved to {self.FILENAME}")

    def get_batch(self, batch_idx, batch_size, actual_size, device):
        """Get fault-free logits for a specific batch."""
        if self.data is None:
            raise RuntimeError(
                "Fault-free logits required for SDC computation. "
                "Run: python script.py --faultfree --logits"
            )

        start = batch_idx * batch_size
        end = start + actual_size
        return self.data["logits"][start:end].to(device)

    @property
    def available(self):
        return self.data is not None


class ModelEvaluator:
    """Evaluate Vision Transformer with optional fault injection."""

    def __init__(self, config):
        self.config = config
        self.model = self._load_model()
        self.dataloader = self._create_dataloader()
        self.criterion = nn.CrossEntropyLoss()
        self.ff_logits = FaultFreeLogits()

    def _load_model(self):
        """Load and prepare the model."""
        model = timm.create_model(self.config.model_name, pretrained=True).to(
            self.config.device
        )
        model.eval()
        return model

    def _create_dataloader(self):
        """Create validation dataloader with transforms."""
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
        """Run model evaluation."""
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

                # Forward pass
                with torch.autocast(device_type="cuda", enabled=self.config.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Save logits for fault-free run
                if save_logits and mode == "faultfree":
                    logits_buffer.append(outputs.cpu())
                    labels_buffer.append(labels.cpu())

                # Compute metrics
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

        # Save results
        if save_logits and logits_buffer:
            self.ff_logits.save(logits_buffer, labels_buffer)

        if compute_metrics:
            self._print_results(mode, metrics.get_results())

    def _print_results(self, mode, results):
        """Print and save evaluation results."""
        if results is None:
            print("No samples evaluated")
            return

        # Console output
        print("\n" + "=" * 50)
        print(f"RESULTS - {mode.upper()} MODE")
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

        # File output
        output_file = f"results_{mode}.txt"
        with open(output_file, "w") as f:
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
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Vision Transformer fault injection evaluation"
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

    if not (args.faultfree or args.faulty):
        print("Please specify --faultfree or --faulty")
        return

    config = Config()
    evaluator = ModelEvaluator(config)

    mode = "faultfree" if args.faultfree else "faulty"
    evaluator.run(mode=mode, compute_metrics=args.metrics, save_logits=args.logits)


if __name__ == "__main__":
    main()
