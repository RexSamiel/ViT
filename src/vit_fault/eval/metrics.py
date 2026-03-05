"""High-level evaluation function and Results dataclass."""

import torch
from dataclasses import dataclass, field
from typing import Optional

from vit_fault.eval.accuracy import AccuracyTracker
from vit_fault.eval.sdc import SDCTracker


@dataclass
class Results:
    """Results from model evaluation."""

    # Accuracy
    top1: float = 0.0
    top5: float = 0.0

    # SDC (if fault-free logits available)
    sdc_rate: Optional[float] = None
    critical_top1: Optional[float] = None
    critical_top5: Optional[float] = None

    # Detection (if detector provided)
    faults_detected: int = 0
    faulty_layers: list[str] = field(default_factory=list)

    # Metadata
    samples: int = 0
    batches: int = 0

    def print(self):
        """Print results summary."""
        print()
        print("=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Samples: {self.samples}")
        print()
        print(f"Accuracy:")
        print(f"  Top-1: {self.top1:.2f}%")
        print(f"  Top-5: {self.top5:.2f}%")

        if self.sdc_rate is not None:
            print()
            print(f"SDC Metrics:")
            print(f"  Logit SDC Rate: {self.sdc_rate:.2f}%")
            print(f"  Critical Top-1: {self.critical_top1:.2f}%")
            print(f"  Critical Top-5: {self.critical_top5:.2f}%")

        if self.faults_detected > 0:
            print()
            print(f"Detection:")
            print(f"  Faults detected: {self.faults_detected}")
            for layer in self.faulty_layers:
                print(f"    - {layer}")
        print("=" * 50)


def evaluate(model, detector=None) -> Results:
    """Evaluate model and compute metrics.

    Args:
        model: Model instance (vit_fault.Model)
        detector: Optional Detector instance for fault detection

    Returns:
        Results with accuracy, SDC, and detection metrics
    """
    # Get the underlying network and batches
    if hasattr(model, "net"):
        net = model.net
        batches = model.get_batches()
        ff_logits = model.ff_logits if model.ff_logits.available else None
    else:
        net = model
        batches = []
        ff_logits = None

    if not batches:
        return Results()

    # Initialize trackers
    acc_tracker = AccuracyTracker()
    sdc_tracker = SDCTracker() if ff_logits else None

    # Run evaluation
    for batch_idx, (images, labels) in enumerate(batches):
        with torch.inference_mode():
            outputs = net(images)

        # Accuracy
        acc_tracker.update_batch(outputs, labels)

        # SDC (if fault-free logits available)
        if sdc_tracker and ff_logits:
            ff_batch = ff_logits.get_batch(batch_idx, len(images), images.device)
            sdc_tracker.update_batch(outputs, ff_batch)

    # Gather results
    acc = acc_tracker.get_results()

    results = Results(
        top1=acc["top1_acc"],
        top5=acc["top5_acc"],
        samples=acc["samples"],
        batches=len(batches),
    )

    # Add SDC if available
    if sdc_tracker:
        sdc = sdc_tracker.get_results()
        results.sdc_rate = sdc["logit_sdc_rate"]
        results.critical_top1 = sdc["critical_top1_sdc_rate"]
        results.critical_top5 = sdc["critical_top5_sdc_rate"]

    # Add detection results
    if detector:
        faults = detector.faults_found
        results.faults_detected = len(faults)
        results.faulty_layers = faults

    return results
