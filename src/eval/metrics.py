"""Evaluation function and Results dataclass."""

import statistics as _stlib
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from eval.accuracy import AccuracyTracker
from eval.sdc import SDCTracker


@dataclass
class Results:
    """Results from model evaluation."""

    # Accuracy
    top1: float = 0.0
    top5: float = 0.0

    # SDC metrics (if fault-free logits available)
    sdc_rate: Optional[float] = None
    msdc: Optional[float] = None
    sdc_1pct: Optional[float] = None
    sdc_5pct: Optional[float] = None
    sdc_10pct: Optional[float] = None
    sdc_15pct: Optional[float] = None
    sdc_20pct: Optional[float] = None
    sdc_25pct: Optional[float] = None
    sdc_50pct: Optional[float] = None
    critical_top1: Optional[float] = None
    critical_top5: Optional[float] = None
    crash_rate: Optional[float] = None

    # Detection (if detector provided)
    faults_detected: int = 0
    faulty_layers: list[str] = field(default_factory=list)

    # Timing
    elapsed_s: float = 0.0  # total forward-pass wall-clock seconds
    ms_per_sample: float = 0.0  # normalised per-sample latency
    batch_times_ms: list[float] = field(default_factory=list)  # per-batch forward time
    layer_ms: dict[str, float] = field(default_factory=dict)  # per-wrapper GPU time

    # Metadata
    samples: int = 0
    batches: int = 0

    def print(self):
        """Print results summary."""
        print()
        print("-" * 60)
        print("EVALUATION RESULTS")
        print("-" * 60)
        print(f"Samples: {self.samples}")
        print(
            f"Time:    {self.elapsed_s * 1000:.1f} ms total  |  {self.ms_per_sample:.3f} ms/sample"
        )
        if self.batch_times_ms:
            bms = self.batch_times_ms
            print(
                f"         per-batch: avg={_stlib.mean(bms):.2f} ms  "
                f"min={min(bms):.2f}  max={max(bms):.2f}  ({len(bms)} batches)"
            )
        if self.layer_ms:
            total_lms = sum(self.layer_ms.values())
            print(
                f"         layer total: {total_lms:.2f} ms across {len(self.layer_ms)} wrapped layers"
            )
            for lname, lms in sorted(self.layer_ms.items()):
                print(f"           {lname}: {lms:.3f} ms")
        print()
        print(f"Accuracy:")
        print(f"  Top-1: {self.top1:.2f}%")
        print(f"  Top-5: {self.top5:.2f}%")

        if self.sdc_rate is not None:
            print()
            print(f"SDC Metrics:")
            print(f"  Logit SDC Rate:   {self.sdc_rate:.2f}%")
            if self.msdc is not None and self.msdc > 0:
                print(f"  MSDC:             {self.msdc:.6f}")
            print()
            print(f"  Threshold-based SDC:")
            if self.sdc_1pct is not None:
                print(f"    ≥ 1%:  {self.sdc_1pct:.2f}%")
            if self.sdc_5pct is not None:
                print(f"    ≥ 5%:  {self.sdc_5pct:.2f}%")
            if self.sdc_10pct is not None:
                print(f"    ≥10%:  {self.sdc_10pct:.2f}%")
            if self.sdc_15pct is not None:
                print(f"    ≥15%:  {self.sdc_15pct:.2f}%")
            if self.sdc_20pct is not None:
                print(f"    ≥20%:  {self.sdc_20pct:.2f}%")
            if self.sdc_25pct is not None:
                print(f"    ≥25%:  {self.sdc_25pct:.2f}%")
            if self.sdc_50pct is not None:
                print(f"    ≥50%:  {self.sdc_50pct:.2f}%")
            print()
            if self.crash_rate is not None and self.crash_rate > 0:
                print(f"  Crash Rate:       {self.crash_rate:.2f}%")
            print(f"  Critical SDC:")
            print(f"    Top-1: {self.critical_top1:.2f}%")
            print(f"    Top-5: {self.critical_top5:.2f}%")

        if self.faults_detected > 0:
            print()
            print(f"Detection:")
            print(f"  Faults detected: {self.faults_detected}")
            for layer in self.faulty_layers:
                print(f"    - {layer}")
        print("-" * 60)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = {
            "top1_acc": self.top1,
            "top5_acc": self.top5,
            "elapsed_s": self.elapsed_s,
            "ms_per_sample": self.ms_per_sample,
            "samples": self.samples,
            "batches": self.batches,
        }
        if self.sdc_rate is not None:
            d["sdc_rate"] = self.sdc_rate
            d["msdc"] = self.msdc
            d["sdc_1pct"] = self.sdc_1pct
            d["sdc_5pct"] = self.sdc_5pct
            d["sdc_10pct"] = self.sdc_10pct
            d["sdc_15pct"] = self.sdc_15pct
            d["sdc_20pct"] = self.sdc_20pct
            d["sdc_25pct"] = self.sdc_25pct
            d["sdc_50pct"] = self.sdc_50pct
            d["critical_top1"] = self.critical_top1
            d["critical_top5"] = self.critical_top5
            d["crash_rate"] = self.crash_rate
        if self.faults_detected > 0:
            d["faults_detected"] = self.faults_detected
            d["faulty_layers"] = self.faulty_layers
        if self.batch_times_ms:
            d["batch_times_ms"] = self.batch_times_ms
        if self.layer_ms:
            d["layer_ms"] = self.layer_ms
        return d


def evaluate(model, detector=None) -> Results:
    """Evaluate model and compute metrics.

    Args:
        model: Model instance (vit_fault.Model)
        detector: Optional Detector instance for fault detection

    Returns:
        Results with accuracy, SDC, and detection metrics
    """
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

    device = next(net.parameters()).device
    cuda_avail = torch.cuda.is_available()

    acc_tracker = AccuracyTracker()
    sdc_tracker = SDCTracker() if ff_logits else None

    for module in net.modules():
        if hasattr(module, "elapsed_ms"):
            module.elapsed_ms = 0.0

    elapsed_s = 0.0
    batch_times_ms: list[float] = []

    for batch_idx, (images, labels) in enumerate(batches):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = net(images)
        if cuda_avail:
            torch.cuda.synchronize()
        batch_s = time.perf_counter() - t0
        elapsed_s += batch_s
        batch_times_ms.append(batch_s * 1000.0)

        acc_tracker.update_batch(outputs, labels)

        if sdc_tracker and ff_logits:
            ff_batch = ff_logits.get_batch(batch_idx, len(images), images.device)
            sdc_tracker.update_batch(outputs, ff_batch)

    # Collect per-layer times from wrappers
    layer_ms: dict[str, float] = {}
    for module in net.modules():
        if (
            hasattr(module, "elapsed_ms")
            and hasattr(module, "name")
            and module.elapsed_ms > 0
        ):
            layer_ms[module.name] = round(module.elapsed_ms, 4)

    acc = acc_tracker.get_results()

    n_samples = acc["samples"]
    results = Results(
        top1=acc["top1_acc"],
        top5=acc["top5_acc"],
        elapsed_s=elapsed_s,
        ms_per_sample=elapsed_s * 1000 / n_samples if n_samples else 0.0,
        batch_times_ms=batch_times_ms,
        layer_ms=layer_ms,
        samples=n_samples,
        batches=len(batches),
    )

    # Add SDC if available
    if sdc_tracker:
        sdc = sdc_tracker.get_results()
        results.sdc_rate = sdc["logit_sdc_rate"]
        results.msdc = sdc["msdc"]
        results.sdc_1pct = sdc["sdc_1pct"]
        results.sdc_5pct = sdc["sdc_5pct"]
        results.sdc_10pct = sdc["sdc_10pct"]
        results.sdc_15pct = sdc["sdc_15pct"]
        results.sdc_20pct = sdc["sdc_20pct"]
        results.sdc_25pct = sdc["sdc_25pct"]
        results.sdc_50pct = sdc["sdc_50pct"]
        results.critical_top1 = sdc["critical_top1_sdc_rate"]
        results.critical_top5 = sdc["critical_top5_sdc_rate"]
        results.crash_rate = sdc["crash_rate"]

    # Add detection results
    if detector:
        faults = detector.get_faults()
        faulty_layers = list({f.layer for f in faults})
        results.faults_detected = len(faulty_layers)
        results.faulty_layers = faulty_layers

    return results
