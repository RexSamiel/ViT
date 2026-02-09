"""Fault injection - engine class and workflow functions."""

import datetime
import json
import math
import os
import time
import torch

from src.core.fault_injection.accuracy import AccuracyTracker
from src.core.fault_injection.sdc import SDCTracker
from src.core.fault_injection.injection import Injector
from src.core.library.utils import resolve_amp


class FaultInjection:
    """Fault injection engine - orchestrates fault injection, accuracy tracking, and SDC analysis.

    Uses Injector for fault injection/restoration, and AccuracyTracker/SDCTracker
    building blocks for metric computation.
    """

    def __init__(self):
        self.injector = Injector()
        self.accuracy = AccuracyTracker()
        self.sdc = SDCTracker()

    def reset(self) -> None:
        """Reset per-run state."""
        self.accuracy.reset()
        self.sdc.reset()

    def reset_aggregation(self) -> None:
        """Reset multi-run aggregation state."""
        self.accuracy.reset_aggregation()
        self.sdc.reset_aggregation()

    def process_batch(self, outputs, labels, batch_idx, runner, compute_sdc) -> None:
        """Process a single batch: update accuracy and SDC metrics."""
        self.accuracy.update_batch(outputs, labels)

        if compute_sdc:
            ff_batch = runner.ff_logits.get_batch(
                batch_idx,
                runner.config.batch_size,
                outputs.size(0),
                runner.config.device,
            )
            faulty_aligned, faultfree_aligned = AccuracyTracker.align_batch_sizes(outputs, ff_batch)
            self.sdc.update_batch(faulty_aligned, faultfree_aligned)

    def get_results(
        self, compute_sdc: bool, fault_info: dict | None, runtime: float, mode: str
    ) -> dict:
        """Compile full results dict for current run."""
        results = self.accuracy.get_results()
        if compute_sdc:
            results.update(self.sdc.get_results())
        results["fault_info"] = fault_info
        results["runtime_sec"] = round(runtime, 2)
        results["mode"] = mode
        return results

    def aggregate_run(self, run_results: dict) -> None:
        """Aggregate a run's results into multi-run statistics."""
        self.accuracy.aggregate_run(run_results)
        self.sdc.aggregate_run(run_results)

    def get_summary(self, total_runtime: float) -> dict:
        """Get full aggregated summary across all runs."""
        return {
            **self.accuracy.get_summary(),
            **self.sdc.get_summary(),
            "total_runtime": round(total_runtime, 2),
        }

    def print_run_results(self, results: dict, runtime: float, model_name: str) -> None:
        """Print results for a single fault injection run."""
        print(
            f"\nResults for {model_name}:\n"
            f"{'-' * 40}\n"
            f"  Top-1 Accuracy: {results['top1_acc']:.2f}%\n"
            f"  Top-5 Accuracy: {results['top5_acc']:.2f}%\n"
            f"  Samples: {results['samples']}"
        )

        if "logit_sdc_rate" in results:
            msdc = f"\n  MSDC Average:            {results['msdc_avg']:.6f}" if not math.isnan(results["msdc_avg"]) else ""
            print(
                f"\nSDC Metrics:\n"
                f"  Logit SDC Rate:          {results['logit_sdc_rate']:.2f}%"
                f"{msdc}\n"
                f"  Critical Top-1 SDC:      {results['critical_top1_sdc_rate']:.2f}%\n"
                f"  Critical Top-5 SDC:      {results['critical_top5_sdc_rate']:.2f}%"
            )

            rel = "\n".join(
                f"  SDC >= {p}%:{' ' * (14 - len(str(p)))}{results[f'sdc_{p}pct']:.2f}%"
                for p in [1, 5, 10, 15, 20, 25, 50]
                if not math.isnan(results[f"sdc_{p}pct"])
            )
            if rel:
                print(f"Relative SDC Thresholds:\n{rel}")

            if results["batches_all_nan"] > 0 or results["batches_partial_nan"] > 0:
                print(
                    f"NaN Statistics:\n"
                    f"  Batches all NaN:         {results['batches_all_nan']}/{results['total_batches']}\n"
                    f"  Batches partial NaN:     {results['batches_partial_nan']}/{results['total_batches']}"
                )

        print(f"\nRuntime: {runtime:.2f}s")

    def print_summary(self, n_runs: int, total_runtime: float) -> None:
        """Print full multi-run summary."""
        print(
            f"\n{'=' * 60}\n"
            f" SUMMARY: {n_runs} runs completed in {total_runtime:.2f}s\n"
            f"{'=' * 60}\n"
        )
        self.accuracy.print_summary()
        print()
        self.sdc.print_summary()
        print("=" * 60)


def run_single(
    runner,
    mode: str = "faultfree",
    save_logits: bool = False,
    fault_params: dict = None,
    fi: FaultInjection = None,
    verbose: bool = True,
) -> dict:
    """Execute a single fault injection evaluation run.

    Args:
        runner: ModelRunner instance
        mode: "faultfree" or "faulty"
        save_logits: Whether to save logits to disk
        fault_params: Dict with fault injection parameters (if mode="faulty")
        fi: Existing FaultInjection instance (for multi-run reuse). Created if None.
        verbose: Whether to print results

    Returns:
        Dictionary with results
    """
    if fi is None:
        fi = FaultInjection()
    use_amp = resolve_amp(runner.config)

    fi.injector.restore()
    fi.reset()

    # Inject fault if in faulty mode
    fault_info = None
    if mode == "faulty" and fault_params:
        fault_info = fi.injector.inject(runner.model, fault_params)
        if verbose:
            print(Injector.format_fault_info(fault_info))

    # Run evaluation
    batches = runner.get_batches()
    compute_sdc = mode == "faulty" and runner.ff_logits.available
    logits_buffer, labels_buffer = [], []

    start_time = time.perf_counter()

    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(batches):
            outputs = runner.inference(images, use_amp)
            fi.process_batch(outputs, labels, batch_idx, runner, compute_sdc)

            if save_logits:
                logits_buffer.append(outputs.cpu())
                labels_buffer.append(labels.cpu())

    runtime = time.perf_counter() - start_time

    if save_logits and logits_buffer:
        runner.ff_logits.save(logits_buffer, labels_buffer)

    results = fi.get_results(compute_sdc, fault_info, runtime, mode)

    if verbose:
        fi.print_run_results(results, runtime, runner.config.model_name)

    return results


def run_multiple(
    runner,
    n_runs: int,
    fault_params: dict = None,
    verbose: bool = True,
    show_info: bool = True,
) -> dict:
    """Execute multiple fault injection runs and aggregate results.

    Args:
        runner: ModelRunner instance
        n_runs: Number of runs to execute
        fault_params: Dict with fault injection parameters
        verbose: Whether to print summary
        show_info: Whether to show per-run info

    Returns:
        Dictionary with aggregated results
    """
    fi = FaultInjection()
    total_start = time.perf_counter()

    for i in range(n_runs):
        if verbose and show_info:
            print(f"\n{'=' * 60}\n Run {i + 1}/{n_runs}\n{'=' * 60}")

        results = run_single(
            runner,
            mode="faulty",
            fault_params=fault_params,
            fi=fi,
            verbose=show_info,
        )
        fi.aggregate_run(results)

    total_runtime = time.perf_counter() - total_start

    if verbose:
        fi.print_summary(n_runs, total_runtime)

    return fi.get_summary(total_runtime)


# =====================================================
# File I/O functions
# =====================================================


def load_base_accuracy(model_key: str) -> dict | None:
    """Load base accuracy from faultfree run if available."""
    base_path = f"results/base_accuracy/{model_key}.json"
    if os.path.exists(base_path):
        try:
            with open(base_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def save_base_accuracy(model_key: str, results: dict) -> None:
    """Save base accuracy from faultfree run."""
    os.makedirs("results/base_accuracy", exist_ok=True)
    path = f"results/base_accuracy/{model_key}.json"

    base_acc = {
        "top1": results.get("top1_acc", 0.0),
        "top5": results.get("top5_acc", 0.0),
        "samples": results.get("samples", 0),
        "timestamp": datetime.datetime.now().isoformat(),
    }

    with open(path, "w") as f:
        json.dump(base_acc, f, indent=2)

    print(f"Base accuracy saved to: {path}")


def save_summary(
    summary: dict,
    config,
    model_key: str,
    mode: str,
    fault_config: dict = None,
    base_accuracy: dict = None,
    seed: int = None,
    total_blocks: int = None,
) -> None:
    """Save experiment summary to JSON."""
    os.makedirs("results/new_runs", exist_ok=True)

    date_str = datetime.datetime.now().strftime("%Y%m%d")
    samples = config.batch_size * (config.max_batches or 500)
    filename = f"{model_key}_{mode}_{samples}samples_{date_str}.json"
    path = f"results/new_runs/{filename}"

    existing = []
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                existing = json.load(f)
                if not isinstance(existing, list):
                    existing = [existing]
        except (json.JSONDecodeError, IOError):
            existing = []

    samples_per_run = config.batch_size * (config.max_batches or 500)

    summary["timestamp"] = datetime.datetime.now().isoformat()
    summary["config"] = {
        "model": model_key,
        "model_name": config.model_name,
        "mode": mode,
        "repeat": fault_config.get("repeat", 1) if fault_config else 1,
        "samples_per_run": samples_per_run,
        "batch_size": config.batch_size,
        "max_batches": config.max_batches,
        "component": fault_config.get("component") if fault_config else None,
        "sub_component": fault_config.get("sub_component") if fault_config else None,
        "block_idx": fault_config.get("block_idx") if fault_config else None,
        "idx": fault_config.get("idx") if fault_config else None,
        "bit_range": fault_config.get("bit_range") if fault_config else None,
        "total_blocks": total_blocks,
    }

    summary["reproducibility"] = {
        "seed": seed,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device": str(config.device),
        "cudnn_version": torch.backends.cudnn.version()
        if torch.cuda.is_available()
        else None,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }

    if base_accuracy:
        summary["base_accuracy"] = base_accuracy

    existing.append(summary)

    with open(path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"Summary saved to: {path}")
