"""Activation analysis - analyzer class and workflow functions."""

import datetime
import json
import os
import time
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

from src.core.library.layers import extract_block_idx, is_excluded
from src.core.activation.hooks import HookManager
from src.core.activation.histogram import (
    sample_and_histogram,
    compute_histogram,
    _BIN_RANGE,
    _NUM_BINS,
)
from src.core.library.ui import format_count
from src.core.library.utils import resolve_amp


class ActivationAnalyzer:
    """Activation analysis engine - data collection, histogram computation, and results.

    Uses HookManager for forward hook lifecycle and activation dispatch.
    """

    EXCLUDE_PATTERNS: list[str] = [
        # Uncomment to exclude:
        # "attn_drop",    # Attention weights (0-1 probabilities)
        # "drop",         # Dropout layers
    ]

    def __init__(self, sampling_percent: float = 1.0):
        self.sampling_percent = max(0.01, min(100.0, sampling_percent))
        self._hook_manager = HookManager()
        self._reset()

    def _reset(self) -> None:
        """Reset all collected data."""
        self.layer_data: dict[int, dict] = {}

        self.global_stats = {
            comp: {"min": float("inf"), "max": float("-inf")}
            for comp in ["input", "output", "block", "mha", "mlp"]
        }

        self._hist_counts: dict[str, np.ndarray] = {
            comp: np.zeros(_NUM_BINS, dtype=np.int64)
            for comp in ["input", "output", "block", "mha", "mlp"]
        }

        self._data_range: dict[str, dict] = {
            comp: {"min": float("inf"), "max": float("-inf")}
            for comp in ["input", "output", "block", "mha", "mlp"]
        }

        self._activation_counts: dict[str, dict] = {
            comp: {"total": 0, "sampled": 0}
            for comp in ["input", "output", "block", "mha", "mlp"]
        }

        self._name_to_idx: dict[str, int] = {}
        self.total_samples = 0
        self.total_batches = 0
        self.num_blocks = 0

    def _record_activation(
        self,
        tensor: torch.Tensor,
        name: str,
        module_type: str,
        component: str,
        block_idx: int | None,
    ) -> None:
        """Record activation values from a layer using incremental histograms.

        This is the callback passed to HookManager. It is called for every
        module output during the forward pass.
        """
        num_elements = tensor.numel()

        if name not in self._name_to_idx:
            self._name_to_idx[name] = len(self._name_to_idx)
        idx = self._name_to_idx[name]

        excluded = is_excluded(name, module_type, self.EXCLUDE_PATTERNS)

        if idx in self.layer_data:
            data = self.layer_data[idx]
            t_min, t_max = tensor.aminmax()
            data["min"] = min(data["min"], t_min.item())
            data["max"] = max(data["max"], t_max.item())
            data["total_activations"] += num_elements
        else:
            t_min, t_max = tensor.aminmax()
            self.layer_data[idx] = {
                "name": name,
                "op_type": module_type,
                "component": component,
                "block_idx": block_idx,
                "min": t_min.item(),
                "max": t_max.item(),
                "excluded": excluded,
                "total_activations": num_elements,
                "sampled_activations": 0,
            }

        if excluded:
            return

        self._activation_counts[component]["total"] += num_elements

        data = self.layer_data[idx]
        self.global_stats[component]["min"] = min(
            self.global_stats[component]["min"], data["min"]
        )
        self.global_stats[component]["max"] = max(
            self.global_stats[component]["max"], data["max"]
        )

        num_sampled, sampled_min, sampled_max = sample_and_histogram(
            tensor,
            self.sampling_percent,
            self._hist_counts[component],
            _BIN_RANGE,
        )

        self._activation_counts[component]["sampled"] += num_sampled
        self.layer_data[idx]["sampled_activations"] += num_sampled

        self._data_range[component]["min"] = min(
            self._data_range[component]["min"], sampled_min
        )
        self._data_range[component]["max"] = max(
            self._data_range[component]["max"], sampled_max
        )

    def collect_data(
        self,
        model: nn.Module,
        batches: tuple,
        inference_fn,
        use_amp: bool,
        verbose: bool = True,
    ) -> None:
        """Collect activation data by running inference with hooks.

        Registers hooks via HookManager, runs batches, and removes hooks.
        This is the single entry point for data collection.

        Args:
            model: The model to analyze
            batches: Tuple of (images, labels) batches
            inference_fn: Callable(images, use_amp) -> outputs
            use_amp: Whether to use automatic mixed precision
            verbose: Whether to print progress
        """
        self._hook_manager.remove()
        self._reset()

        for name, _ in model.named_modules():
            idx = extract_block_idx(name)
            if idx is not None:
                self.num_blocks = max(self.num_blocks, idx + 1)

        hook_count = self._hook_manager.register(model, self._record_activation)

        pct_str = (
            f"{self.sampling_percent:.2f}%"
            if self.sampling_percent < 1
            else f"{self.sampling_percent:.1f}%"
        )
        if verbose:
            print(f"Registered hooks on {hook_count} modules (sampling {pct_str})")

        # Process batches
        total_batches = len(batches)
        if verbose:
            print(f"Processing {total_batches} batches")

        with torch.inference_mode():
            for batch_idx, (images, _) in enumerate(batches):
                _ = inference_fn(images, use_amp)
                self.total_samples += images.size(0)
                self.total_batches += 1
                self._hook_manager.reset_block_tracking()

                if verbose and (batch_idx + 1) % max(1, total_batches // 10) == 0:
                    progress = 100 * (batch_idx + 1) / total_batches
                    print(
                        f"  Progress: {progress:.0f}% ({batch_idx + 1}/{total_batches} batches)"
                    )

        # Clean up hooks
        self._hook_manager.remove()

    def get_results(self) -> dict:
        """Generate results dictionary for JSON output."""
        layers_output = {}
        for idx, data in self.layer_data.items():
            if not data.get("excluded"):
                layers_output[str(idx)] = {
                    "name": data["name"],
                    "op_type": data["op_type"],
                    "component": data["component"],
                    "block_idx": data["block_idx"],
                    "min": data["min"],
                    "max": data["max"],
                    "total_activations": data.get("total_activations", 0),
                    "sampled_activations": data.get("sampled_activations", 0),
                }

        distributions = {}
        for comp in ["input", "output", "block", "mha", "mlp"]:
            hist = compute_histogram(
                self._hist_counts[comp],
                _BIN_RANGE,
                self._data_range[comp],
                self._activation_counts[comp],
            )
            if hist:
                distributions[comp] = hist

        comp_counts = defaultdict(int)
        for data in self.layer_data.values():
            if not data.get("excluded"):
                comp_counts[data["component"]] += 1

        total_activations = sum(c["total"] for c in self._activation_counts.values())
        total_sampled = sum(c["sampled"] for c in self._activation_counts.values())

        return {
            "layers": layers_output,
            "distributions": distributions,
            "ranges": {
                comp: {
                    "global_min": self.global_stats[comp]["min"]
                    if self.global_stats[comp]["min"] != float("inf")
                    else None,
                    "global_max": self.global_stats[comp]["max"]
                    if self.global_stats[comp]["max"] != float("-inf")
                    else None,
                }
                for comp in ["input", "output", "block", "mha", "mlp"]
            },
            "statistics": {
                "total_samples": self.total_samples,
                "total_batches": self.total_batches,
                "total_layers": len(self.layer_data),
                "num_blocks": self.num_blocks,
                "sampling_percent": self.sampling_percent,
                "num_input_layers": comp_counts["input"],
                "num_output_layers": comp_counts["output"],
                "num_block_layers": comp_counts["block"],
                "num_mha_layers": comp_counts["mha"],
                "num_mlp_layers": comp_counts["mlp"],
            },
            "activation_counts": {
                "total_activations": total_activations,
                "total_sampled": total_sampled,
                "sampling_ratio": round(total_sampled / total_activations * 100, 4)
                if total_activations > 0
                else 0,
                "by_component": {
                    comp: {
                        "total": self._activation_counts[comp]["total"],
                        "sampled": self._activation_counts[comp]["sampled"],
                    }
                    for comp in ["input", "output", "block", "mha", "mlp"]
                },
            },
        }

    def print_results(self) -> None:
        """Print summary to terminal."""
        counts = defaultdict(int)
        for d in self.layer_data.values():
            if not d.get("excluded"):
                counts[d["component"]] += 1

        total_activations = sum(c["total"] for c in self._activation_counts.values())
        total_sampled = sum(c["sampled"] for c in self._activation_counts.values())

        ratio_line = f"\n  Actual ratio:  {total_sampled / total_activations * 100:>11.4f}%" if total_activations > 0 else ""
        print(
            f"\nActivation Analysis Results:\n"
            f"{'-' * 50}\n"
            f"Total layers: {len(self.layer_data)}\n"
            f"Sampling: {self.sampling_percent:.2f}%\n"
            f"\nActivation counts:\n"
            f"  Total found:   {format_count(total_activations):>12}\n"
            f"  Total sampled: {format_count(total_sampled):>12}"
            f"{ratio_line}"
        )

        comp_rows = "\n".join(
            f"  {comp.upper():<8} {counts[comp]:>7} {format_count(self._activation_counts[comp]['total']):>12} {format_count(self._activation_counts[comp]['sampled']):>12}"
            for comp in ["input", "block", "mha", "mlp", "output"]
        )
        print(
            f"\nLayers and activations by component:\n"
            f"  {'Component':<8} {'Layers':>7} {'Total Acts':>12} {'Sampled':>12}\n"
            f"  {'-' * 43}\n"
            f"{comp_rows}"
        )

        ranges = "\n".join(
            f"  {comp.upper():6}: [{self.global_stats[comp]['min']:>10.2f}, {self.global_stats[comp]['max']:>10.2f}]"
            for comp in ["input", "mha", "mlp", "block", "output"]
            if self.global_stats[comp]["min"] != float("inf")
        )
        print(f"\nValue ranges:\n{ranges}\n\nSamples processed: {self.total_samples}")

    def print_layer_ranges(self) -> None:
        """Print per-layer activation ranges."""
        header = (
            f"\nPer-Layer Activation Ranges:\n"
            f"{'=' * 120}\n"
            f"{'Idx':<5} {'Blk':<4} {'Comp':<6} {'Min':>10} {'Max':>10} {'Total':>10} {'Sampled':>10} {'Type':<15} {'Name'}\n"
            f"{'-' * 120}"
        )

        rows = "\n".join(
            f"{idx:<5} {(str(d['block_idx']) if d['block_idx'] is not None else '-'):<4} "
            f"{d['component']:<6} {d['min']:>10.2f} {d['max']:>10.2f} "
            f"{format_count(d.get('total_activations', 0)):>10} {format_count(d.get('sampled_activations', 0)):>10} "
            f"{d['op_type']:<15} {d['name'][-40:] if len(d['name']) > 40 else d['name']}"
            f"{' [EXCLUDED]' if d.get('excluded') else ''}"
            for idx in sorted(self.layer_data.keys())
            for d in [self.layer_data[idx]]
        )

        print(f"{header}\n{rows}")


def run(
    runner, sampling_percent: float = 1.0, verbose: bool = True
) -> tuple[dict, ActivationAnalyzer]:
    """Execute a single activation analysis run.

    Args:
        runner: ModelRunner instance
        sampling_percent: Percentage of activations to sample per layer (0.01-100)
        verbose: Whether to print output

    Returns:
        Tuple of (results_dict, analyzer) - analyzer is returned for optional
        print_layer_ranges() call by the caller.
    """
    analyzer = ActivationAnalyzer(sampling_percent=sampling_percent)
    use_amp = resolve_amp(runner.config)

    if verbose:
        print(
            f"\n{'=' * 60}\n"
            f" Activation Analysis: {runner.config.model_name}\n"
            f"{'=' * 60}"
        )

    batches = runner.get_batches()

    if verbose:
        print(
            f"Processing {len(batches)} batches ({runner.config.batch_size} samples each)\n"
            f"{'-' * 40}"
        )

    start_time = time.perf_counter()

    analyzer.collect_data(
        model=runner.model,
        batches=batches,
        inference_fn=runner.inference,
        use_amp=use_amp,
        verbose=verbose,
    )

    runtime = time.perf_counter() - start_time

    results = analyzer.get_results()
    results["runtime_sec"] = round(runtime, 2)
    results["model"] = runner.config.model_name

    if verbose:
        print(f"{'-' * 40}")
        analyzer.print_results()
        print(f"\nRuntime: {runtime:.2f}s\n{'=' * 60}")

    return results, analyzer


def save_results(results: dict, config, model_key: str) -> None:
    """Save activation analysis results to JSON.

    Args:
        results: Results dictionary from run()
        config: Config object with batch_size, max_batches, model_name
        model_key: Model identifier for filename
    """
    os.makedirs("results/new_runs", exist_ok=True)

    date_str = datetime.datetime.now().strftime("%Y%m%d")
    samples = config.batch_size * (config.max_batches or 500)
    filename = f"activations_{model_key}_{samples}samples_{date_str}.json"
    path = f"results/new_runs/{filename}"

    results["timestamp"] = datetime.datetime.now().isoformat()
    results["config"] = {
        "model": model_key,
        "model_name": config.model_name,
        "batch_size": config.batch_size,
        "max_batches": config.max_batches,
        "samples": samples,
    }

    ranges = results.get("ranges", {})
    for component in ["block", "mha", "mlp"]:
        if component in ranges and "layers" in ranges[component]:
            layers = ranges[component]["layers"]
            ranges[component]["layers"] = {str(k): v for k, v in layers.items()}

    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nActivation results saved to: {path}")
