import datetime
import json
import os
import time
import torch.nn as nn
import numpy as np
from collections import defaultdict

from src.core.library.histogram import (
    compute_histogram,
    get_histogram_config,
)
from src.core.library.ui import format_count
from src.core.library.utils import resolve_amp
from src.core.parameter_analysis.collector import DataCollector


class ParameterAnalyzer:
    """Unified parameter analysis engine for activations and weights.

    Supports two analysis types:
    - "aa": Activation analysis (requires forward passes with data)
    - "wa": Weight analysis (static, no data needed)

    Common functionality:
    - Histogram computation across components
    - Result dictionary generation
    - Statistics printing
    - JSON export

    Type-specific logic is delegated to collector modules.
    """

    def __init__(self, analysis_type: str, sampling_percent: float = 1.0):
        """Initialize parameter analyzer.

        Args:
            analysis_type: "aa" for activations or "wa" for weights
            sampling_percent: Sampling percentage for activations (0.01-100),
                            ignored for weight analysis
        """
        if analysis_type not in ["aa", "wa"]:
            raise ValueError(
                f"analysis_type must be 'aa' or 'wa', got '{analysis_type}'"
            )

        self.analysis_type = analysis_type
        self.sampling_percent = (
            max(0.01, min(100.0, sampling_percent)) if analysis_type == "aa" else 100.0
        )

        self.collector = DataCollector(self.analysis_type, self.sampling_percent)

        self.components = self.collector.get_components()

        self.bin_range, self.bin_resolution, self.num_bins = get_histogram_config(
            analysis_type
        )

        self._reset()

    def _reset(self) -> None:
        """Reset all collected data."""
        self.data_dict: dict[int, dict] = {}

        self.global_stats = {
            comp: {"min": float("inf"), "max": float("-inf")}
            for comp in self.components
        }

        self._hist_counts: dict[str, np.ndarray] = {
            comp: np.zeros(self.num_bins, dtype=np.int64) for comp in self.components
        }

        self._data_range: dict[str, dict] = {
            comp: {"min": float("inf"), "max": float("-inf")}
            for comp in self.components
        }

        self._element_counts: dict[str, dict] = {
            comp: {"total": 0, "sampled": 0} for comp in self.components
        }

        self._name_to_idx: dict[str, int] = {}
        self.num_blocks = 0

    def collect_data(
        self,
        model: nn.Module,
        batches: tuple | None = None,
        inference_fn=None,
        use_amp: bool = False,
        verbose: bool = True,
    ) -> None:
        """Collect parameter data (activations or weights).

        For activation analysis (aa):
            - Requires batches, inference_fn, and use_amp
            - Registers hooks and runs forward passes

        For weight analysis (wa):
            - Only needs model
            - Iterates through model.named_parameters()

        Args:
            model: The model to analyze
            batches: Tuple of (images, labels) batches (aa only)
            inference_fn: Callable(images, use_amp) -> outputs (aa only)
            use_amp: Whether to use automatic mixed precision (aa only)
            verbose: Whether to print progress
        """
        self._reset()
        num_blocks_ref = [0]
        if self.analysis_type == "aa":
            if batches is None or inference_fn is None:
                raise ValueError(
                    "Activation analysis requires batches and inference_fn"
                )

            self.collector.collect(
                model=model,
                batches=batches,
                inference_fn=inference_fn,
                use_amp=use_amp,
                verbose=verbose,
                data_dict=self.data_dict,
                name_to_idx=self._name_to_idx,
                global_stats=self.global_stats,
                hist_counts=self._hist_counts,
                data_range=self._data_range,
                element_counts=self._element_counts,
                num_blocks_ref=num_blocks_ref,
            )
        else:
            self.collector.collect(
                model=model,
                verbose=verbose,
                data_dict=self.data_dict,
                name_to_idx=self._name_to_idx,
                global_stats=self.global_stats,
                hist_counts=self._hist_counts,
                data_range=self._data_range,
                element_counts=self._element_counts,
                num_blocks_ref=num_blocks_ref,
            )

        self.num_blocks = num_blocks_ref[0]

    def get_results(self) -> dict:
        """Generate results dictionary for JSON output.

        Returns a standardized dictionary with:
        - data items (layers/parameters)
        - distributions (histograms by component)
        - ranges (global min/max by component)
        - statistics (counts and metadata)
        - element counts (total/sampled by component)
        """
        items_key = "layers" if self.analysis_type == "aa" else "parameters"
        items_output = {}

        for idx, data in self.data_dict.items():
            if data.get("excluded"):
                continue

            item = {
                "name": data["name"],
                "component": data["component"],
                "block_idx": data["block_idx"],
                "min": data["min"],
                "max": data["max"],
                "total_elements": data.get("total_elements", 0),
                "sampled_elements": data.get("sampled_elements", 0),
            }

            if self.analysis_type == "aa":
                item["op_type"] = data.get("op_type")
            else:
                item["shape"] = data.get("shape")

            items_output[str(idx)] = item

        distributions = {}
        for comp in self.components:
            hist = compute_histogram(
                self._hist_counts[comp],
                self.bin_range,
                self._data_range[comp],
                self._element_counts[comp],
                self.bin_resolution,
            )
            if hist:
                distributions[comp] = hist

        ranges = {
            comp: {
                "global_min": self.global_stats[comp]["min"]
                if self.global_stats[comp]["min"] != float("inf")
                else None,
                "global_max": self.global_stats[comp]["max"]
                if self.global_stats[comp]["max"] != float("-inf")
                else None,
            }
            for comp in self.components
        }

        # Count items by component
        comp_counts = defaultdict(int)
        for data in self.data_dict.values():
            if not data.get("excluded"):
                comp_counts[data["component"]] += 1

        # Build statistics
        statistics = {
            "total_items": len(
                [d for d in self.data_dict.values() if not d.get("excluded")]
            ),
            "num_blocks": self.num_blocks,
        }

        for comp in self.components:
            if comp != "all":
                key = f"num_{comp}_items"
                statistics[key] = comp_counts[comp]

        if self.analysis_type == "aa":
            statistics["total_samples"] = self.collector.total_samples
            statistics["total_batches"] = self.collector.total_batches
            statistics["sampling_percent"] = self.sampling_percent

        total_elements = sum(c["total"] for c in self._element_counts.values())
        total_sampled = sum(c["sampled"] for c in self._element_counts.values())

        if self.analysis_type == "wa":
            total_elements = sum(
                c["total"] for comp, c in self._element_counts.items() if comp != "all"
            )
            total_sampled = sum(
                c["sampled"]
                for comp, c in self._element_counts.items()
                if comp != "all"
            )

        element_counts = {
            "total_elements": total_elements,
            "total_sampled": total_sampled,
            "sampling_ratio": round(total_sampled / total_elements * 100, 4)
            if total_elements > 0
            else 0,
            "by_component": {
                comp: {
                    "total": self._element_counts[comp]["total"],
                    "sampled": self._element_counts[comp]["sampled"],
                }
                for comp in self.components
            },
        }

        return {
            items_key: items_output,
            "distributions": distributions,
            "ranges": ranges,
            "statistics": statistics,
            "element_counts": element_counts,
        }

    def print_results(self) -> None:
        """Print summary to terminal."""
        counts = defaultdict(int)
        for d in self.data_dict.values():
            if not d.get("excluded"):
                counts[d["component"]] += 1

        total_elements = sum(c["total"] for c in self._element_counts.values())
        total_sampled = sum(c["sampled"] for c in self._element_counts.values())

        if self.analysis_type == "wa":
            total_elements = sum(
                c["total"] for comp, c in self._element_counts.items() if comp != "all"
            )
            total_sampled = sum(
                c["sampled"]
                for comp, c in self._element_counts.items()
                if comp != "all"
            )

        title = (
            "Activation Analysis Results:"
            if self.analysis_type == "aa"
            else "Weight Analysis Results:"
        )
        item_label = "layers" if self.analysis_type == "aa" else "parameters"
        element_label = "activations" if self.analysis_type == "aa" else "weights"

        header = f"\n{title}\n{'-' * 50}\nTotal {item_label}: {len(self.data_dict)}\n"

        if self.analysis_type == "aa":
            header += f"Sampling: {self.sampling_percent:.2f}%\n"

        ratio_line = (
            f"\n  Actual ratio:  {total_sampled / total_elements * 100:>11.4f}%"
            if total_elements > 0 and self.analysis_type == "aa"
            else ""
        )

        counts_section = (
            f"\n{element_label.capitalize()} counts:\n"
            f"  Total found:   {format_count(total_elements):>12}\n"
            f"  Total sampled: {format_count(total_sampled):>12}"
            f"{ratio_line}"
        )

        print(header + counts_section)

        comp_rows = "\n".join(
            f"  {comp.upper():<12} {counts[comp]:>7} {format_count(self._element_counts[comp]['total']):>12} {format_count(self._element_counts[comp]['sampled']):>12}"
            for comp in self.components
            if comp != "all"
            and (counts[comp] > 0 or self._element_counts[comp]["total"] > 0)
        )

        breakdown_header = (
            f"\n{item_label.capitalize()} and {element_label} by component:\n"
            f"  {'Component':<12} {item_label.capitalize()[:7]:>7} {'Total':>12} {'Sampled':>12}\n"
            f"  {'-' * 47}\n"
        )

        print(breakdown_header + comp_rows)

        # Value ranges
        ranges = "\n".join(
            f"  {comp.upper():12}: [{self.global_stats[comp]['min']:>10.6f}, {self.global_stats[comp]['max']:>10.6f}]"
            for comp in self.components
            if self.global_stats[comp]["min"] != float("inf")
        )
        print(f"\nValue ranges:\n{ranges}")

        # Type-specific footer
        if self.analysis_type == "aa":
            print(f"\nSamples processed: {self.collector.total_samples}")

    def print_details(self) -> None:
        """Print per-item details (layers for aa, parameters for wa)."""
        item_label = "Layer" if self.analysis_type == "aa" else "Parameter"
        element_label = "Activations" if self.analysis_type == "aa" else "Weights"

        header = (
            f"\nPer-{item_label} Ranges:\n"
            f"{'=' * 120}\n"
            f"{'Idx':<5} {'Blk':<4} {'Comp':<12} {'Min':>12} {'Max':>12} {'Total':>10} {'Sampled':>10} "
        )

        if self.analysis_type == "aa":
            header += f"{'Type':<15} {'Name'}\n"
        else:
            header += f"{'Shape':<20} {'Name'}\n"

        header += f"{'-' * 120}"

        rows = []
        for idx in sorted(self.data_dict.keys()):
            d = self.data_dict[idx]

            row = (
                f"{idx:<5} {(str(d['block_idx']) if d['block_idx'] is not None else '-'):<4} "
                f"{d['component']:<12} {d['min']:>12.6f} {d['max']:>12.6f} "
                f"{format_count(d.get('total_elements', 0)):>10} {format_count(d.get('sampled_elements', 0)):>10} "
            )

            if self.analysis_type == "aa":
                row += f"{d.get('op_type', ''):<15} "
                name_display = d["name"][-40:] if len(d["name"]) > 40 else d["name"]
                row += name_display
                if d.get("excluded"):
                    row += " [EXCLUDED]"
            else:
                shape_str = str(d.get("shape", ""))[:20]
                row += f"{shape_str:<20} "
                name_display = d["name"][-50:] if len(d["name"]) > 50 else d["name"]
                row += name_display

            rows.append(row)

        print(f"{header}\n" + "\n".join(rows))

    def cleanup(self) -> None:
        """Clean up collector resources."""
        if hasattr(self, "collector"):
            self.collector.cleanup()

    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except AttributeError:
            pass


def run(
    runner,
    analysis_type: str,
    sampling_percent: float = 1.0,
    verbose: bool = True,
) -> tuple[dict, ParameterAnalyzer]:
    """Execute a single parameter analysis run.

    Args:
        runner: ModelRunner instance
        analysis_type: "aa" for activations or "wa" for weights
        sampling_percent: Percentage of activations to sample (aa only, 0.01-100)
        verbose: Whether to print output

    Returns:
        Tuple of (results_dict, analyzer) - analyzer is returned for optional
        print_details() call by the caller.
    """
    analyzer = ParameterAnalyzer(
        analysis_type=analysis_type,
        sampling_percent=sampling_percent,
    )

    type_label = "Activation" if analysis_type == "aa" else "Weight"
    print(
        f"\n{'*' * 60}\n {type_label} Analysis: {runner.config.model_name}\n{'*' * 60}"
    )

    start_time = time.perf_counter() if analysis_type == "aa" else None

    if analysis_type == "aa":
        use_amp = resolve_amp(runner.config)
        batches = runner.get_batches()

        print(
            f"Processing {len(batches)} batches ({runner.config.batch_size} samples each)\n"
            f"{'-' * 40}"
        )

        analyzer.collect_data(
            model=runner.model,
            batches=batches,
            inference_fn=runner.inference,
            use_amp=use_amp,
            verbose=verbose,
        )

        runtime = time.perf_counter() - start_time
    else:  # wa
        analyzer.collect_data(
            model=runner.model,
            verbose=verbose,
        )
        runtime = None

    results = analyzer.get_results()
    results["model"] = runner.config.model_name

    if runtime is not None:
        results["runtime_sec"] = round(runtime, 2)

    if verbose:
        print(f"{'-' * 40}")
        analyzer.print_results()
        if runtime is not None:
            print(f"\nRuntime: {runtime:.2f}s")
        print(f"\n{'=' * 60}")

    return results, analyzer


def save_results(
    results: dict, config, model_key: str, analysis_type: str, output_dir: str = None
) -> None:
    """Save parameter analysis results to JSON.

    Args:
        results: Results dictionary from run()
        config: Config object with batch_size, max_batches, model_name
        model_key: Model identifier for filename
        analysis_type: "aa" or "wa" for file organization
        output_dir: Custom output directory (default: results/data/new_runs)
    """
    output_dir = output_dir or "results/data/new_runs"
    os.makedirs(output_dir, exist_ok=True)

    date_str = datetime.datetime.now().strftime("%Y%m%d")
    if analysis_type == "aa":
        samples = config.batch_size * (config.max_batches or 500)
        filename = f"activations_{model_key}_{samples}samples_{date_str}.json"
    else:  # wa
        filename = f"weights_{model_key}_{date_str}.json"
    path = f"{output_dir}/{filename}"

    # Add metadata
    results["timestamp"] = datetime.datetime.now().isoformat()
    results["config"] = {
        "model": model_key,
        "model_name": config.model_name,
    }

    if analysis_type == "aa":
        results["config"]["batch_size"] = config.batch_size
        results["config"]["max_batches"] = config.max_batches
        results["config"]["samples"] = config.batch_size * (config.max_batches or 500)

    # Save to file
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    type_label = "Activation" if analysis_type == "aa" else "Weight"
    print(f"\n{type_label} results saved to: {path}")
