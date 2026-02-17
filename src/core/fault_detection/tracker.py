"""Detection result tracking across all monitored layers.

Accumulates per-layer detection values (sum, avg, min) from multiple batches
and compares them against saved baselines to produce the final results table.

Each metric is tracked independently using a running accumulation strategy:

* **sum / avg**: tracked via a running grand-sum and element count so the
  mean can be computed without storing all tensors.
* **min**: tracked via running minimum (element-wise ``torch.min`` across
  batches reduced to a scalar).

NaN / Inf handling
------------------
If *any* current or baseline value is NaN or Inf the comparison row is
immediately flagged as ``fault_detected=True`` with ``rel_diff=inf``.
NaN/Inf values in the current run are a strong indicator of numerical
instability caused by a fault (e.g. a weight element set to +/-Inf or a
catastrophic cancellation).

Block ordering
--------------
Results are sorted numerically by block index rather than lexicographically.
Lexicographic ordering would place Block10 before Block2; numeric ordering
preserves the natural block sequence.
"""

from __future__ import annotations

import math
import re

import torch

# Ordered list of metric keys that every layer stores.
# Input-based (from detector weight rows) + Output-based (computed from original output)
METRIC_KEYS: tuple[str, ...] = ("sum_input", "avg_input", "sum", "avg", "min")


# ---------------------------------------------------------------------------
# Sort helpers
# ---------------------------------------------------------------------------


def _sort_key(layer_name: str) -> tuple[int, str]:
    """Produce a sort key that orders by block number numerically.

    Layers without a ``Block<N>`` prefix are sorted to the end.

    Args:
        layer_name: e.g. ``"Block10.qkv"``, ``"Block2.fc1"``.

    Returns:
        ``(block_index, remainder)`` tuple.  Block10 -> ``(10, "qkv")``,
        Block2 -> ``(2, "fc1")``.  Unknown format -> ``(999999, layer_name)``.

    Examples:
        >>> sorted(["Block10.qkv", "Block2.fc1", "Block1.proj"], key=_sort_key)
        ['Block1.proj', 'Block2.fc1', 'Block10.qkv']
    """
    match = re.search(r"Block(\d+)\.(\w+)", layer_name)
    if match:
        return (int(match.group(1)), match.group(2))
    return (999_999, layer_name)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class DetectionTracker:
    """Accumulate detection values across batches and compare against baseline.

    One ``DetectionTracker`` instance lives for the duration of a single
    inference run (faultfree or faulty).  For each layer being monitored it
    maintains per-metric running accumulators so that the final scalar per
    metric can be computed at the end without storing all tensors in memory.

    Accumulation strategy per metric:

    * ``sum`` / ``avg``: running grand-sum + element count -> mean at the end.
    * ``min``: running scalar minimum (updated via ``torch.min``).

    Typical usage inside the inference loop::

        tracker = DetectionTracker()

        for images, labels in batches:
            outputs = model(images)
            for layer_name, neuron in neurons.items():
                vals = neuron.get_detection_values()
                # vals == {"sum": tensor, "avg": tensor, "min": tensor}
                tracker.update(layer_name, vals)

        means = tracker.get_means()
        # means == {"Block0.qkv": {"sum": 1.23, "avg": 0.005, "min": -2.5}, ...}

    Args:
        threshold: Relative difference above which a layer/metric is considered
                   fault-detected (default: 0.1, i.e. 10%).
    """

    def __init__(self, threshold: float = 0.1) -> None:
        self.threshold = threshold

        # Running grand-sums for 'sum' and 'avg' metrics:
        #   _metric_sums[layer_name][metric_key] -> scalar double tensor
        self._metric_sums: dict[str, dict[str, torch.Tensor]] = {}
        # Element counts for computing the mean:
        #   _metric_counts[layer_name][metric_key] -> int
        self._metric_counts: dict[str, dict[str, int]] = {}
        # Running minimum for 'min' metric:
        #   _metric_mins[layer_name] -> scalar double tensor
        self._metric_mins: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def update(self, layer_name: str, values: dict[str, torch.Tensor] | None) -> None:
        """Record detection values for one batch.

        Each tensor is reduced to a scalar so memory usage stays constant
        regardless of batch count.

        Args:
            layer_name: Identifier such as ``"Block3.fc1"``.
            values: Dict with keys from METRIC_KEYS mapping to tensors.
                    If ``None`` or empty, the call is a no-op.
        """
        if not values:
            return

        # --- sum/avg metrics: accumulate via running grand-sum + count ---
        # This includes both input-based (sum_input, avg_input) and output-based (sum, avg)
        for metric in ("sum_input", "avg_input", "sum", "avg"):
            tensor = values.get(metric)
            if tensor is None:
                continue

            batch_sum = tensor.double().sum()
            batch_count = tensor.numel()

            if layer_name not in self._metric_sums:
                self._metric_sums[layer_name] = {}
                self._metric_counts[layer_name] = {}

            if metric not in self._metric_sums[layer_name]:
                self._metric_sums[layer_name][metric] = batch_sum.detach()
                self._metric_counts[layer_name][metric] = batch_count
            else:
                self._metric_sums[layer_name][metric] = (
                    self._metric_sums[layer_name][metric] + batch_sum.detach()
                )
                self._metric_counts[layer_name][metric] += batch_count

        # --- min: running scalar minimum ---
        min_tensor = values.get("min")
        if min_tensor is not None:
            batch_min = min_tensor.double().min()
            if layer_name not in self._metric_mins:
                self._metric_mins[layer_name] = batch_min.detach()
            else:
                self._metric_mins[layer_name] = torch.min(
                    self._metric_mins[layer_name], batch_min.detach()
                )

    def reset(self) -> None:
        """Clear all accumulated state (call between runs)."""
        self._metric_sums.clear()
        self._metric_counts.clear()
        self._metric_mins.clear()

    # ------------------------------------------------------------------
    # Derived statistics
    # ------------------------------------------------------------------

    def get_means(self) -> dict[str, dict[str, float]]:
        """Return the mean (or min) scalar per layer per metric.

        For sum/avg metrics the grand-sum is divided by the total element count
        to produce a mean. For ``min`` the running minimum is returned directly.

        Returns:
            Nested dict with all metric keys from METRIC_KEYS.
        """
        result: dict[str, dict[str, float]] = {}

        # Collect all layer names from both accumulators.
        all_layers = sorted(
            set(self._metric_sums.keys()) | set(self._metric_mins.keys())
        )

        for layer_name in all_layers:
            entry: dict[str, float] = {}

            # sum/avg metrics via mean of the grand-sum
            for metric in ("sum_input", "avg_input", "sum", "avg"):
                sums = self._metric_sums.get(layer_name, {})
                counts = self._metric_counts.get(layer_name, {})
                if metric in sums and counts.get(metric, 0) > 0:
                    entry[metric] = (sums[metric] / counts[metric]).item()

            # min as running minimum
            if layer_name in self._metric_mins:
                entry["min"] = self._metric_mins[layer_name].item()

            if entry:
                result[layer_name] = entry

        return result

    @property
    def layer_names(self) -> list[str]:
        """Sorted list of layers that have been updated at least once."""
        return sorted(
            set(self._metric_sums.keys()) | set(self._metric_mins.keys())
        )

    # ------------------------------------------------------------------
    # Comparison against baseline
    # ------------------------------------------------------------------

    def compare(
        self,
        baseline_means: dict[str, dict[str, float]],
    ) -> list[dict]:
        """Compare current means against baseline means for all metrics.

        Each (layer, metric) pair produces one result row.

        NaN / Inf rules applied before the threshold test:

        * If **current_val** is NaN or Inf -> ``fault_detected=True``,
          ``rel_diff=inf``.  A fault can corrupt a weight to +/-Inf or
          produce NaN via 0 * Inf; either case is a definitive fault signal.
        * If **baseline_val** is NaN or Inf -> ``fault_detected=True``,
          ``rel_diff=inf``.  This should not occur in a healthy baseline but
          is handled defensively.

        Results are sorted by block number numerically (Block0, Block1, ...,
        Block9, Block10, ...) rather than lexicographically.

        Args:
            baseline_means: Nested dict of the form
                ``{layer_name: {"sum": float, "avg": float, "min": float}}``
                as returned by :meth:`get_means` on a faultfree run.

        Returns:
            List of result dicts, one per (layer, metric) pair, with keys:

            * ``layer_name``     - e.g. ``"Block0.qkv"``
            * ``metric``         - one of ``"sum"``, ``"avg"``, ``"min"``
            * ``baseline_val``   - float
            * ``current_val``    - float
            * ``abs_diff``       - ``|current - baseline|``
            * ``rel_diff``       - ``abs_diff / (|baseline| + 1e-8)``
            * ``fault_detected`` - bool, True when rel_diff > threshold OR
                                   either value is NaN / Inf
        """
        current_means = self.get_means()
        results: list[dict] = []

        all_layers = sorted(set(baseline_means) | set(current_means))

        for layer_name in all_layers:
            baseline_layer = baseline_means.get(layer_name, {})
            current_layer = current_means.get(layer_name, {})

            for metric in METRIC_KEYS:
                baseline_val = baseline_layer.get(metric, float("nan"))
                current_val = current_layer.get(metric, float("nan"))

                # --- NaN / Inf guard (checked before threshold comparison) ---
                current_bad = math.isnan(current_val) or math.isinf(current_val)
                baseline_bad = math.isnan(baseline_val) or math.isinf(baseline_val)

                if current_bad or baseline_bad:
                    # Any non-finite value is treated as a fault signal.
                    abs_diff = float("inf")
                    rel_diff = float("inf")
                    fault_detected = True
                else:
                    abs_diff = abs(current_val - baseline_val)
                    rel_diff = abs_diff / (abs(baseline_val) + 1e-8)
                    fault_detected = rel_diff > self.threshold

                results.append(
                    {
                        "layer_name": layer_name,
                        "metric": metric,
                        "baseline_val": baseline_val,
                        "current_val": current_val,
                        "abs_diff": abs_diff,
                        "rel_diff": rel_diff,
                        "fault_detected": fault_detected,
                    }
                )

        # Sort by block number numerically, then by layer type within a block.
        results.sort(key=lambda r: _sort_key(r["layer_name"]))

        return results

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    @staticmethod
    def print_results(results: list[dict], threshold: float) -> None:
        """Print a formatted table of detection results.

        Columns: Layer | Metric | Baseline | Current | Abs Diff | Rel Diff | Status

        Args:
            results: Output of :meth:`compare`.
            threshold: The threshold used for detection (shown in the header).
        """
        if not results:
            print("No detection results to display.")
            return

        # Dynamic column width for layer names
        w_layer = max(len(r["layer_name"]) for r in results)
        w_layer = max(w_layer, len("Layer"))

        sep_char = "="
        divider_char = "-"
        col_metric = 6   # "Metric"
        col_val = 13     # "Baseline" / "Current"
        col_diff = 12    # "Abs Diff" / "Rel Diff"
        col_status = 8   # "Status"

        header = (
            f"{'Layer':<{w_layer}} | "
            f"{'Metric':<{col_metric}} | "
            f"{'Baseline':>{col_val}} | "
            f"{'Current':>{col_val}} | "
            f"{'Abs Diff':>{col_diff}} | "
            f"{'Rel Diff':>{col_diff}} | "
            f"{'Status':<{col_status}}"
        )
        total_width = len(header)
        sep = sep_char * total_width
        divider = divider_char * total_width

        print(f"\n{sep}")
        print(f"Fault Detection Results (threshold={threshold:.3f})")
        print(sep)
        print(header)
        print(divider)

        n_flagged_metrics = 0
        flagged_layers: set[str] = set()

        for r in results:
            layer = r["layer_name"]
            metric = r["metric"]
            bval = r["baseline_val"]
            cval = r["current_val"]
            adiff = r["abs_diff"]
            rdiff = r["rel_diff"]
            detected = r["fault_detected"]

            def _fmt(v: float, w: int) -> str:
                if math.isnan(v):
                    return f"{'NaN':>{w}}"
                if math.isinf(v):
                    return f"{'Inf':>{w}}"
                return f"{v:>{w}.4f}"

            status = "FAULT" if detected else "OK"
            if detected:
                n_flagged_metrics += 1
                flagged_layers.add(layer)

            print(
                f"{layer:<{w_layer}} | "
                f"{metric:<{col_metric}} | "
                f"{_fmt(bval, col_val)} | "
                f"{_fmt(cval, col_val)} | "
                f"{_fmt(adiff, col_diff)} | "
                f"{_fmt(rdiff, col_diff)} | "
                f"{status:<{col_status}}"
            )

        print(divider)
        n_total = len(results)
        n_layers_flagged = len(flagged_layers)
        n_total_layers = len({r["layer_name"] for r in results})
        print(
            f"Summary: {n_flagged_metrics}/{n_total} metrics flagged "
            f"across {n_layers_flagged}/{n_total_layers} layers\n"
        )
