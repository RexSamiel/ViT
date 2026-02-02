"""
Activation Analyzer for Vision Transformers.

Captures activation values from transformer layers using forward hooks,
samples a configurable percentage of values, and computes integer-binned
distributions for visualization.

Workflow:
1. Register hooks on all modules
2. Run inference - hooks capture activations
3. Sample percentage of values from each layer (random)
4. Sort and count values by integer bins
5. Output JSON with layer info, min/max, and distributions
"""

import torch
import torch.nn as nn
import numpy as np
import re
from typing import Any
from collections import defaultdict


# =============================================================================
# EXCLUSION PATTERNS - Layers to exclude from distributions
# =============================================================================
EXCLUDE_PATTERNS: list[str] = [
    # Uncomment to exclude:
    # "attn_drop",    # Attention weights (0-1 probabilities)
    # "drop",         # Dropout layers
]


class ActivationAnalyzer:
    """Simple activation analyzer with percentage-based sampling."""

    def __init__(self, sampling_percent: float = 1.0):
        """
        Args:
            sampling_percent: Percentage of activations to sample per layer (0.01-100)
        """
        self.sampling_percent = max(0.01, min(100.0, sampling_percent))
        self.reset()

    def reset(self) -> None:
        """Reset all collected data."""
        # Per-layer data: {layer_idx: {name, component, min, max, values}}
        self.layer_data: dict[int, dict] = {}

        # Global stats per component
        self.global_stats = {
            "input": {"min": float("inf"), "max": float("-inf")},
            "output": {"min": float("inf"), "max": float("-inf")},
            "block": {"min": float("inf"), "max": float("-inf")},
            "mha": {"min": float("inf"), "max": float("-inf")},
            "mlp": {"min": float("inf"), "max": float("-inf")},
        }

        # All sampled values per component (for histogram)
        self._values: dict[str, list] = {
            "input": [],
            "output": [],
            "block": [],
            "mha": [],
            "mlp": [],
        }

        # Activation counts: total found vs sampled
        self._activation_counts: dict[str, dict] = {
            comp: {"total": 0, "sampled": 0}
            for comp in ["input", "output", "block", "mha", "mlp"]
        }

        self._hooks: list = []
        self._name_to_idx: dict[str, int] = {}
        self._max_block_seen: int = -1
        self.total_samples = 0
        self.total_batches = 0
        self.num_blocks = 0

    def _extract_block_idx(self, name: str) -> int | None:
        """Extract block index from module name."""
        name_lower = name.lower()

        # Swin-style: layers.X.blocks.Y
        swin_match = re.search(r"layers\.(\d+)\.blocks\.(\d+)", name_lower)
        if swin_match:
            return int(swin_match.group(1)) * 100 + int(swin_match.group(2))

        # Standard ViT: blocks.X
        for pattern in [r"blocks\.(\d+)", r"blocks_(\d+)", r"layer_(\d+)"]:
            match = re.search(pattern, name_lower)
            if match:
                return int(match.group(1))
        return None

    def _classify_component(self, name: str, block_idx: int | None) -> str:
        """Classify module into: input, output, block, mha, or mlp."""
        name_lower = name.lower()

        if block_idx is None:
            return "input" if self._max_block_seen < 0 else "output"

        if "attn" in name_lower:
            return "mha"
        if "mlp" in name_lower:
            return "mlp"
        return "block"

    def _is_excluded(self, name: str, module_type: str) -> bool:
        """Check if layer should be excluded from distributions."""
        name_lower = name.lower()
        type_lower = module_type.lower()
        return any(
            p.lower() in name_lower or p.lower() in type_lower for p in EXCLUDE_PATTERNS
        )

    def _record_activation(
        self,
        tensor: torch.Tensor,
        name: str,
        module_type: str,
        component: str,
        block_idx: int | None,
    ) -> None:
        """Record activation values from a layer."""
        num_elements = tensor.numel()

        # Assign stable index to layer
        if name not in self._name_to_idx:
            self._name_to_idx[name] = len(self._name_to_idx)
        idx = self._name_to_idx[name]

        excluded = self._is_excluded(name, module_type)

        # Update or create layer data
        if idx in self.layer_data:
            data = self.layer_data[idx]
            # Use aminmax for efficiency (single kernel call)
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

        # Track total activations for this component
        self._activation_counts[component]["total"] += num_elements

        # Update global stats
        data = self.layer_data[idx]
        self.global_stats[component]["min"] = min(
            self.global_stats[component]["min"], data["min"]
        )
        self.global_stats[component]["max"] = max(
            self.global_stats[component]["max"], data["max"]
        )

        # Sample percentage of values randomly
        sample_count = max(1, int(num_elements * self.sampling_percent / 100))

        # Track sampled count
        actual_sampled = min(sample_count, num_elements)
        self._activation_counts[component]["sampled"] += actual_sampled
        self.layer_data[idx]["sampled_activations"] += actual_sampled

        # Efficient sampling: reshape handles non-contiguous tensors
        flat = tensor.detach().reshape(-1)
        if sample_count < num_elements:
            # Use randint for faster index generation than randperm
            indices = torch.randint(
                0, num_elements, (sample_count,), device=flat.device
            )
            sampled = flat[indices].float().cpu().numpy()
        else:
            sampled = flat.float().cpu().numpy()

        # Extend values list (numpy array, not list conversion)
        self._values[component].append(sampled)

    def register_hooks(self, model: nn.Module) -> int:
        """Register forward hooks on all modules."""
        self.remove_hooks()
        self.reset()

        # Count blocks
        for name, _ in model.named_modules():
            idx = self._extract_block_idx(name)
            if idx is not None:
                self.num_blocks = max(self.num_blocks, idx + 1)

        # Register hooks
        count = 0
        for name, module in model.named_modules():
            if name == "":  # Skip root
                continue

            block_idx = self._extract_block_idx(name)

            def make_hook(n: str, b: int | None):
                def hook(mod: nn.Module, inp: Any, out: Any) -> None:
                    if isinstance(out, tuple):
                        out = out[0]
                    if not isinstance(out, torch.Tensor) or out.numel() <= 1:
                        return

                    if b is not None and b > self._max_block_seen:
                        self._max_block_seen = b

                    component = self._classify_component(n, b)
                    self._record_activation(
                        out, n, mod.__class__.__name__, component, b
                    )

                return hook

            handle = module.register_forward_hook(make_hook(name, block_idx))
            self._hooks.append(handle)
            count += 1

        pct_str = (
            f"{self.sampling_percent:.2f}%"
            if self.sampling_percent < 1
            else f"{self.sampling_percent:.1f}%"
        )
        print(f"Registered hooks on {count} modules (sampling {pct_str})")
        return count

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def run_inference(self, model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """Run inference (hooks will capture activations)."""
        self._max_block_seen = -1
        return model(inputs)

    def update(self, batch_size: int) -> None:
        """Update counters after a batch."""
        self.total_samples += batch_size
        self.total_batches += 1
        self._max_block_seen = -1

    def _compute_histogram(self, component: str) -> dict:
        """Compute integer-binned histogram for a component."""
        values = self._values[component]
        if not values:
            return {}

        # Concatenate all numpy arrays efficiently
        arr = np.concatenate(values) if len(values) > 1 else values[0]
        arr = arr[np.isfinite(arr)]

        if arr.size == 0:
            return {}

        # Get range
        data_min = float(arr.min())
        data_max = float(arr.max())

        # Integer binning: floor each value to get integer bin
        int_min = int(np.floor(data_min))
        int_max = int(np.ceil(data_max))

        # Create bins with width 1
        bins = np.arange(int_min, int_max + 2, 1.0)  # +2 for right edge
        counts, edges = np.histogram(arr, bins=bins)

        # Bin centers are the integers
        centers = (edges[:-1] + edges[1:]) / 2

        # Get activation counts for this component
        counts_info = self._activation_counts[component]

        return {
            "bin_edges": edges.tolist(),
            "bin_centers": centers.tolist(),
            "counts": counts.tolist(),
            "total_sampled": len(arr),
            "total_activations": counts_info["total"],
            "data_range": [data_min, data_max],
        }

    def get_results(self) -> dict:
        """Generate results dictionary for JSON output."""
        # Prepare layer data
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

        # Compute histograms
        distributions = {}
        for comp in ["input", "output", "block", "mha", "mlp"]:
            hist = self._compute_histogram(comp)
            if hist:
                distributions[comp] = hist

        # Count layers per component
        comp_counts = defaultdict(int)
        for data in self.layer_data.values():
            if not data.get("excluded"):
                comp_counts[data["component"]] += 1

        # Calculate total activations across all components
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

    def _format_count(self, n: int) -> str:
        """Format large numbers with K/M/B suffixes."""
        if n >= 1_000_000_000:
            return f"{n / 1_000_000_000:.2f}B"
        if n >= 1_000_000:
            return f"{n / 1_000_000:.2f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}K"
        return str(n)

    def print_results(self) -> None:
        """Print summary to terminal."""
        print("\nActivation Analysis Results:")
        print("-" * 50)

        counts = defaultdict(int)
        for d in self.layer_data.values():
            if not d.get("excluded"):
                counts[d["component"]] += 1

        print(f"Total layers: {len(self.layer_data)}")
        print(f"Sampling: {self.sampling_percent:.2f}%")

        # Calculate totals
        total_activations = sum(c["total"] for c in self._activation_counts.values())
        total_sampled = sum(c["sampled"] for c in self._activation_counts.values())

        print(f"\nActivation counts:")
        print(f"  Total found:   {self._format_count(total_activations):>12}")
        print(f"  Total sampled: {self._format_count(total_sampled):>12}")
        if total_activations > 0:
            ratio = total_sampled / total_activations * 100
            print(f"  Actual ratio:  {ratio:>11.4f}%")

        print(f"\nLayers and activations by component:")
        print(f"  {'Component':<8} {'Layers':>7} {'Total Acts':>12} {'Sampled':>12}")
        print(f"  {'-' * 43}")
        for comp in ["input", "block", "mha", "mlp", "output"]:
            c = self._activation_counts[comp]
            print(
                f"  {comp.upper():<8} {counts[comp]:>7} {self._format_count(c['total']):>12} {self._format_count(c['sampled']):>12}"
            )

        print("\nValue ranges:")
        for comp in ["input", "mha", "mlp", "block", "output"]:
            s = self.global_stats[comp]
            if s["min"] != float("inf"):
                print(f"  {comp.upper():6}: [{s['min']:>10.2f}, {s['max']:>10.2f}]")

        print(f"\nSamples processed: {self.total_samples}")

    def print_layer_ranges(self) -> None:
        """Print per-layer activation ranges."""
        print("\nPer-Layer Activation Ranges:")
        print("=" * 120)
        print(
            f"{'Idx':<5} {'Blk':<4} {'Comp':<6} {'Min':>10} {'Max':>10} {'Total':>10} {'Sampled':>10} {'Type':<15} {'Name'}"
        )
        print("-" * 120)

        for idx in sorted(self.layer_data.keys()):
            d = self.layer_data[idx]
            name = d["name"][-40:] if len(d["name"]) > 40 else d["name"]
            blk = str(d["block_idx"]) if d["block_idx"] is not None else "-"
            status = " [EXCLUDED]" if d.get("excluded") else ""
            total = self._format_count(d.get("total_activations", 0))
            sampled = self._format_count(d.get("sampled_activations", 0))
            print(
                f"{idx:<5} {blk:<4} {d['component']:<6} {d['min']:>10.2f} {d['max']:>10.2f} {total:>10} {sampled:>10} {d['op_type']:<15} {name}{status}"
            )
