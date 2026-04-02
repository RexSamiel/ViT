"""Activation analysis for ViT models."""

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
from collections import defaultdict


class ActivationAnalyzer:
    """Analyzes activation ranges and distributions during inference.

    Hooks into model layers to capture min/max values and optionally
    builds histograms of activation distributions.

    Example:
        analyzer = ActivationAnalyzer(model)
        analyzer.run(num_batches=10)
        analyzer.save("results/activations_vit_tiny.json")
        analyzer.remove()
    """

    def __init__(self, model, include_histogram: bool = True, histogram_bins: int = 1000):
        """
        Args:
            model: Model instance (vit_fault.Model) or nn.Module
            include_histogram: Build activation distributions
            histogram_bins: Number of bins for histograms
        """
        if hasattr(model, "net"):
            self.net = model.net
            self.model = model
            self.model_name = model.name
        else:
            self.net = model
            self.model = None
            self.model_name = "unknown"

        self.include_histogram = include_histogram
        self.histogram_bins = histogram_bins
        self.hooks = []
        self.layer_data = {}
        self.distributions = defaultdict(lambda: {
            "counts": None,
            "bin_edges": None,
            "total_values_seen": 0,
            "data_range": [float("inf"), float("-inf")],
        })
        self._setup_hooks()

    def _classify_layer(self, name: str) -> tuple[str, Optional[int]]:
        """Classify layer into component type and block index."""
        if "blocks." in name:
            parts = name.split(".")
            try:
                block_idx = int(parts[parts.index("blocks") + 1])
            except (ValueError, IndexError):
                block_idx = None

            if ".attn." in name:
                return "mha", block_idx
            elif ".mlp." in name:
                return "mlp", block_idx
            else:
                return "block", block_idx
        elif "patch_embed" in name or "cls_token" in name or "pos_embed" in name:
            return "input", None
        elif "head" in name or "fc_norm" in name or "norm." in name:
            return "output", None
        else:
            return "other", None

    def _setup_hooks(self):
        """Register forward hooks on all layers."""
        layer_idx = 0

        def make_hook(name, idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                if not isinstance(output, torch.Tensor):
                    return

                with torch.no_grad():
                    min_val = output.min().item()
                    max_val = output.max().item()

                    # Update layer data
                    if idx not in self.layer_data:
                        component, block_idx = self._classify_layer(name)
                        self.layer_data[idx] = {
                            "min": min_val,
                            "max": max_val,
                            "component": component,
                            "name": name,
                            "op_type": module.__class__.__name__,
                            "block_idx": block_idx,
                            "shape": list(output.shape),
                        }
                    else:
                        self.layer_data[idx]["min"] = min(self.layer_data[idx]["min"], min_val)
                        self.layer_data[idx]["max"] = max(self.layer_data[idx]["max"], max_val)

                    # Update histogram
                    if self.include_histogram:
                        component = self.layer_data[idx]["component"]
                        self._update_histogram(component, output)
            return hook

        for name, module in self.net.named_modules():
            if name:  # Skip root module
                hook = module.register_forward_hook(make_hook(name, layer_idx))
                self.hooks.append(hook)
                layer_idx += 1

    def _update_histogram(self, component: str, tensor: torch.Tensor):
        """Update histogram for a component."""
        data = tensor.detach().float().flatten().cpu()

        # Update data range
        t_min, t_max = data.min().item(), data.max().item()
        dist = self.distributions[component]
        dist["data_range"][0] = min(dist["data_range"][0], t_min)
        dist["data_range"][1] = max(dist["data_range"][1], t_max)
        dist["total_values_seen"] += data.numel()

        # Simple histogram accumulation
        if dist["counts"] is None:
            # Initialize with reasonable range
            hist_min = max(-2000, t_min)
            hist_max = min(2000, t_max)
            dist["bin_edges"] = torch.linspace(hist_min, hist_max, self.histogram_bins + 1)
            dist["counts"] = torch.zeros(self.histogram_bins)

        # Accumulate counts
        counts = torch.histc(data, bins=self.histogram_bins,
                            min=dist["bin_edges"][0].item(),
                            max=dist["bin_edges"][-1].item())
        dist["counts"] += counts

    def run(self, num_batches: Optional[int] = None) -> dict:
        """Run analysis on model batches.

        Args:
            num_batches: Number of batches to process (None = all available)

        Returns:
            Analysis results dictionary
        """
        if self.model is None:
            raise ValueError("Model instance required for batch iteration")

        batches = self.model.get_batches()
        if num_batches:
            batches = batches[:num_batches]

        total_samples = 0
        for i, (images, _) in enumerate(batches):
            print(f"  Analyzing batch {i + 1}/{len(batches)}", end="\r")
            with torch.inference_mode():
                _ = self.net(images)
            total_samples += images.shape[0]

        print(f"  Analyzed {total_samples} samples across {len(batches)} batches")
        return self.get_results()

    def get_results(self) -> dict:
        """Get analysis results as dictionary."""
        # Compute block aggregated data
        block_agg = self._compute_block_aggregated()

        # Format distributions
        distributions = {}
        for comp, data in self.distributions.items():
            if data["counts"] is not None:
                edges = data["bin_edges"]
                centers = ((edges[:-1] + edges[1:]) / 2).tolist()
                distributions[comp] = {
                    "bin_centers": centers,
                    "counts": data["counts"].tolist(),
                    "total_values_seen": data["total_values_seen"],
                    "data_range": data["data_range"],
                }

        return {
            "layers": {str(k): v for k, v in self.layer_data.items()},
            "block_aggregated": block_agg,
            "distributions": distributions,
            "statistics": {
                "total_layers": len(self.layer_data),
                "num_blocks": len(block_agg),
            },
        }

    def _compute_block_aggregated(self) -> dict:
        """Compute per-block aggregated min/max for each component."""
        block_agg = defaultdict(lambda: {
            "mha": {"min": float("inf"), "max": float("-inf"), "last_layer_idx": -1},
            "mlp": {"min": float("inf"), "max": float("-inf"), "last_layer_idx": -1},
            "block": {"min": float("inf"), "max": float("-inf"), "last_layer_idx": -1},
        })

        for idx, info in self.layer_data.items():
            block_idx = info.get("block_idx")
            if block_idx is None:
                continue

            comp = info["component"]
            if comp not in ["mha", "mlp", "block"]:
                continue

            block_data = block_agg[str(block_idx)]
            block_data[comp]["min"] = min(block_data[comp]["min"], info["min"])
            block_data[comp]["max"] = max(block_data[comp]["max"], info["max"])
            block_data[comp]["last_layer_idx"] = max(block_data[comp]["last_layer_idx"], idx)

            # Also update block envelope
            block_data["block"]["min"] = min(block_data["block"]["min"], info["min"])
            block_data["block"]["max"] = max(block_data["block"]["max"], info["max"])
            block_data["block"]["last_layer_idx"] = max(block_data["block"]["last_layer_idx"], idx)

        # Clean up infinite values
        result = {}
        for block_key, data in block_agg.items():
            result[block_key] = {}
            for comp, comp_data in data.items():
                if comp_data["min"] != float("inf"):
                    result[block_key][comp] = comp_data

        return result

    def save(self, path: str):
        """Save analysis results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        results = self.get_results()
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Saved activation analysis to {path}")

    def remove(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
