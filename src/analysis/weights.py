"""Weight parameter analysis for ViT models."""

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
from collections import defaultdict


class WeightAnalyzer:
    """Analyzes weight parameter ranges and distributions.

    Iterates through model parameters to capture min/max values
    and builds histograms of weight distributions.

    Example:
        analyzer = WeightAnalyzer(model)
        results = analyzer.run()
        analyzer.save("results/weights_vit_tiny.json")
    """

    def __init__(self, model, histogram_bins: int = 1000):
        """
        Args:
            model: Model instance (vit_fault.Model) or nn.Module
            histogram_bins: Number of bins for histograms
        """
        if hasattr(model, "net"):
            self.net = model.net
            self.model_name = model.name
        else:
            self.net = model
            self.model_name = "unknown"

        self.histogram_bins = histogram_bins
        self.param_data = {}
        self.distributions = defaultdict(lambda: {
            "counts": None,
            "bin_edges": None,
            "total_values_seen": 0,
            "data_range": [float("inf"), float("-inf")],
        })

    def _classify_param(self, name: str) -> tuple[str, Optional[int]]:
        """Classify parameter into component type and block index."""
        if "blocks." in name:
            parts = name.split(".")
            try:
                block_idx = int(parts[parts.index("blocks") + 1])
            except (ValueError, IndexError):
                block_idx = None

            if ".attn." in name:
                return "attention", block_idx
            elif ".mlp." in name:
                return "mlp", block_idx
            elif ".norm" in name:
                return "norm", block_idx
            else:
                return "other", block_idx
        elif "patch_embed" in name:
            return "patch_embed", None
        elif "head" in name or "fc_norm" in name:
            return "classifier", None
        elif "cls_token" in name or "pos_embed" in name:
            return "embedding", None
        else:
            return "other", None

    def _update_histogram(self, component: str, tensor: torch.Tensor):
        """Update histogram for a component."""
        data = tensor.detach().float().flatten().cpu()

        # Update data range
        t_min, t_max = data.min().item(), data.max().item()
        dist = self.distributions[component]
        dist["data_range"][0] = min(dist["data_range"][0], t_min)
        dist["data_range"][1] = max(dist["data_range"][1], t_max)
        dist["total_values_seen"] += data.numel()

        # Histogram accumulation
        if dist["counts"] is None:
            # Weight ranges are typically smaller than activations
            hist_min = max(-10, t_min)
            hist_max = min(10, t_max)
            dist["bin_edges"] = torch.linspace(hist_min, hist_max, self.histogram_bins + 1)
            dist["counts"] = torch.zeros(self.histogram_bins)

        counts = torch.histc(data, bins=self.histogram_bins,
                            min=dist["bin_edges"][0].item(),
                            max=dist["bin_edges"][-1].item())
        dist["counts"] += counts

    def run(self) -> dict:
        """Run weight analysis.

        Returns:
            Analysis results dictionary
        """
        print(f"Analyzing weights for {self.model_name}...")
        param_idx = 0
        total_params = 0

        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue

            with torch.no_grad():
                min_val = param.min().item()
                max_val = param.max().item()
                num_params = param.numel()

            component, block_idx = self._classify_param(name)

            self.param_data[param_idx] = {
                "name": name,
                "component": component,
                "block_idx": block_idx,
                "shape": list(param.shape),
                "num_params": num_params,
                "min": min_val,
                "max": max_val,
                "mean": param.mean().item(),
                "std": param.std().item(),
            }

            self._update_histogram(component, param)
            total_params += num_params
            param_idx += 1

        print(f"  Analyzed {param_idx} parameters ({total_params:,} total values)")
        return self.get_results()

    def get_results(self) -> dict:
        """Get analysis results as dictionary."""
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

        # Compute statistics per component
        component_stats = defaultdict(lambda: {"count": 0, "min": float("inf"), "max": float("-inf")})
        num_blocks = 0
        for idx, info in self.param_data.items():
            comp = info["component"]
            component_stats[comp]["count"] += 1
            component_stats[comp]["min"] = min(component_stats[comp]["min"], info["min"])
            component_stats[comp]["max"] = max(component_stats[comp]["max"], info["max"])
            if info["block_idx"] is not None:
                num_blocks = max(num_blocks, info["block_idx"] + 1)

        return {
            "parameters": {str(k): v for k, v in self.param_data.items()},
            "distributions": distributions,
            "component_stats": dict(component_stats),
            "statistics": {
                "total_parameters": len(self.param_data),
                "total_values": sum(p["num_params"] for p in self.param_data.values()),
                "num_blocks": num_blocks,
            },
        }

    def save(self, path: str):
        """Save analysis results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        results = self.get_results()
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Saved weight analysis to {path}")
