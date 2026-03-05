import torch.nn as nn
from src.core.library.layers import get_linear_layers
from src.core.fault_detection.neuron import (
    NeuroChecker,
    wrap_layer,
    unwrap_layer,
)
from src.core.fault_detection.save_and_load import (
    get_or_compute_checker_weights,
    NeuroWeights,
)


class FaultDetector:
    """Manages checker neurons across linear layers."""

    def __init__(self, model: nn.Module, threshold: float = 1e-6):
        """
        Args:
            model: The model to protect
            threshold: Detection threshold for flagging faults
        """
        self.model = model
        self.threshold = threshold
        self.wrapped: dict[str, NeuroChecker] = {}
        self.preloaded_weights: dict | None = None

    def load_weights(self, model_key: str, force_recompute: bool = False):
        """Load or compute checker weights from disk.

        Args:
            model_key: Model identifier (e.g., "vit_tiny")
            force_recompute: If True, recompute even if cached on disk
        """
        self.preloaded_weights = get_or_compute_checker_weights(
            self.model, model_key, force_recompute
        )
        print(f"Loaded neuro weights for {len(self.preloaded_weights)} layers")

    def apply(self, layer_filter: str = "all") -> list[str]:
        """Wrap layers with checker neurons."""
        layers = get_linear_layers(self.model)
        names = []

        for name in layers.keys():
            if layer_filter != "all" and layer_filter not in name:
                continue

            layer_weights = None
            if self.preloaded_weights and name in self.preloaded_weights:
                w = self.preloaded_weights[name]
                if isinstance(w, NeuroWeights):
                    layer_weights = {
                        "checker_row": w.checker_row,
                        "checker_bias": w.checker_bias,
                    }

            self.wrapped[name] = wrap_layer(self.model, name, layer_weights)
            names.append(name)

        source = "preloaded" if self.preloaded_weights else "computed"
        print(f"Applied neuro checkers to {len(names)} layers ({source} weights)")
        return names

    def remove(self):
        """Remove all wrappers."""
        for name in list(self.wrapped.keys()):
            unwrap_layer(self.model, name)
        self.wrapped.clear()

    def print_values(self, relative: bool = True):
        """Print checker neuron value vs expected value for each layer.

        Args:
            relative: If True, use relative difference (diff/expected) for threshold
        """
        print()
        header = f"{'Layer':<30} {'Checker':>12} {'Expected':>12} {'Diff':>12}"
        if relative:
            header += f" {'RelDiff':>12}"
        print(header)
        print("-" * (70 + (14 if relative else 0)))

        for name, wrapper in self.wrapped.items():
            c = wrapper.checker_val
            e = wrapper.expected_val
            diff = c - e

            rel_diff = abs(diff / e) if abs(e) > 1e-10 else abs(diff)

            line = f"{name:<30} {c:>12.6f} {e:>12.6f} {diff:>12.6f}"
            if relative:
                line += f" {rel_diff:>12.2e}"

            is_fault = (
                rel_diff > self.threshold if relative else abs(diff) > self.threshold
            )
            if is_fault:
                line += "  <- Fault"

            print(line)

    def get_fault_locations(self) -> dict[str, dict]:
        """Get fault info for each layer.

        Returns:
            Dict mapping layer name to fault info
        """
        results = {}
        for name, wrapper in self.wrapped.items():
            diff = wrapper.checker_val - wrapper.expected_val
            results[name] = {
                "diff": diff,
                "is_fault": abs(diff) > self.threshold,
            }
        return results
