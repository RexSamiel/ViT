"""Detector - Manages checker neurons across model layers."""

import torch
from pathlib import Path
from dataclasses import dataclass

from vit_fault.detection.checker import NeuroChecker
from vit_fault.core.layers import get_linear_layers, filter_layers, set_layer, get_layer


WEIGHTS_DIR = Path("data/weights")


@dataclass
class CheckerWeights:
    """Precomputed checker weights for a layer."""

    checker_row: torch.Tensor
    checker_bias: torch.Tensor | None


class Detector:
    """Manages fault detection across linear layers.

    Example:
        detector = Detector(model, layers="fc1", threshold=0.1)
        # Run inference...
        faults = detector.check()
        detector.print_results()
        detector.remove()
    """

    def __init__(
        self,
        model,
        layers: str = "all",
        threshold: float = 0.1,
        load_weights: bool = True,
    ):
        """
        Args:
            model: Model instance (vit_fault.Model) or nn.Module
            layers: Layer filter ("all", "fc1", "fc2", "qkv", "proj")
            threshold: Relative difference threshold for fault detection
            load_weights: Load precomputed weights if available
        """
        # Handle both Model wrapper and raw nn.Module
        if hasattr(model, "net"):
            self.model = model.net
            self.model_name = model.name
        else:
            self.model = model
            self.model_name = "unknown"

        self.layer_filter = layers
        self.threshold = threshold
        self.wrapped: dict[str, NeuroChecker] = {}
        self._weights: dict[str, CheckerWeights] | None = None

        # Load weights and wrap layers
        if load_weights:
            self._load_weights()
        self._wrap_layers()

    def _load_weights(self):
        """Load precomputed checker weights from disk."""
        path = WEIGHTS_DIR / f"neuro_{self.model_name}.pt"
        if path.exists():
            data = torch.load(path, weights_only=True)
            self._weights = {
                name: CheckerWeights(d["checker_row"], d["checker_bias"])
                for name, d in data.items()
            }
            print(f"Loaded checker weights for {len(self._weights)} layers")

    def _save_weights(self):
        """Save computed checker weights to disk."""
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        path = WEIGHTS_DIR / f"neuro_{self.model_name}.pt"

        data = {}
        for name, checker in self.wrapped.items():
            data[name] = {
                "checker_row": checker.checker_row.squeeze(0),
                "checker_bias": checker.checker_bias.squeeze(0)
                if checker.checker_bias is not None
                else None,
            }
        torch.save(data, path)
        print(f"Saved checker weights to {path}")

    def _wrap_layers(self):
        """Wrap matching layers with checker neurons."""
        all_layers = get_linear_layers(self.model)
        target_layers = filter_layers(all_layers, self.layer_filter)

        for name in target_layers:
            original = get_layer(self.model, name)

            weights = self._weights.get(name) if self._weights else None
            if weights:
                checker = NeuroChecker(
                    original,
                    checker_row=weights.checker_row,
                    checker_bias=weights.checker_bias,
                )
            else:
                checker = NeuroChecker(original)

            set_layer(self.model, name, checker)
            self.wrapped[name] = checker

        if not self._weights and self.wrapped:
            self._save_weights()

        print(f"Applied checkers to {len(self.wrapped)} layers")

    def remove(self):
        """Remove all checker wrappers, restoring original layers."""
        for name, checker in self.wrapped.items():
            set_layer(self.model, name, checker.original)
        self.wrapped.clear()

    def check(self) -> dict[str, bool]:
        """Check for faults in all wrapped layers.

        Returns:
            Dict mapping layer names to fault status (True = fault detected)
        """
        return {
            name: checker.rel_diff > self.threshold
            for name, checker in self.wrapped.items()
        }

    @property
    def faults_found(self) -> list[str]:
        """List of layer names where faults were detected."""
        return [name for name, is_fault in self.check().items() if is_fault]

    def get_values(self) -> dict[str, dict]:
        """Get detailed values for all wrapped layers.

        Returns:
            Dict with checker_val, expected_val, diff, rel_diff for each layer
        """
        return {
            name: {
                "checker": checker.checker_val,
                "expected": checker.expected_val,
                "diff": checker.diff,
                "rel_diff": checker.rel_diff,
                "is_fault": checker.rel_diff > self.threshold,
            }
            for name, checker in self.wrapped.items()
        }

    def print_results(self):
        """Print detection results in a formatted table."""
        print()
        print("-" * 75)
        print("FAULT DETECTION RESULTS")
        print("-" * 75)
        print(f"Threshold: {self.threshold:.2e}")
        print()
        print(
            f"{'Layer':<30} {'Checker':>12} {'Expected':>12} {'RelDiff':>12} {'Status':>8}"
        )
        print("-" * 75)

        for name, checker in self.wrapped.items():
            is_fault = checker.rel_diff > self.threshold
            status = "FAULT" if is_fault else "OK"
            status_color = "\033[91m" if is_fault else "\033[92m"  # Red or Green
            reset = "\033[0m"

            print(
                f"{name:<30} "
                f"{checker.checker_val:>12.6f} "
                f"{checker.expected_val:>12.6f} "
                f"{checker.rel_diff:>12.2e} "
                f"{status_color}{status:>8}{reset}"
            )

        faults = self.faults_found
        print("-" * 75)
        if faults:
            print(f"\033[91mFaults detected: {len(faults)}/{len(self.wrapped)}\033[0m")
        else:
            print(
                f"\033[92mNo faults detected ({len(self.wrapped)} layers checked)\033[0m"
            )
        print("=" * 75)
