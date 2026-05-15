"""Baseline timer — wraps nn.Linear layers with no detection logic.

Use as the reference measurement to compare against CheckOne and Checksum:

    python -m cli -m vit_tiny -w 3 -r 20 hr --detect all --method baseline  --time
    python -m cli -m vit_tiny -w 3 -r 20 hr --detect all --method checkone  --time
    python -m cli -m vit_tiny -w 3 -r 20 hr --detect all --method checksum  --time
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.layers import unwrap_layers, wrap_layers


class _Wrapper(nn.Module):
    """Pass-through wrapper — identical forward to the original layer."""

    def __init__(self, original: nn.Linear, name: str):
        super().__init__()
        self.original = original
        self.name = name

    @property
    def weight(self):
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.original.weight, self.original.bias)


class Baseline:
    """Timer-only detector with the same interface as CheckOne and Checksum.

    Wraps the same linear layers with a pass-through forward so batch-level
    timing reflects the true cost of unmodified inference through wrapped layers.
    """

    name = "baseline"

    def __init__(self, model, layers: str = "all", correction: str | None = None):
        if hasattr(model, "net"):
            self.model = model.net
            self.model_name = model.name
        else:
            self.model = model
            self.model_name = "unknown"

        self.wrapped = wrap_layers(self.model, _Wrapper, layers)
        print(f"[{self.name}] Wrapped {len(self.wrapped)} layers")

    def load(self, path=None, verbose: bool = False) -> bool:
        return True

    def save(
        self, path=None, include_weights: bool = False, save_calibration: bool = True
    ):
        pass

    def reset(self):
        pass

    def get_faults(self) -> list:
        return []

    def get_weight_faults(self) -> list:
        return []

    def check(self) -> dict[str, bool]:
        return {name: False for name in self.wrapped}

    def remove(self):
        unwrap_layers(self.model, self.wrapped)
        self.wrapped.clear()

    def print_results(self):
        print(f"FAULT DETECTION ({self.name}) — pass-through, no detection")

    def print_summary(self, layer_stats: dict):
        pass

    def print_timing_summary(self, all_runs: list[dict]):
        """Print timing: total and per-sample mean ± std across runs."""
        timed = [r for r in all_runs if r.get("times_ms")]
        if not timed:
            return
        per_run_ms = [sum(r["times_ms"]) for r in timed]
        per_run_mpb = [ms / len(r["times_ms"]) for ms, r in zip(per_run_ms, timed)]
        total_ms = sum(per_run_ms)
        avg_mpb = sum(per_run_mpb) / len(per_run_mpb)
        std_mpb = (
            (sum((v - avg_mpb) ** 2 for v in per_run_mpb) / (len(per_run_mpb) - 1))
            ** 0.5
            if len(per_run_mpb) > 1
            else 0.0
        )
        print(
            f"\nDetection Timing — {self.name} ({len(timed)} runs):\n"
            f"  Total:         {total_ms:.1f} ms  ({total_ms / 1000:.2f} s)\n"
            f"  Avg per batch: {avg_mpb:.4f} ms  ±{std_mpb:.4f} ms std"
        )

    def get_values(self) -> dict:
        return {
            "method": self.name,
            "layers_checked": len(self.wrapped),
            "weight_faults": 0,
            "faults": [],
        }
