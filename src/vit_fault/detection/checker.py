"""NeuroChecker - Mean-based checker neuron for fault detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuroChecker(nn.Module):
    """Wraps a Linear layer with a mean-based checker neuron.

    The checker neuron computes the mean of weight columns and uses it
    to verify the output. If weights are corrupted, the checker output
    will differ from the expected value.
    """

    def __init__(
        self,
        original: nn.Linear,
        checker_row: torch.Tensor | None = None,
        checker_bias: torch.Tensor | None = None,
    ):
        """
        Args:
            original: The linear layer to wrap
            checker_row: Precomputed checker weights (mean of columns)
            checker_bias: Precomputed checker bias (mean of bias)
        """
        super().__init__()
        self.original = original
        self.out_features = original.out_features

        if checker_row is not None:
            self.checker_row = (
                checker_row.unsqueeze(0) if checker_row.dim() == 1 else checker_row
            )
        else:
            self.checker_row = original.weight.data.mean(dim=0, keepdim=True).clone()

        if checker_bias is not None:
            self.checker_bias = (
                checker_bias.unsqueeze(0) if checker_bias.dim() == 0 else checker_bias
            )
        elif original.bias is not None:
            self.checker_bias = original.bias.data.mean().unsqueeze(0).clone()
        else:
            self.checker_bias = None

        self.checker_val = 0.0
        self.expected_val = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with checker neuron computation.

        Returns the original output (checker neuron is internal only).
        """
        # Extended weight: original rows + checker row
        extended_weight = torch.vstack([self.original.weight, self.checker_row])

        if self.original.bias is not None and self.checker_bias is not None:
            extended_bias = torch.cat([self.original.bias, self.checker_bias])
        else:
            extended_bias = None

        # Single forward pass with extended weights
        extended_out = F.linear(x, extended_weight, extended_bias)

        # Split output
        out = extended_out[..., : self.out_features]
        checker_out = extended_out[..., self.out_features]

        # Store values for fault detection
        self.checker_val = checker_out.mean().item()
        self.expected_val = out.mean(dim=-1).mean().item()

        return out

    @property
    def diff(self) -> float:
        """Difference between checker and expected values."""
        return self.checker_val - self.expected_val

    @property
    def rel_diff(self) -> float:
        """Relative difference (normalized by expected value)."""
        if abs(self.expected_val) > 1e-10:
            return abs(self.diff / self.expected_val)
        return abs(self.diff)
